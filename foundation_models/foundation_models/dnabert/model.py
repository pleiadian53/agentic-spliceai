"""
DNABERT-2 Model Wrapper

Uses HuggingFace transformers (AutoModel + AutoTokenizer with trust_remote_code).
Supports MPS, CPU, and CUDA.

DNABERT-2 is a bidirectional model with BPE tokenization -- tokens do NOT
map 1:1 to nucleotides.  The encode() method returns per-TOKEN embeddings
(special tokens stripped).  For the sparse exon classifier, taking the
center token is a reasonable approximation since BPE tokens are roughly
evenly distributed across the sequence.

Installation:
    pip install transformers einops
    # On CUDA: pip install triton  (for FlashAttention, optional)
"""

import logging
from typing import List, Optional, Union

import torch

from foundation_models.base import BaseEmbeddingModel, ModelMetadata
from foundation_models.dnabert.config import DNABERT2Config

logger = logging.getLogger(__name__)


class DNABERT2Model(BaseEmbeddingModel):
    """Wrapper for DNABERT-2 foundation model.

    Supports MPS (Apple Silicon), CPU, and CUDA.  Uses BPE tokenization --
    output positions correspond to BPE tokens, not individual nucleotides.

    Example::

        >>> model = DNABERT2Model()
        >>> embeddings = model.encode("ATCGATCG...")
        >>> print(embeddings.shape)  # [num_tokens, 768]
    """

    def __init__(
        self,
        config: Optional[DNABERT2Config] = None,
        **kwargs,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = DNABERT2Config(**kwargs)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load DNABERT-2 model from HuggingFace.

        Uses ``trust_remote_code=True`` — DNABERT-2 has custom tokenizer
        and model code on HuggingFace.
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        model_id = self.config.model_id
        device = self.config.device

        # Suppress verbose HuggingFace HTTP request logging during model load
        logging.getLogger("httpx").setLevel(logging.WARNING)

        logger.info(
            "Loading DNABERT-2 %s (model=%s, device=%s)...",
            self.config.model_variant, model_id, device,
        )

        # DNABERT-2 uses custom tokenizer code on HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )

        # --- Manual model loading (bypass from_pretrained) ---
        #
        # DNABERT-2's custom bert_layers.py has two incompatibilities with
        # transformers 5.x:
        #   1. config.pad_token_id missing → AttributeError
        #   2. ALiBi tensor creation during __init__ → meta-device crash
        #
        # Neither low_cpu_mem_usage=False nor dtype= fixes the meta-device
        # issue reliably.  Instead, we follow the SpliceBERT pattern:
        # resolve the model class, instantiate on CPU, load weights manually.
        from transformers import AutoConfig
        from huggingface_hub import hf_hub_download

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = self.tokenizer.pad_token_id or 0

        # DNABERT-2's custom flash_attn_triton.py uses tl.dot(q, k, trans_b=True)
        # which was removed in newer Triton (3.x).  Patch the cached file
        # to use tl.dot(q, tl.trans(k)) instead before importing the model.
        self._patch_triton_compat(model_id)

        # Resolve DNABERT-2's custom model class from downloaded code
        model_cls = self._resolve_model_class(model_id, config)

        # Instantiate on CPU (no meta-device context → ALiBi works)
        self.model = model_cls(config)

        # Load pretrained weights — remap state dict keys
        state_dict = self._load_state_dict(model_id)
        remapped = self._remap_state_dict(state_dict)

        # Diagnostic: key matching
        model_keys = set(self.model.state_dict().keys())
        remapped_keys = set(remapped.keys())
        matched = model_keys & remapped_keys
        logger.info(
            "State dict: %d remapped, %d model, %d matched, %d missing, %d unexpected",
            len(remapped_keys), len(model_keys), len(matched),
            len(model_keys - remapped_keys), len(remapped_keys - model_keys),
        )

        result = self.model.load_state_dict(remapped, strict=False)
        if result.missing_keys:
            logger.info("Missing keys (OK if pooler only): %s", result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected keys (not loaded!): %s", result.unexpected_keys)

        if device not in ("mps", "cpu"):
            self.model = self.model.half()
        self.model = self.model.to(device)
        self.model.eval()

        # Diagnostic: verify tokenizer with a known sequence
        test_seq = "ATCGATCG"
        test_tokens = self.tokenizer(test_seq, return_tensors="pt")
        token_ids = test_tokens["input_ids"][0].tolist()
        logger.info(
            "Tokenizer diagnostic: '%s' -> %d tokens, ids=%s",
            test_seq, len(token_ids), token_ids,
        )

        # Diagnostic: check embedding variance (should NOT be near-zero)
        with torch.no_grad():
            test_input = {k: v.to(device) for k, v in test_tokens.items()}
            test_out = self.model(**test_input)[0][0]  # last_hidden_state[0]
            emb_std = test_out.float().std().item()
            emb_mean = test_out.float().mean().item()
            pos_var = test_out.float().var(dim=0).mean().item()
            logger.info(
                "Embedding diagnostic: mean=%.4f, std=%.4f, "
                "cross-position variance=%.6f",
                emb_mean, emb_std, pos_var,
            )

        logger.info(
            "DNABERT-2 %s loaded on %s (hidden_dim=%d, tokenization=BPE)",
            self.config.model_variant, device, self.hidden_dim,
        )

    @staticmethod
    def _resolve_model_class(model_id: str, config):
        """Resolve DNABERT-2's custom BertModel class from HuggingFace.

        DNABERT-2 ships custom ``bert_layers.py`` via ``trust_remote_code``.
        We need the actual class to instantiate on CPU directly (bypassing
        transformers' meta-device initialization in ``from_pretrained``).

        Uses the config's ``auto_map`` to find the class module path, then
        loads it via ``get_class_from_dynamic_module``.
        """
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # config.auto_map looks like:
        # {"AutoConfig": "configuration_bert.BertConfig",
        #  "AutoModel": "bert_layers.BertModel"}
        auto_map = getattr(config, "auto_map", {})
        class_ref = auto_map.get("AutoModel")
        if class_ref is None:
            raise ImportError(
                f"DNABERT-2 config at {model_id} has no auto_map['AutoModel']. "
                f"Available keys: {list(auto_map.keys())}"
            )

        # class_ref is e.g. "bert_layers.BertModel"
        model_cls = get_class_from_dynamic_module(
            class_ref, model_id, trust_remote_code=True,
        )
        return model_cls

    @staticmethod
    def _patch_triton_compat(model_id: str) -> None:
        """Patch DNABERT-2's cached flash_attn_triton.py for Triton 3.x.

        The cached remote code uses ``tl.dot(q, k, trans_b=True)`` which
        was removed in Triton 3.x.  Replace with ``tl.dot(q, tl.trans(k))``
        in-place so the model can use flash attention on modern Triton.
        """
        from pathlib import Path

        # Find the cached flash_attn_triton.py
        cache_dir = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
        if not cache_dir.exists():
            return

        patched = False
        for triton_file in cache_dir.rglob("flash_attn_triton.py"):
            # Only patch DNABERT-2's file
            if "DNABERT" not in str(triton_file) and "dnabert" not in str(triton_file).lower():
                continue
            content = triton_file.read_text()
            if "trans_b=True" in content:
                new_content = content.replace(
                    "tl.dot(q, k, trans_b=True)",
                    "tl.dot(q, tl.trans(k))",
                )
                triton_file.write_text(new_content)
                patched = True
                logger.info("Patched %s: tl.dot trans_b → tl.trans", triton_file.name)

        if not patched:
            logger.debug("No flash_attn_triton.py files needed Triton patching")

    @staticmethod
    def _load_state_dict(model_id: str) -> dict:
        """Download and load DNABERT-2 weights."""
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(model_id, "model.safetensors")
            from safetensors.torch import load_file
            return load_file(path)
        except Exception:
            path = hf_hub_download(model_id, "pytorch_model.bin")
            return torch.load(path, map_location="cpu", weights_only=True)

    @staticmethod
    def _remap_state_dict(state_dict: dict) -> dict:
        """Remap DNABERT-2 checkpoint keys to match the model.

        The checkpoint uses a ``bert.`` prefix (e.g. ``bert.embeddings.*``)
        and includes ``cls.predictions.*`` (LM head) keys.  The model
        expects no prefix and has no LM head.
        """
        remapped = {}
        for key, value in state_dict.items():
            new_key = key
            # Strip 'bert.' prefix
            if new_key.startswith("bert."):
                new_key = new_key[len("bert."):]
            # Skip LM head keys
            if new_key.startswith("cls.") or key.startswith("cls."):
                continue
            remapped[new_key] = value
        return remapped

    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension of model."""
        return self.config.hidden_dim

    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device

    def encode(
        self,
        sequences: Union[str, List[str]],
        layer: Optional[str] = None,
    ) -> torch.Tensor:
        """Encode DNA sequences to per-token embeddings.

        DNABERT-2 uses BPE tokenization.  The returned tensor has shape
        ``[num_tokens, hidden_dim]``, NOT ``[num_nucleotides, hidden_dim]``.
        For the sparse exon classifier, taking ``emb[center_idx, :]`` is a
        reasonable approximation since BPE tokens are roughly evenly
        distributed across the sequence.

        Special tokens ([CLS], [SEP]) are stripped from the output.

        Args:
            sequences: DNA sequence(s) (A/C/G/T, no spaces needed).
            layer: Ignored -- DNABERT-2 does not support layer selection.

        Returns:
            Tensor of shape ``[num_tokens, hidden_dim]`` for a single sequence,
            or ``[batch, max_tokens, hidden_dim]`` for multiple sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False

        # DNABERT-2 BPE tokenizer takes raw DNA strings directly --
        # no space separation or T→U conversion needed.
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs[0] is last_hidden_state: [batch, tokens, hidden_dim]
        # Use [0] instead of .last_hidden_state for compatibility with
        # DNABERT-2's custom model code which may not return BaseModelOutput.
        embeddings = outputs[0]

        if single_input:
            # Strip [CLS] (first) and [SEP] (last) special tokens
            emb = embeddings[0]  # [tokens_with_special, hidden_dim]
            return emb[1:-1]  # [num_tokens, hidden_dim]

        # Batch: strip special tokens per sequence using attention mask
        results = []
        attention_mask = inputs.get("attention_mask")
        for i in range(embeddings.size(0)):
            emb_i = embeddings[i]
            if attention_mask is not None:
                real_len = attention_mask[i].sum().item()
                emb_i = emb_i[1:real_len - 1]  # skip CLS, stop before SEP
            else:
                emb_i = emb_i[1:-1]
            results.append(emb_i)

        # Pad to same length for stacking
        max_len = max(r.size(0) for r in results)
        padded = torch.zeros(
            len(results), max_len, self.hidden_dim,
            device=self.device, dtype=results[0].dtype,
        )
        for i, r in enumerate(results):
            padded[i, :r.size(0)] = r

        return padded

    def metadata(self) -> ModelMetadata:
        """Return metadata describing this DNABERT-2 model instance."""
        return ModelMetadata(
            name=f"dnabert2-{self.config.model_variant}",
            model_type="bidirectional",
            hidden_dim=self.hidden_dim,
            max_context=self.config.context_length,
            tokenization="bpe",
            supports_layer_selection=False,
        )
