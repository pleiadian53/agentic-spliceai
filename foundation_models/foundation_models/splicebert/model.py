"""
SpliceBERT Model Wrapper

Uses HuggingFace transformers (BertModel + AutoTokenizer).
Supports MPS, CPU, and CUDA.

SpliceBERT is a bidirectional BERT model — every position's embedding
already encodes full left+right context.  Unlike causal models (Evo2,
HyenaDNA), there is no need for dual-strand extraction to capture
downstream context.

Loads directly from HuggingFace (multimolecule/splicebert) as a standard
BertModel — no ``multimolecule`` package required, avoiding its fragile
transformers version constraints.

Installation:
    pip install transformers sentencepiece
"""

import logging
from typing import List, Optional, Union

import torch

from foundation_models.base import BaseEmbeddingModel, ModelMetadata
from foundation_models.splicebert.config import SpliceBERTConfig

logger = logging.getLogger(__name__)


class SpliceBERTModel(BaseEmbeddingModel):
    """Wrapper for SpliceBERT foundation model via ``multimolecule``.

    Supports MPS (Apple Silicon), CPU, and CUDA.  Uses single-nucleotide
    tokenization — each output position maps 1:1 to an input nucleotide.

    Example::

        >>> model = SpliceBERTModel()
        >>> embeddings = model.encode("ATCGATCG...")
        >>> print(embeddings.shape)  # [seq_len, 512]
    """

    def __init__(
        self,
        config: Optional[SpliceBERTConfig] = None,
        **kwargs,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SpliceBERTConfig(**kwargs)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load SpliceBERT model from HuggingFace.

        Loads directly as a BertModel — no ``multimolecule`` package needed.
        SpliceBERT is architecturally a standard BERT with single-nucleotide
        tokenization, so ``BertModel.from_pretrained()`` works directly.
        """
        try:
            from transformers import BertModel, BertTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )

        model_id = self.config.model_id
        device = self.config.device

        # Suppress verbose HuggingFace HTTP request logging during model load
        logging.getLogger("httpx").setLevel(logging.WARNING)

        logger.info(
            "Loading SpliceBERT %s (model=%s, device=%s)...",
            self.config.model_variant, model_id, device,
        )

        try:
            # Use BertTokenizer directly — AutoTokenizer in transformers 5.x
            # routes to the fast tokenizer even with use_fast=False, which fails
            # on SpliceBERT's sentencepiece-based vocab.
            self.tokenizer = BertTokenizer.from_pretrained(
                model_id, do_lower_case=False,
            )
        except Exception as exc:
            if "sentencepiece" in str(exc).lower():
                raise ImportError(
                    "SpliceBERT tokenizer requires sentencepiece. "
                    "Install with: pip install sentencepiece"
                ) from exc
            raise

        # Load as standard BertModel with key remapping.
        # SpliceBERT's checkpoint uses "model." prefix and "layer_norm" naming,
        # while BertModel expects no prefix and "LayerNorm".
        # Can't pass state_dict + model_id together in transformers 5.x,
        # so we load config first, create the model, then load weights.
        from transformers import BertConfig

        bert_config = BertConfig.from_pretrained(model_id)
        logger.info(
            "BertConfig: hidden=%d, layers=%d, heads=%d, vocab=%d",
            bert_config.hidden_size, bert_config.num_hidden_layers,
            bert_config.num_attention_heads, bert_config.vocab_size,
        )
        self.model = BertModel(bert_config)
        remapped = self._remap_state_dict(model_id)

        # Log key counts for debugging
        model_keys = set(self.model.state_dict().keys())
        remapped_keys = set(remapped.keys())
        matched = model_keys & remapped_keys
        logger.info(
            "State dict: %d remapped, %d model, %d matched, "
            "%d missing, %d unexpected",
            len(remapped_keys), len(model_keys), len(matched),
            len(model_keys - remapped_keys), len(remapped_keys - model_keys),
        )
        if len(matched) == 0:
            # Log first few keys from each to diagnose
            logger.error("NO keys matched! Sample model keys: %s",
                         list(model_keys)[:5])
            logger.error("Sample remapped keys: %s",
                         list(remapped_keys)[:5])

        result = self.model.load_state_dict(remapped, strict=False)
        if result.missing_keys:
            logger.info("Missing keys (OK if only pooler): %s",
                        result.missing_keys)
        if result.unexpected_keys:
            logger.warning("Unexpected keys (not loaded!): %s",
                           result.unexpected_keys)
        self.model = self.model.to(device)
        self.model.eval()

        # Diagnostic: verify tokenizer produces correct nucleotide IDs
        # Use U instead of T — SpliceBERT's vocab is RNA convention
        test_seq = "ACGUACGU"
        test_tokens = self.tokenizer(" ".join(test_seq), return_tensors="pt")
        logger.info(
            "Tokenizer diagnostic: '%s' -> ids=%s, tokens=%s",
            test_seq,
            test_tokens["input_ids"][0].tolist(),
            self.tokenizer.convert_ids_to_tokens(
                test_tokens["input_ids"][0].tolist()
            ),
        )

        # Diagnostic: check embedding variance (should NOT be near-zero)
        with torch.no_grad():
            test_input = test_tokens.to(device)
            test_out = self.model(**test_input).last_hidden_state[0]
            emb_std = test_out.std().item()
            emb_mean = test_out.mean().item()
            # Check if different positions have different embeddings
            pos_var = test_out.var(dim=0).mean().item()
            logger.info(
                "Embedding diagnostic: mean=%.4f, std=%.4f, "
                "cross-position variance=%.6f",
                emb_mean, emb_std, pos_var,
            )

        logger.info(
            "SpliceBERT %s loaded on %s (hidden_dim=%d)",
            self.config.model_variant, device, self.hidden_dim,
        )

    @staticmethod
    def _remap_state_dict(model_id: str) -> dict:
        """Load and remap SpliceBERT checkpoint keys for standard BertModel.

        SpliceBERT checkpoint uses:
          - ``model.`` prefix on all encoder/embedding keys
          - ``layer_norm`` instead of ``LayerNorm``
          - ``lm_head.*`` keys (masked LM head, not needed for embeddings)

        Standard BertModel expects:
          - No ``model.`` prefix
          - ``LayerNorm`` (capital L, capital N)
        """
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        # Download and load the safetensors file
        path = hf_hub_download(model_id, "model.safetensors")
        raw = load_file(path)

        remapped = {}
        for key, value in raw.items():
            new_key = key
            # Strip "model." prefix
            if new_key.startswith("model."):
                new_key = new_key[len("model."):]
            # Rename layer_norm -> LayerNorm
            new_key = new_key.replace("layer_norm", "LayerNorm")
            # Skip lm_head keys (not part of BertModel)
            if new_key.startswith("lm_head"):
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
        """Encode DNA sequences to per-nucleotide embeddings.

        SpliceBERT uses single-nucleotide tokenization (N=5, A=6, C=7, G=8,
        T/U=9) so each output position maps 1:1 to an input nucleotide.
        Special tokens ([CLS], [SEP]) are stripped from the output.

        Args:
            sequences: DNA sequence(s) to encode.
            layer: Ignored — SpliceBERT does not support layer selection.

        Returns:
            Tensor of shape ``[seq_len, hidden_dim]`` for a single sequence,
            or ``[batch, seq_len, hidden_dim]`` for multiple sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False

        # SpliceBERT was trained on pre-mRNA (RNA convention): vocab has U, not T.
        # Convert T→U before tokenization so thymine doesn't map to <unk>.
        # Also space-separate nucleotides — BertTokenizer splits on whitespace
        # first, so without spaces the whole sequence becomes a single [UNK].
        spaced = [" ".join(s.replace("T", "U").replace("t", "u")) for s in sequences]
        inputs = self.tokenizer(
            spaced,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # outputs.last_hidden_state: [batch, seq_len_with_special, hidden_dim]
        # Strip [CLS] (position 0) and [SEP] (last non-pad position)
        embeddings = outputs.last_hidden_state

        if single_input:
            # Single sequence: strip CLS and SEP, return per-nucleotide
            emb = embeddings[0]  # [seq_len_with_special, hidden_dim]
            # CLS is at index 0, SEP is at the last non-padding position.
            # With single input and no padding, layout is: [CLS] n1 n2 ... nL [SEP]
            return emb[1:-1]  # [seq_len, hidden_dim]

        # Batch: strip CLS token; handle variable-length SEP via attention mask
        results = []
        attention_mask = inputs.get("attention_mask")
        for i in range(embeddings.size(0)):
            emb_i = embeddings[i]  # [seq_len_with_special, hidden_dim]
            if attention_mask is not None:
                # Find last real token position (SEP), keep [1:-1] of real tokens
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
        """Return metadata describing this SpliceBERT model instance."""
        return ModelMetadata(
            name=(self.config.model_variant
                  if self.config.model_variant != "splicebert"
                  else "splicebert"),
            model_type="bidirectional",
            hidden_dim=self.hidden_dim,
            max_context=self.config.context_length,
            tokenization="character",
            supports_layer_selection=False,
        )
