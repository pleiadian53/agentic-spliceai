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

        # float32 on MPS (float16 causes issues on Apple Silicon),
        # float16 on CUDA for efficiency
        model_dtype = torch.float32 if device in ("mps", "cpu") else torch.float16

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
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
