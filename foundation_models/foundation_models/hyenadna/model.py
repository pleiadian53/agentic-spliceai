"""
HyenaDNA Model Wrapper

Uses HuggingFace transformers (AutoModel + AutoTokenizer).
Supports MPS, CPU, and CUDA.
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from foundation_models.base import BaseEmbeddingModel, ModelMetadata
from foundation_models.hyenadna.config import HyenaDNAConfig

logger = logging.getLogger(__name__)


class HyenaDNAModel(BaseEmbeddingModel):
    """Wrapper for HyenaDNA foundation model via HuggingFace transformers.

    Supports MPS (Apple Silicon), CPU, and CUDA. Uses character-level
    tokenization at single-nucleotide resolution.

    Example::

        >>> model = HyenaDNAModel()  # auto-detects MPS/CUDA/CPU
        >>> embeddings = model.encode("ATCGATCG...")
        >>> print(embeddings.shape)  # [seq_len, hidden_dim]
    """

    def __init__(
        self,
        config: Optional[HyenaDNAConfig] = None,
        **kwargs,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = HyenaDNAConfig(**kwargs)
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        """Load HyenaDNA model from HuggingFace."""
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

        logger.info("Loading HyenaDNA %s (model=%s, device=%s)...",
                    self.config.model_size, model_id, device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )

        # Use float32 on MPS (float16 causes issues on Apple Silicon)
        model_dtype = torch.float32 if device == "mps" else torch.float16

        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=model_dtype,
        )
        self.model = self.model.to(device)
        self.model.eval()

        logger.info("HyenaDNA %s loaded on %s", self.config.model_size, device)

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

        Args:
            sequences: DNA sequence(s) to encode.

        Returns:
            Tensor of shape ``[seq_len, hidden_dim]`` for a single sequence,
            or ``[batch, seq_len, hidden_dim]`` for multiple sequences.
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False

        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]

        if single_input:
            if "attention_mask" in inputs:
                mask = inputs["attention_mask"][0].bool()
                return embeddings[0][mask]  # [seq_len, hidden_dim]
            return embeddings[0]  # no padding for single input

        return embeddings

    def get_likelihood(self, sequence: str) -> float:
        """Get sequence log-likelihood.

        Note: HyenaDNA is trained with next-token prediction, so we
        compute autoregressive log-likelihood from the model's logits.

        Args:
            sequence: DNA sequence.

        Returns:
            Log-likelihood of the sequence.
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # HyenaDNA's output may not have a language model head by default.
        # If using AutoModel (not AutoModelForCausalLM), logits may not be
        # directly available. Fall back to embedding-based approximation.
        if hasattr(outputs, "logits"):
            logits = outputs.logits  # [batch, seq_len, vocab]
            shift_logits = logits[0, :-1, :]
            shift_targets = inputs["input_ids"][0, 1:]
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, shift_targets.unsqueeze(-1)
            ).squeeze(-1)
            return token_log_probs.sum().item()

        raise NotImplementedError(
            "get_likelihood() requires a model with a language model head. "
            "Load with AutoModelForCausalLM instead of AutoModel."
        )

    def compute_delta_likelihood(self, ref_seq: str, alt_seq: str) -> float:
        """Compute delta log-likelihood for variant effect prediction.

        Args:
            ref_seq: Reference sequence.
            alt_seq: Alternate sequence (with variant).

        Returns:
            Change in log-likelihood (alt - ref).
        """
        ref_ll = self.get_likelihood(ref_seq)
        alt_ll = self.get_likelihood(alt_seq)
        return alt_ll - ref_ll

    def metadata(self) -> ModelMetadata:
        """Return metadata describing this HyenaDNA model instance."""
        return ModelMetadata(
            name=f"hyenadna-{self.config.model_size}",
            model_type="causal",
            hidden_dim=self.hidden_dim,
            max_context=self.config.context_length,
            tokenization="character",
            supports_layer_selection=False,
        )
