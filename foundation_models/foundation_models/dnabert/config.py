"""
Configuration for DNABERT-2 model.

DNABERT-2 is a multi-species genome foundation model using BPE tokenization
and ALiBi positional encoding (no hard position limit).  117M parameters.

Reference: Zhou et al. 2024, "DNABERT-2: Efficient Foundation Model and
Benchmark For Multi-Species Genome", ICLR 2024.
"""

from dataclasses import dataclass
from typing import Literal, Optional


# Model variants: name -> (max_context_nt, hidden_dim)
# ALiBi has no hard limit, but trained on 128 nt.  Practical limit ~4000 bp.
MODEL_VARIANTS = {
    "117M": (4000, 768),
}


@dataclass
class DNABERT2Config:
    """Configuration for DNABERT-2 foundation model.

    DNABERT-2 supports CPU, MPS, and CUDA via HuggingFace transformers
    with ``trust_remote_code=True``.

    Attributes:
        model_variant: Model variant name (currently only "117M").
        device: Device to load on ("auto", "cuda", "mps", "cpu").
        max_length: Maximum sequence length in nucleotides.
        batch_size: Batch size for inference.
    """

    model_variant: str = "117M"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    max_length: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Validate config and set defaults."""
        if self.model_variant not in MODEL_VARIANTS:
            available = ", ".join(sorted(MODEL_VARIANTS.keys()))
            raise ValueError(
                f"Unknown model_variant '{self.model_variant}'. "
                f"Available: {available}"
            )

        if self.max_length is None:
            self.max_length = MODEL_VARIANTS[self.model_variant][0]

        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

    @property
    def model_id(self) -> str:
        """HuggingFace model ID."""
        return "zhihan1996/DNABERT-2-117M"

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension for this model variant."""
        return MODEL_VARIANTS[self.model_variant][1]

    @property
    def context_length(self) -> int:
        """Maximum context length for this model variant."""
        return MODEL_VARIANTS[self.model_variant][0]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_variant": self.model_variant,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
        }
