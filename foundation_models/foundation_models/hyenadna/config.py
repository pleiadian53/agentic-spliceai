"""
Configuration for HyenaDNA models.
"""

import warnings
from dataclasses import dataclass
from typing import Literal, Optional


# Model variants: name -> (HF repo suffix, max context, hidden dim)
MODEL_VARIANTS = {
    "tiny-1k": ("hyenadna-tiny-1k-seqlen-hf", 1_024, 128),
    "small-32k": ("hyenadna-small-32k-seqlen-hf", 32_768, 256),
    "medium-160k": ("hyenadna-medium-160k-seqlen-hf", 160_000, 256),
    "medium-450k": ("hyenadna-medium-450k-seqlen-hf", 450_000, 256),
    "large-1m": ("hyenadna-large-1m-seqlen-hf", 1_000_000, 256),
}


@dataclass
class HyenaDNAConfig:
    """Configuration for HyenaDNA foundation model.

    HyenaDNA supports MPS, CPU, and CUDA via HuggingFace transformers.

    Attributes:
        model_size: Model variant (e.g. "medium-160k", "large-1m").
        device: Device to load on ("auto", "cuda", "mps", "cpu").
        max_length: Maximum sequence length (auto-set from model variant).
        batch_size: Batch size for inference.
    """

    model_size: str = "medium-160k"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    max_length: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        """Validate config and set defaults."""
        if self.model_size not in MODEL_VARIANTS:
            available = ", ".join(sorted(MODEL_VARIANTS.keys()))
            raise ValueError(
                f"Unknown model_size '{self.model_size}'. "
                f"Available: {available}"
            )

        variant = MODEL_VARIANTS[self.model_size]
        if self.max_length is None:
            self.max_length = variant[1]

        if self.device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.device == "mps" and self.model_size == "large-1m":
            warnings.warn(
                "HyenaDNA large-1m on MPS with 16GB RAM may OOM at full "
                "context length. Consider medium-160k for local development.",
                stacklevel=2,
            )

    @property
    def model_id(self) -> str:
        """HuggingFace model ID."""
        return f"LongSafari/{MODEL_VARIANTS[self.model_size][0]}"

    @property
    def hidden_dim(self) -> int:
        """Hidden dimension for this model variant."""
        return MODEL_VARIANTS[self.model_size][2]

    @property
    def context_length(self) -> int:
        """Maximum context length for this model variant."""
        return MODEL_VARIANTS[self.model_size][1]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_size": self.model_size,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "model_id": self.model_id,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def for_local_mac(cls, **kwargs) -> "HyenaDNAConfig":
        """Preset for MacBook Pro M1/M2/M3 (MPS)."""
        defaults = {
            "model_size": "medium-160k",
            "device": "mps",
            "batch_size": 1,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def for_gpu(cls, model_size: str = "large-1m", **kwargs) -> "HyenaDNAConfig":
        """Preset for CUDA GPU."""
        defaults = {
            "model_size": model_size,
            "device": "cuda",
            "batch_size": 8,
        }
        defaults.update(kwargs)
        return cls(**defaults)
