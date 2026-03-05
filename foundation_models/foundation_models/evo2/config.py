"""
Configuration for Evo2 models.

Evo2 requires CUDA (Linux + GPU). For MPS/CPU, use HyenaDNA instead.
"""

import warnings
from dataclasses import dataclass
from typing import Literal, Optional


# Default embedding layers (last block MLP output)
_DEFAULT_EMBEDDING_LAYERS = {
    "7b": "blocks.27.mlp.l3",   # 28 blocks (0-27)
    "40b": "blocks.63.mlp.l3",  # 64 blocks (0-63)
}

# Known hidden dimensions
HIDDEN_DIMS = {
    "7b": 2560,
    "40b": 5120,
}


@dataclass
class Evo2Config:
    """Configuration for Evo2 foundation model (CUDA only).

    Evo2 uses the official ``evo2`` Python package from Arc Institute,
    which requires CUDA. For Apple Silicon / CPU, use HyenaDNA.

    Attributes:
        model_size: Evo2 model size ("7b" or "40b").
        checkpoint_name: Evo2 checkpoint name (e.g. "evo2_7b").
        max_length: Maximum sequence length per chunk.
        batch_size: Batch size for inference.
        embedding_layer: Internal layer name for embedding extraction.
    """

    model_size: Literal["7b", "40b"] = "7b"
    checkpoint_name: Optional[str] = None
    max_length: int = 32768
    batch_size: int = 1
    embedding_layer: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate config and set defaults."""
        import torch

        if not torch.cuda.is_available():
            warnings.warn(
                "Evo2 requires CUDA (Linux + GPU). CUDA is not available on "
                "this system. Model loading will fail.\n"
                "For MPS/CPU, use HyenaDNA instead:\n"
                "  from foundation_models.hyenadna import HyenaDNAModel",
                stacklevel=2,
            )

        if self.checkpoint_name is None:
            self.checkpoint_name = f"evo2_{self.model_size}"

        if self.embedding_layer is None:
            self.embedding_layer = _DEFAULT_EMBEDDING_LAYERS[self.model_size]

        if self.max_length > 1_000_000:
            warnings.warn(
                f"max_length={self.max_length} exceeds 1M. "
                "This will be very slow and may cause OOM errors.",
                stacklevel=2,
            )

    @property
    def hidden_dim(self) -> int:
        """Known hidden dimension for this model size."""
        return HIDDEN_DIMS[self.model_size]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_size": self.model_size,
            "checkpoint_name": self.checkpoint_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "embedding_layer": self.embedding_layer,
        }

    @classmethod
    def for_gpu_pod(
        cls, model_size: Literal["7b", "40b"] = "40b", **kwargs
    ) -> "Evo2Config":
        """Preset for GPU pod (A40, A100, H100)."""
        defaults = {
            "model_size": model_size,
            "max_length": 131072,
            "batch_size": 4 if model_size == "40b" else 8,
        }
        defaults.update(kwargs)
        return cls(**defaults)
