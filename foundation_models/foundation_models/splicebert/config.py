"""
Configuration for SpliceBERT models.

SpliceBERT is a BERT encoder pre-trained on 2 million pre-mRNA sequences from
72 vertebrate species (65 billion nucleotides).  It uses single-nucleotide
tokenization and supports sequences up to 1024 nt.

Reference: Chen & Zheng 2024, Briefings in Bioinformatics 25(3).
"""

from dataclasses import dataclass
from typing import Literal, Optional


# Model variants: name -> (max_context, hidden_dim)
MODEL_VARIANTS = {
    "splicebert": (1024, 512),
    "splicebert.510nt": (510, 512),
}


@dataclass
class SpliceBERTConfig:
    """Configuration for SpliceBERT foundation model.

    SpliceBERT supports CPU, MPS, and CUDA via HuggingFace transformers
    (``multimolecule`` package).

    Attributes:
        model_variant: Model variant name.
        device: Device to load on ("auto", "cuda", "mps", "cpu").
        max_length: Maximum sequence length in nucleotides.
        batch_size: Batch size for inference.
    """

    model_variant: str = "splicebert"
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
        return f"multimolecule/{self.model_variant}"

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
