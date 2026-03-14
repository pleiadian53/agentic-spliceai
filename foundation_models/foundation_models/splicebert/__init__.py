"""SpliceBERT model wrapper — BERT pre-trained on 2M pre-mRNA sequences."""

from foundation_models.splicebert.config import SpliceBERTConfig
from foundation_models.splicebert.model import SpliceBERTModel

__all__ = [
    "SpliceBERTConfig",
    "SpliceBERTModel",
]
