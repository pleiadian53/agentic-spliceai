"""
HyenaDNA Foundation Model Integration

HyenaDNA is a long-range genomic foundation model (~54M params) using
Hyena operators for sub-quadratic sequence modeling. Supports MPS, CPU,
and CUDA via standard HuggingFace transformers.

Key features:
- Up to 1M bp context window (model-size dependent)
- Single-nucleotide resolution (character-level tokenizer)
- MPS/Apple Silicon compatible (unlike Evo2 which requires CUDA)
- Same research lineage as Evo2 (Hyena → StripedHyena → Evo2)

References:
- Paper: https://arxiv.org/abs/2306.15794
- GitHub: https://github.com/HazyResearch/hyena-dna
- Models: https://huggingface.co/LongSafari
"""

from foundation_models.hyenadna.config import HyenaDNAConfig
from foundation_models.hyenadna.model import HyenaDNAModel
from foundation_models.hyenadna.embedder import HyenaDNAEmbedder

__all__ = [
    "HyenaDNAConfig",
    "HyenaDNAModel",
    "HyenaDNAEmbedder",
]
