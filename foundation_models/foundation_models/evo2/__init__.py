"""
Evo2 Foundation Model Integration

Arc Institute's Evo2 is a 7B/40B parameter genomic foundation model
trained on 9.3T tokens from all domains of life.

Key features:
- 1 million bp context window
- Single-nucleotide resolution
- Zero-shot variant effect prediction
- Exon/intron classification via embeddings

References:
- Paper: https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1
- GitHub: https://github.com/ArcInstitute/evo2
- HuggingFace: arc-institute/evo2-7b, arc-institute/evo2-40b
"""

from foundation_models.evo2.config import Evo2Config
from foundation_models.evo2.model import Evo2Model, load_evo2_model
from foundation_models.evo2.embedder import Evo2Embedder
from foundation_models.evo2.classifier import ExonClassifier

__all__ = [
    "Evo2Config",
    "Evo2Model",
    "load_evo2_model",
    "Evo2Embedder",
    "ExonClassifier",
]
