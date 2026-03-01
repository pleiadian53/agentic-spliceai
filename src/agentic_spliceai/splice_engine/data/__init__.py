"""Cross-layer data utilities.

Gene sampling and other data utilities shared across base_layer, meta_layer,
agentic_layer, and applications.
"""

from .sampling import sample_random_genes, get_protein_coding_genes

__all__ = [
    'sample_random_genes',
    'get_protein_coding_genes',
]
