"""Cross-layer data utilities.

Gene sampling, homology detection, and other data utilities shared across
base_layer, meta_layer, agentic_layer, and applications.
"""

from .sampling import sample_random_genes, get_protein_coding_genes
from .homology import (
    detect_gene_families_by_name,
    detect_paralogs_by_alignment,
    get_paralog_groups,
)

__all__ = [
    'sample_random_genes',
    'get_protein_coding_genes',
    'detect_gene_families_by_name',
    'detect_paralogs_by_alignment',
    'get_paralog_groups',
]
