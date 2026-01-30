"""Data loading and preparation for base layer.

This package provides utilities for:
- Data types for gene manifests and prediction results
- Loading gene annotations from GTF files
- Loading splice site annotations
- Loading and extracting genomic sequences

Exports:
    GeneManifestEntry: Single gene manifest entry
    GeneManifest: Gene processing manifest
    PredictionResult: Container for prediction results
    
    Genomic extraction functions (standalone, no meta-spliceai dependency):
    - extract_gene_annotations
    - extract_transcript_annotations
    - extract_exon_annotations
    - extract_splice_sites
    - extract_all_annotations
    
    Sequence extraction functions:
    - extract_gene_sequences
    - extract_sequence_from_fasta
    - one_hot_encode
    - reverse_complement
"""

from .types import (
    GeneManifestEntry,
    GeneManifest,
    PredictionResult,
)

from .genomic_extraction import (
    parse_gtf_attributes,
    iter_gtf_records,
    extract_gene_annotations,
    extract_transcript_annotations,
    extract_exon_annotations,
    extract_splice_sites_from_exons,
    extract_splice_sites,
    extract_all_annotations,
)

from .sequence_extraction import (
    read_fasta_index,
    extract_sequence_from_fasta,
    extract_gene_sequences,
    reverse_complement,
    one_hot_encode,
    load_chromosome_sequences,
)

from .position_types import (
    PositionType,
    GeneCoordinates,
    absolute_to_relative,
    relative_to_absolute,
    validate_position_range,
    infer_position_type,
    convert_positions_batch,
)

from .preparation import (
    prepare_gene_data,
    prepare_splice_site_annotations,
    load_gene_annotations,
    extract_sequences,
    filter_by_genes,
    filter_by_chromosomes,
    normalize_chromosome_names,
    get_gene_count,
    get_genes_by_chromosome,
    get_missing_sequences,
    validate_gene_data,
)

__all__ = [
    # Data types
    'GeneManifestEntry',
    'GeneManifest',
    'PredictionResult',
    # Position types and conversions
    'PositionType',
    'GeneCoordinates',
    'absolute_to_relative',
    'relative_to_absolute',
    'validate_position_range',
    'infer_position_type',
    'convert_positions_batch',
    # Genomic extraction
    'parse_gtf_attributes',
    'iter_gtf_records',
    'extract_gene_annotations',
    'extract_transcript_annotations',
    'extract_exon_annotations',
    'extract_splice_sites_from_exons',
    'extract_splice_sites',
    'extract_all_annotations',
    # Sequence extraction
    'read_fasta_index',
    'extract_sequence_from_fasta',
    'extract_gene_sequences',
    'reverse_complement',
    'one_hot_encode',
    'load_chromosome_sequences',
    # Data preparation (Phase 2)
    'prepare_gene_data',
    'prepare_splice_site_annotations',
    'load_gene_annotations',
    'extract_sequences',
    'filter_by_genes',
    'filter_by_chromosomes',
    'normalize_chromosome_names',
    'get_gene_count',
    'get_genes_by_chromosome',
    'get_missing_sequences',
    'validate_gene_data',
]
