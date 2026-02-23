"""
foundation_models.utils
=======================

Shared utilities for genomic foundation model workflows.

Modules
-------
quantization:
    Device detection, bitsandbytes / native PyTorch INT8-INT4 quantization,
    and memory estimation helpers.

chunking:
    DNA sequence chunking, embedding stitching, per-nucleotide exon label
    derivation from splice_sites_enhanced.tsv, and labeled-window generation
    for training data pipelines.
"""

from .quantization import (
    get_optimal_device,
    get_optimal_dtype,
    get_device_memory_gb,
    recommend_quantization,
    get_bnb_quantization_config,
    quantize_model_dynamic,
    apply_quantization,
    estimate_model_memory_gb,
    print_quantization_summary,
)

from .chunking import (
    SequenceChunk,
    LabeledWindow,
    chunk_sequence,
    iter_chunks,
    stitch_embeddings,
    load_gene_sequences_parquet,
    make_chunks_for_gene,
    build_exon_labels,
    generate_labeled_windows,
    estimate_chunk_count,
    estimate_window_count,
)

__all__ = [
    # quantization
    "get_optimal_device",
    "get_optimal_dtype",
    "get_device_memory_gb",
    "recommend_quantization",
    "get_bnb_quantization_config",
    "quantize_model_dynamic",
    "apply_quantization",
    "estimate_model_memory_gb",
    "print_quantization_summary",
    # chunking
    "SequenceChunk",
    "LabeledWindow",
    "chunk_sequence",
    "iter_chunks",
    "stitch_embeddings",
    "load_gene_sequences_parquet",
    "make_chunks_for_gene",
    "build_exon_labels",
    "generate_labeled_windows",
    "estimate_chunk_count",
    "estimate_window_count",
]
