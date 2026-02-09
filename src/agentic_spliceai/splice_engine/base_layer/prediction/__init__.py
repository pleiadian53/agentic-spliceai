"""Prediction workflow and evaluation for base layer.

This package provides:
- Core prediction functions for SpliceAI/OpenSpliceAI
- Input sequence preparation
- Model loading utilities
- Evaluation against ground truth annotations
"""

from .core import (
    prepare_input_sequence,
    predict_with_model,
    predict_splice_sites_for_genes,
    load_spliceai_models,
    normalize_strand,
    SPLICEAI_CONTEXT,
    SPLICEAI_BLOCK_SIZE,
)

from .evaluation import (
    evaluate_splice_site_predictions,
    add_derived_features,
    filter_annotations_by_transcript,
    splice_site_gap_analysis,
)

__all__ = [
    # Core prediction
    'prepare_input_sequence',
    'predict_with_model',
    'predict_splice_sites_for_genes',
    'load_spliceai_models',
    'normalize_strand',
    'SPLICEAI_CONTEXT',
    'SPLICEAI_BLOCK_SIZE',
    # Evaluation
    'evaluate_splice_site_predictions',
    'add_derived_features',
    'filter_annotations_by_transcript',
    'splice_site_gap_analysis',
]
