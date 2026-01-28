"""
Meta models package for splice site prediction.

This package provides workflows and utilities for meta-learning based
splice site prediction, including data preparation and model inference.
"""

from .utils import normalize_chromosome_names, determine_target_chromosomes
from .workflows import (
    # Data preparation
    prepare_gene_annotations,
    prepare_splice_site_annotations,
    prepare_genomic_sequences,
    handle_overlapping_genes,
    load_spliceai_models,
    # Prediction workflows
    run_enhanced_splice_prediction_workflow,
    run_base_model_predictions,
    validate_workflow_config
)

__all__ = [
    # Utils
    'normalize_chromosome_names',
    'determine_target_chromosomes',
    # Data preparation
    'prepare_gene_annotations',
    'prepare_splice_site_annotations',
    'prepare_genomic_sequences',
    'handle_overlapping_genes',
    'load_spliceai_models',
    # Prediction workflows
    'run_enhanced_splice_prediction_workflow',
    'run_base_model_predictions',
    'validate_workflow_config',
]
