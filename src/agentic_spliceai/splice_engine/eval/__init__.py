"""Cross-layer evaluation utilities.

Metrics extraction, output writing, display functions, and gene-level
splitting strategies for splice site evaluation. Reusable across
base_layer, meta_layer, agentic_layer, and UI applications.
"""

from .metrics import compute_site_metrics, extract_metrics_from_eval
from .output import EvaluationOutputWriter
from .display import print_eval_results, print_dual_comparison
from .splitting import (
    GeneSplit,
    ChromosomeSplitConfig,
    SPLIT_PRESETS,
    build_gene_split,
    gene_chromosomes_from_dataframe,
    gene_chromosomes_from_gtf,
)
from .calibration import (
    compute_ece,
    compute_mce,
    compute_brier_decomposition,
    reliability_curve,
    compare_calibration,
    print_calibration_comparison,
    BrierDecomposition,
    ReliabilityCurve,
    CalibrationComparison,
)

__all__ = [
    'compute_site_metrics',
    'extract_metrics_from_eval',
    'EvaluationOutputWriter',
    'print_eval_results',
    'print_dual_comparison',
    'GeneSplit',
    'ChromosomeSplitConfig',
    'SPLIT_PRESETS',
    'build_gene_split',
    'gene_chromosomes_from_dataframe',
    'gene_chromosomes_from_gtf',
    'compute_ece',
    'compute_mce',
    'compute_brier_decomposition',
    'reliability_curve',
    'compare_calibration',
    'print_calibration_comparison',
    'BrierDecomposition',
    'ReliabilityCurve',
    'CalibrationComparison',
]
