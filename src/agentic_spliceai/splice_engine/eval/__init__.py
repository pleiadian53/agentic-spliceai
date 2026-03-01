"""Cross-layer evaluation utilities.

Metrics extraction, output writing, and display functions for splice site
evaluation results. Reusable across base_layer, meta_layer, agentic_layer,
and UI applications.
"""

from .metrics import compute_site_metrics, extract_metrics_from_eval
from .output import EvaluationOutputWriter
from .display import print_eval_results, print_dual_comparison

__all__ = [
    'compute_site_metrics',
    'extract_metrics_from_eval',
    'EvaluationOutputWriter',
    'print_eval_results',
    'print_dual_comparison',
]
