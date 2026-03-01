"""Cross-layer evaluation metrics extraction and serialization.

Converts evaluation results (containing Polars DataFrames from
base_layer.prediction.evaluation) into JSON-serializable dictionaries.
Reusable across base_layer, meta_layer, agentic_layer, and UI applications.
"""

import polars as pl
from typing import Dict, Any


def compute_site_metrics(positions_df: pl.DataFrame, site_type: str = "overall") -> Dict[str, Any]:
    """Compute TP/FP/FN counts and derived metrics for a splice site type.

    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame with 'pred_type' and 'splice_type' columns.
        Output of evaluate_splice_site_predictions().
    site_type : str
        One of 'donor', 'acceptor', or 'overall'.

    Returns
    -------
    dict
        Keys: tp, fp, fn, recall, precision, f1 (all JSON-serializable).
    """
    if site_type == "overall":
        df = positions_df
    else:
        df = positions_df.filter(pl.col('splice_type') == site_type)

    tp = len(df.filter(pl.col('pred_type') == 'TP'))
    fp = len(df.filter(pl.col('pred_type') == 'FP'))
    fn = len(df.filter(pl.col('pred_type') == 'FN'))

    total_true = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'recall': round(recall, 4),
        'precision': round(precision, 4),
        'f1': round(f1, 4),
    }


def extract_metrics_from_eval(eval_results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Convert evaluation results (with Polars DFs) to JSON-serializable dict.

    Processes evaluation results keyed by filter mode (e.g., 'canonical', 'all')
    and extracts per-site-type metrics, PR metrics, top-k accuracy, and
    windowed recall.

    Parameters
    ----------
    eval_results : dict
        Mapping of filter_mode -> result dict. Each result dict must contain
        'positions_df' (pl.DataFrame) and optionally 'pr_metrics',
        'topk_accuracy', 'windowed_recall'.

    Returns
    -------
    dict
        JSON-serializable nested dict of metrics keyed by filter mode.
    """
    metrics = {}
    for filter_mode, result in eval_results.items():
        positions_df = result['positions_df']
        mode_metrics = {}

        for site_type in ['donor', 'acceptor', 'overall']:
            mode_metrics[site_type] = compute_site_metrics(positions_df, site_type)

        pr = result.get('pr_metrics', {})
        if pr:
            mode_metrics['pr_metrics'] = {
                'macro_ap': round(pr.get('macro_ap', 0), 4),
                'macro_pr_auc': round(pr.get('macro_pr_auc', 0), 4),
                'donor_ap': round(pr.get('donor_ap', 0), 4),
                'acceptor_ap': round(pr.get('acceptor_ap', 0), 4),
            }

        topk = result.get('topk_accuracy', {})
        if topk and 'overall' in topk:
            mode_metrics['topk_accuracy'] = {
                str(k): round(v, 4) for k, v in topk['overall'].items()
            }

        wr = result.get('windowed_recall', {})
        if wr and 'overall' in wr:
            mode_metrics['windowed_recall'] = {
                str(k): round(v, 4) for k, v in wr['overall'].items()
            }

        metrics[filter_mode] = mode_metrics
    return metrics
