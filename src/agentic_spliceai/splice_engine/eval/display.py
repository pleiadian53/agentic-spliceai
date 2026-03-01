"""Display functions for evaluation results.

Formatted console output for evaluation metrics. Complements the existing
display utilities in splice_engine/utils/display.py but is specific to
evaluation result structures.
"""

import polars as pl
from typing import Dict, Optional

from .metrics import compute_site_metrics


def print_eval_results(
    positions_df: pl.DataFrame,
    pr_metrics: Optional[Dict] = None,
    topk_accuracy: Optional[Dict] = None,
    windowed_recall: Optional[Dict] = None,
    indent: int = 3,
) -> None:
    """Print concise evaluation results for one annotation filter.

    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame with pred_type and splice_type columns.
    pr_metrics : dict, optional
        PR metrics from compute_pr_metrics().
    topk_accuracy : dict, optional
        Top-k accuracy from compute_topk_accuracy().
    windowed_recall : dict, optional
        Windowed recall from compute_windowed_recall().
    indent : int
        Number of spaces for indentation.
    """
    pad = " " * indent
    if positions_df.height == 0:
        print(f"{pad}(no positions evaluated)")
        return

    for site_type in ['donor', 'acceptor']:
        m = compute_site_metrics(positions_df, site_type)
        total_true = m['tp'] + m['fn']
        print(f"\n{pad}{site_type.capitalize()} sites: {m['tp']}/{total_true} detected "
              f"(Recall: {m['recall']:.1%}, Precision: {m['precision']:.1%}, F1: {m['f1']:.1%})")

    # Overall
    m = compute_site_metrics(positions_df, 'overall')
    print(f"\n{pad}Overall: Recall={m['recall']:.1%}, Precision={m['precision']:.1%}, F1={m['f1']:.1%}")

    if pr_metrics:
        print(f"{pad}Macro AP: {pr_metrics['macro_ap']:.4f}, Macro PR-AUC: {pr_metrics['macro_pr_auc']:.4f}")

    if topk_accuracy:
        for k in [1.0, 2.0]:
            overall_acc = topk_accuracy['overall'].get(k, 0.0)
            if overall_acc > 0:
                print(f"{pad}Top-k accuracy (k={k:.0f}x): {overall_acc:.1%}")

    if windowed_recall:
        wr_2 = windowed_recall['overall'].get(2, 0.0)
        if wr_2 > 0:
            print(f"{pad}Windowed recall (+-2bp): {wr_2:.1%}")


def print_dual_comparison(eval_results: Dict[str, Dict]) -> None:
    """Print side-by-side comparison of canonical vs all-transcript metrics.

    Parameters
    ----------
    eval_results : dict
        Must contain 'canonical' and 'all' keys with evaluation result dicts,
        each having a 'positions_df' key.
    """
    print(f"\n{'='*70}")
    print(f"  Dual-Metric Summary: Base Model Performance vs Full Landscape")
    print(f"{'='*70}")

    can = eval_results.get('canonical', {})
    all_ = eval_results.get('all', {})

    can_pdf = can.get('positions_df', pl.DataFrame())
    all_pdf = all_.get('positions_df', pl.DataFrame())

    if can_pdf.height == 0 or all_pdf.height == 0:
        print("  (insufficient data for comparison)")
        return

    print(f"\n  {'Metric':<35} {'Canonical':>12} {'All Transcripts':>16}")
    print(f"  {'-'*35} {'-'*12} {'-'*16}")

    for site_type in ['donor', 'acceptor', 'overall']:
        can_m = compute_site_metrics(can_pdf, site_type)
        all_m = compute_site_metrics(all_pdf, site_type)

        can_total = can_m['tp'] + can_m['fn']
        all_total = all_m['tp'] + all_m['fn']

        label = site_type.capitalize()
        can_str = f"{can_m['tp']}/{can_total} ({can_m['recall']:.0%})"
        all_str = f"{all_m['tp']}/{all_total} ({all_m['recall']:.0%})"

        print(f"  {label + ' recall':<35} {can_str:>12} {all_str:>16}")

    # Show the gap
    can_total_all = len(can_pdf.filter(pl.col('pred_type').is_in(['TP', 'FN'])))
    all_total_all = len(all_pdf.filter(pl.col('pred_type').is_in(['TP', 'FN'])))
    gap = all_total_all - can_total_all

    print(f"\n  The gap of {gap} additional sites across alternative transcripts")
    print(f"  represents what meta/agentic layers need to address.")
    print(f"{'='*70}")
