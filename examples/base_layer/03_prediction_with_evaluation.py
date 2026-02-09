#!/usr/bin/env python
"""Phase 1 Example: Splice Site Prediction with Evaluation Metrics.

This example demonstrates:
1. Running splice site predictions
2. Loading ground truth annotations
3. Evaluating predictions with dual metrics (canonical + all transcripts)
4. Computing performance metrics (F1, precision, recall, PR-AUC)
5. Gap analysis: what the meta/agentic layers need to address

Usage:
    python 03_prediction_with_evaluation.py --gene TP53 --model spliceai
    python 03_prediction_with_evaluation.py --gene TP53 --model spliceai --threshold 0.3
    python 03_prediction_with_evaluation.py --genes TP53 --model spliceai --gap-analysis

Example output:
    Canonical Transcript Recall: 90.0% (9/10 donor, 9/10 acceptor)
    All-Transcript Recall:       40.0% (9/23 donor, 9/22 acceptor)
    
    Gap Analysis:
      14 missed donor sites:
        8 single-transcript (no signal) - alternative isoform-specific
        6 multi-transcript (moderate signal) - meta layer can recover
"""

import sys
import argparse
from pathlib import Path

import polars as pl

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_splice_site_annotations,
    prepare_gene_data
)
from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
    evaluate_splice_site_predictions,
    compute_pr_metrics,
    compute_topk_accuracy,
    compute_windowed_recall,
    filter_annotations_by_transcript,
    splice_site_gap_analysis,
)
from agentic_spliceai.splice_engine.base_layer.prediction.core import (
    predict_splice_sites_for_genes,
    load_spliceai_models
)


def _print_eval_results(positions_df, pr_metrics, topk_accuracy, windowed_recall):
    """Print concise evaluation results for one annotation filter."""
    if positions_df.height == 0:
        print("   (no positions evaluated)")
        return
    
    for site_type in ['donor', 'acceptor']:
        site_df = positions_df.filter(pl.col('splice_type') == site_type)
        tp = len(site_df.filter(pl.col('pred_type') == 'TP'))
        fp = len(site_df.filter(pl.col('pred_type') == 'FP'))
        fn = len(site_df.filter(pl.col('pred_type') == 'FN'))
        
        total_true = tp + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / total_true if total_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n   {site_type.capitalize()} sites: {tp}/{total_true} detected "
              f"(Recall: {recall:.1%}, Precision: {precision:.1%}, F1: {f1:.1%})")
    
    # Overall
    all_tp = len(positions_df.filter(pl.col('pred_type') == 'TP'))
    all_fp = len(positions_df.filter(pl.col('pred_type') == 'FP'))
    all_fn = len(positions_df.filter(pl.col('pred_type') == 'FN'))
    
    total_true = all_tp + all_fn
    overall_p = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    overall_r = all_tp / total_true if total_true > 0 else 0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0
    
    print(f"\n   Overall: Recall={overall_r:.1%}, Precision={overall_p:.1%}, F1={overall_f1:.1%}")
    
    # PR-AUC
    if pr_metrics:
        print(f"   Macro AP: {pr_metrics['macro_ap']:.4f}, Macro PR-AUC: {pr_metrics['macro_pr_auc']:.4f}")
    
    # Top-k (just show k=1.0 and k=2.0)
    if topk_accuracy:
        for m in [1.0, 2.0]:
            overall_acc = topk_accuracy['overall'].get(m, 0.0)
            if overall_acc > 0:
                print(f"   Top-k accuracy (k={m:.0f}x): {overall_acc:.1%}")
    
    # Windowed recall (just k=2)
    if windowed_recall:
        wr_2 = windowed_recall['overall'].get(2, 0.0)
        if wr_2 > 0:
            print(f"   Windowed recall (±2bp): {wr_2:.1%}")


def _print_dual_comparison(eval_results):
    """Print side-by-side comparison of canonical vs all-transcript metrics."""
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
        if site_type == 'overall':
            can_tp = len(can_pdf.filter(pl.col('pred_type') == 'TP'))
            can_fn = len(can_pdf.filter(pl.col('pred_type') == 'FN'))
            all_tp = len(all_pdf.filter(pl.col('pred_type') == 'TP'))
            all_fn = len(all_pdf.filter(pl.col('pred_type') == 'FN'))
        else:
            can_tp = len(can_pdf.filter((pl.col('pred_type') == 'TP') & (pl.col('splice_type') == site_type)))
            can_fn = len(can_pdf.filter((pl.col('pred_type') == 'FN') & (pl.col('splice_type') == site_type)))
            all_tp = len(all_pdf.filter((pl.col('pred_type') == 'TP') & (pl.col('splice_type') == site_type)))
            all_fn = len(all_pdf.filter((pl.col('pred_type') == 'FN') & (pl.col('splice_type') == site_type)))
        
        can_total = can_tp + can_fn
        all_total = all_tp + all_fn
        can_recall = can_tp / can_total if can_total > 0 else 0
        all_recall = all_tp / all_total if all_total > 0 else 0
        
        label = site_type.capitalize()
        can_str = f"{can_tp}/{can_total} ({can_recall:.0%})"
        all_str = f"{all_tp}/{all_total} ({all_recall:.0%})"
        
        print(f"  {label + ' recall':<35} {can_str:>12} {all_str:>16}")
    
    # Show the gap
    can_total_all = len(can_pdf.filter(pl.col('pred_type').is_in(['TP', 'FN'])))
    all_total_all = len(all_pdf.filter(pl.col('pred_type').is_in(['TP', 'FN'])))
    gap = all_total_all - can_total_all
    
    print(f"\n  The gap of {gap} additional sites across alternative transcripts")
    print(f"  represents what meta/agentic layers need to address.")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 Example: Prediction with Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--gene",
        help="Single gene to predict (e.g., TP53, BRCA1)"
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        help="Multiple genes to predict (e.g., BRCA1 TP53 MYC)"
    )
    parser.add_argument(
        "--model",
        default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Splice site detection threshold (default: 0.5)"
    )
    parser.add_argument(
        "--consensus-window",
        type=int,
        default=2,
        help="Consensus window for matching predictions (default: 2)"
    )
    parser.add_argument(
        "--transcript-filter",
        default="dual",
        choices=["all", "canonical", "shared", "dual"],
        help="Transcript filtering for evaluation (default: dual = canonical + all)"
    )
    parser.add_argument(
        "--gap-analysis",
        action="store_true",
        help="Run splice site gap analysis (shows what meta/agentic layers need to address)"
    )
    args = parser.parse_args()
    
    # Determine gene list
    if args.gene:
        gene_list = [args.gene]
    elif args.genes:
        gene_list = args.genes
    else:
        print("❌ Error: Must specify --gene or --genes")
        return 1
    
    print("=" * 80)
    print("Phase 1 Example: Splice Site Prediction with Evaluation")
    print("=" * 80)
    print(f"\nGenes: {', '.join(gene_list)}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Consensus window: ±{args.consensus_window}bp")
    print()
    
    # Get model-specific resources (build, annotation source, paths)
    # This automatically determines the correct genomic build and annotation source
    # based on what the model was trained on
    from agentic_spliceai.splice_engine.resources import get_model_resources
    
    model_resources = get_model_resources(args.model)
    build = model_resources.build
    annotation_source = model_resources.annotation_source
    
    print(f"📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Build: {build}")
    print(f"   Annotation: {annotation_source}")
    print()
    
    # Step 1: Load ground truth annotations
    print("📂 Loading ground truth splice site annotations...")
    
    # Get build-specific annotations directory
    # Structure: data/<annotation_source>/<build>/splice_sites_enhanced.tsv
    # This ensures GRCh37 and GRCh38 annotations are kept separate
    annotations_dir = model_resources.get_annotations_dir(create=True)
    
    annotations_result = prepare_splice_site_annotations(
        output_dir=str(annotations_dir),
        genes=gene_list,
        build=build,
        annotation_source=annotation_source,
        verbosity=1
    )
    
    annotations_df = annotations_result['splice_sites_df']
    print(f"✓ Loaded {len(annotations_df)} ground truth splice sites")
    
    # Count by type
    donor_count = len(annotations_df.filter(pl.col('site_type') == 'donor'))
    acceptor_count = len(annotations_df.filter(pl.col('site_type') == 'acceptor'))
    print(f"   Donor: {donor_count}, Acceptor: {acceptor_count}")
    print()
    
    # Step 2: Prepare gene data and run predictions
    print("🧬 Preparing gene data...")
    genes_df = prepare_gene_data(
        genes=gene_list,
        build=build,
        annotation_source=annotation_source,
        verbosity=1
    )
    print(f"✓ Prepared {len(genes_df)} genes")
    print()
    
    print("🔧 Loading models...")
    models = load_spliceai_models(
        model_type=args.model,
        build=build,
        verbosity=1
    )
    print()
    
    print("🧬 Running predictions...")
    predictions = predict_splice_sites_for_genes(
        gene_df=genes_df,
        models=models,
        context=10000,
        output_format='dict',
        verbosity=1
    )
    print()
    
    # =========================================================================
    # Step 3: Dual-metric evaluation (canonical + all transcripts)
    # =========================================================================
    # Base models like SpliceAI primarily predict canonical splice sites.
    # Evaluating against all transcripts deflates recall because alternative 
    # isoforms contribute sites the base model genuinely cannot predict.
    # We report BOTH metrics to clearly separate base model performance from
    # the gap that meta/agentic layers need to fill.
    # =========================================================================
    
    use_dual = args.transcript_filter == 'dual'
    
    # Determine which evaluations to run
    eval_configs = []
    if use_dual:
        eval_configs = [
            ('canonical', 'Canonical Transcript (base model performance)'),
            ('all', 'All Transcripts (includes alternative isoforms)'),
        ]
    else:
        eval_configs = [
            (args.transcript_filter, f'Transcript filter: {args.transcript_filter}'),
        ]
    
    eval_results = {}
    
    for filter_mode, label in eval_configs:
        print(f"\n{'─'*70}")
        print(f"📊 Evaluating: {label}")
        print(f"{'─'*70}")
        
        # Filter annotations
        filtered_annot = filter_annotations_by_transcript(
            annotations_df, mode=filter_mode, verbosity=1
        )
        
        # Count unique sites
        n_annotations = filtered_annot.height
        site_col = 'splice_type' if 'splice_type' in filtered_annot.columns else 'site_type'
        for st_name in ['donor', 'acceptor']:
            st_annot = filtered_annot.filter(pl.col(site_col).str.to_lowercase() == st_name)
            n_unique = st_annot['position'].n_unique() if st_annot.height > 0 else 0
            print(f"   {st_name.capitalize()}: {n_unique} unique positions ({st_annot.height} annotations)")
        
        # Run evaluation
        error_df, positions_df, pr_metrics = evaluate_splice_site_predictions(
            predictions=predictions,
            annotations_df=filtered_annot,
            threshold=args.threshold,
            consensus_window=args.consensus_window,
            collect_tn=True,
            no_tn_sampling=True,
            verbosity=0,
            return_pr_metrics=True
        )
        
        # Compute additional metrics
        topk_accuracy = compute_topk_accuracy(
            predictions=predictions,
            annotations_df=filtered_annot,
            k_multipliers=[0.5, 1.0, 2.0, 4.0],
            min_score=0.1,
            verbosity=0
        )
        
        windowed_recall = compute_windowed_recall(
            predictions=predictions,
            annotations_df=filtered_annot,
            k_values=[1, 2, 5, 10, 20],
            min_score=0.1,
            verbosity=0
        )
        
        eval_results[filter_mode] = {
            'label': label,
            'error_df': error_df,
            'positions_df': positions_df,
            'pr_metrics': pr_metrics,
            'topk_accuracy': topk_accuracy,
            'windowed_recall': windowed_recall,
        }
        
        # Print concise results for this filter
        _print_eval_results(positions_df, pr_metrics, topk_accuracy, windowed_recall)
    
    # =========================================================================
    # Step 4: Gap analysis (what meta/agentic layers need to address)
    # =========================================================================
    if args.gap_analysis or use_dual:
        print()
        gap_results = splice_site_gap_analysis(
            predictions=predictions,
            annotations_df=annotations_df,
            threshold=args.threshold,
            nearby_window=5,
            verbosity=1
        )
    
    # Print comparison summary if dual mode
    if use_dual and len(eval_results) == 2:
        _print_dual_comparison(eval_results)
    
    # Sample FP/FN examples (use the 'all' or primary evaluation)
    primary_key = 'all' if 'all' in eval_results else list(eval_results.keys())[0]
    positions_df = eval_results[primary_key]['positions_df']
    
    all_fp = len(positions_df.filter(pl.col('pred_type') == 'FP'))
    all_fn = len(positions_df.filter(pl.col('pred_type') == 'FN'))
    
    if all_fp > 0:
        print(f"\n⚠️  False Positive Examples (first 5):")
        fp_df = positions_df.filter(pl.col('pred_type') == 'FP').head(5)
        for row in fp_df.iter_rows(named=True):
            site = row['splice_type']
            score = row[f'{site}_score']
            print(f"   {row['gene_name']} {row['chrom']}:{row['position']} ({site}, score={score:.4f})")
    
    if all_fn > 0:
        print(f"\n❌ False Negative Examples (first 5):")
        fn_df = positions_df.filter(pl.col('pred_type') == 'FN').head(5)
        for row in fn_df.iter_rows(named=True):
            site = row['splice_type']
            score = row[f'{site}_score']
            print(f"   {row['gene_name']} {row['chrom']}:{row['position']} ({site}, score={score:.4f})")
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    print(f"\n💡 Tips:")
    print(f"   --threshold: Adjust to trade off precision vs recall")
    print(f"   --gap-analysis: See what meta/agentic layers need to fix")
    print(f"   --transcript-filter: 'canonical', 'shared', 'all', or 'dual' (default)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
