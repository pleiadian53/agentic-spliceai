#!/usr/bin/env python
"""Phase 1 Example: Splice Site Prediction with Evaluation Metrics.

This example demonstrates:
1. Running splice site predictions
2. Loading ground truth annotations
3. Evaluating predictions (TP/FP/FN/TN)
4. Computing performance metrics (F1, precision, recall, PR-AUC)

Usage:
    python 03_prediction_with_evaluation.py --gene BRCA1
    python 03_prediction_with_evaluation.py --gene TP53 --model openspliceai --threshold 0.5
    python 03_prediction_with_evaluation.py --genes BRCA1 TP53 MYC

Example output:
    ✅ Prediction successful!
    ⏱️  Runtime: 3.42s
    📊 Positions predicted: 19,070
    
    🎯 Splice Sites Detected:
       Donor sites: 12 (TP: 10, FP: 2, FN: 1)
       Acceptor sites: 13 (TP: 11, FP: 2, FN: 1)
    
    📈 Performance Metrics:
       Precision: 0.8462
       Recall: 0.9167
       F1 Score: 0.8800
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
from agentic_spliceai.splice_engine.base_layer.data.preparation import prepare_splice_site_annotations
from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
    evaluate_splice_site_predictions,
    compute_pr_metrics
)
from agentic_spliceai.splice_engine.base_layer.prediction.core import predict_splice_sites_for_genes
from agentic_spliceai.splice_engine.base_layer.prediction.core import load_spliceai_models
from agentic_spliceai.splice_engine.base_layer.data.preparation import load_gene_annotations, extract_sequences
import polars as pl


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
    
    # Determine build and annotation source from model
    build = 'GRCh38' if args.model == 'openspliceai' else 'GRCh37'
    annotation_source = 'mane' if args.model == 'openspliceai' else 'ensembl'
    
    print(f"📋 Configuration:")
    print(f"   Build: {build}")
    print(f"   Annotation: {annotation_source}")
    print()
    
    # Step 1: Load ground truth annotations
    print("📂 Loading ground truth splice site annotations...")
    
    from pathlib import Path
    cache_dir = Path.home() / ".cache" / "agentic-spliceai" / "annotations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_result = prepare_splice_site_annotations(
        output_dir=str(cache_dir),
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
    genes_df = load_gene_annotations()
    genes_df = genes_df.filter(pl.col('gene_name').is_in(gene_list))
    genes_df = extract_sequences(genes_df)
    print(f"✓ Prepared {len(genes_df)} genes")
    print()
    
    print("🔧 Loading models...")
    models = load_spliceai_models(model_type=args.model, verbosity=1)
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
    
    # Step 3: Evaluate predictions
    print("📊 Evaluating predictions...")
    error_df, positions_df, pr_metrics = evaluate_splice_site_predictions(
        predictions=predictions,
        annotations_df=annotations_df,
        threshold=args.threshold,
        consensus_window=args.consensus_window,
        collect_tn=True,
        no_tn_sampling=True,  # Keep all positions for analysis
        verbosity=1,
        return_pr_metrics=True
    )
    
    # Step 4: Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    print(f"✅ Prediction and evaluation complete!")
    print(f"\n📊 Total positions analyzed: {len(positions_df):,}")
    
    # Splice site detection statistics
    print(f"\n🎯 Splice Sites Detected:")
    
    for site_type in ['donor', 'acceptor']:
        site_df = positions_df.filter(pl.col('splice_type') == site_type)
        
        tp = len(site_df.filter(pl.col('pred_type') == 'TP'))
        fp = len(site_df.filter(pl.col('pred_type') == 'FP'))
        fn = len(site_df.filter(pl.col('pred_type') == 'FN'))
        tn = len(site_df.filter(pl.col('pred_type') == 'TN'))
        
        detected = tp + fp
        true_total = tp + fn
        
        print(f"\n   {site_type.capitalize()} sites:")
        print(f"      True positives (TP):  {tp:>6}")
        print(f"      False positives (FP): {fp:>6}")
        print(f"      False negatives (FN): {fn:>6}")
        print(f"      True negatives (TN):  {tn:>6,}")
        print(f"      ─────────────────────────")
        print(f"      Detected: {detected} / {true_total} true sites")
        
        # Site-specific metrics
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"      Precision: {precision:.4f}")
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"      Recall: {recall:.4f}")
        
        if tp + fp > 0 and tp + fn > 0:
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"      F1 Score: {f1:.4f}")
    
    # Overall performance metrics
    print(f"\n📈 Overall Performance Metrics:")
    
    all_tp = len(positions_df.filter(pl.col('pred_type') == 'TP'))
    all_fp = len(positions_df.filter(pl.col('pred_type') == 'FP'))
    all_fn = len(positions_df.filter(pl.col('pred_type') == 'FN'))
    all_tn = len(positions_df.filter(pl.col('pred_type') == 'TN'))
    
    if all_tp + all_fp > 0:
        overall_precision = all_tp / (all_tp + all_fp)
        print(f"   Precision: {overall_precision:.4f}")
    
    if all_tp + all_fn > 0:
        overall_recall = all_tp / (all_tp + all_fn)
        print(f"   Recall: {overall_recall:.4f}")
    
    if all_tp + all_fp > 0 and all_tp + all_fn > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        print(f"   F1 Score: {overall_f1:.4f}")
    
    # PR-AUC metrics
    if pr_metrics:
        print(f"\n📊 PR-AUC Metrics (continuous scores):")
        print(f"   Donor AP: {pr_metrics['donor_ap']:.4f}")
        print(f"   Donor PR-AUC: {pr_metrics['donor_pr_auc']:.4f}")
        print(f"   Acceptor AP: {pr_metrics['acceptor_ap']:.4f}")
        print(f"   Acceptor PR-AUC: {pr_metrics['acceptor_pr_auc']:.4f}")
        print(f"   ─────────────────────────")
        print(f"   Macro AP: {pr_metrics['macro_ap']:.4f}")
        print(f"   Macro PR-AUC: {pr_metrics['macro_pr_auc']:.4f}")
    
    # Sample predictions
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
    print(f"\n💡 Tip: Adjust --threshold to trade off precision vs recall")
    print(f"   Lower threshold → Higher recall (fewer FN, more FP)")
    print(f"   Higher threshold → Higher precision (fewer FP, more FN)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
