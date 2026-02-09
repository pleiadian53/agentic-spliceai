#!/usr/bin/env python
"""Check coordinate consistency between base model predictions and annotations.

This diagnostic tool helps identify position offset issues by:
1. Running predictions on a sample of genes
2. Comparing predictions to ground truth annotations
3. Detecting optimal position adjustments (per site type, per strand)
4. Reporting metrics before and after adjustment

Use Cases:
-----------
- Adding a new base model → Check if adjustments are needed
- Poor performance debugging → Is this a coordinate issue?
- Switching genome builds → Verify alignment
- Model comparison → Are both models aligned correctly?

Examples:
---------
# Check SpliceAI on specific genes
python check_coordinate_consistency.py --model spliceai --genes TP53 BRCA1

# Quick check using random sample
python check_coordinate_consistency.py --model openspliceai --sample 20

# Check with specific chromosome
python check_coordinate_consistency.py --model spliceai --sample 10 --chromosome chr17

# Save adjustment file
python check_coordinate_consistency.py --model spliceai --sample 20 --save-adjustments
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

import argparse
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import polars as pl
import numpy as np

from agentic_spliceai.splice_engine.resources import get_model_resources
from agentic_spliceai.splice_engine.base_layer.data import (
    prepare_gene_data,
    prepare_splice_site_annotations
)
from agentic_spliceai.splice_engine.base_layer.prediction import (
    load_spliceai_models,
    predict_splice_sites_for_genes
)


def detect_position_offsets(
    predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    threshold: float = 0.5,
    max_offset: int = 5,
    verbosity: int = 1
) -> Dict[str, Dict[str, int]]:
    """Detect optimal position offsets by comparing predictions to ground truth.
    
    Tests various position offsets and finds the one that maximizes recall.
    
    Parameters
    ----------
    predictions_df : pl.DataFrame
        Model predictions with columns: position, gene_name, strand, donor_prob, acceptor_prob
    ground_truth_df : pl.DataFrame
        Ground truth splice sites with columns: position, gene_name, strand, site_type
    threshold : float
        Score threshold for calling a prediction positive
    max_offset : int
        Maximum offset to test (tests from -max_offset to +max_offset)
    verbosity : int
        Output verbosity level
        
    Returns
    -------
    dict
        Detected offsets: {'donor': {'plus': int, 'minus': int}, 
                          'acceptor': {'plus': int, 'minus': int}}
    """
    if verbosity >= 1:
        print("\n🔍 Detecting optimal position offsets...")
        print(f"   Testing offsets from -{max_offset} to +{max_offset} bp")
    
    # Initialize results
    best_offsets = {
        'donor': {'plus': 0, 'minus': 0},
        'acceptor': {'plus': 0, 'minus': 0}
    }
    
    # For each combination of site_type and strand
    for site_type in ['donor', 'acceptor']:
        for strand in ['+', '-']:
            # Filter ground truth for this combination
            gt_sites = ground_truth_df.filter(
                (pl.col('site_type') == site_type) & 
                (pl.col('strand') == strand)
            )
            
            if len(gt_sites) == 0:
                continue
            
            # Get true positions
            true_positions = set(gt_sites['position'].to_list())
            
            # Test each offset
            best_recall = 0
            best_offset = 0
            
            for offset in range(-max_offset, max_offset + 1):
                # Get predictions with this offset applied
                prob_col = f'{site_type}_prob'
                
                # Filter predictions by strand and threshold
                preds = predictions_df.filter(
                    (pl.col('strand') == strand) &
                    (pl.col(prob_col) >= threshold)
                )
                
                if len(preds) == 0:
                    continue
                
                # Apply offset to predicted positions
                pred_positions = set((pl.Series(preds['position']) + offset).to_list())
                
                # Calculate recall
                matches = len(true_positions & pred_positions)
                recall = matches / len(true_positions) if len(true_positions) > 0 else 0
                
                if recall > best_recall:
                    best_recall = recall
                    best_offset = offset
            
            # Store best offset
            strand_key = 'plus' if strand == '+' else 'minus'
            best_offsets[site_type][strand_key] = best_offset
            
            if verbosity >= 2:
                print(f"   {site_type.capitalize()} {strand} strand: {best_offset:+d} bp (recall: {best_recall:.1%})")
    
    return best_offsets


def apply_offsets(
    predictions_df: pl.DataFrame,
    offsets: Dict[str, Dict[str, int]]
) -> pl.DataFrame:
    """Apply position offsets to predictions.
    
    Parameters
    ----------
    predictions_df : pl.DataFrame
        Predictions with position, strand, donor_prob, acceptor_prob columns
    offsets : dict
        Offsets to apply: {'donor': {'plus': int, 'minus': int}, ...}
        
    Returns
    -------
    pl.DataFrame
        Predictions with adjusted positions
    """
    # Create adjusted dataframe
    adjusted = predictions_df.clone()
    
    # Apply offsets based on strand and dominant site type
    for strand, strand_key in [('+', 'plus'), ('-', 'minus')]:
        strand_mask = adjusted['strand'] == strand
        
        # Determine which site type is more likely for each position
        # and apply the corresponding offset
        donor_offset = offsets['donor'][strand_key]
        acceptor_offset = offsets['acceptor'][strand_key]
        
        # For simplicity, apply the offset based on which score is higher
        # In a full implementation, you'd create separate predictions for each type
        donor_dominant = adjusted['donor_prob'] > adjusted['acceptor_prob']
        
        # Apply donor offset where donor is dominant and on this strand
        donor_mask = strand_mask & donor_dominant
        if donor_mask.sum() > 0 and donor_offset != 0:
            adjusted = adjusted.with_columns(
                pl.when(donor_mask)
                .then(pl.col('position') + donor_offset)
                .otherwise(pl.col('position'))
                .alias('position')
            )
        
        # Apply acceptor offset where acceptor is dominant and on this strand
        acceptor_mask = strand_mask & ~donor_dominant
        if acceptor_mask.sum() > 0 and acceptor_offset != 0:
            adjusted = adjusted.with_columns(
                pl.when(acceptor_mask)
                .then(pl.col('position') + acceptor_offset)
                .otherwise(pl.col('position'))
                .alias('position')
            )
    
    return adjusted


def evaluate_predictions(
    predictions_df: pl.DataFrame,
    ground_truth_df: pl.DataFrame,
    threshold: float = 0.5,
    consensus_window: int = 0
) -> Dict[str, float]:
    """Evaluate prediction accuracy.
    
    Parameters
    ----------
    predictions_df : pl.DataFrame
        Predictions with position, donor_prob, acceptor_prob columns
    ground_truth_df : pl.DataFrame
        Ground truth with position, site_type columns
    threshold : float
        Score threshold
    consensus_window : int
        Window around true site to consider a match (0 = exact match)
        
    Returns
    -------
    dict
        Metrics: {'recall': float, 'precision': float, 'f1': float, ...}
    """
    # Get true sites by type
    true_donors = set(ground_truth_df.filter(pl.col('site_type') == 'donor')['position'].to_list())
    true_acceptors = set(ground_truth_df.filter(pl.col('site_type') == 'acceptor')['position'].to_list())
    
    # Get predicted sites
    pred_donors = set(predictions_df.filter(pl.col('donor_prob') >= threshold)['position'].to_list())
    pred_acceptors = set(predictions_df.filter(pl.col('acceptor_prob') >= threshold)['position'].to_list())
    
    # Calculate matches (with window if specified)
    if consensus_window > 0:
        # Windowed matching
        donor_tp = sum(1 for pos in true_donors 
                      if any(abs(pos - p) <= consensus_window for p in pred_donors))
        acceptor_tp = sum(1 for pos in true_acceptors 
                         if any(abs(pos - p) <= consensus_window for p in pred_acceptors))
    else:
        # Exact matching
        donor_tp = len(true_donors & pred_donors)
        acceptor_tp = len(true_acceptors & pred_acceptors)
    
    # Calculate metrics
    total_tp = donor_tp + acceptor_tp
    total_fn = (len(true_donors) - donor_tp) + (len(true_acceptors) - acceptor_tp)
    total_fp = (len(pred_donors) - donor_tp) + (len(pred_acceptors) - acceptor_tp)
    
    total_true = len(true_donors) + len(true_acceptors)
    total_pred = len(pred_donors) + len(pred_acceptors)
    
    recall = total_tp / total_true if total_true > 0 else 0
    precision = total_tp / total_pred if total_pred > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': total_tp,
        'fn': total_fn,
        'fp': total_fp,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'n_true': total_true,
        'n_pred': total_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description='Check coordinate consistency for base models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Base model to check (spliceai, openspliceai)')
    parser.add_argument('--genes', nargs='+',
                       help='Specific genes to check (e.g., TP53 BRCA1)')
    parser.add_argument('--sample', type=int,
                       help='Number of random genes to sample')
    parser.add_argument('--chromosome', type=str,
                       help='Chromosome to sample from (e.g., chr17)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Score threshold (default: 0.5)')
    parser.add_argument('--max-offset', type=int, default=5,
                       help='Maximum offset to test in bp (default: 5)')
    parser.add_argument('--save-adjustments', action='store_true',
                       help='Save detected adjustments to file')
    parser.add_argument('--output', type=str,
                       help='Output file for adjustments (default: auto-generated)')
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                       help='Output verbosity (0=quiet, 1=normal, 2=verbose)')
    
    args = parser.parse_args()
    
    # Banner
    print("=" * 80)
    print("Coordinate Consistency Check")
    print("=" * 80)
    print()
    
    # Get model resources
    resources = get_model_resources(args.model)
    build = resources.build
    annotation_source = resources.annotation_source
    
    print(f"📋 Configuration:")
    print(f"   Model: {args.model}")
    print(f"   Build: {build}")
    print(f"   Annotation: {annotation_source}")
    print()
    
    # Determine gene list
    if args.genes:
        gene_list = args.genes
        print(f"📊 Sample: {len(gene_list)} specified genes")
        print(f"   Genes: {', '.join(gene_list)}")
    elif args.sample:
        # For now, use predefined sample (in full implementation, would sample from GTF)
        print(f"⚠️  Random sampling not yet implemented, using TP53 as example")
        gene_list = ['TP53']
    else:
        print("❌ Error: Must specify either --genes or --sample")
        return 1
    
    if args.chromosome:
        print(f"   Chromosome: {args.chromosome}")
    print()
    
    # Load ground truth
    print("📂 Loading ground truth annotations...")
    annotations_dir = resources.get_annotations_dir()
    annotations_result = prepare_splice_site_annotations(
        output_dir=str(annotations_dir),
        genes=gene_list,
        build=build,
        annotation_source=annotation_source,
        verbosity=0
    )
    
    ground_truth_df = annotations_result['splice_sites_df']
    print(f"✓ Loaded {len(ground_truth_df)} ground truth splice sites")
    print(f"   Donors: {len(ground_truth_df.filter(pl.col('site_type') == 'donor'))}")
    print(f"   Acceptors: {len(ground_truth_df.filter(pl.col('site_type') == 'acceptor'))}")
    print()
    
    # Prepare gene data
    print("🧬 Preparing gene data...")
    genes_df = prepare_gene_data(
        genes=gene_list,
        build=build,
        annotation_source=annotation_source,
        verbosity=0
    )
    print(f"✓ Loaded {len(genes_df)} genes")
    print()
    
    # Load models and run predictions
    print("🔧 Loading models and running predictions...")
    models = load_spliceai_models(model_type=args.model, build=build, verbosity=0)
    print(f"✓ Loaded {len(models)} model(s)")
    
    predictions_df = predict_splice_sites_for_genes(
        gene_df=genes_df,
        models=models,
        context=10000,
        output_format='dataframe',
        verbosity=0
    )
    print(f"✓ Generated {len(predictions_df)} predictions")
    print()
    
    # Detect offsets
    offsets = detect_position_offsets(
        predictions_df=predictions_df,
        ground_truth_df=ground_truth_df,
        threshold=args.threshold,
        max_offset=args.max_offset,
        verbosity=args.verbosity
    )
    
    # Report detected offsets
    print()
    print("=" * 80)
    print("📐 Detected Position Offsets")
    print("=" * 80)
    print()
    
    any_offset = False
    for site_type in ['donor', 'acceptor']:
        print(f"{site_type.capitalize()} sites:")
        for strand_key, strand_symbol in [('plus', '+'), ('minus', '-')]:
            offset = offsets[site_type][strand_key]
            status = "✅ (aligned)" if offset == 0 else f"⚠️  (needs adjustment)"
            print(f"   {strand_symbol} strand: {offset:+3d} bp {status}")
            if offset != 0:
                any_offset = True
        print()
    
    # Evaluate before and after
    print("=" * 80)
    print("📈 Performance Metrics")
    print("=" * 80)
    print()
    
    # Before adjustment (exact match)
    metrics_before_exact = evaluate_predictions(
        predictions_df, ground_truth_df, 
        threshold=args.threshold, consensus_window=0
    )
    
    # Before adjustment (windowed)
    metrics_before_window = evaluate_predictions(
        predictions_df, ground_truth_df,
        threshold=args.threshold, consensus_window=2
    )
    
    print("Before adjustment:")
    print(f"   Exact match:")
    print(f"      Recall:    {metrics_before_exact['recall']:6.1%}")
    print(f"      Precision: {metrics_before_exact['precision']:6.1%}")
    print(f"      F1 Score:  {metrics_before_exact['f1']:6.1%}")
    print(f"   Windowed (±2bp):")
    print(f"      Recall:    {metrics_before_window['recall']:6.1%}")
    print()
    
    if any_offset:
        # Apply offsets and re-evaluate
        adjusted_predictions = apply_offsets(predictions_df, offsets)
        
        metrics_after = evaluate_predictions(
            adjusted_predictions, ground_truth_df,
            threshold=args.threshold, consensus_window=0
        )
        
        print("After adjustment:")
        print(f"   Exact match:")
        print(f"      Recall:    {metrics_after['recall']:6.1%} ", end="")
        
        # Show improvement
        improvement = metrics_after['recall'] - metrics_before_exact['recall']
        if improvement > 0:
            print(f"(+{improvement:.1%} ✅)")
        else:
            print()
        
        print(f"      Precision: {metrics_after['precision']:6.1%}")
        print(f"      F1 Score:  {metrics_after['f1']:6.1%}")
        print()
    
    # Check for suspicious pattern: exact == windowed (suggests methodology issue)
    windowed_equals_exact = (metrics_before_window['recall'] == metrics_before_exact['recall'])
    
    # Recommendation
    print("=" * 80)
    print("💡 Recommendation")
    print("=" * 80)
    print()
    
    if windowed_equals_exact and metrics_before_exact['recall'] < 0.7:
        print("⚠️  WARNING: Diagnostic tool limitation detected")
        print()
        print("   Windowed recall equals exact recall, AND recall is low.")
        print("   This suggests the model may need SCORE ARRAY adjustment,")
        print("   not simple POSITION offset adjustment.")
        print()
        print("   This tool tests position shifts AFTER peak detection, but")
        print("   some models (like SpliceAI) need the entire score array")
        print("   rolled BEFORE peak detection.")
        print()
        print("   📚 For SpliceAI specifically:")
        print("      Known offsets exist (+2bp donor/+strand, +1bp donor/-strand, etc.)")
        print("      These require score array rolling, not position shifting.")
        print()
        print("   🔧 Next steps:")
        print("      1. Check if you're using SpliceAI (known to need array rolling)")
        print("      2. See: dev/base_layer/SPLICE_COORDINATE_ADJUSTMENT_TODO.md")
        print("      3. Port the full MetaSpliceAI adjustment system")
        print()
        print("   ⚠️  Do NOT trust the '0bp offset' result above if using SpliceAI!")
        print()
    elif not any_offset:
        print("✅ Coordinates are well-aligned (no adjustment needed)")
        print()
        print("   The model's predictions align correctly with the annotations.")
        print("   No position adjustments are required.")
        if windowed_equals_exact:
            print()
            print("   ℹ️  Note: Windowed recall equals exact recall.")
            print("   This means predictions are either exact matches or far off (>2bp).")
    else:
        print("⚠️  Position adjustments recommended")
        print()
        print("   The model's predictions are systematically offset from annotations.")
        print("   Applying the detected adjustments will improve recall by")
        print(f"   {improvement:.1%} ({metrics_before_exact['recall']:.1%} → {metrics_after['recall']:.1%})")
        print()
        
        # Save adjustments if requested
        if args.save_adjustments:
            output_file = args.output
            if not output_file:
                model_dir = resources.get_eval_dir().parent
                output_file = model_dir / "coordinate_adjustments.json"
            else:
                output_file = Path(output_file)
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(offsets, f, indent=2)
            
            print(f"💾 Adjustment file saved: {output_file}")
            print()
            print("   To use these adjustments in predictions, pass this file")
            print("   to the prediction workflow or apply them manually.")
    
    print()
    print("=" * 80)
    print("✅ Check complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
