#!/usr/bin/env python
"""Phase 1 Example: Single Gene Splice Site Prediction.

Demonstrates the complete Phase 1 workflow:
1. Load genomic resources (GTF, FASTA)
2. Extract gene data
3. Load base model (OpenSpliceAI)
4. Generate splice site predictions
5. Display results

Usage:
    python 01_phase1_prediction.py --gene BRCA1
    python 01_phase1_prediction.py --gene TP53 --model openspliceai

Example:
    python 01_phase1_prediction.py --gene BRCA1
"""

import argparse
import sys
from pathlib import Path
import time

import polars as pl

# Add project to path - using marker-based root finding (no fragile parent.parent.parent!)
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner


def main():
    """Run Phase 1 prediction example."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Predict splice sites for a single gene"
    )
    parser.add_argument(
        "--gene",
        required=True,
        help="Gene symbol (e.g., BRCA1, TP53)"
    )
    parser.add_argument(
        "--model",
        default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 1 Example: Single Gene Splice Site Prediction")
    print("=" * 80)
    print(f"\nGene: {args.gene}")
    print(f"Model: {args.model}")
    
    # Initialize runner
    print("\n🔧 Initializing BaseModelRunner...")
    runner = BaseModelRunner()
    
    # Run prediction
    print(f"🧬 Running splice site prediction for {args.gene}...")
    
    result = runner.run_single_model(
        model_name=args.model,
        target_genes=[args.gene],
        test_name=f"example_{args.gene}",
        mode="test",
        coverage="gene_subset",
        verbosity=1
    )
    
    elapsed = result.runtime_seconds
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    if result.success:
        print(f"✅ Prediction successful!")
        print(f"\n⏱️  Runtime: {elapsed:.2f}s")
        print(f"📊 Positions predicted: {len(result.positions):,}")
        print(f"🧬 Genes processed: {len(result.processed_genes)}")
        
        if result.processed_genes:
            print(f"   - {', '.join(sorted(result.processed_genes))}")
        
        # Only show missing genes if we actually have zero predictions
        if result.missing_genes and len(result.positions) == 0:
            print(f"\n⚠️  Missing genes: {', '.join(sorted(result.missing_genes))}")
        
        # Show predicted splice sites (most interesting positions)
        print(f"\n🎯 Top Predicted Splice Sites:")
        
        # Add max splice probability column for sorting
        positions_with_max = result.positions.with_columns([
            pl.max_horizontal(['donor_prob', 'acceptor_prob']).alias('max_splice_prob')
        ])
        
        # Get top donor sites
        top_donors = positions_with_max.filter(
            pl.col('donor_prob') > pl.col('acceptor_prob')
        ).sort('donor_prob', descending=True).head(5)
        
        if len(top_donors) > 0:
            print(f"\n   Top 5 Donor Sites:")
            for row in top_donors.iter_rows(named=True):
                pos = row['position']
                score = row['donor_prob']
                chrom = row['seqname']
                print(f"      {chrom}:{pos:>10,}  score={score:.6f}")
        
        # Get top acceptor sites
        top_acceptors = positions_with_max.filter(
            pl.col('acceptor_prob') > pl.col('donor_prob')
        ).sort('acceptor_prob', descending=True).head(5)
        
        if len(top_acceptors) > 0:
            print(f"\n   Top 5 Acceptor Sites:")
            for row in top_acceptors.iter_rows(named=True):
                pos = row['position']
                score = row['acceptor_prob']
                chrom = row['seqname']
                print(f"      {chrom}:{pos:>10,}  score={score:.6f}")
        
        # Count sites above threshold
        threshold = 0.5
        high_donor = len(positions_with_max.filter(pl.col('donor_prob') >= threshold))
        high_acceptor = len(positions_with_max.filter(pl.col('acceptor_prob') >= threshold))
        
        print(f"\n📊 Splice Sites Detected (threshold={threshold}):")
        print(f"      Donor sites: {high_donor}")
        print(f"      Acceptor sites: {high_acceptor}")
        print(f"      Total: {high_donor + high_acceptor}")
        
        # Show a couple TN examples
        low_prob_positions = positions_with_max.filter(
            (pl.col('donor_prob') < 0.01) & (pl.col('acceptor_prob') < 0.01)
        ).head(3)
        
        if len(low_prob_positions) > 0:
            print(f"\n✅ Example True Negatives (low scores):")
            for row in low_prob_positions.iter_rows(named=True):
                pos = row['position']
                donor = row['donor_prob']
                acceptor = row['acceptor_prob']
                chrom = row['seqname']
                print(f"      {chrom}:{pos:>10,}  donor={donor:.2e}, acceptor={acceptor:.2e}")
        
        # Score distribution statistics
        print(f"\n📈 Score Distribution:")
        print(f"   Donor prob - max: {result.positions['donor_prob'].max():.6f}, mean: {result.positions['donor_prob'].mean():.6f}")
        print(f"   Acceptor prob - max: {result.positions['acceptor_prob'].max():.6f}, mean: {result.positions['acceptor_prob'].mean():.6f}")
    else:
        print(f"❌ Prediction failed: {result.error}")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
