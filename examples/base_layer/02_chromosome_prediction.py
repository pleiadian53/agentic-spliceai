#!/usr/bin/env python
"""Phase 1 Example: Chromosome-Wide Splice Site Prediction.

Demonstrates prediction on multiple genes from a chromosome:
1. Load genomic resources
2. Filter genes by chromosome
3. Run predictions on N genes
4. Display aggregate results

Usage:
    python 02_chromosome_prediction.py --chromosome chr21 --genes 10
    python 02_chromosome_prediction.py --chromosome chr17 --genes 5 --model openspliceai

Example:
    python 02_chromosome_prediction.py --chromosome chr21 --genes 10
"""

import argparse
import sys
from pathlib import Path
import time

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    load_gene_annotations,
    filter_by_chromosomes,
)
from agentic_spliceai.splice_engine.resources.model_resources import get_model_resources
import polars as pl


def main():
    """Run Phase 1 chromosome prediction example."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Predict splice sites for multiple genes on a chromosome"
    )
    parser.add_argument(
        "--chromosome",
        required=True,
        help="Chromosome (e.g., chr21, chr17)"
    )
    parser.add_argument(
        "--genes",
        type=int,
        default=10,
        help="Number of genes to process (default: 10)"
    )
    parser.add_argument(
        "--model",
        default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 1 Example: Chromosome-Wide Splice Site Prediction")
    print("=" * 80)
    print(f"\nChromosome: {args.chromosome}")
    print(f"Number of genes: {args.genes}")
    print(f"Model: {args.model}")
    print()
    
    # Get model-specific resources (ensures consistent GTF for gene selection and prediction)
    model_resources = get_model_resources(args.model)
    registry = model_resources.get_registry()

    print(f"Build: {model_resources.build}")
    print(f"Annotation Source: {model_resources.annotation_source}")
    print()

    # Load gene annotations for chromosome
    print(f"📂 Loading gene annotations for {args.chromosome}...")

    # Get GTF path from model-specific registry
    gtf_path = registry.get_gtf_path()
    
    print(f"   GTF: {gtf_path}")
    
    # Load annotations filtered to target chromosome (more efficient)
    genes_df = load_gene_annotations(
        gtf_path=gtf_path,
        chromosomes=[args.chromosome],
        verbosity=1
    )
    
    # Select N genes (filter out empty gene names)
    valid_genes = genes_df.filter(pl.col('gene_name') != '').filter(pl.col('gene_name').is_not_null())
    gene_list = valid_genes['gene_name'].head(args.genes).to_list()
    
    print(f"✓ Found {len(genes_df)} genes on {args.chromosome}")
    print(f"✓ Selected {len(gene_list)} genes for prediction:")
    print(f"   {', '.join(gene_list)}")
    print()
    
    # Initialize runner
    print("🔧 Initializing BaseModelRunner...")
    runner = BaseModelRunner()
    
    # Run prediction
    print(f"🧬 Running splice site prediction for {len(gene_list)} genes...")
    
    result = runner.run_single_model(
        model_name=args.model,
        target_genes=gene_list,
        test_name=f"example_{args.chromosome}",
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
        print(f"📊 Total positions predicted: {len(result.positions):,}")
        print(f"🧬 Genes processed: {len(result.processed_genes)}/{len(gene_list)}")
        print(f"⚡ Throughput: {len(result.positions) / elapsed:.0f} positions/second")
        
        if result.processed_genes:
            print(f"\n✅ Successfully processed:")
            for gene in sorted(result.processed_genes):
                # Try both gene_id and gene_name columns
                if 'gene_name' in result.positions.columns:
                    gene_positions = result.positions.filter(
                        (pl.col('gene_name') == gene) | (pl.col('gene_id') == gene)
                    )
                else:
                    gene_positions = result.positions.filter(pl.col('gene_id') == gene)
                
                # Count splice sites for this gene
                threshold = 0.5
                gene_donors = len(gene_positions.filter(pl.col('donor_prob') >= threshold))
                gene_acceptors = len(gene_positions.filter(pl.col('acceptor_prob') >= threshold))
                
                print(f"   - {gene}: {len(gene_positions):,} positions, {gene_donors} donors, {gene_acceptors} acceptors")
        
        # Only show missing genes if they truly weren't processed
        if result.missing_genes and len(result.positions) < len(gene_list) * 1000:
            print(f"\n⚠️  Missing genes: {', '.join(sorted(result.missing_genes))}")
        
        # Aggregate splice site statistics
        threshold = 0.5
        total_donors = len(result.positions.filter(pl.col('donor_prob') >= threshold))
        total_acceptors = len(result.positions.filter(pl.col('acceptor_prob') >= threshold))
        
        print(f"\n📊 Aggregate Splice Site Detection (threshold={threshold}):")
        print(f"   Total donor sites: {total_donors}")
        print(f"   Total acceptor sites: {total_acceptors}")
        print(f"   Total sites: {total_donors + total_acceptors}")
        
        # Summary statistics
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
