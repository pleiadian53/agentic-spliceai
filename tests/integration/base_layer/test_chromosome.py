#!/usr/bin/env python
"""
Test script for full chromosome prediction.

Tests BaseModelRunner on a complete chromosome (chr21 - medium sized).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

def test_chromosome_prediction():
    """Test prediction on chromosome 21 (medium-sized chromosome)."""
    print("=" * 80)
    print("Chromosome Test: Full Chromosome Prediction (chr21)")
    print("=" * 80)
    
    # Initialize runner
    runner = BaseModelRunner()
    
    # Run prediction on chromosome 21
    print("\n🧪 Testing BaseModelRunner on full chromosome...")
    print("   Model: openspliceai")
    print("   Chromosome: 21 (~200-300 genes)")
    print("   Test: phase1_chr21")
    print()
    
    # Note: run_single_model doesn't support chromosomes parameter yet
    # For now, we'll get all MANE genes and filter in post-processing
    # TODO: Add chromosome filtering support to BaseModelConfig
    
    result = runner.run_single_model(
        model_name='openspliceai',
        target_genes=None,  # All genes on chr21 (filter handled internally)
        test_name='phase1_chr21',
        mode='test',
        verbosity=2
    )
    
    # Filter to chr21 if we got other chromosomes
    if result.positions is not None and result.positions.height > 0:
        import polars as pl
        if 'seqname' in result.positions.columns:
            result.positions = result.positions.filter(
                (pl.col('seqname') == 'chr21') | (pl.col('seqname') == '21')
            )
    
    # Check results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Processed genes: {len(result.processed_genes)}")
    print(f"Missing genes: {len(result.missing_genes)}")
    print(f"Positions predicted: {result.positions.height if result.positions is not None else 0}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.positions is not None and result.positions.height > 0:
        print(f"\nFirst 10 predictions:")
        print(result.positions.head(10))
        
        print(f"\nLast 10 predictions:")
        print(result.positions.tail(10))
        
        print(f"\nColumns: {result.positions.columns}")
        print(f"Shape: {result.positions.shape}")
        
        # Show some statistics
        print(f"\n📊 Statistics:")
        print(f"   Total positions: {result.positions.height:,}")
        print(f"   Unique genes: {result.positions['gene_id'].n_unique()}")
        
        # Show genes processed
        if len(result.processed_genes) > 0:
            genes_list = sorted(list(result.processed_genes))[:10]
            print(f"\n   First 10 genes processed: {', '.join(genes_list)}")
    
    # Verdict
    print("\n" + "=" * 80)
    if result.success and result.positions is not None and result.positions.height > 0:
        print("✅ Chromosome Test PASSED!")
        print(f"   Successfully predicted on {len(result.processed_genes)} genes")
        print(f"   Total predictions: {result.positions.height:,} positions")
        return True
    else:
        print("❌ Chromosome Test FAILED!")
        if result.error:
            print(f"   Error: {result.error}")
        return False
    print("=" * 80)

if __name__ == '__main__':
    success = test_chromosome_prediction()
    sys.exit(0 if success else 1)
