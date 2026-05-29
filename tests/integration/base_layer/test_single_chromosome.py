#!/usr/bin/env python
"""
Test script for single chromosome prediction (simplified).

This test manually specifies chr21 genes from MANE annotations.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

# Sample of chr21 genes from MANE (you can add more)
CHR21_GENES = [
    'RUNX1',      # AML1 - Acute myeloid leukemia
    'APP',        # Amyloid beta precursor protein (Alzheimer's)
    'SOD1',       # Superoxide dismutase (ALS)
    'DYRK1A',     # Dual-specificity kinase (Down syndrome)
    'ITSN1',      # Intersectin 1
    'COL6A1',     # Collagen Type VI Alpha 1
    'COL6A2',     # Collagen Type VI Alpha 2  
    'ADAMTS1',    # ADAM metallopeptidase
    'CBS',        # Cystathionine beta-synthase
    'CRYAA',      # Crystallin Alpha A
]

def test_chr21_genes():
    """Test prediction on selected chr21 genes."""
    print("=" * 80)
    print(f"Chromosome 21 Test: {len(CHR21_GENES)} Selected Genes")
    print("=" * 80)
    
    # Initialize runner
    runner = BaseModelRunner()
    
    # Run prediction
    print(f"\n🧪 Testing {len(CHR21_GENES)} chr21 genes...")
    print(f"   Model: openspliceai")
    print(f"   Genes: {', '.join(CHR21_GENES[:5])}...")
    print(f"   Test: phase1_chr21_selected")
    print()
    
    result = runner.run_single_model(
        model_name='openspliceai',
        target_genes=CHR21_GENES,
        test_name='phase1_chr21_selected',
        mode='test',
        verbosity=2
    )
    
    # Check results
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Runtime: {result.runtime_seconds:.2f}s")
    print(f"Processed genes: {len(result.processed_genes)}")
    print(f"Missing genes: {len(result.missing_genes)}")
    
    if result.missing_genes:
        print(f"Missing: {', '.join(sorted(result.missing_genes))}")
    
    print(f"Positions predicted: {result.positions.height if result.positions is not None else 0}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.positions is not None and result.positions.height > 0:
        print(f"\nFirst 5 predictions:")
        print(result.positions.head(5))
        
        print(f"\n📊 Statistics:")
        print(f"   Total positions: {result.positions.height:,}")
        print(f"   Unique genes: {result.positions['gene_id'].n_unique()}")
        print(f"   Unique chromosomes: {result.positions['seqname'].n_unique()}")
        
        # Verify all on chr21
        chromosomes = result.positions['seqname'].unique().to_list()
        print(f"   Chromosomes found: {chromosomes}")
        
        # Show genes processed
        if len(result.processed_genes) > 0:
            genes_list = sorted(list(result.processed_genes))
            print(f"\n   Genes processed ({len(genes_list)}):")
            for gene in genes_list:
                print(f"     - {gene}")
    
    # Verdict
    print("\n" + "=" * 80)
    if result.success and result.positions is not None and result.positions.height > 0:
        print("✅ Chr21 Test PASSED!")
        print(f"   Successfully predicted on {len(result.processed_genes)}/{len(CHR21_GENES)} genes")
        print(f"   Total predictions: {result.positions.height:,} positions")
        return True
    else:
        print("❌ Chr21 Test FAILED!")
        if result.error:
            print(f"   Error: {result.error}")
        return False
    print("=" * 80)

if __name__ == '__main__':
    success = test_chr21_genes()
    sys.exit(0 if success else 1)
