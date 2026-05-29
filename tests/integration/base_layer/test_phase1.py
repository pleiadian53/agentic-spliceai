#!/usr/bin/env python
"""
Test script for Phase 1: BaseModelRunner integration.

This script tests the newly wired-up BaseModelRunner to ensure it can:
1. Load gene annotations
2. Extract sequences
3. Run predictions
4. Return properly formatted results
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

def test_single_gene_prediction():
    """Test prediction on a single gene (BRCA1)."""
    print("=" * 80)
    print("Phase 1 Test: Single Gene Prediction")
    print("=" * 80)
    
    # Initialize runner
    runner = BaseModelRunner()
    
    # Run prediction on BRCA1
    print("\n🧪 Testing BaseModelRunner.run_single_model()...")
    print("   Model: openspliceai")
    print("   Gene: BRCA1")
    print("   Test: phase1_brca1")
    print()
    
    result = runner.run_single_model(
        model_name='openspliceai',
        target_genes=['BRCA1'],
        test_name='phase1_brca1',
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
    print(f"Positions predicted: {result.positions.height if result.positions is not None else 0}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    if result.positions is not None and result.positions.height > 0:
        print(f"\nFirst 5 predictions:")
        print(result.positions.head(5))
        
        print(f"\nColumns: {result.positions.columns}")
        print(f"\nShape: {result.positions.shape}")
    
    # Verdict
    print("\n" + "=" * 80)
    if result.success and result.positions is not None and result.positions.height > 0:
        print("✅ Phase 1 Test PASSED!")
        print("   BaseModelRunner successfully integrated with prediction pipeline")
        return True
    else:
        print("❌ Phase 1 Test FAILED!")
        if result.error:
            print(f"   Error: {result.error}")
        return False
    print("=" * 80)

if __name__ == '__main__':
    success = test_single_gene_prediction()
    sys.exit(0 if success else 1)
