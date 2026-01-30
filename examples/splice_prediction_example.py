"""
Example: Splice Site Prediction with Agentic-SpliceAI

This script demonstrates how to use the splice prediction capabilities
integrated from meta-spliceai.
"""

import sys
from pathlib import Path

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent))
from _example_utils import setup_example_environment
setup_example_environment()


def example_1_basic_prediction():
    """Example 1: Basic gene prediction."""
    print("\n" + "="*60)
    print("Example 1: Basic Gene Prediction")
    print("="*60)
    
    from agentic_spliceai.splice_engine import predict_splice_sites
    
    # Predict splice sites for ALS-related genes
    print("\nPredicting splice sites for ALS-related genes...")
    results = predict_splice_sites(
        genes=["UNC13A", "STMN2", "TARDBP"],
        base_model="openspliceai",
        verbosity=1
    )
    
    if results.get("success"):
        positions = results["positions"]
        print(f"\n✓ Prediction successful!")
        print(f"  Total positions analyzed: {positions.height}")
        
        # Show high-confidence donors
        import polars as pl
        donors = positions.filter(
            (pl.col("splice_type") == "donor") & 
            (pl.col("donor_score") > 0.9)
        )
        print(f"  High-confidence donors (>0.9): {donors.height}")
        
        # Show sample
        print("\nSample predictions:")
        print(positions.head(5))
    else:
        print(f"\n✗ Prediction failed: {results.get('error')}")


def example_2_api_usage():
    """Example 2: Using the high-level API."""
    print("\n" + "="*60)
    print("Example 2: High-Level API Usage")
    print("="*60)
    
    from agentic_spliceai.splice_engine.api import SplicePredictionAPI
    
    # Initialize API
    print("\nInitializing Splice Prediction API...")
    api = SplicePredictionAPI(base_model="openspliceai", verbosity=1)
    
    # Predict for genes
    print("\nPredicting for BRCA1...")
    results = api.predict_genes(["BRCA1"])
    
    if results.get("success"):
        # Get high-confidence predictions
        print("\nFiltering high-confidence predictions...")
        high_conf = api.get_high_confidence_predictions(results, threshold=0.9)
        print(f"High-confidence predictions: {high_conf.height}")
        
        # Get error positions
        print("\nAnalyzing prediction errors...")
        errors = api.get_error_positions(results)
        print(f"Total errors (FP + FN): {errors.height}")
        
        # Export to file
        output_path = "brca1_predictions.csv"
        print(f"\nExporting predictions to {output_path}...")
        api.export_predictions(results, output_path, format="csv")
        print(f"✓ Exported to {output_path}")
    else:
        print(f"\n✗ Prediction failed: {results.get('error')}")


def example_3_quick_functions():
    """Example 3: Quick convenience functions."""
    print("\n" + "="*60)
    print("Example 3: Quick Convenience Functions")
    print("="*60)
    
    from agentic_spliceai.splice_engine.api import quick_predict, predict_and_filter
    
    # Quick prediction
    print("\nQuick prediction for TP53...")
    results = quick_predict(genes=["TP53"], verbosity=1)
    
    if results.get("success"):
        positions = results["positions"]
        print(f"✓ Found {positions.height} positions")
        
        # Predict and filter in one step
        print("\nPredict and filter (confidence > 0.95)...")
        high_conf = predict_and_filter(
            genes=["TP53"],
            confidence_threshold=0.95,
            verbosity=0
        )
        print(f"✓ High-confidence predictions: {high_conf.height}")
        print("\nSample:")
        print(high_conf.head(3))
    else:
        print(f"\n✗ Prediction failed: {results.get('error')}")


def example_4_custom_config():
    """Example 4: Custom configuration."""
    print("\n" + "="*60)
    print("Example 4: Custom Configuration")
    print("="*60)
    
    from agentic_spliceai.splice_engine import run_base_model_predictions
    
    print("\nRunning with custom configuration...")
    results = run_base_model_predictions(
        base_model="openspliceai",
        target_genes=["UNC13A"],
        mode="test",
        coverage="gene_subset",
        threshold=0.5,
        verbosity=2,
        no_tn_sampling=False,  # Sample true negatives
        save_nucleotide_scores=False  # Don't save per-nucleotide scores
    )
    
    if results.get("success"):
        print("\n✓ Prediction successful!")
        
        # Access different result components
        positions = results["positions"]
        errors = results["error_analysis"]
        sequences = results.get("analysis_sequences")
        manifest = results.get("gene_manifest")
        
        print(f"\nResult components:")
        print(f"  Positions: {positions.height if positions is not None else 0}")
        print(f"  Errors: {errors.height if errors is not None else 0}")
        print(f"  Sequences: {sequences.height if sequences is not None else 0}")
        print(f"  Manifest: {manifest.height if manifest is not None else 0}")
        
        # Show paths
        paths = results.get("paths", {})
        print(f"\nOutput paths:")
        for key, value in paths.items():
            if value and "artifact" in key:
                print(f"  {key}: {value}")
    else:
        print(f"\n✗ Prediction failed: {results.get('error')}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("SPLICE SITE PREDICTION EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the splice prediction capabilities")
    print("integrated from meta-spliceai into agentic-spliceai.")
    print("\nNote: Requires meta-spliceai to be installed.")
    print("="*70)
    
    try:
        # Check if meta-spliceai is available
        import meta_spliceai
        print("\n✓ meta-spliceai is installed")
    except ImportError:
        print("\n✗ meta-spliceai is not installed!")
        print("\nPlease install meta-spliceai:")
        print("  cd /Users/pleiadian53/work/meta-spliceai")
        print("  mamba activate agentic-spliceai")
        print("  pip install -e .")
        return 1
    
    # Run examples
    try:
        example_1_basic_prediction()
        example_2_api_usage()
        example_3_quick_functions()
        example_4_custom_config()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED")
        print("="*70)
        return 0
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
