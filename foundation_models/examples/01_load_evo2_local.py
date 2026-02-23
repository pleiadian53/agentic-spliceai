#!/usr/bin/env python3
"""
Example: Load Evo2 on M1 MacBook Pro

This script demonstrates loading Evo2 7B with INT8 quantization
on a MacBook Pro with M1/M2/M3 chip.

Usage:
    python examples/01_load_evo2_local.py
"""

import sys
from pathlib import Path

# Add foundation_models to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from foundation_models.evo2 import Evo2Config, Evo2Model, Evo2Embedder


def main():
    print("=" * 60)
    print("Evo2 on M1 Mac - Quick Test")
    print("=" * 60)
    print()
    
    # Configuration for M1 Mac
    print("Step 1: Configure for M1 Mac")
    config = Evo2Config.for_local_mac()
    print(f"  Model: {config.model_id}")
    print(f"  Size: {config.model_size}")
    print(f"  Device: {config.effective_device}")
    print(f"  Quantize: {config.quantize} ({config.quantization_bits}-bit)")
    print()
    
    # Load model
    print("Step 2: Load Evo2 model")
    print("  (This may take 2-3 minutes on first run)")
    print()
    
    try:
        embedder = Evo2Embedder(config=config)
        print()
        print("✓ Model loaded successfully!")
        print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you installed dependencies:")
        print("   conda activate aspliceai-evo2-local")
        print("   pip install -e .")
        print("   pip install -e ./foundation_models")
        print()
        print("2. Check internet connection (model downloads from HuggingFace)")
        print()
        print("3. Check disk space (~20GB needed for model weights)")
        print()
        return
    
    # Test encoding
    print("Step 3: Test encoding a short DNA sequence")
    sequence = "ATCGATCGATCGATCGATCG" * 10  # 200 bp
    print(f"  Sequence length: {len(sequence)} bp")
    print(f"  Sequence: {sequence[:50]}...")
    print()
    
    print("  Encoding...")
    embeddings = embedder.encode(sequence)
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    print(f"    - Sequence length: {embeddings.shape[0]} positions")
    print(f"    - Hidden dimension: {embeddings.shape[1]}")
    print()
    
    # Test longer sequence
    print("Step 4: Test encoding a longer sequence (chunking)")
    long_sequence = "ATCGATCGATCG" * 1000  # 12kb
    print(f"  Sequence length: {len(long_sequence)} bp")
    print()
    
    print("  Encoding with chunking (max_length=8192)...")
    long_embeddings = embedder.encode(
        long_sequence,
        chunk_size=8192,
        overlap=512,
    )
    print(f"  ✓ Embeddings shape: {long_embeddings.shape}")
    print()
    
    # Summary
    print("=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print()
    print("Evo2 is working on your M1 Mac. You can now:")
    print()
    print("1. Extract embeddings for genes:")
    print("   python examples/02_extract_embeddings.py")
    print()
    print("2. Train exon classifier:")
    print("   python examples/03_train_classifier.py")
    print()
    print("3. See tutorial notebook:")
    print("   jupyter notebook notebooks/evo2_exon_classifier.ipynb")
    print()
    
    # Performance note
    print("Performance Notes:")
    print(f"  - Inference speed: ~{len(sequence) / 10:.0f} bp/second (estimated)")
    print("  - Memory usage: ~8GB RAM")
    print("  - Recommended max sequence: 32,768 bp")
    print()
    print("For faster inference, use GPU pod with Evo2 40B!")
    print()


if __name__ == "__main__":
    main()
