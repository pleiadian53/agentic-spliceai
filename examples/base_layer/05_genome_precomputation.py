#!/usr/bin/env python
"""Phase 3 Example: Genome Precomputation for Meta-Layer Training.

Demonstrates the production-mode workflow that precomputes base-layer
predictions in the registry-managed directory structure:

    data/{annotation_source}/{build}/{model}_eval/precomputed/

These artifacts serve as input for the meta layer, which consumes
per-nucleotide donor_prob, acceptor_prob, and neither_prob scores.

Key differences from test-mode workflow (04_chunked_prediction.py):
- Output goes to registry path (stable, annotation/build/model-specific)
- Production mode (immutable artifacts)
- No timestamp — designed for resume across sessions

Usage:
    # Single chromosome demo
    python 05_genome_precomputation.py --chromosomes chr22 --chunk-size 100

    # Multiple chromosomes
    python 05_genome_precomputation.py --chromosomes chr21 chr22 --chunk-size 500

    # All chromosomes (genome-wide)
    python 05_genome_precomputation.py --all --chunk-size 500 --resume

    # Resume an interrupted run
    python 05_genome_precomputation.py --all --chunk-size 500 --resume

    # Use SpliceAI instead of OpenSpliceAI
    python 05_genome_precomputation.py --chromosomes chr22 --model spliceai --chunk-size 100

Example:
    python 05_genome_precomputation.py --chromosomes chr22 --chunk-size 100
"""

import argparse
import sys
from pathlib import Path

# Add project to path - using marker-based root finding (no fragile parent.parent.parent!)
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.config import create_workflow_config
from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow


def main():
    """Run genome precomputation workflow."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Genome Precomputation for Meta-Layer Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to precompute (e.g., chr22, or 21 22)"
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_chromosomes",
        help="Precompute all canonical chromosomes (chr1-22, chrX, chrY)"
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=100,
        help="Genes per chunk (default: 100)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints"
    )
    args = parser.parse_args()

    if not args.chromosomes and not args.all_chromosomes:
        parser.error("at least one of --chromosomes or --all is required")

    # Resolve chromosome list
    ALL_CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]
    if args.all_chromosomes:
        chromosomes = ALL_CANONICAL
    else:
        chromosomes = [
            c if c.startswith("chr") else f"chr{c}" for c in args.chromosomes
        ]

    print("=" * 70)
    print("Phase 3 Example: Genome Precomputation for Meta-Layer Training")
    print("=" * 70)
    print(f"\n📋 Chromosomes: {', '.join(chromosomes)}")
    print(f"🧬 Model: {args.model}")
    print(f"📦 Chunk size: {args.chunk_size} genes")
    print(f"🔄 Resume: {args.resume}")

    # Create config with mode='production'
    # This routes output to: data/{annotation_source}/{build}/{model}_eval/precomputed/
    config = create_workflow_config(
        base_model=args.model,
        chunk_size=args.chunk_size,
        mode="production",
        resume=args.resume,
    )

    print(f"\n📁 Registry eval dir: {config.eval_dir}")
    print(f"📁 Output dir:        {config.output_dir}")
    print(f"🔒 Mode: {config.mode}")

    # Run workflow
    workflow = PredictionWorkflow(config)
    result = workflow.run(chromosomes=chromosomes)

    # Display results
    print("\n" + "=" * 70)
    if result.success:
        summary = result.manifest.get_summary()
        print("✅ Precomputation completed successfully!")
        print(f"\n📊 Results:")
        print(f"   Total predictions: {result.predictions.height:,} positions")
        print(f"   Chunks: {result.chunks_processed} processed, "
              f"{result.chunks_skipped} resumed")
        print(f"   Runtime: {result.runtime_seconds:.1f}s")
        print(f"\n📋 Gene Manifest:")
        print(f"   Processed: {summary['processed_genes']}")
        print(f"   Failed: {summary['failed_genes']}")

        # Show meta-layer input artifacts
        output_path = Path(result.output_dir)
        print(f"\n🔬 Meta-Layer Input Artifacts:")
        for name in ["predictions.tsv", "manifest.tsv", "summary.json"]:
            artifact = output_path / name
            if artifact.exists():
                size_mb = artifact.stat().st_size / (1024 * 1024)
                print(f"   {artifact}  ({size_mb:,.1f} MB)")
            else:
                print(f"   {artifact}  (not yet aggregated)")

        if not args.resume:
            print(f"\n💡 Tip: Resume interrupted runs with --resume flag")
    else:
        print(f"❌ Precomputation failed: {result.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
