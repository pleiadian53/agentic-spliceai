#!/usr/bin/env python
"""Phase 3 Example: Chunked Prediction Workflow.

Demonstrates the chunked prediction pipeline with:
1. Gene-level chunking (configurable batch size)
2. Per-chunk checkpointing with resume capability
3. Raw prediction persistence (meta layer input)
4. Gene processing manifest tracking

Supports flexible target selection: specific genes, whole chromosomes,
or both (genes are additive to chromosome-based selection).

This uses test mode (default) with timestamped output for ad-hoc exploration.
For registry-managed output suitable for meta-layer training, see
05_genome_precomputation.py (production mode).

Usage:
    # Specific genes
    python 04_chunked_prediction.py --genes BRCA1 TP53 MYC EGFR --chunk-size 2

    # Whole chromosome
    python 04_chunked_prediction.py --chromosomes chr22 --chunk-size 100

    # Multiple chromosomes
    python 04_chunked_prediction.py --chromosomes chr21 chr22 --chunk-size 200

    # Genes + chromosomes (union: all genes on chr22, plus BRCA1 from chr17)
    python 04_chunked_prediction.py --chromosomes chr22 --genes BRCA1 --chunk-size 100

    # Resume from previous run
    python 04_chunked_prediction.py --chromosomes chr22 --chunk-size 100 --resume --output-dir <prev_output>

Example:
    python 04_chunked_prediction.py --genes BRCA1 TP53 MYC EGFR --chunk-size 2
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
    """Run chunked prediction workflow example."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Chunked Prediction Workflow (test mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--genes", nargs="+", default=None,
        help="Gene symbols to predict (e.g., BRCA1 TP53 MYC EGFR)"
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to process (e.g., chr22, or 21 22)"
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=2,
        help="Genes per chunk (default: 2, small for demo)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: auto-generated with timestamp)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints (requires --output-dir)"
    )
    args = parser.parse_args()

    if not args.genes and not args.chromosomes:
        parser.error("at least one of --genes or --chromosomes is required")

    # Normalise chromosome names
    chromosomes = None
    if args.chromosomes:
        chromosomes = [
            c if c.startswith("chr") else f"chr{c}" for c in args.chromosomes
        ]

    print("=" * 70)
    print("Phase 3 Example: Chunked Prediction Workflow")
    print("=" * 70)
    if args.genes:
        print(f"\n📋 Genes: {', '.join(args.genes)}")
    if chromosomes:
        print(f"📋 Chromosomes: {', '.join(chromosomes)}")
    print(f"🧬 Model: {args.model}")
    print(f"📦 Chunk size: {args.chunk_size} genes")
    print(f"🔄 Resume: {args.resume}")

    # Create workflow config (test mode — timestamped output)
    config = create_workflow_config(
        base_model=args.model,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    # Run workflow
    workflow = PredictionWorkflow(config)
    result = workflow.run(genes=args.genes, chromosomes=chromosomes)

    # Display results
    print("\n" + "=" * 70)
    if result.success:
        summary = result.manifest.get_summary()
        print("✅ Workflow completed successfully!")
        print(f"\n📊 Results:")
        print(f"   Total predictions: {result.predictions.height:,} positions")
        print(f"   Chunks: {result.chunks_processed} processed, "
              f"{result.chunks_skipped} resumed")
        print(f"   Runtime: {result.runtime_seconds:.1f}s")
        print(f"\n📋 Gene Manifest:")
        print(f"   Processed: {summary['processed_genes']}")
        print(f"   Failed: {summary['failed_genes']}")
        print(f"\n📁 Output: {result.output_dir}")

        # Show artifact files
        output_path = Path(result.output_dir)
        if output_path.exists():
            artifacts = sorted(output_path.iterdir())
            print(f"\n📄 Artifacts ({len(artifacts)} files):")
            for f in artifacts:
                size_kb = f.stat().st_size / 1024
                print(f"   {f.name}  ({size_kb:,.0f} KB)")

        # Demonstrate resume capability
        if not args.resume:
            gene_flag = f" --genes {' '.join(args.genes)}" if args.genes else ""
            chrom_flag = f" --chromosomes {' '.join(chromosomes)}" if chromosomes else ""
            print(f"\n💡 Tip: Re-run with --resume to skip completed chunks:")
            print(f"   python {Path(__file__).name}{gene_flag}{chrom_flag} "
                  f"--chunk-size {args.chunk_size} --resume "
                  f"--output-dir {result.output_dir}")
    else:
        print(f"❌ Workflow failed: {result.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
