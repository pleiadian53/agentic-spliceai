#!/usr/bin/env python
"""Genome-scale base model precomputation for meta-layer training.

Runs a base model (OpenSpliceAI, SpliceAI, or foundation model-derived
classifiers) on gene annotations and saves per-nucleotide splice site
scores as prediction parquets.  These artifacts serve as input for the
meta-layer (both position-level M*-P and sequence-level M*-S models).

Supports multiple annotation sources on the same genome build:
- **MANE** (default for OpenSpliceAI): canonical protein-coding transcripts
- **Ensembl**: comprehensive annotations including alternative isoforms
  (enables M2a evaluation at Ensembl-only splice sites)

Output directory is resolved automatically based on the annotation source
and base model:
    data/{annotation_source}/{build}/{model}_eval/precomputed/

Usage:
    # OpenSpliceAI on MANE (default)
    python 05_genome_precomputation.py --all --chunk-size 500

    # OpenSpliceAI on Ensembl (for M2a evaluation)
    python 05_genome_precomputation.py --all --chunk-size 500 \\
        --annotation-source ensembl

    # SpliceAI on Ensembl/GRCh37
    python 05_genome_precomputation.py --all --model spliceai

    # Specific chromosomes, resume after interruption
    python 05_genome_precomputation.py --chromosomes chr21 chr22 --resume

    # Custom GTF (overrides annotation source resolution)
    python 05_genome_precomputation.py --chromosomes chr22 \\
        --gtf /path/to/custom.gtf
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.models.config import (
    create_workflow_config,
)
from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow


ALL_CANONICAL = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def main():
    parser = argparse.ArgumentParser(
        description="Genome-scale base model precomputation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to precompute (e.g., chr22, or 21 22)",
    )
    parser.add_argument(
        "--all", action="store_true", dest="all_chromosomes",
        help="Precompute all canonical chromosomes (chr1-22, chrX, chrY)",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model (default: openspliceai)",
    )
    parser.add_argument(
        "--annotation-source", default=None,
        choices=["mane", "ensembl"],
        help="Gene annotation source.  Default: model's training source "
             "(mane for openspliceai, ensembl for spliceai).  Override to "
             "run a model on a different annotation set.",
    )
    parser.add_argument(
        "--gtf", type=Path, default=None,
        help="Explicit GTF path (overrides --annotation-source resolution)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Genes per chunk (default: 500)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing checkpoints",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory (default: auto-resolved from registry)",
    )
    args = parser.parse_args()

    if not args.chromosomes and not args.all_chromosomes:
        parser.error("at least one of --chromosomes or --all is required")

    # Resolve chromosome list
    if args.all_chromosomes:
        chromosomes = ALL_CANONICAL
    else:
        chromosomes = [
            c if c.startswith("chr") else f"chr{c}" for c in args.chromosomes
        ]

    # Build config
    config_kwargs = dict(
        base_model=args.model,
        chunk_size=args.chunk_size,
        mode="production",
        resume=args.resume,
    )
    if args.annotation_source:
        config_kwargs["override_annotation_source"] = args.annotation_source
    if args.gtf:
        config_kwargs["gtf_file"] = str(args.gtf)
    if args.output_dir:
        config_kwargs["output_dir"] = str(args.output_dir)

    config = create_workflow_config(**config_kwargs)

    # Determine effective annotation source for display
    effective_source = args.annotation_source or config.annotation_source

    # Warn about cross-build combinations.  All base models are
    # sequence-based so cross-build predictions are valid (the model
    # sees DNA, not coordinates), but the user should be aware.
    MODEL_NATIVE_BUILD = {
        "openspliceai": ("GRCh38", "mane"),
        "spliceai": ("GRCh37", "ensembl"),
    }
    native_build, native_source = MODEL_NATIVE_BUILD.get(
        args.model, (config.genomic_build, effective_source)
    )
    if effective_source != native_source:
        print(f"NOTE: {args.model} was trained on {native_source}/{native_build}, "
              f"but running on {effective_source}/{config.genomic_build} annotations.")
        if config.genomic_build != native_build:
            print(f"  Build mismatch: model is {native_build}-native, "
                  f"sequences are {config.genomic_build}.")
            print(f"  This is valid (sequence-based model) but may show "
                  f"build-specific performance differences.")
        print()

    print("=" * 70)
    print("Genome Precomputation for Meta-Layer Training")
    print("=" * 70)
    print(f"  Model:       {args.model}")
    print(f"  Annotations: {effective_source} / {config.genomic_build}")
    if args.gtf:
        print(f"  GTF:         {args.gtf}")
    print(f"  Chromosomes: {len(chromosomes)}")
    print(f"  Chunk size:  {args.chunk_size} genes")
    print(f"  Resume:      {args.resume}")
    print(f"  Output:      {config.output_dir}")

    # Run workflow
    workflow = PredictionWorkflow(config)
    result = workflow.run(chromosomes=chromosomes)

    # Display results
    print("\n" + "=" * 70)
    if result.success:
        summary = result.manifest.get_summary()
        print("Precomputation completed successfully!")
        print(f"  Predictions: {result.predictions.height:,} positions")
        print(f"  Chunks:      {result.chunks_processed} processed, "
              f"{result.chunks_skipped} resumed")
        print(f"  Runtime:     {result.runtime_seconds:.1f}s")
        print(f"  Processed:   {summary['processed_genes']} genes")
        print(f"  Failed:      {summary['failed_genes']} genes")
        print(f"  Output:      {result.output_dir}")
    else:
        print(f"Workflow failed: {result.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
