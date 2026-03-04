#!/usr/bin/env python
"""Feature Engineering Example 4: Genome-Scale Workflow.

Demonstrates FeatureWorkflow for production-scale feature engineering
on precomputed base layer predictions:
1. Load predictions from Phase 3 output (predictions.tsv or chunk files)
2. Apply FeaturePipeline per chromosome with atomic writes
3. Resume support for interrupted runs
4. Output as parquet or TSV with per-chromosome files

Prerequisite: Run base_layer/05_genome_precomputation.py first to generate
predictions at data/mane/GRCh38/openspliceai_eval/precomputed/.

Usage:
    # Single chromosome (default)
    python 04_genome_scale_workflow.py --chromosomes chr22

    # Multiple chromosomes
    python 04_genome_scale_workflow.py --chromosomes chr21 chr22

    # Custom modalities and output
    python 04_genome_scale_workflow.py --chromosomes chr22 \\
        --modalities base_scores genomic \\
        --output-format tsv

    # Resume an interrupted run
    python 04_genome_scale_workflow.py --chromosomes chr22 --resume

Example:
    python 04_genome_scale_workflow.py --chromosomes chr22
"""

import argparse
import json
import sys
from pathlib import Path

from agentic_spliceai.splice_engine.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeatureWorkflow,
)
from agentic_spliceai.splice_engine.resources import get_model_resources


def main():
    """Run genome-scale feature engineering workflow."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 4: Genome-Scale Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=["chr22"],
        help="Chromosomes to process (default: chr22)",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model (default: openspliceai)",
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["base_scores", "genomic"],
        help="Modalities to apply (default: base_scores genomic)",
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Input predictions directory (default: auto-resolved from registry)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: {input_dir}/../analysis_sequences/)",
    )
    parser.add_argument(
        "--output-format", default="parquet",
        choices=["parquet", "tsv"],
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output files",
    )
    args = parser.parse_args()

    # Normalize chromosome names
    chromosomes = [
        c if c.startswith("chr") else f"chr{c}" for c in args.chromosomes
    ]

    print("=" * 70)
    print("Feature Engineering Example 4: Genome-Scale Workflow")
    print("=" * 70)
    print(f"\n📋 Chromosomes: {', '.join(chromosomes)}")
    print(f"🧬 Model: {args.model}")
    print(f"🔬 Modalities: {args.modalities}")
    print(f"📄 Output format: {args.output_format}")
    print(f"🔄 Resume: {args.resume}")

    # ---------------------------------------------------------------
    # Step 1: Resolve input directory
    # ---------------------------------------------------------------
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        resources = get_model_resources(args.model)
        registry = resources.get_registry()
        input_dir = registry.get_base_model_eval_dir(args.model) / "precomputed"

    if not input_dir.exists():
        print(f"\n❌ Input directory not found: {input_dir}")
        print(f"   Run base_layer/05_genome_precomputation.py first to generate predictions.")
        return 1

    print(f"\n📁 Input: {input_dir}")

    # ---------------------------------------------------------------
    # Step 2: Configure and run workflow
    # ---------------------------------------------------------------
    pipeline_config = FeaturePipelineConfig(
        base_model=args.model,
        modalities=args.modalities,
        output_format=args.output_format,
        verbosity=1,
    )

    workflow = FeatureWorkflow(
        pipeline_config=pipeline_config,
        input_dir=input_dir,
        output_dir=args.output_dir,
        resume=args.resume,
    )

    print(f"📁 Output: {workflow.output_dir}")

    # Show available modalities
    print(f"\n🔧 Available modalities: {FeaturePipeline.available_modalities()}")
    print(f"   Active: {args.modalities}")

    print(f"\n🚀 Running feature workflow...")
    result = workflow.run(chromosomes=chromosomes)

    # ---------------------------------------------------------------
    # Step 3: Display results
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    if result.success:
        print("✅ Feature workflow completed!")
        print(f"\n📊 Results:")
        print(f"   Total positions:      {result.total_positions:,}")
        print(f"   Chromosomes processed: {result.chromosomes_processed}")
        if result.chromosomes_skipped:
            print(f"   Chromosomes resumed:  {result.chromosomes_skipped}")
        print(f"   Runtime:              {result.runtime_seconds:.1f}s")

        # Show output files
        print(f"\n📁 Output Files:")
        ext = "parquet" if args.output_format == "parquet" else "tsv"
        for chrom in chromosomes:
            path = workflow.output_dir / f"analysis_sequences_{chrom}.{ext}"
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"   {path.name}  ({size_mb:,.1f} MB)")

        # Show schema
        print(f"\n📋 Output Schema (by modality):")
        for modality, cols in result.pipeline_schema.items():
            print(f"   {modality}: {len(cols)} columns")

        # Show summary JSON
        summary_path = workflow.output_dir / "feature_summary.json"
        if summary_path.exists():
            print(f"\n📄 Summary: {summary_path}")
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"   Timestamp: {summary.get('timestamp', 'N/A')}")
            print(f"   Modalities: {summary.get('pipeline_config', {}).get('modalities', [])}")

        if not args.resume:
            print(f"\n💡 Tip: Re-run with --resume to skip completed chromosomes")
    else:
        print(f"❌ Feature workflow failed: {result.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
