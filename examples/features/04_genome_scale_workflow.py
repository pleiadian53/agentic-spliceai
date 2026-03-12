#!/usr/bin/env python
"""Feature Engineering Example 4: Genome-Scale Workflow.

Demonstrates FeatureWorkflow for production-scale feature engineering
on precomputed base layer predictions:
1. Auto-detect or generate base-layer predictions (self-sufficient)
2. Apply FeaturePipeline per chromosome with atomic writes
3. Resume support for interrupted runs
4. Output as parquet or TSV with per-chromosome files

If precomputed predictions exist (from base_layer/05_genome_precomputation.py),
they are reused. If not, predictions are generated on-demand and saved to the
registry path for future reuse.

Usage:
    # Single chromosome (default) — auto-predicts if needed
    python 04_genome_scale_workflow.py --chromosomes chr22

    # Multiple chromosomes
    python 04_genome_scale_workflow.py --chromosomes chr21 chr22

    # Custom modalities and output
    python 04_genome_scale_workflow.py --chromosomes chr22 \\
        --modalities base_scores genomic \\
        --output-format tsv

    # Resume an interrupted run (skips chromosomes with existing output)
    python 04_genome_scale_workflow.py --chromosomes chr22 --resume

    # Add chromosomes incrementally (predictions are merged, new chroms processed)
    python 04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 --resume

    # Use SpliceAI (GRCh37) — chromosomes auto-normalized to bare numbers
    python 04_genome_scale_workflow.py --chromosomes chr21 chr22 --model spliceai --sample

    # Control prediction chunk size (for on-demand prediction)
    python 04_genome_scale_workflow.py --chromosomes chr22 --chunk-size 200

    # --- Position sampling (reduces storage ~100-170x) ---

    # Enable sampling with defaults (score>0.01, 0.5% background, ~1.5 GB genome)
    python 04_genome_scale_workflow.py --chromosomes chr22 --sample

    # Sampling with proximity window (retains local context for CNN meta-layers)
    python 04_genome_scale_workflow.py --chromosomes chr22 --sample \\
        --sample-window 50

    # Tighter sampling (~0.4 GB genome: score>0.01, 0.1% background)
    python 04_genome_scale_workflow.py --chromosomes chr22 --sample \\
        --sample-bg-rate 0.001

    # Broader splice capture with context (score>0.001, window=100, 0.1% bg)
    python 04_genome_scale_workflow.py --chromosomes chr22 --sample \\
        --sample-threshold 0.001 --sample-window 100 --sample-bg-rate 0.001

    # --- All chromosomes (full genome) ---

    # Process all 24 canonical chromosomes (1-22 + X + Y)
    python 04_genome_scale_workflow.py --chromosomes all --model spliceai --sample --resume

    nohup python -u examples/features/04_genome_scale_workflow.py \
    --chromosomes all --model spliceai --sample --resume \
    > /workspace/output/spliceai_genome_features.log 2>&1 &

    # --- Fresh run (redo from scratch) ---

    # Write to a new output directory (avoids --resume picking up old files)
    python 04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 \\
        --sample --output-dir /path/to/new/output

    # Or remove existing artifacts first, then run without --resume
    #   rm -rf data/mane/GRCh38/openspliceai_eval/analysis_sequences/
    python 04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 --sample

Example:
    python 04_genome_scale_workflow.py --chromosomes chr22
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from agentic_spliceai.splice_engine.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    FeatureWorkflow,
)
from agentic_spliceai.splice_engine.base_layer.data import normalize_chromosomes_for_build
from agentic_spliceai.splice_engine.resources import ensure_chrom_column, get_model_resources

log = logging.getLogger(__name__)


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Natural sort key: autosomes numerically, then X, Y, M."""
    bare = chrom.replace("chr", "")
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    try:
        return (int(bare), "")
    except ValueError:
        return (order.get(bare, 99), bare)


def _ensure_predictions(
    model: str,
    chromosomes: list[str],
    chunk_size: int,
) -> Path:
    """Ensure base-layer predictions exist, generating them on-demand if needed.

    New chromosome predictions are generated in a temporary directory, then
    merged into the registry-managed production path. This avoids chunk index
    conflicts with existing predictions from other chromosomes.

    Returns the production precomputed directory path.
    """
    import shutil
    import tempfile

    import polars as pl

    from agentic_spliceai.splice_engine.base_layer.models.config import (
        create_workflow_config,
    )
    from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow
    from agentic_spliceai.splice_engine.resources import get_model_resources

    # Resolve the production output directory
    resources = get_model_resources(model)
    registry = resources.get_registry()
    production_dir = registry.get_base_model_eval_dir(model) / "precomputed"

    print(f"\n   Predicting {', '.join(chromosomes)} with {model} "
          f"(chunk_size={chunk_size})...")

    # Run predictions in a temp directory to avoid chunk index conflicts
    with tempfile.TemporaryDirectory(prefix="spliceai_predict_") as tmp_dir:
        cfg = create_workflow_config(
            base_model=model,
            chunk_size=chunk_size,
            mode="test",
            coverage="full_genome",
            output_dir=tmp_dir,
        )

        workflow = PredictionWorkflow(cfg)
        result = workflow.run(chromosomes=chromosomes)

        if not result.success:
            raise RuntimeError(f"Prediction failed: {result.error}")

        new_predictions = result.predictions
        n_new = new_predictions.height if new_predictions is not None else 0
        print(f"   Generated: {n_new:,} positions in {result.runtime_seconds:.0f}s")

        # Merge into production directory
        production_dir.mkdir(parents=True, exist_ok=True)
        agg_file = production_dir / "predictions.tsv"

        if agg_file.exists() and new_predictions is not None and new_predictions.height > 0:
            # Append new predictions to existing file
            existing = pl.read_csv(agg_file, separator="\t")
            # Normalize: legacy files use 'seqname', new output uses 'chrom'
            existing = ensure_chrom_column(existing)
            new_predictions = ensure_chrom_column(new_predictions)
            merged = pl.concat([existing, new_predictions])
            merged.write_csv(agg_file, separator="\t")
            print(f"   Merged into {agg_file.name}: "
                  f"{existing.height:,} + {new_predictions.height:,} = {merged.height:,}")
        elif new_predictions is not None and new_predictions.height > 0:
            # No existing file — write directly
            new_predictions.write_csv(agg_file, separator="\t")
            print(f"   Saved: {n_new:,} positions to {agg_file}")

    print(f"   Production path: {production_dir}")
    return production_dir


def main():
    """Run genome-scale feature engineering workflow."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering Example 4: Genome-Scale Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=["chr22"],
        help="Chromosomes to process (default: chr22). "
             "Use 'all' for all canonical autosomes + sex chromosomes.",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model (default: openspliceai)",
    )
    parser.add_argument(
        "--modalities", nargs="+",
        default=["base_scores", "annotation", "genomic"],
        help="Modalities to apply (default: base_scores annotation genomic)",
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
        "--chunk-size", type=int, default=200,
        help="Genes per chunk for on-demand prediction (default: 200)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output files",
    )
    # Position sampling — reduces storage from ~271 GB to ~1-5 GB for full genome
    parser.add_argument(
        "--sample", action="store_true",
        help="Enable splice-aware position sampling (reduces storage ~100x)",
    )
    parser.add_argument(
        "--sample-threshold", type=float, default=0.01,
        help="Score threshold for splice site retention (default: 0.01)",
    )
    parser.add_argument(
        "--sample-window", type=int, default=0,
        help="Proximity window (bp) around splice sites to retain (default: 0). "
             "Set to 50-200 for local context around splice sites.",
    )
    parser.add_argument(
        "--sample-bg-rate", type=float, default=0.005,
        help="Background sampling rate for distant positions (default: 0.005 = 0.5%%)",
    )
    args = parser.parse_args()

    # Resolve "all" to canonical chromosomes for this build
    resources = get_model_resources(args.model)
    if args.chromosomes == ["all"]:
        from agentic_spliceai.splice_engine.base_layer.data import get_canonical_chromosomes
        all_chroms = get_canonical_chromosomes(resources.build, include_mito=False)
        # Keep only the build's native format (chr-prefixed or bare)
        normalized = set(normalize_chromosomes_for_build(list(all_chroms), resources.build))
        chromosomes = sorted(normalized, key=_chrom_sort_key)
    else:
        # Normalize user-provided names to match the model's build convention
        # (GRCh38/MANE → chr22, GRCh37/Ensembl → 22)
        chromosomes = normalize_chromosomes_for_build(args.chromosomes, resources.build)

    print("=" * 70)
    print("Feature Engineering Example 4: Genome-Scale Workflow")
    print("=" * 70)
    print(f"\n  Chromosomes: {', '.join(chromosomes)}")
    print(f"  Model: {args.model}")
    print(f"  Modalities: {args.modalities}")
    print(f"  Output format: {args.output_format}")
    print(f"  Resume: {args.resume}")
    if args.sample:
        print(f"  Sampling: ON (threshold={args.sample_threshold}, "
              f"window={args.sample_window}bp, bg_rate={args.sample_bg_rate})")

    # ---------------------------------------------------------------
    # Step 1: Resolve or generate predictions
    # ---------------------------------------------------------------
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"\n  Input directory not found: {input_dir}")
            return 1
    else:
        registry = resources.get_registry()
        input_dir = registry.get_base_model_eval_dir(args.model) / "precomputed"

    if not input_dir.exists():
        print(f"\n  No precomputed predictions found at: {input_dir}")
        print(f"  Generating predictions on-demand (saved for reuse)...")
        input_dir = _ensure_predictions(args.model, chromosomes, args.chunk_size)
    else:
        # Check if requested chromosomes are available in existing predictions
        import polars as pl

        agg_file = input_dir / "predictions.tsv"
        if agg_file.exists():
            lazy = pl.scan_csv(agg_file, separator="\t")
            # Normalize: legacy files use 'seqname', new output uses 'chrom'
            cols = lazy.collect_schema().names()
            if "chrom" not in cols and "seqname" in cols:
                lazy = lazy.rename({"seqname": "chrom"})
            available = (
                lazy.select("chrom")
                .unique()
                .collect()["chrom"]
                .to_list()
            )
            # Normalize available chroms to the build's convention for comparison
            available_norm = set(normalize_chromosomes_for_build(
                available, resources.build
            ))
            missing = [c for c in chromosomes if c not in available_norm]
            if missing:
                print(f"\n  Precomputed predictions found but missing: "
                      f"{', '.join(missing)}")
                print(f"  Generating missing chromosomes on-demand...")
                _ensure_predictions(args.model, missing, args.chunk_size)
                # Predictions are appended; FeatureWorkflow will re-load

    print(f"\n  Input: {input_dir}")

    # ---------------------------------------------------------------
    # Step 2: Configure and run workflow
    # ---------------------------------------------------------------
    pipeline_config = FeaturePipelineConfig(
        base_model=args.model,
        modalities=args.modalities,
        output_format=args.output_format,
        verbosity=1,
    )

    # Position sampling (optional — for storage-efficient genome-scale runs)
    sampling_config = None
    if args.sample:
        from agentic_spliceai.splice_engine.features import PositionSamplingConfig
        sampling_config = PositionSamplingConfig(
            enabled=True,
            score_threshold=args.sample_threshold,
            proximity_window=args.sample_window,
            background_rate=args.sample_bg_rate,
        )

    workflow = FeatureWorkflow(
        pipeline_config=pipeline_config,
        input_dir=input_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        sampling_config=sampling_config,
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
