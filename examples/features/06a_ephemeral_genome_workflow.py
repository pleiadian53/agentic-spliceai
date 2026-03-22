#!/usr/bin/env python
"""Feature Engineering Example 6a: Ephemeral Genome Workflow.

Interleaved per-chromosome processing: predict → feature-engineer → delete
raw predictions. This keeps disk usage bounded to one chromosome at a time
instead of accumulating ~100 GB of predictions for the full genome.

The key difference from 06_multimodal_genome_workflow.py:
  - 06.py: predict ALL chromosomes first, then feature-engineer ALL
  - 06a.py: for each chromosome: predict → feature-engineer → delete predictions

Use this script for:
  - Full-genome runs on pods with limited disk
  - Memory-constrained environments (only one chromosome in memory at a time)
  - Production pipelines where only analysis_sequences matter

Use 06.py when:
  - You want to keep raw predictions for re-analysis with different configs
  - You're processing a few chromosomes interactively

Usage:
    # Full genome, ephemeral mode (delete predictions after feature engineering)
    python 06a_ephemeral_genome_workflow.py \\
        --config configs/full_stack.yaml --chromosomes all --ephemeral

    # Keep predictions (same as 06.py but interleaved per-chromosome)
    python 06a_ephemeral_genome_workflow.py \\
        --config configs/full_stack.yaml --chromosomes all --resume

    # GPU pod with memory monitoring
    nohup python -u 06a_ephemeral_genome_workflow.py \\
        --config configs/full_stack.yaml --chromosomes all \\
        --ephemeral --resume --memory-limit 30 \\
        > /workspace/output/features.log 2>&1 &
"""

import argparse
import json
import logging
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from agentic_spliceai.splice_engine.base_layer.data import normalize_chromosomes_for_build
from agentic_spliceai.splice_engine.features import FeaturePipeline, FeatureWorkflow
from agentic_spliceai.splice_engine.resources import ensure_chrom_column, get_model_resources

# Local config loader (sibling file)
sys.path.insert(0, str(Path(__file__).parent))
from config_loader import load_workflow_config  # noqa: E402

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    """Natural sort key: autosomes numerically, then X, Y, M."""
    bare = chrom.replace("chr", "")
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    try:
        return (int(bare), "")
    except ValueError:
        return (order.get(bare, 99), bare)


def _resolve_chromosomes(
    model: str,
    cli_chroms: list[str] | None,
    default_chroms: list[str],
) -> list[str]:
    """Resolve chromosome list from CLI > YAML default, normalizing for build."""
    resources = get_model_resources(model)
    raw = cli_chroms if cli_chroms is not None else default_chroms

    if raw == ["all"]:
        from agentic_spliceai.splice_engine.base_layer.data import get_canonical_chromosomes

        all_chroms = get_canonical_chromosomes(resources.build, include_mito=False)
        normalized = set(normalize_chromosomes_for_build(list(all_chroms), resources.build))
        return sorted(normalized, key=_chrom_sort_key)

    return normalize_chromosomes_for_build(raw, resources.build)


def _predict_chromosome(
    model: str,
    chrom: str,
    chunk_size: int,
) -> Optional[Path]:
    """Generate predictions for one chromosome, save as parquet.

    Returns the parquet file path, or None if predictions were empty.
    """
    import polars as pl

    from agentic_spliceai.splice_engine.base_layer.models.config import create_workflow_config
    from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow

    resources = get_model_resources(model)
    registry = resources.get_registry()
    production_dir = registry.get_base_model_eval_dir(model) / "precomputed"
    production_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   Predicting {chrom} with {model} (chunk_size={chunk_size})...")

    with tempfile.TemporaryDirectory(prefix="spliceai_predict_") as tmp_dir:
        cfg = create_workflow_config(
            base_model=model,
            chunk_size=chunk_size,
            mode="test",
            coverage="full_genome",
            output_dir=tmp_dir,
        )

        workflow = PredictionWorkflow(cfg)
        result = workflow.run(chromosomes=[chrom])

        if not result.success:
            raise RuntimeError(f"Prediction failed for {chrom}: {result.error}")

        predictions = result.predictions
        if predictions is None or predictions.height == 0:
            print(f"   No predictions for {chrom}")
            return None

        predictions = ensure_chrom_column(predictions)
        parquet_path = production_dir / f"predictions_{chrom}.parquet"
        predictions.write_parquet(parquet_path)
        size_mb = parquet_path.stat().st_size / (1024 * 1024)
        print(f"   Saved: {predictions.height:,} positions -> "
              f"{parquet_path.name} ({size_mb:.1f} MB) "
              f"in {result.runtime_seconds:.0f}s")

    return parquet_path


def _has_predictions(input_dir: Path, chrom: str) -> bool:
    """Check if predictions exist for a chromosome (parquet or legacy TSV)."""
    # Check per-chromosome parquet first
    if (input_dir / f"predictions_{chrom}.parquet").exists():
        return True

    # Fall back to legacy TSV
    agg_file = input_dir / "predictions.tsv"
    if not agg_file.exists():
        return False

    import polars as pl
    lazy = pl.scan_csv(agg_file, separator="\t")
    cols = lazy.collect_schema().names()
    chrom_col = "chrom" if "chrom" in cols else "seqname"
    available = lazy.select(chrom_col).unique().collect()[chrom_col].to_list()
    return chrom in available


def _resolve_input_dir(model: str, cli_input_dir: Path | None) -> Path:
    """Resolve the input predictions directory."""
    if cli_input_dir is not None:
        if not cli_input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {cli_input_dir}")
        return cli_input_dir

    resources = get_model_resources(model)
    registry = resources.get_registry()
    return registry.get_base_model_eval_dir(model) / "precomputed"


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    """Run interleaved per-chromosome feature engineering workflow."""
    parser = argparse.ArgumentParser(
        description="Ephemeral Multimodal Feature Engineering "
                    "(interleaved per-chromosome processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config source (mutually exclusive)
    cfg_group = parser.add_mutually_exclusive_group()
    cfg_group.add_argument(
        "--config", type=Path, default=None,
        help="Path to YAML config file",
    )
    cfg_group.add_argument(
        "--config-name", default="default",
        help="Config variant name: default, full_stack, ... "
             "(resolves configs/{name}.yaml)",
    )

    # Overrides
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Override chromosomes (default: from YAML). Use 'all' for full genome.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override base model (default: from YAML)",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=None,
        help="Override input predictions directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Override genes per chunk for on-demand prediction",
    )

    # Behavior flags
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip chromosomes with existing analysis_sequences output files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process chromosomes even if output exists",
    )
    parser.add_argument(
        "--ephemeral", action="store_true",
        help="Delete raw prediction parquets after feature engineering succeeds "
             "for each chromosome. Saves disk but prevents re-running features "
             "without re-predicting.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and show what would run, without executing",
    )
    parser.add_argument(
        "--memory-limit", type=float, default=None, metavar="GB",
        help="Memory limit in GB. Exit gracefully before OOM.",
    )

    args = parser.parse_args()

    if args.resume and args.force:
        parser.error("--resume and --force are mutually exclusive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Step 1: Load and validate config ──────────────────────────────
    overrides = {}
    if args.model:
        overrides["pipeline.base_model"] = args.model

    try:
        pipeline_config, sampling_config, workflow_opts = load_workflow_config(
            config_path=args.config,
            config_name=args.config_name,
            overrides=overrides,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"Config error: {e}")
        return 1

    model = pipeline_config.base_model
    chunk_size = args.chunk_size or workflow_opts.get("chunk_size", 200)
    default_chroms = workflow_opts.get("chromosomes", ["chr22"])
    chromosomes = _resolve_chromosomes(model, args.chromosomes, default_chroms)
    input_dir = _resolve_input_dir(model, args.input_dir)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / "analysis_sequences"

    # ── Print summary ─────────────────────────────────────────────────
    config_source = args.config or f"configs/{args.config_name}.yaml"
    print("=" * 70)
    print("Ephemeral Multimodal Feature Engineering (per-chromosome)")
    print("=" * 70)
    print(f"  Config:       {config_source}")
    print(f"  Model:        {model}")
    print(f"  Chromosomes:  {', '.join(chromosomes)}")
    print(f"  Modalities:   {pipeline_config.modalities}")
    print(f"  Output:       {pipeline_config.output_format}")
    print(f"  Sampling:     {'ON' if sampling_config else 'OFF'}")
    if sampling_config:
        print(f"    threshold={sampling_config.score_threshold}, "
              f"window={sampling_config.proximity_window}bp, "
              f"bg_rate={sampling_config.background_rate}")
    print(f"  Resume:       {args.resume}")
    print(f"  Ephemeral:    {args.ephemeral}")
    print(f"  Registered:   {FeaturePipeline.available_modalities()}")

    # Show per-modality column counts
    print(f"\n  Modality output schema:")
    pipeline = FeaturePipeline(pipeline_config)
    schema = pipeline.get_output_schema()
    total_cols = 0
    for mod_name, cols in schema.items():
        total_cols += len(cols)
        print(f"    {mod_name}: +{len(cols)} columns")
    print(f"    Total: {total_cols} feature columns")
    print(f"\n  Input:  {input_dir}")
    print(f"  Output: {output_dir}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Config validated. {len(chromosomes)} chromosomes queued.")
        return 0

    # ── Step 2: Interleaved per-chromosome processing ─────────────────
    t0 = time.time()
    chroms_processed = []
    chroms_skipped = []
    chroms_failed = []
    total_positions = 0
    predictions_deleted = 0
    fmt = pipeline_config.output_format
    ext = "parquet" if fmt == "parquet" else "tsv"

    for i, chrom in enumerate(chromosomes, 1):
        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(chromosomes)}] {chrom}")
        print(f"{'─' * 70}")

        # Check if analysis_sequences already exist (resume support)
        analysis_path = output_dir / f"analysis_sequences_{chrom}.{ext}"
        if args.resume and analysis_path.exists():
            print(f"  ✓ Skipping (analysis_sequences exists)")
            chroms_skipped.append(chrom)
            continue

        if args.force and analysis_path.exists():
            analysis_path.unlink()
            print(f"  Force: removed {analysis_path.name}")

        # 2a. Ensure predictions exist for this chromosome
        if not _has_predictions(input_dir, chrom):
            _predict_chromosome(model, chrom, chunk_size)

        # 2b. Run feature engineering for this chromosome
        workflow = FeatureWorkflow(
            pipeline_config=pipeline_config,
            input_dir=input_dir,
            output_dir=output_dir,
            resume=False,  # We handle resume above
            sampling_config=sampling_config,
            memory_limit_gb=args.memory_limit,
        )

        result = workflow.run(chromosomes=[chrom])

        if result.success and result.chromosomes_processed:
            chroms_processed.append(chrom)
            total_positions += result.total_positions
            size_mb = analysis_path.stat().st_size / (1024 * 1024) if analysis_path.exists() else 0
            print(f"  ✓ {result.total_positions:,} positions -> "
                  f"{analysis_path.name} ({size_mb:.1f} MB)")

            # 2c. Ephemeral: delete raw predictions after successful feature engineering
            if args.ephemeral:
                pred_path = input_dir / f"predictions_{chrom}.parquet"
                if pred_path.exists():
                    pred_size_mb = pred_path.stat().st_size / (1024 * 1024)
                    pred_path.unlink()
                    predictions_deleted += 1
                    print(f"  ✗ Deleted {pred_path.name} ({pred_size_mb:.1f} MB)")
        else:
            chroms_failed.append(chrom)
            print(f"  ✗ Failed: {result.error}")
            # Do NOT delete predictions on failure — allow retry

    # ── Step 3: Report results ────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print("Ephemeral Feature Workflow — Summary")
    print(f"{'=' * 70}")
    print(f"  Processed:     {len(chroms_processed)} chromosomes")
    if chroms_skipped:
        print(f"  Resumed:       {len(chroms_skipped)} chromosomes")
    if chroms_failed:
        print(f"  Failed:        {len(chroms_failed)} chromosomes: {chroms_failed}")
    print(f"  Total positions: {total_positions:,}")
    print(f"  Runtime:       {elapsed:.1f}s ({elapsed/60:.1f} min)")
    if args.ephemeral:
        print(f"  Predictions deleted: {predictions_deleted}")

    # Show output files
    print(f"\n  Output Files:")
    total_size = 0
    for chrom in chromosomes:
        path = output_dir / f"analysis_sequences_{chrom}.{ext}"
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"    {path.name}  ({size_mb:,.1f} MB)")
    print(f"    Total: {total_size:,.1f} MB")

    # Show schema
    print(f"\n  Schema (by modality):")
    for modality, cols in schema.items():
        print(f"    {modality}: {len(cols)} columns")

    print(f"{'=' * 70}")

    return 1 if chroms_failed else 0


if __name__ == "__main__":
    sys.exit(main())
