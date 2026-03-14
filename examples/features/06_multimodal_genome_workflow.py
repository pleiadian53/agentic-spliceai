#!/usr/bin/env python
"""Feature Engineering Example 6: YAML-Driven Multimodal Genome Workflow.

Thin orchestration script that loads a YAML config and runs FeatureWorkflow.
All modality configuration lives in the YAML file — adding a new modality
requires editing only the YAML, not this script.

Config files live in configs/ (sibling directory). The script dynamically
resolves modality config classes from the FeaturePipeline registry — no
modality-specific imports needed.

Usage:
    # Default config (base_scores + annotation + genomic, chr22)
    python 06_multimodal_genome_workflow.py

    # With conservation scores (full_stack config)
    python 06_multimodal_genome_workflow.py --config configs/full_stack.yaml

    # Config variant by name (searches configs/{name}.yaml)
    python 06_multimodal_genome_workflow.py --config-name full_stack

    # Override chromosomes and model from CLI
    python 06_multimodal_genome_workflow.py --chromosomes chr21 chr22 --model spliceai

    # All 24 canonical chromosomes
    python 06_multimodal_genome_workflow.py --chromosomes all --resume

    # Validate config without running (check modality registration, print summary)
    python 06_multimodal_genome_workflow.py --config configs/full_stack.yaml --dry-run

    # Force re-process (delete existing output, then run)
    python 06_multimodal_genome_workflow.py --chromosomes chr22 --force

    # Full genome on GPU pod
    nohup python -u examples/features/06_multimodal_genome_workflow.py \\
        --config configs/full_stack.yaml --chromosomes all --resume \\
        > /workspace/output/features.log 2>&1 &

Example:
    python 06_multimodal_genome_workflow.py --dry-run
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

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


def _ensure_predictions(
    model: str,
    chromosomes: list[str],
    chunk_size: int,
) -> Path:
    """Ensure base-layer predictions exist, generating on-demand if needed.

    New chromosome predictions are generated in a temporary directory, then
    merged into the registry-managed production path.

    Returns the production precomputed directory path.
    """
    import polars as pl

    from agentic_spliceai.splice_engine.base_layer.models.config import create_workflow_config
    from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow

    resources = get_model_resources(model)
    registry = resources.get_registry()
    production_dir = registry.get_base_model_eval_dir(model) / "precomputed"

    print(f"\n   Predicting {', '.join(chromosomes)} with {model} "
          f"(chunk_size={chunk_size})...")

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
            existing = pl.read_csv(agg_file, separator="\t")
            existing = ensure_chrom_column(existing)
            new_predictions = ensure_chrom_column(new_predictions)
            merged = pl.concat([existing, new_predictions])
            merged.write_csv(agg_file, separator="\t")
            print(f"   Merged: {existing.height:,} + {new_predictions.height:,} "
                  f"= {merged.height:,}")
        elif new_predictions is not None and new_predictions.height > 0:
            new_predictions.write_csv(agg_file, separator="\t")
            print(f"   Saved: {n_new:,} positions to {agg_file}")

    print(f"   Production path: {production_dir}")
    return production_dir


def _resolve_input_dir(model: str, cli_input_dir: Path | None) -> Path:
    """Resolve the input predictions directory."""
    if cli_input_dir is not None:
        if not cli_input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {cli_input_dir}")
        return cli_input_dir

    resources = get_model_resources(model)
    registry = resources.get_registry()
    return registry.get_base_model_eval_dir(model) / "precomputed"


def _ensure_all_chromosomes(
    input_dir: Path,
    model: str,
    chromosomes: list[str],
    chunk_size: int,
) -> Path:
    """Check for missing chromosomes in predictions and generate if needed."""
    import polars as pl

    if not input_dir.exists():
        print(f"\n  No precomputed predictions at: {input_dir}")
        print(f"  Generating on-demand...")
        return _ensure_predictions(model, chromosomes, chunk_size)

    agg_file = input_dir / "predictions.tsv"
    if not agg_file.exists():
        print(f"\n  No predictions.tsv in: {input_dir}")
        print(f"  Generating on-demand...")
        return _ensure_predictions(model, chromosomes, chunk_size)

    # Check which chromosomes are available
    resources = get_model_resources(model)
    lazy = pl.scan_csv(agg_file, separator="\t")
    cols = lazy.collect_schema().names()
    if "chrom" not in cols and "seqname" in cols:
        lazy = lazy.rename({"seqname": "chrom"})

    available = (
        lazy.select("chrom").unique().collect()["chrom"].to_list()
    )
    available_norm = set(normalize_chromosomes_for_build(available, resources.build))
    missing = [c for c in chromosomes if c not in available_norm]

    if missing:
        print(f"\n  Missing chromosomes: {', '.join(missing)}")
        print(f"  Generating on-demand...")
        _ensure_predictions(model, missing, chunk_size)

    return input_dir


def _handle_force(
    output_dir: Path,
    chromosomes: list[str],
    output_format: str,
) -> None:
    """Delete existing output files for requested chromosomes (--force)."""
    ext = "parquet" if output_format == "parquet" else "tsv"
    removed = 0
    for chrom in chromosomes:
        path = output_dir / f"analysis_sequences_{chrom}.{ext}"
        if path.exists():
            path.unlink()
            removed += 1
    summary = output_dir / "feature_summary.json"
    if summary.exists():
        summary.unlink()
    if removed:
        print(f"  Force: removed {removed} existing output file(s)")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    """Run YAML-driven multimodal feature engineering workflow."""
    parser = argparse.ArgumentParser(
        description="YAML-Driven Multimodal Feature Engineering",
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

    # Overrides — take precedence over YAML
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
        help="Skip chromosomes with existing output files",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-process chromosomes even if output exists (deletes old files)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and show what would run, without executing",
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

    # ── Print summary ─────────────────────────────────────────────────
    config_source = args.config or f"configs/{args.config_name}.yaml"
    print("=" * 70)
    print("Multimodal Feature Engineering (YAML-driven)")
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
    print(f"  Force:        {args.force}")
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

    if args.dry_run:
        print(f"\n  [DRY RUN] Config validated successfully. Exiting.")
        return 0

    # ── Step 2: Resolve or generate predictions ───────────────────────
    input_dir = _resolve_input_dir(model, args.input_dir)
    input_dir = _ensure_all_chromosomes(input_dir, model, chromosomes, chunk_size)
    print(f"\n  Input: {input_dir}")

    # ── Step 3: Handle --force ────────────────────────────────────────
    # We need the output_dir to clean it. FeatureWorkflow defaults it if None,
    # so we compute the default here to handle --force before workflow.run().
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_dir.parent / "analysis_sequences"

    if args.force and output_dir.exists():
        _handle_force(output_dir, chromosomes, pipeline_config.output_format)

    # ── Step 4: Run workflow ──────────────────────────────────────────
    workflow = FeatureWorkflow(
        pipeline_config=pipeline_config,
        input_dir=input_dir,
        output_dir=output_dir,
        resume=args.resume,
        sampling_config=sampling_config,
    )

    print(f"  Output: {workflow.output_dir}")
    print(f"\n  Running feature workflow...")
    result = workflow.run(chromosomes=chromosomes)

    # ── Step 5: Report results ────────────────────────────────────────
    print("\n" + "=" * 70)
    if result.success:
        print("Feature workflow completed successfully.")
        print(f"\n  Results:")
        print(f"    Total positions:      {result.total_positions:,}")
        print(f"    Chromosomes processed: {result.chromosomes_processed}")
        if result.chromosomes_skipped:
            print(f"    Chromosomes resumed:  {result.chromosomes_skipped}")
        print(f"    Runtime:              {result.runtime_seconds:.1f}s")

        # Show output files
        ext = "parquet" if pipeline_config.output_format == "parquet" else "tsv"
        print(f"\n  Output Files:")
        for chrom in chromosomes:
            path = workflow.output_dir / f"analysis_sequences_{chrom}.{ext}"
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"    {path.name}  ({size_mb:,.1f} MB)")

        # Show schema
        print(f"\n  Schema (by modality):")
        for modality, cols in result.pipeline_schema.items():
            print(f"    {modality}: {len(cols)} columns")

        # Show summary file
        summary_path = workflow.output_dir / "feature_summary.json"
        if summary_path.exists():
            print(f"\n  Summary: {summary_path}")
            with open(summary_path) as f:
                summary = json.load(f)
            print(f"    Timestamp:  {summary.get('timestamp', 'N/A')}")
            print(f"    Modalities: "
                  f"{summary.get('pipeline_config', {}).get('modalities', [])}")
    else:
        print(f"Feature workflow failed: {result.error}")
        return 1

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
