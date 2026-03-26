#!/usr/bin/env python
"""Verify position alignment of multimodal feature artifacts.

Checks that feature values from different modalities are correctly
aligned to the same genomic positions. Catches issues like:

- Row shuffling (labels don't match scores)
- Join misalignment (wrong features attached to positions)
- Schema drift (missing or extra columns after augmentation)
- Value range violations (probabilities outside [0,1], negative counts)
- Build inconsistencies (GRCh37 data in GRCh38 artifacts)

Usage:
    # Verify a single chromosome
    python verify_feature_alignment.py --chromosomes chr22

    # Verify all available chromosomes
    python verify_feature_alignment.py --chromosomes all

    # Verify specific chromosomes with custom output dir
    python verify_feature_alignment.py --chromosomes chr1 chr22 \
        --output-dir /path/to/analysis_sequences/

    # Use SpliceAI (GRCh37) build
    python verify_feature_alignment.py --model spliceai --chromosomes chr22

    # Verbose mode: show per-modality null summary and full stats
    python verify_feature_alignment.py --chromosomes chr22 --verbose

Example:
    python verify_feature_alignment.py --chromosomes chr22
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from agentic_spliceai.splice_engine.features import (
    FeaturePipeline,
    PositionAlignmentVerifier,
    verify_artifact,
    verify_artifact_directory,
)
from agentic_spliceai.splice_engine.resources import get_model_resources


log = logging.getLogger(__name__)


def _resolve_output_dir(model: str, cli_dir: Path | None) -> Path:
    """Resolve the analysis_sequences directory."""
    if cli_dir is not None:
        return cli_dir
    resources = get_model_resources(model)
    registry = resources.get_registry()
    return registry.get_base_model_eval_dir(model) / "analysis_sequences"


def _resolve_pipeline_schema(model: str) -> dict[str, list[str]]:
    """Build expected schema from the pipeline registry."""
    from agentic_spliceai.splice_engine.features.pipeline import (
        FeaturePipelineConfig,
    )

    # Use all registered modalities to build the full expected schema
    config = FeaturePipelineConfig(
        base_model=model,
        modalities=FeaturePipeline.available_modalities(),
    )
    pipeline = FeaturePipeline(config)
    return pipeline.get_output_schema()


def _print_report(
    chrom: str, report: "VerificationReport", verbose: bool = False
) -> None:
    """Print a verification report for one chromosome."""
    status = "\033[32mPASSED\033[0m" if report.passed else "\033[31mFAILED\033[0m"
    n_pos = report.stats.get("n_positions", "?")
    n_cols = report.stats.get("n_columns", "?")

    print(f"\n  {chrom}: {status} ({n_pos:,} positions, {n_cols} columns)")

    for err in report.errors:
        print(f"    \033[31m[ERROR]\033[0m {err}")

    for warn in report.warnings:
        print(f"    \033[33m[WARN]\033[0m  {warn}")

    if verbose:
        # Label-score alignment stats
        for key in [
            "donor_sites_count", "donor_sites_mean_donor_score",
            "donor_sites_mean_acceptor_score",
            "acceptor_sites_count", "acceptor_sites_mean_donor_score",
            "acceptor_sites_mean_acceptor_score",
        ]:
            if key in report.stats:
                print(f"    {key}: {report.stats[key]}")

        # Modality presence
        if "modalities_present" in report.stats:
            present = report.stats["modalities_present"]
            missing = report.stats.get("modalities_missing", [])
            print(f"    Modalities present ({len(present)}): {present}")
            if missing:
                print(f"    Modalities missing ({len(missing)}): {missing}")

        # Overlapping genes
        if "overlapping_gene_positions" in report.stats:
            n_overlap = report.stats["overlapping_gene_positions"]
            pairs = report.stats.get("overlapping_gene_pairs", [])
            print(f"    Overlapping gene positions: {n_overlap}")
            for pair in pairs:
                print(f"      {pair}")

        # Null summary
        if "null_summary" in report.stats:
            null_sum = report.stats["null_summary"]
            for mod, total_nulls in sorted(null_sum.items()):
                if total_nulls > 0:
                    print(f"    {mod}: {total_nulls:,} total nulls")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify position alignment of multimodal feature artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=["chr22"],
        help="Chromosomes to verify. Use 'all' for all available. "
             "Default: chr22",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        help="Base model name for build/path resolution (default: openspliceai)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override analysis_sequences directory path",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed statistics per chromosome",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve paths
    output_dir = _resolve_output_dir(args.model, args.output_dir)
    if not output_dir.exists():
        print(f"Error: output directory not found: {output_dir}")
        return 1

    resources = get_model_resources(args.model)
    build = resources.build
    pipeline_schema = _resolve_pipeline_schema(args.model)

    # Resolve chromosome list
    chroms = None if args.chromosomes == ["all"] else args.chromosomes

    # Print header
    print("=" * 60)
    print("Feature Artifact Position Alignment Verification")
    print("=" * 60)
    print(f"  Directory:  {output_dir}")
    print(f"  Build:      {build}")
    print(f"  Model:      {args.model}")
    print(f"  Chromosomes: {'all' if chroms is None else ', '.join(chroms)}")
    print(f"  Expected modalities ({len(pipeline_schema)}):")
    for mod, cols in pipeline_schema.items():
        print(f"    {mod}: {len(cols)} columns")

    # Run verification
    reports = verify_artifact_directory(
        str(output_dir),
        build=build,
        pipeline_schema=pipeline_schema,
        chromosomes=chroms,
    )

    if not reports:
        print("\n  No artifacts found to verify.")
        return 1

    # Print results
    n_passed = 0
    n_failed = 0
    total_positions = 0

    for chrom, report in sorted(reports.items()):
        _print_report(chrom, report, verbose=args.verbose)
        if report.passed:
            n_passed += 1
        else:
            n_failed += 1
        total_positions += report.stats.get("n_positions", 0)

    # Summary
    print("\n" + "=" * 60)
    print(f"  Total: {n_passed + n_failed} chromosomes, "
          f"{total_positions:,} positions")
    print(f"  Passed: {n_passed}")
    if n_failed:
        print(f"  \033[31mFailed: {n_failed}\033[0m")
    else:
        print(f"  \033[32mAll checks passed.\033[0m")
    print("=" * 60)

    return 1 if n_failed else 0


if __name__ == "__main__":
    sys.exit(main())
