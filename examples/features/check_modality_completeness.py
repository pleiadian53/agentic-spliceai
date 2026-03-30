#!/usr/bin/env python
"""Check modality completeness across feature artifacts.

Scans existing feature parquets to determine:
- Which chromosomes have artifacts (and which are missing)
- Which modalities are present per chromosome (column-level detection)
- Which columns have all-null values (data source may have failed)
- Suggested remediation: --augment (new modalities) or --refresh (recompute)

Complements ``verify_feature_alignment.py`` which checks data integrity
(value ranges, alignment, leakage). This script checks *coverage* — do we
have the data we expect?

Usage:
    # Check all available artifacts (default: openspliceai)
    python check_modality_completeness.py

    # Check specific chromosomes
    python check_modality_completeness.py --chromosomes chr1 chr22

    # Check against a specific YAML config (not just registry)
    python check_modality_completeness.py --config configs/full_stack.yaml

    # Show per-column null rates
    python check_modality_completeness.py --verbose

    # Show suggested commands to fill gaps
    python check_modality_completeness.py --suggest

Example:
    python check_modality_completeness.py --suggest
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl

from agentic_spliceai.splice_engine.features import FeaturePipeline
from agentic_spliceai.splice_engine.resources import get_model_resources

log = logging.getLogger(__name__)

# Canonical chromosomes (natural sort order)
CANONICAL_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModalityStatus:
    """Status of one modality in one chromosome."""

    name: str
    expected_columns: list[str]
    present_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    all_null_columns: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.missing_columns) == 0

    @property
    def is_present(self) -> bool:
        return len(self.present_columns) > 0

    @property
    def has_all_null(self) -> bool:
        return len(self.all_null_columns) > 0


@dataclass
class ChromosomeReport:
    """Completeness report for one chromosome."""

    chrom: str
    n_positions: int
    n_columns: int
    size_mb: float
    modalities: dict[str, ModalityStatus] = field(default_factory=dict)

    @property
    def complete_modalities(self) -> list[str]:
        return [m for m, s in self.modalities.items() if s.is_complete]

    @property
    def partial_modalities(self) -> list[str]:
        return [m for m, s in self.modalities.items()
                if s.is_present and not s.is_complete]

    @property
    def missing_modalities(self) -> list[str]:
        return [m for m, s in self.modalities.items() if not s.is_present]

    @property
    def all_null_modalities(self) -> list[str]:
        """Modalities where ALL expected columns are null (data source failed)."""
        result = []
        for m, s in self.modalities.items():
            if s.is_present and s.all_null_columns and len(s.all_null_columns) == len(s.present_columns):
                result.append(m)
        return result

    @property
    def is_fully_complete(self) -> bool:
        return (len(self.missing_modalities) == 0
                and len(self.partial_modalities) == 0)


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_expected_schema(
    model: str,
    config_path: Optional[Path] = None,
) -> dict[str, list[str]]:
    """Build expected modality → columns mapping.

    If config_path is given, uses only modalities listed in that YAML.
    Otherwise, uses all registered modalities.
    """
    if config_path is not None:
        sys.path.insert(0, str(Path(__file__).parent))
        from config_loader import load_workflow_config  # noqa: E402

        pipeline_config, _, _ = load_workflow_config(config_path=config_path)
        pipeline = FeaturePipeline(pipeline_config)
        return pipeline.get_output_schema()

    from agentic_spliceai.splice_engine.features.pipeline import FeaturePipelineConfig

    config = FeaturePipelineConfig(
        base_model=model,
        modalities=FeaturePipeline.available_modalities(),
    )
    pipeline = FeaturePipeline(config)
    return pipeline.get_output_schema()


def analyze_chromosome(
    path: Path,
    expected_schema: dict[str, list[str]],
    check_nulls: bool = True,
) -> ChromosomeReport:
    """Analyze modality completeness for one chromosome artifact."""
    chrom = path.stem.replace("analysis_sequences_", "")
    size_mb = path.stat().st_size / (1024 * 1024)

    # Read schema (fast — no data loaded)
    parquet_schema = pl.read_parquet_schema(path)
    actual_cols = set(parquet_schema.keys())
    n_cols = len(actual_cols)

    # Row count
    n_positions = pl.scan_parquet(path).select(pl.len()).collect().item()

    report = ChromosomeReport(
        chrom=chrom,
        n_positions=n_positions,
        n_columns=n_cols,
        size_mb=size_mb,
    )

    # Check null columns if requested (requires reading data)
    null_cols: set[str] = set()
    if check_nulls:
        null_counts = pl.read_parquet(path).null_count()
        for col in actual_cols:
            if col in null_counts.columns:
                n_null = null_counts[col][0]
                if n_null == n_positions:
                    null_cols.add(col)

    # Check each modality
    for mod_name, expected_cols in expected_schema.items():
        present = [c for c in expected_cols if c in actual_cols]
        missing = [c for c in expected_cols if c not in actual_cols]
        all_null = [c for c in present if c in null_cols]

        report.modalities[mod_name] = ModalityStatus(
            name=mod_name,
            expected_columns=expected_cols,
            present_columns=present,
            missing_columns=missing,
            all_null_columns=all_null,
        )

    return report


def suggest_remediation(
    reports: dict[str, ChromosomeReport],
    missing_chroms: list[str],
    expected_schema: dict[str, list[str]],
    config_name: str = "full_stack",
) -> list[str]:
    """Generate suggested commands to fill coverage gaps."""
    suggestions: list[str] = []

    # 1. Missing chromosomes entirely
    if missing_chroms:
        chrom_str = " ".join(missing_chroms)
        suggestions.append(
            f"# Generate features for {len(missing_chroms)} missing chromosome(s):\n"
            f"python 06_multimodal_genome_workflow.py \\\n"
            f"    --config configs/{config_name}.yaml \\\n"
            f"    --chromosomes {chrom_str} \\\n"
            f"    --memory-limit 5"
        )

    # 2. Chromosomes missing entire modalities → --augment
    augment_mods: dict[str, list[str]] = {}  # modality → list of chroms
    for chrom, report in reports.items():
        for mod in report.missing_modalities:
            augment_mods.setdefault(mod, []).append(chrom)

    if augment_mods:
        # Find chroms that need augmentation
        all_augment_chroms = set()
        for chroms in augment_mods.values():
            all_augment_chroms.update(chroms)
        chrom_str = " ".join(sorted(all_augment_chroms, key=_chrom_sort_key))
        mod_str = ", ".join(sorted(augment_mods.keys()))
        suggestions.append(
            f"# Augment {len(all_augment_chroms)} chromosome(s) with missing modalities ({mod_str}):\n"
            f"# Ensure these modalities are in your YAML config, then:\n"
            f"python 06_multimodal_genome_workflow.py \\\n"
            f"    --config configs/{config_name}.yaml \\\n"
            f"    --chromosomes {chrom_str} \\\n"
            f"    --augment"
        )

    # 3. Modalities with all-null values → --refresh
    refresh_mods: dict[str, list[str]] = {}  # modality → list of chroms
    for chrom, report in reports.items():
        for mod in report.all_null_modalities:
            refresh_mods.setdefault(mod, []).append(chrom)

    if refresh_mods:
        for mod, chroms in refresh_mods.items():
            chrom_str = " ".join(sorted(chroms, key=_chrom_sort_key))
            suggestions.append(
                f"# Refresh {mod} on {len(chroms)} chromosome(s) (all columns are null):\n"
                f"python 06_multimodal_genome_workflow.py \\\n"
                f"    --config configs/{config_name}.yaml \\\n"
                f"    --chromosomes {chrom_str} \\\n"
                f"    --refresh {mod}"
            )

    # 4. Partial modalities (some columns present, some missing) → --refresh
    partial_mods: dict[str, list[str]] = {}
    for chrom, report in reports.items():
        for mod in report.partial_modalities:
            partial_mods.setdefault(mod, []).append(chrom)

    if partial_mods:
        for mod, chroms in partial_mods.items():
            chrom_str = " ".join(sorted(chroms, key=_chrom_sort_key))
            suggestions.append(
                f"# Refresh {mod} on {len(chroms)} chromosome(s) (partial columns — config may have changed):\n"
                f"python 06_multimodal_genome_workflow.py \\\n"
                f"    --config configs/{config_name}.yaml \\\n"
                f"    --chromosomes {chrom_str} \\\n"
                f"    --refresh {mod}"
            )

    return suggestions


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    bare = chrom.replace("chr", "")
    order = {"X": 23, "Y": 24, "M": 25, "MT": 25}
    try:
        return (int(bare), "")
    except ValueError:
        return (order.get(bare, 99), bare)


def print_report(
    reports: dict[str, ChromosomeReport],
    missing_chroms: list[str],
    expected_schema: dict[str, list[str]],
    verbose: bool = False,
) -> int:
    """Print completeness report. Returns number of issues found."""
    n_expected_mods = len(expected_schema)
    total_expected_cols = sum(len(cols) for cols in expected_schema.values())
    issues = 0

    # Summary table
    print(f"\n{'Chrom':<8} {'Rows':>10} {'Cols':>6} {'MB':>7}  "
          f"{'Complete':>8} {'Partial':>8} {'Missing':>8} {'AllNull':>8}  Status")
    print("-" * 95)

    for chrom in sorted(reports.keys(), key=_chrom_sort_key):
        r = reports[chrom]
        n_complete = len(r.complete_modalities)
        n_partial = len(r.partial_modalities)
        n_missing = len(r.missing_modalities)
        n_all_null = len(r.all_null_modalities)

        if r.is_fully_complete and n_all_null == 0:
            status = f"\033[32mOK ({n_complete}/{n_expected_mods})\033[0m"
        elif n_missing > 0:
            status = f"\033[31mMISSING: {r.missing_modalities}\033[0m"
            issues += 1
        elif n_partial > 0:
            status = f"\033[33mPARTIAL: {r.partial_modalities}\033[0m"
            issues += 1
        elif n_all_null > 0:
            status = f"\033[33mALL-NULL: {r.all_null_modalities}\033[0m"
            issues += 1
        else:
            status = f"\033[32mOK ({n_complete}/{n_expected_mods})\033[0m"

        print(f"{chrom:<8} {r.n_positions:>10,} {r.n_columns:>6} {r.size_mb:>7.1f}  "
              f"{n_complete:>8} {n_partial:>8} {n_missing:>8} {n_all_null:>8}  {status}")

    # Missing chromosomes
    if missing_chroms:
        print()
        for chrom in missing_chroms:
            print(f"{chrom:<8} {'—':>10} {'—':>6} {'—':>7}  "
                  f"{'—':>8} {'—':>8} {'—':>8} {'—':>8}  "
                  f"\033[31mNO ARTIFACT\033[0m")
            issues += 1

    # Verbose: per-column null details
    if verbose:
        for chrom in sorted(reports.keys(), key=_chrom_sort_key):
            r = reports[chrom]
            has_issues = (r.all_null_modalities or r.partial_modalities
                          or r.missing_modalities)
            if not has_issues:
                continue

            print(f"\n  {chrom} details:")
            for mod_name, status in r.modalities.items():
                if not status.is_present:
                    print(f"    {mod_name}: \033[31mMISSING\033[0m "
                          f"({len(status.expected_columns)} expected columns)")
                elif status.all_null_columns:
                    n_null = len(status.all_null_columns)
                    n_total = len(status.present_columns)
                    print(f"    {mod_name}: {n_null}/{n_total} columns all-null: "
                          f"{status.all_null_columns}")
                elif status.missing_columns:
                    print(f"    {mod_name}: missing columns: "
                          f"{status.missing_columns}")

    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check modality completeness across feature artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to check (default: all canonical). "
             "Use 'all' explicitly for all 24.",
    )
    parser.add_argument(
        "--model", default="openspliceai",
        help="Base model for path resolution (default: openspliceai)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override analysis_sequences directory",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config to check against (uses its modality list). "
             "Default: all registered modalities.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show per-column null details for incomplete chromosomes",
    )
    parser.add_argument(
        "--suggest", action="store_true",
        help="Print suggested commands to fill coverage gaps",
    )
    parser.add_argument(
        "--skip-null-check", action="store_true",
        help="Skip all-null column detection (faster, schema-only check)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        resources = get_model_resources(args.model)
        registry = resources.get_registry()
        output_dir = registry.get_base_model_eval_dir(args.model) / "analysis_sequences"

    if not output_dir.exists():
        print(f"Error: directory not found: {output_dir}")
        return 1

    # Build expected schema
    expected_schema = build_expected_schema(args.model, args.config)

    # Determine which chromosomes to check
    target_chroms = CANONICAL_CHROMS
    if args.chromosomes is not None and args.chromosomes != ["all"]:
        target_chroms = [
            c if c.startswith("chr") else f"chr{c}"
            for c in args.chromosomes
        ]

    # Scan artifacts
    existing_files = {
        f.stem.replace("analysis_sequences_", ""): f
        for f in output_dir.glob("analysis_sequences_*.parquet")
    }

    existing_chroms = [c for c in target_chroms if c in existing_files]
    missing_chroms = [c for c in target_chroms if c not in existing_files]

    # Determine config name for suggestions
    config_name = "full_stack"
    if args.config is not None:
        config_name = args.config.stem

    # Header
    print("=" * 70)
    print("Modality Completeness Check")
    print("=" * 70)
    print(f"  Directory:    {output_dir}")
    print(f"  Model:        {args.model}")
    print(f"  Config:       {args.config or '(all registered modalities)'}")
    print(f"  Target:       {len(target_chroms)} chromosomes")
    print(f"  Found:        {len(existing_chroms)} artifacts, "
          f"{len(missing_chroms)} missing")
    print(f"  Null check:   {'skip' if args.skip_null_check else 'enabled'}")
    print(f"\n  Expected modalities ({len(expected_schema)}):")
    for mod, cols in expected_schema.items():
        print(f"    {mod}: {len(cols)} columns")
    total_expected = sum(len(c) for c in expected_schema.values())
    print(f"    Total: {total_expected} columns")

    # Analyze each chromosome
    reports: dict[str, ChromosomeReport] = {}
    for i, chrom in enumerate(sorted(existing_chroms, key=_chrom_sort_key), 1):
        path = existing_files[chrom]
        if not args.skip_null_check:
            log.info("Analyzing %s (%d/%d)...", chrom, i, len(existing_chroms))
        reports[chrom] = analyze_chromosome(
            path, expected_schema, check_nulls=not args.skip_null_check,
        )

    # Print report
    issues = print_report(reports, missing_chroms, expected_schema, args.verbose)

    # Totals
    total_positions = sum(r.n_positions for r in reports.values())
    total_size = sum(r.size_mb for r in reports.values())
    print(f"\n  Total: {total_positions:,} positions, {total_size:,.1f} MB "
          f"across {len(reports)} chromosomes")

    # Suggestions
    if args.suggest:
        suggestions = suggest_remediation(
            reports, missing_chroms, expected_schema, config_name,
        )
        if suggestions:
            print(f"\n{'=' * 70}")
            print("Suggested Commands")
            print("=" * 70)
            for i, cmd in enumerate(suggestions, 1):
                print(f"\n  [{i}] {cmd}")
        else:
            print(f"\n  \033[32mNo gaps found — all modalities complete.\033[0m")

    # Final status
    print(f"\n{'=' * 70}")
    if issues == 0:
        print(f"  \033[32mAll {len(reports)} chromosomes complete "
              f"({len(expected_schema)} modalities each).\033[0m")
        if missing_chroms:
            print(f"  \033[33m{len(missing_chroms)} chromosomes have no "
                  f"artifacts yet: {missing_chroms}\033[0m")
    else:
        print(f"  \033[33m{issues} issue(s) found.\033[0m "
              f"Run with --suggest for remediation commands.")
    print("=" * 70)

    return 1 if issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
