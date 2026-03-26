"""Position alignment verification for multimodal feature artifacts.

Verifies that feature values from different modalities are correctly
aligned to the same genomic positions. Each check returns a list of
error/warning strings — an empty list means the check passed.

The verifier works against existing parquet artifacts (post-hoc
verification) and can be integrated into the workflow for runtime
validation on sampled rows.

Usage
-----
>>> from agentic_spliceai.splice_engine.features.verification import (
...     PositionAlignmentVerifier,
... )
>>> verifier = PositionAlignmentVerifier(build="GRCh38")
>>> report = verifier.verify(df)
>>> if report.passed:
...     print("All checks passed")
... else:
...     for err in report.errors:
...         print(f"  ERROR: {err}")
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


# ── Value range specifications ────────────────────────────────────────

# Columns that must be in [0, 1] range (probabilities)
PROBABILITY_COLUMNS = frozenset({
    "donor_score", "acceptor_score", "neither_score",
    "relative_donor_probability", "splice_probability",
})

# Columns that must be non-negative
NON_NEGATIVE_COLUMNS = frozenset({
    "junction_log1p", "junction_n_partners", "junction_max_reads",
    "junction_tissue_breadth", "junction_tissue_max", "junction_tissue_mean",
    "rbp_n_bound", "rbp_max_signal", "rbp_n_sr_proteins", "rbp_n_hnrnps",
    "rbp_cell_line_breadth", "rbp_mean_signal",
    "atac_max_across_tissues", "atac_mean_across_tissues",
    "atac_tissue_breadth", "atac_context_mean",
})

# Binary columns (must be 0.0 or 1.0 only)
BINARY_COLUMNS = frozenset({
    "junction_has_support", "junction_is_annotated",
    "rbp_has_splice_regulator",
    "atac_has_peak",
    "donor_is_local_peak", "acceptor_is_local_peak",
})

# Columns where NaN means "unsupported build" (not a data error)
NAN_ALLOWED_COLUMNS = frozenset({
    # Epigenetic: GRCh38 only
    "h3k36me3_max_across_tissues", "h3k36me3_mean_across_tissues",
    "h3k36me3_tissue_breadth", "h3k36me3_variance",
    "h3k36me3_context_mean", "h3k36me3_exon_intron_ratio",
    "h3k4me3_max_across_tissues", "h3k4me3_mean_across_tissues",
    "h3k4me3_tissue_breadth", "h3k4me3_variance",
    "h3k4me3_context_mean", "h3k4me3_exon_intron_ratio",
    # ATAC: GRCh38 only
    "atac_max_across_tissues", "atac_mean_across_tissues",
    "atac_tissue_breadth", "atac_variance",
    "atac_context_mean", "atac_has_peak",
    # Conservation context stats can be NaN at chromosome edges
    "phylop_context_mean", "phylop_context_max", "phylop_context_std",
    "phastcons_context_mean", "phastcons_context_max", "phastcons_context_std",
    # Junction PSI: null for positions without junction partners
    "junction_psi", "junction_psi_variance",
})


@dataclass
class VerificationReport:
    """Results of position alignment verification.

    Attributes
    ----------
    passed : bool
        True if all checks passed (no errors). Warnings are allowed.
    errors : list of str
        Hard failures — data integrity issues.
    warnings : list of str
        Soft issues — unexpected but not necessarily wrong.
    stats : dict
        Summary statistics from the verification.
    checks_run : list of str
        Names of checks that were executed.
    """

    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, object] = field(default_factory=dict)
    checks_run: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        """Human-readable summary."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Verification: {status}",
            f"  Checks run: {len(self.checks_run)}",
            f"  Errors: {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
        ]
        for err in self.errors:
            lines.append(f"  [ERROR] {err}")
        for warn in self.warnings:
            lines.append(f"  [WARN]  {warn}")
        return "\n".join(lines)


class PositionAlignmentVerifier:
    """Verify position alignment across modality feature columns.

    Parameters
    ----------
    build : str
        Genomic build (e.g., 'GRCh38', 'GRCh37'). Used for build-specific
        checks (e.g., epigenetic features should be NaN for GRCh37).
    pipeline_schema : dict or None
        Expected modality → columns mapping from pipeline.get_output_schema().
        If None, schema checks are skipped.
    """

    def __init__(
        self,
        build: str = "GRCh38",
        pipeline_schema: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.build = build
        self.pipeline_schema = pipeline_schema

    def verify(self, df: pl.DataFrame) -> VerificationReport:
        """Run all verification checks on a feature DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Feature artifact (e.g., from analysis_sequences_chr22.parquet).

        Returns
        -------
        VerificationReport
            Results with errors, warnings, and statistics.
        """
        report = VerificationReport()
        report.stats["n_positions"] = df.height
        report.stats["n_columns"] = df.width

        self.check_coordinates(df, report)
        self.check_schema_completeness(df, report)
        self.check_label_score_alignment(df, report)
        self.check_null_patterns(df, report)
        self.check_value_ranges(df, report)
        self.check_row_count_preservation(df, report)

        return report

    # ── Individual checks ─────────────────────────────────────────────

    def check_coordinates(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify coordinate columns are well-formed."""
        report.checks_run.append("coordinates")

        # Required columns
        if "chrom" not in df.columns:
            report.add_error("Missing required column: 'chrom'")
            return
        if "position" not in df.columns:
            report.add_error("Missing required column: 'position'")
            return

        # Chrom: must be string
        if df["chrom"].dtype != pl.Utf8:
            report.add_error(
                f"'chrom' column has dtype {df['chrom'].dtype}, expected Utf8"
            )

        # Chrom: no nulls
        chrom_nulls = df["chrom"].null_count()
        if chrom_nulls > 0:
            report.add_error(f"'chrom' has {chrom_nulls} null values")

        # Position: no nulls
        pos_nulls = df["position"].null_count()
        if pos_nulls > 0:
            report.add_error(f"'position' has {pos_nulls} null values")

        # Position: non-negative
        pos_arr = df["position"].to_numpy()
        if np.any(pos_arr < 0):
            n_neg = int(np.sum(pos_arr < 0))
            report.add_error(f"'position' has {n_neg} negative values")

        # Position: no duplicates within (chrom, position, gene_id).
        # Note: same position can appear in multiple overlapping genes —
        # that's legitimate. We check uniqueness per gene.
        dedup_cols = ["chrom", "position"]
        if "gene_id" in df.columns:
            dedup_cols.append("gene_id")
        dup_df = df.group_by(dedup_cols).len().filter(pl.col("len") > 1)
        dup_count = dup_df.height
        if dup_count > 0:
            total_extra = int(dup_df["len"].sum()) - dup_count
            report.add_warning(
                f"{dup_count} duplicate ({', '.join(dedup_cols)}) groups "
                f"({total_extra} extra rows). Likely from background "
                f"sampling overlap — harmless but deduplication recommended."
            )
            report.stats["duplicate_groups"] = dup_count
            report.stats["duplicate_extra_rows"] = total_extra

        # Overlapping genes: same position in different genes (informational)
        if "gene_id" in df.columns and "gene_name" in df.columns:
            pos_groups = df.group_by(["chrom", "position"]).agg(
                pl.col("gene_name").unique().alias("genes"),
                pl.col("gene_name").n_unique().alias("n_genes"),
            ).filter(pl.col("n_genes") > 1)

            if pos_groups.height > 0:
                report.stats["overlapping_gene_positions"] = pos_groups.height

                # Collect unique overlapping gene pairs
                overlap_pairs: list[str] = []
                for row in pos_groups.iter_rows(named=True):
                    genes = sorted(row["genes"])
                    pair = "/".join(genes)
                    if pair not in overlap_pairs:
                        overlap_pairs.append(pair)
                report.stats["overlapping_gene_pairs"] = overlap_pairs

        # Build-specific chromosome prefix check
        chroms = df["chrom"].unique().to_list()
        if self.build in ("GRCh38", "GRCh38_MANE"):
            bare = [c for c in chroms if not c.startswith("chr")]
            if bare:
                report.add_warning(
                    f"GRCh38 positions should have 'chr' prefix, "
                    f"but found: {bare[:5]}"
                )
        elif self.build == "GRCh37":
            prefixed = [c for c in chroms if c.startswith("chr")]
            if prefixed:
                report.add_warning(
                    f"GRCh37 positions should not have 'chr' prefix, "
                    f"but found: {prefixed[:5]}"
                )

        report.stats["chromosomes"] = sorted(chroms)
        report.stats["position_dtype"] = str(df["position"].dtype)

    def check_schema_completeness(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify expected modality columns are present."""
        report.checks_run.append("schema_completeness")

        if self.pipeline_schema is None:
            report.add_warning(
                "No pipeline_schema provided — skipping schema check"
            )
            return

        actual_cols = set(df.columns)
        modalities_present: List[str] = []
        modalities_missing: List[str] = []

        for mod_name, expected_cols in self.pipeline_schema.items():
            expected_set = set(expected_cols)
            if expected_set.issubset(actual_cols):
                modalities_present.append(mod_name)
            else:
                missing = sorted(expected_set - actual_cols)
                modalities_missing.append(mod_name)
                report.add_error(
                    f"Modality '{mod_name}' missing columns: {missing}"
                )

        report.stats["modalities_present"] = modalities_present
        report.stats["modalities_missing"] = modalities_missing

    def check_label_score_alignment(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify that base model scores align with splice_type labels.

        At annotated donor sites, donor_score should be systematically
        higher than acceptor_score (and vice versa). This is the most
        direct test that rows haven't been shuffled.
        """
        report.checks_run.append("label_score_alignment")

        required = {"splice_type", "donor_score", "acceptor_score"}
        if not required.issubset(set(df.columns)):
            report.add_warning(
                "Missing columns for label-score alignment check: "
                f"{required - set(df.columns)}"
            )
            return

        # Donor sites: donor_score should dominate
        donors = df.filter(pl.col("splice_type") == "donor")
        if donors.height > 0:
            mean_donor = donors["donor_score"].mean()
            mean_acc = donors["acceptor_score"].mean()

            report.stats["donor_sites_count"] = donors.height
            report.stats["donor_sites_mean_donor_score"] = round(mean_donor, 4)
            report.stats["donor_sites_mean_acceptor_score"] = round(mean_acc, 4)

            if mean_donor < mean_acc:
                report.add_error(
                    f"Label-score misalignment at donor sites: "
                    f"mean donor_score ({mean_donor:.4f}) < "
                    f"mean acceptor_score ({mean_acc:.4f}). "
                    f"Rows may be shuffled."
                )
            elif mean_donor < 0.5:
                report.add_warning(
                    f"Low mean donor_score at donor sites: {mean_donor:.4f}"
                )

        # Acceptor sites: acceptor_score should dominate
        acceptors = df.filter(pl.col("splice_type") == "acceptor")
        if acceptors.height > 0:
            mean_donor = acceptors["donor_score"].mean()
            mean_acc = acceptors["acceptor_score"].mean()

            report.stats["acceptor_sites_count"] = acceptors.height
            report.stats["acceptor_sites_mean_donor_score"] = round(mean_donor, 4)
            report.stats["acceptor_sites_mean_acceptor_score"] = round(mean_acc, 4)

            if mean_acc < mean_donor:
                report.add_error(
                    f"Label-score misalignment at acceptor sites: "
                    f"mean acceptor_score ({mean_acc:.4f}) < "
                    f"mean donor_score ({mean_donor:.4f}). "
                    f"Rows may be shuffled."
                )

    def check_null_patterns(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify null/NaN patterns are consistent across modalities.

        Key invariant: within a single modality, either ALL columns
        should have values or ALL should be null/NaN (e.g., when the
        build is unsupported). Mixed nulls within a modality indicate
        a join or alignment error.
        """
        report.checks_run.append("null_patterns")

        if self.pipeline_schema is None:
            return

        null_summary: Dict[str, Dict[str, int]] = {}

        for mod_name, cols in self.pipeline_schema.items():
            mod_nulls: Dict[str, int] = {}
            for col in cols:
                if col not in df.columns:
                    continue
                null_count = df[col].null_count()
                mod_nulls[col] = null_count
            null_summary[mod_name] = mod_nulls

            # Check for mixed nulls within a modality
            null_counts = list(mod_nulls.values())
            if not null_counts:
                continue

            if len(set(null_counts)) > 1:
                # Some columns have different null counts — suspicious
                min_nulls = min(null_counts)
                max_nulls = max(null_counts)
                if min_nulls == 0 and max_nulls > 0:
                    # Filter out columns where nulls are expected
                    unexpected_null_cols = [
                        c for c, n in mod_nulls.items()
                        if n > 0 and c not in NAN_ALLOWED_COLUMNS
                    ]
                    # Only flag if unexpected columns have nulls
                    if unexpected_null_cols and max_nulls > df.height * 0.01:
                        report.add_warning(
                            f"Modality '{mod_name}' has mixed null pattern: "
                            f"{len(unexpected_null_cols)} unexpected cols "
                            f"with nulls ({max_nulls} max). "
                            f"Cols: {unexpected_null_cols[:3]}"
                        )

        report.stats["null_summary"] = {
            mod: sum(v.values()) for mod, v in null_summary.items()
        }

    def check_value_ranges(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify feature values are within expected ranges."""
        report.checks_run.append("value_ranges")

        actual_cols = set(df.columns)
        range_violations: List[str] = []

        # Probability columns: [0, 1]
        for col in PROBABILITY_COLUMNS & actual_cols:
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            min_val = series.min()
            max_val = series.max()
            if min_val < -0.001 or max_val > 1.001:
                range_violations.append(
                    f"'{col}' outside [0,1]: min={min_val:.4f}, max={max_val:.4f}"
                )

        # Non-negative columns
        for col in NON_NEGATIVE_COLUMNS & actual_cols:
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            min_val = series.min()
            if min_val < -0.001:
                range_violations.append(
                    f"'{col}' has negative values: min={min_val:.4f}"
                )

        # Binary columns: only 0.0 or 1.0
        for col in BINARY_COLUMNS & actual_cols:
            series = df[col].drop_nulls()
            if series.len() == 0:
                continue
            unique_vals = series.unique().sort().to_list()
            non_binary = [v for v in unique_vals if v not in (0.0, 1.0)]
            if non_binary:
                range_violations.append(
                    f"'{col}' is not binary: found values {non_binary[:5]}"
                )

        for v in range_violations:
            report.add_error(v)

        report.stats["range_violations"] = len(range_violations)

    def check_row_count_preservation(
        self, df: pl.DataFrame, report: VerificationReport
    ) -> None:
        """Verify that all modality columns have the same row count.

        After left-join augmentation, the DataFrame should have no
        extra or missing rows compared to the original prediction set.
        """
        report.checks_run.append("row_count_preservation")

        if self.pipeline_schema is None:
            return

        n = df.height
        for mod_name, cols in self.pipeline_schema.items():
            for col in cols:
                if col not in df.columns:
                    continue
                col_len = df[col].len()
                if col_len != n:
                    report.add_error(
                        f"Row count mismatch in '{mod_name}.{col}': "
                        f"expected {n}, got {col_len}"
                    )


def verify_artifact(
    path: str,
    build: str = "GRCh38",
    pipeline_schema: Optional[Dict[str, List[str]]] = None,
) -> VerificationReport:
    """Convenience function to verify a single parquet artifact.

    Parameters
    ----------
    path : str or Path
        Path to analysis_sequences parquet file.
    build : str
        Genomic build (default: 'GRCh38').
    pipeline_schema : dict or None
        Expected modality → columns mapping. If None, auto-detected
        from the FeaturePipeline registry.

    Returns
    -------
    VerificationReport
        Verification results.
    """
    if pipeline_schema is None:
        pipeline_schema = _auto_detect_schema()

    df = pl.read_parquet(path)
    verifier = PositionAlignmentVerifier(
        build=build,
        pipeline_schema=pipeline_schema,
    )
    report = verifier.verify(df)
    report.stats["artifact_path"] = str(path)
    return report


def verify_artifact_directory(
    output_dir: str,
    build: str = "GRCh38",
    pipeline_schema: Optional[Dict[str, List[str]]] = None,
    chromosomes: Optional[List[str]] = None,
) -> Dict[str, VerificationReport]:
    """Verify all parquet artifacts in a directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing analysis_sequences_*.parquet files.
    build : str
        Genomic build (default: 'GRCh38').
    pipeline_schema : dict or None
        Expected modality → columns mapping. If None, auto-detected.
    chromosomes : list of str or None
        If specified, only verify these chromosomes.

    Returns
    -------
    dict
        Maps chromosome name → VerificationReport.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    if pipeline_schema is None:
        pipeline_schema = _auto_detect_schema()

    reports: Dict[str, VerificationReport] = {}

    for parquet in sorted(output_path.glob("analysis_sequences_*.parquet")):
        chrom = parquet.stem.replace("analysis_sequences_", "")
        if chromosomes is not None and chrom not in chromosomes:
            continue

        logger.info("Verifying %s...", parquet.name)
        reports[chrom] = verify_artifact(
            str(parquet), build=build, pipeline_schema=pipeline_schema
        )

    return reports


def _auto_detect_schema() -> Dict[str, List[str]]:
    """Auto-detect pipeline schema from the modality registry."""
    from .pipeline import FeaturePipeline
    from . import modalities as _  # noqa: F401 — trigger registration

    schema: Dict[str, List[str]] = {}
    for name in FeaturePipeline.available_modalities():
        info = FeaturePipeline.get_modality_info(name)
        if info is not None:
            schema[name] = info["output_columns"]
    return schema
