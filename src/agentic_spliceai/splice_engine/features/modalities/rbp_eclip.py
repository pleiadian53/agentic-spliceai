"""RBP eCLIP modality — RNA-binding protein binding evidence from ENCODE.

Extracts per-position features from ENCODE eCLIP narrowPeak data across
~150 RBPs in K562 and HepG2 cell lines. Features capture active
post-transcriptional regulation at splice sites — SR proteins recruit
spliceosomes (exon inclusion) while hnRNPs block splice sites (exon
skipping).

Unlike conservation/epigenetic modalities (bigWig, per-base signal),
eCLIP data is **interval-based** (narrowPeak BED format). Each peak
is a ~50-200bp region of enriched RBP binding. The modality performs
interval overlap at transform time using batched numpy searchsorted.

GRCh38 only — ENCODE eCLIP is aligned to GRCh38. For GRCh37 builds,
all columns are filled with 0.0 and a warning is logged.

Requires pre-aggregated peaks parquet from
``scripts/aggregate_eclip_peaks.py``.

See Also
--------
examples/features/docs/rbp-eclip-tutorial.md
    Full tutorial on RBP binding, eCLIP data, and feature interpretation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── Splice regulator family definitions ──────────────────────────────
# Used for family-level aggregate features. Curated from ENCODE eCLIP
# targets cross-referenced with known splice regulatory function.

SPLICE_REGULATOR_FAMILIES: dict[str, frozenset[str]] = {
    # SR proteins: bind exonic splicing enhancers (ESEs), recruit
    # U1/U2 snRNP, promote exon inclusion
    "sr_proteins": frozenset({
        "SRSF1", "SRSF3", "SRSF4", "SRSF5", "SRSF7", "SRSF9",
        "SRSF10", "SRSF11", "TRA2A", "TRA2B",
    }),
    # hnRNPs: bind exonic/intronic splicing silencers (ESSs/ISSs),
    # generally promote exon skipping
    "hnrnps": frozenset({
        "HNRNPA1", "HNRNPA2B1", "HNRNPC", "HNRNPD", "HNRNPF",
        "HNRNPH1", "HNRNPK", "HNRNPL", "HNRNPM", "HNRNPU",
    }),
    # Core splice regulators with well-characterized mechanisms
    "core_regulators": frozenset({
        "RBFOX2",   # YGCAYG motif, brain/muscle splicing
        "PTBP1",    # Polypyrimidine tract binding, represses adult exons
        "U2AF1",    # U2AF small subunit, 3' splice site recognition
        "U2AF2",    # U2AF large subunit, polypyrimidine tract binding
        "ELAVL1",   # HuR, AU-rich element binding, mRNA stability
        "QKI",      # Quaking, ACUAA motif, neural/glial splicing
        "MBNL1",    # Muscleblind, YGCY motif, microsatellite expansion
        "SF3B4",    # U2 snRNP component, branch point recognition
        "PRPF8",    # Tri-snRNP component, catalytic core
    }),
}

# Union of all families for quick membership testing
ALL_SPLICE_REGULATORS: frozenset[str] = frozenset().union(
    *SPLICE_REGULATOR_FAMILIES.values()
)


@dataclass
class RBPEclipConfig(ModalityConfig):
    """Configuration for the RBP eCLIP modality.

    Attributes
    ----------
    base_model : str
        Base model name for build resolution (e.g., 'openspliceai').
    eclip_data_path : Path or None
        Path to pre-aggregated eCLIP peaks parquet. If None,
        auto-resolved from the genomic registry.
    aggregation : str
        'summarized' (default): Cross-RBP summary statistics (~8 cols).
        'detailed': Per-RBP features (future).
    cell_lines : tuple of str
        Cell lines to include. Default: K562 and HepG2.
    min_neg_log10_pvalue : float
        Minimum -log10(pValue) threshold for filtering peaks.
        Default: 2.0 (-log10(0.01)), IDR-filtered peaks only.
    batch_size : int
        Positions per batch in interval overlap computation.
        Controls memory usage during vectorized overlap.
    """

    base_model: str = "openspliceai"
    eclip_data_path: Optional[Path] = None
    aggregation: str = "summarized"
    cell_lines: tuple[str, ...] = ("K562", "HepG2")
    min_neg_log10_pvalue: float = 2.0
    batch_size: int = 5000


class RBPEclipModality(Modality):
    """RBP binding site features from ENCODE eCLIP data.

    Loads pre-aggregated eCLIP narrowPeak data and performs interval
    overlap at transform time. Each genomic position is annotated with
    how many RBPs have a binding peak overlapping it, the strength of
    binding, and which regulatory families are represented.

    Features are **sparse**: most positions have zero RBP binding.
    Non-overlapping positions get 0.0 for all columns (absence of
    binding is informative — the meta-layer learns from it).
    """

    def __init__(self, config: RBPEclipConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: RBPEclipConfig = self.config  # type: ignore[assignment]
        self._peaks_df: Optional[pl.DataFrame] = None
        self._build: Optional[str] = None

        # Pre-compute family sets for fast membership testing
        self._sr_set = SPLICE_REGULATOR_FAMILIES["sr_proteins"]
        self._hnrnp_set = SPLICE_REGULATOR_FAMILIES["hnrnps"]
        self._regulator_set = ALL_SPLICE_REGULATORS

    @property
    def meta(self) -> ModalityMeta:
        cols = self._compute_output_columns()
        return ModalityMeta(
            name="rbp_eclip",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position"}),
            optional_inputs=frozenset({"strand"}),
            description=(
                "RBP eCLIP binding evidence from ENCODE "
                "(K562/HepG2, ~150 RBPs)."
            ),
        )

    @classmethod
    def default_config(cls) -> RBPEclipConfig:
        return RBPEclipConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)

        if self._cfg.aggregation not in ("summarized", "detailed"):
            errors.append(
                f"Invalid aggregation mode: '{self._cfg.aggregation}'. "
                f"Must be 'summarized' or 'detailed'."
            )

        # Try to resolve data path — warn but don't fail
        try:
            path = self._resolve_eclip_path()
            if path is not None and not path.exists():
                logger.warning(
                    "eCLIP data not found at %s — "
                    "RBP features will be zero-filled.",
                    path,
                )
        except Exception:
            logger.warning(
                "Could not resolve eCLIP data path. "
                "RBP features will be zero-filled."
            )

        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add RBP eCLIP binding features to the DataFrame.

        Loads pre-aggregated peaks, performs per-chromosome interval
        overlap, and left-joins aggregated features onto the input
        DataFrame.
        """
        peaks = self._get_peaks()
        n_positions = df.height

        if peaks is None or peaks.height == 0:
            logger.warning(
                "No eCLIP peak data available. "
                "Filling all RBP columns with 0.0."
            )
            return self._fill_defaults(df)

        build = self._resolve_build()
        logger.info(
            "RBP eCLIP modality: %d positions, build=%s, "
            "%d peaks, cell_lines=%s, aggregation=%s",
            n_positions, build, peaks.height,
            self._cfg.cell_lines, self._cfg.aggregation,
        )

        # Build per-position feature index via interval overlap
        rbp_index = self._build_position_features(df, peaks)

        if rbp_index is None or rbp_index.height == 0:
            return self._fill_defaults(df)

        # Left join: keep all prediction rows
        output_cols = list(self._compute_output_columns())
        if df["position"].dtype != rbp_index["position"].dtype:
            rbp_index = rbp_index.with_columns(
                pl.col("position").cast(df["position"].dtype)
            )
        df = df.join(rbp_index, on=["chrom", "position"], how="left")

        # Fill nulls: 0.0 for all columns (no NaN columns in RBP features)
        fill_exprs = [
            pl.col(c).fill_null(0.0)
            for c in output_cols
            if c in df.columns
        ]
        if fill_exprs:
            df = df.with_columns(fill_exprs)

        # Ensure all declared output columns exist
        for col in output_cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(0.0).alias(col))

        logger.info(
            "RBP eCLIP modality: added %d columns",
            len(self.meta.output_columns),
        )
        return df

    # ------------------------------------------------------------------
    # Output column computation
    # ------------------------------------------------------------------

    def _compute_output_columns(self) -> list[str]:
        """Compute output column names based on config."""
        if self._cfg.aggregation == "summarized":
            return [
                "rbp_n_bound",
                "rbp_max_signal",
                "rbp_max_neg_log10_pvalue",
                "rbp_has_splice_regulator",
                "rbp_n_sr_proteins",
                "rbp_n_hnrnps",
                "rbp_cell_line_breadth",
                "rbp_mean_signal",
            ]
        else:
            # Detailed mode: per-RBP columns (future extension)
            logger.warning(
                "Detailed aggregation mode not yet implemented. "
                "Falling back to summarized."
            )
            return [
                "rbp_n_bound",
                "rbp_max_signal",
                "rbp_max_neg_log10_pvalue",
                "rbp_has_splice_regulator",
                "rbp_n_sr_proteins",
                "rbp_n_hnrnps",
                "rbp_cell_line_breadth",
                "rbp_mean_signal",
            ]

    # ------------------------------------------------------------------
    # Core interval overlap logic
    # ------------------------------------------------------------------

    def _build_position_features(
        self, df: pl.DataFrame, peaks: pl.DataFrame
    ) -> Optional[pl.DataFrame]:
        """Build per-position feature DataFrame via interval overlap.

        For each chromosome in the input DataFrame, finds eCLIP peaks
        overlapping each query position and aggregates features.

        Returns
        -------
        pl.DataFrame or None
            DataFrame with columns (chrom, position, rbp_n_bound, ...)
            or None if no overlaps found.
        """
        chromosomes = df["chrom"].unique().to_list()
        chrom_results: list[pl.DataFrame] = []

        for chrom in chromosomes:
            chrom_peaks = peaks.filter(pl.col("chrom") == chrom)
            if chrom_peaks.height == 0:
                continue

            chrom_mask = df["chrom"] == chrom
            positions = df.filter(chrom_mask)["position"].to_numpy()

            if len(positions) == 0:
                continue

            result = self._interval_overlap_vectorized(
                chrom, positions, chrom_peaks
            )
            if result is not None and result.height > 0:
                chrom_results.append(result)

        if not chrom_results:
            return None

        return pl.concat(chrom_results)

    def _interval_overlap_vectorized(
        self,
        chrom: str,
        positions: np.ndarray,
        peaks_df: pl.DataFrame,
    ) -> Optional[pl.DataFrame]:
        """Vectorized interval overlap using numpy searchsorted.

        For each position, finds all eCLIP peaks where
        start <= position < end and aggregates binding features.

        Uses batched processing to control memory.

        Parameters
        ----------
        chrom : str
            Chromosome name (for output DataFrame).
        positions : np.ndarray
            Sorted query positions for this chromosome.
        peaks_df : pl.DataFrame
            Peaks for this chromosome with columns:
            start, end, rbp, cell_line, signal_value, neg_log10_pvalue.

        Returns
        -------
        pl.DataFrame or None
            Per-position features, or None if no overlaps.
        """
        # Sort peaks by start for searchsorted
        peaks_sorted = peaks_df.sort("start")
        starts = peaks_sorted["start"].to_numpy().astype(np.int64)
        ends = peaks_sorted["end"].to_numpy().astype(np.int64)
        signals = peaks_sorted["signal_value"].to_numpy().astype(np.float64)
        pvalues = peaks_sorted["neg_log10_pvalue"].to_numpy().astype(np.float64)
        rbps = peaks_sorted["rbp"].to_list()
        cell_lines = peaks_sorted["cell_line"].to_list()

        # Pre-compute family membership flags per peak
        is_sr = np.array([r in self._sr_set for r in rbps], dtype=bool)
        is_hnrnp = np.array([r in self._hnrnp_set for r in rbps], dtype=bool)
        is_regulator = np.array(
            [r in self._regulator_set for r in rbps], dtype=bool
        )

        n = len(positions)
        n_bound = np.zeros(n, dtype=np.float64)
        max_signal = np.zeros(n, dtype=np.float64)
        max_pvalue = np.zeros(n, dtype=np.float64)
        has_regulator = np.zeros(n, dtype=np.float64)
        n_sr = np.zeros(n, dtype=np.float64)
        n_hnrnp = np.zeros(n, dtype=np.float64)
        cell_breadth = np.zeros(n, dtype=np.float64)
        mean_signal = np.zeros(n, dtype=np.float64)

        batch_size = self._cfg.batch_size
        positions_i64 = positions.astype(np.int64)

        for b_start in range(0, n, batch_size):
            b_end = min(b_start + batch_size, n)
            batch_pos = positions_i64[b_start:b_end]

            # For each position, find peaks where start <= pos
            # searchsorted(side='right') gives index of first start > pos
            right_bounds = np.searchsorted(starts, batch_pos, side="right")

            for i, (pos, rb) in enumerate(zip(batch_pos, right_bounds)):
                if rb == 0:
                    # No peaks have start <= pos
                    continue

                # Candidate peaks: indices [0, rb) have start <= pos
                # Filter: end > pos (peak still covers this position)
                candidate_ends = ends[:rb]
                valid_mask = candidate_ends > pos

                if not valid_mask.any():
                    continue

                idx = b_start + i

                # Aggregate over overlapping peaks
                valid_signals = signals[:rb][valid_mask]
                valid_pvals = pvalues[:rb][valid_mask]
                valid_is_sr = is_sr[:rb][valid_mask]
                valid_is_hnrnp = is_hnrnp[:rb][valid_mask]
                valid_is_reg = is_regulator[:rb][valid_mask]

                # Unique RBPs (count distinct RBP names)
                valid_rbps = [rbps[j] for j in range(rb) if valid_mask[j]]
                valid_cls = [cell_lines[j] for j in range(rb) if valid_mask[j]]

                n_bound[idx] = len(set(valid_rbps))
                max_signal[idx] = float(np.max(valid_signals))
                max_pvalue[idx] = float(np.max(valid_pvals))
                mean_signal[idx] = float(np.mean(valid_signals))
                has_regulator[idx] = float(valid_is_reg.any())
                n_sr[idx] = float(len(set(
                    r for r, s in zip(valid_rbps, valid_is_sr) if s
                )))
                n_hnrnp[idx] = float(len(set(
                    r for r, h in zip(valid_rbps, valid_is_hnrnp) if h
                )))
                cell_breadth[idx] = float(len(set(valid_cls)))

        # Only return positions that had at least one overlap
        has_overlap = n_bound > 0
        if not has_overlap.any():
            return None

        overlap_positions = positions[has_overlap]

        logger.debug(
            "eCLIP overlap on %s: %d/%d positions have binding",
            chrom, int(has_overlap.sum()), n,
        )

        return pl.DataFrame({
            "chrom": [chrom] * int(has_overlap.sum()),
            "position": overlap_positions,
            "rbp_n_bound": n_bound[has_overlap],
            "rbp_max_signal": max_signal[has_overlap],
            "rbp_max_neg_log10_pvalue": max_pvalue[has_overlap],
            "rbp_has_splice_regulator": has_regulator[has_overlap],
            "rbp_n_sr_proteins": n_sr[has_overlap],
            "rbp_n_hnrnps": n_hnrnp[has_overlap],
            "rbp_cell_line_breadth": cell_breadth[has_overlap],
            "rbp_mean_signal": mean_signal[has_overlap],
        })

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _get_peaks(self) -> Optional[pl.DataFrame]:
        """Get the eCLIP peaks DataFrame (cached)."""
        if self._peaks_df is not None:
            return self._peaks_df

        path = self._resolve_eclip_path()
        if path is None or not path.exists():
            return None

        self._peaks_df = self._load_peaks(path)
        return self._peaks_df

    def _load_peaks(self, path: Path) -> pl.DataFrame:
        """Load and filter eCLIP peaks from parquet.

        Applies cell line filter, significance threshold, and
        chromosome normalization.
        """
        logger.info("Loading eCLIP peaks from %s", path)
        df = pl.read_parquet(path)

        initial_count = df.height
        logger.info("Loaded %d raw peaks", initial_count)

        # Filter by cell lines
        if self._cfg.cell_lines:
            df = df.filter(pl.col("cell_line").is_in(list(self._cfg.cell_lines)))
            logger.info(
                "After cell line filter (%s): %d peaks",
                self._cfg.cell_lines, df.height,
            )

        # Filter by significance threshold
        if self._cfg.min_neg_log10_pvalue > 0:
            df = df.filter(
                pl.col("neg_log10_pvalue") >= self._cfg.min_neg_log10_pvalue
            )
            logger.info(
                "After pvalue filter (>= %.1f): %d peaks",
                self._cfg.min_neg_log10_pvalue, df.height,
            )

        # Normalize chromosome names to match build convention
        df = self._normalize_chrom(df)

        logger.info(
            "eCLIP peaks ready: %d peaks, %d RBPs, %d cell lines, "
            "%d chromosomes",
            df.height,
            df["rbp"].n_unique(),
            df["cell_line"].n_unique(),
            df["chrom"].n_unique(),
        )

        return df

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _fill_defaults(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all RBP columns with 0.0 (no data available)."""
        for col_name in self._compute_output_columns():
            df = df.with_columns(pl.lit(0.0).alias(col_name))
        return df

    def _normalize_chrom(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize chromosome names to match the build convention."""
        build = self._resolve_build()

        # GRCh38 / MANE uses chr prefix; GRCh37 / Ensembl does not
        uses_chr = build in ("GRCh38", "GRCh38_MANE")

        if uses_chr:
            df = df.with_columns(
                pl.when(pl.col("chrom").str.starts_with("chr"))
                .then(pl.col("chrom"))
                .otherwise(pl.concat_str([pl.lit("chr"), pl.col("chrom")]))
                .alias("chrom")
            )
        else:
            df = df.with_columns(
                pl.col("chrom").str.replace("^chr", "").alias("chrom")
            )

        return df

    def _resolve_build(self) -> str:
        """Resolve genomic build from base_model (cached)."""
        if self._build is not None:
            return self._build

        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        self._build = resources.build
        return self._build

    def _resolve_eclip_path(self) -> Optional[Path]:
        """Resolve eCLIP data file path.

        Resolution order:
        1. Explicit config path (eclip_data_path)
        2. Registry auto-resolution
        3. Fallback to rbp_data/ subdirectory in build dir
        """
        if self._cfg.eclip_data_path is not None:
            return Path(self._cfg.eclip_data_path)

        try:
            from agentic_spliceai.splice_engine.resources import get_model_resources

            resources = get_model_resources(self._cfg.base_model)
            registry = resources.get_registry()

            # Try registry first (may not be registered yet)
            try:
                path = registry.resolve("eclip_peaks")
                if path is not None and Path(path).exists():
                    return Path(path)
            except (ValueError, KeyError):
                pass  # Not registered — fall through to convention-based lookup

            # Fallback: rbp_data/ subdirectory
            build_dir = Path(registry.stash)
            eclip_parquet = build_dir / "rbp_data" / "eclip_peaks.parquet"
            if eclip_parquet.exists():
                logger.info("Using eCLIP peaks: %s", eclip_parquet)
                return eclip_parquet

            return None
        except Exception:
            return None
