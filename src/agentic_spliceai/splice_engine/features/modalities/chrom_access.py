"""Chromatin accessibility modality — ATAC-seq open chromatin features from ENCODE.

Extracts chromatin accessibility signals from ENCODE ATAC-seq
fold-change-over-control bigWig tracks across a panel of cell types.

Uses Strategy B (cross-tissue summary statistics): max, mean, breadth,
variance, context_mean, and a binary has_peak indicator. This captures
whether a genomic position is physically accessible to regulatory factors
(spliceosome, RBPs, transcription factors) — complementary to histone
marks which indicate chromatin *state* but not *accessibility* directly.

A position can have high H3K36me3 (transcribed exon body) while being
nucleosome-occupied (not accessible). The meta-layer can learn these
interaction patterns when both modalities are present.

GRCh38 only — ENCODE ATAC-seq signal tracks are aligned to GRCh38.
For GRCh37 builds, all columns are filled with NaN and a warning is logged.

Requires ``pyBigWig`` (optional dependency).

See Also
--------
examples/features/docs/chromatin-accessibility-tutorial.md
    Full tutorial on chromatin accessibility, ATAC-seq data, and
    feature interpretation.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── ENCODE ATAC-seq track registry (GRCh38 only) ─────────────────────
# Fold-change-over-control bigWig files, replicate-pooled.
# Keys: build → cell_line → {url, accession, experiment, filename}
# All URLs are ENCODE S3 direct links (stable, no redirects).

ATAC_TRACK_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "GRCh38": {
        "K562": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2021/02/24/"
                "b1a5f637-dba2-4272-b09e-86316d037135/ENCFF754EAC.bigWig"
            ),
            "accession": "ENCFF754EAC",
            "experiment": "ENCSR483RKN",
            "filename": "ENCFF754EAC.bigWig",
        },
        "GM12878": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2021/02/24/"
                "4c49a77a-edb2-4766-8328-d44ebe7446eb/ENCFF487LOB.bigWig"
            ),
            "accession": "ENCFF487LOB",
            "experiment": "ENCSR095QNB",
            "filename": "ENCFF487LOB.bigWig",
        },
        "HepG2": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2021/02/24/"
                "bfb21d9d-a2eb-4ce1-a536-236ccc97eef8/ENCFF664EJT.bigWig"
            ),
            "accession": "ENCFF664EJT",
            "experiment": "ENCSR042AWH",
            "filename": "ENCFF664EJT.bigWig",
        },
        "A549": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2021/03/16/"
                "0b01a7f6-3cf2-4f6e-b49a-675703bf3776/ENCFF872SDF.bigWig"
            ),
            "accession": "ENCFF872SDF",
            "experiment": "ENCSR032RGS",
            "filename": "ENCFF872SDF.bigWig",
        },
        "IMR-90": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2021/02/24/"
                "4dd00198-47b5-4ac6-9ac1-f11e2804800c/ENCFF282RNO.bigWig"
            ),
            "accession": "ENCFF282RNO",
            "experiment": "ENCSR200OML",
            "filename": "ENCFF282RNO.bigWig",
        },
    },
    # GRCh37: No ENCODE ATAC-seq signal tracks available for hg19.
    # The modality gracefully degrades (fills NaN, logs warning).
}

# Default panel: ENCODE Tier 1 (K562, GM12878) + liver + lung + fibroblast
DEFAULT_ATAC_CELL_LINES = ("K562", "GM12878", "HepG2", "A549", "IMR-90")


def _check_pybigwig() -> None:
    """Verify pyBigWig is importable."""
    try:
        import pyBigWig  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyBigWig is required for the chrom_access modality. Install with:\n"
            "  pip install pyBigWig\n"
            "If you get ABI errors, force-reinstall:\n"
            "  pip install --no-cache-dir --force-reinstall pyBigWig"
        )


@dataclass
class ChromAccessConfig(ModalityConfig):
    """Configuration for the chromatin accessibility modality.

    Attributes
    ----------
    base_model : str
        Base model name for build resolution (e.g., 'openspliceai').
    cell_lines : tuple of str
        Cell lines to query. Default: 5-cell panel (K562, GM12878,
        HepG2, A549, IMR-90).
    window : int
        Half-window (bp) for context features. ATAC-seq peaks are
        ~200bp (nucleosome scale). Default: 150.
    aggregation : str
        'summarized' (Strategy B, default): cross-tissue summary stats.
    signal_threshold : float
        Minimum fold-change signal to count a tissue as "accessible"
        for the tissue_breadth feature. Default: 2.0.
    peak_threshold : float
        Higher fold-change threshold for the binary has_peak feature.
        Default: 3.0.
    cache_dir : Path or None
        Local bigWig cache directory. If None, uses remote ENCODE URLs.
    remote_fallback : bool
        Fall back to remote URLs when local files not found.
    """

    base_model: str = "openspliceai"
    cell_lines: tuple[str, ...] = DEFAULT_ATAC_CELL_LINES
    window: int = 150
    aggregation: str = "summarized"
    signal_threshold: float = 2.0
    peak_threshold: float = 3.0
    cache_dir: Optional[Path] = None
    remote_fallback: bool = True


class ChromAccessModality(Modality):
    """Chromatin accessibility features from ENCODE ATAC-seq data.

    Extracts fold-change-over-control signals from ENCODE ATAC-seq
    bigWig tracks across a panel of cell types.

    In 'summarized' mode (default), produces cross-tissue statistics:
    max, mean, breadth, variance, context_mean, and has_peak.

    GRCh38 only. For GRCh37, fills all columns with NaN.
    """

    def __init__(self, config: ChromAccessConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: ChromAccessConfig = self.config  # type: ignore[assignment]
        self._build: Optional[str] = None

    @property
    def meta(self) -> ModalityMeta:
        cols = self._compute_output_columns()
        return ModalityMeta(
            name="chrom_access",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position"}),
            optional_inputs=frozenset(),
            description=(
                "Chromatin accessibility (ATAC-seq) from ENCODE "
                "across multiple cell types."
            ),
        )

    def _compute_output_columns(self) -> list[str]:
        """Compute output column names based on config."""
        if self._cfg.aggregation == "summarized":
            return [
                "atac_max_across_tissues",
                "atac_mean_across_tissues",
                "atac_tissue_breadth",
                "atac_variance",
                "atac_context_mean",
                "atac_has_peak",
            ]
        # Only summarized mode is supported
        return []

    @classmethod
    def default_config(cls) -> ChromAccessConfig:
        return ChromAccessConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        try:
            _check_pybigwig()
        except ImportError as e:
            errors.append(str(e))

        if self._cfg.aggregation != "summarized":
            errors.append(
                f"Invalid aggregation mode: '{self._cfg.aggregation}'. "
                f"Only 'summarized' is supported for chrom_access."
            )

        build = self._resolve_build()
        if build not in ATAC_TRACK_REGISTRY:
            logger.warning(
                "No ATAC-seq tracks for build '%s'. All chrom_access "
                "features will be NaN. Use GRCh38 (openspliceai) for "
                "chromatin accessibility features.",
                build,
            )
        else:
            available_cl = ATAC_TRACK_REGISTRY[build]
            for cl in self._cfg.cell_lines:
                if cl not in available_cl:
                    errors.append(
                        f"Cell line '{cl}' not in ATAC-seq registry for "
                        f"{build}. Available: {list(available_cl.keys())}"
                    )

        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add chromatin accessibility features to the DataFrame."""
        _check_pybigwig()

        build = self._resolve_build()
        n_positions = df.height

        logger.info(
            "Chrom access modality: %d positions, build=%s, "
            "cell_lines=%s, aggregation=%s",
            n_positions, build,
            self._cfg.cell_lines, self._cfg.aggregation,
        )

        # Graceful degradation for unsupported builds
        if build not in ATAC_TRACK_REGISTRY:
            logger.warning(
                "Build '%s' has no ENCODE ATAC-seq tracks. "
                "Filling all chrom_access columns with NaN.", build,
            )
            return self._fill_nan(df)

        df = self._transform_summarized(df, build)

        logger.info(
            "Chrom access modality: added %d columns",
            len(self.meta.output_columns),
        )
        return df

    # ── Summarized mode (Strategy B) ──────────────────────────────────

    def _transform_summarized(
        self, df: pl.DataFrame, build: str
    ) -> pl.DataFrame:
        """Cross-tissue summary statistics for ATAC-seq signal."""
        n = df.height

        # Collect per-cell-line scores at each position
        # Shape: (n_positions, n_cell_lines)
        cellline_scores = np.full(
            (n, len(self._cfg.cell_lines)), np.nan
        )
        # Context means (pooled across cell lines for final context_mean)
        cellline_context = np.full(
            (n, len(self._cfg.cell_lines)), np.nan
        )

        n_cl = len(self._cfg.cell_lines)
        for cl_idx, cl in enumerate(self._cfg.cell_lines):
            track_info = ATAC_TRACK_REGISTRY[build].get(cl)
            if track_info is None:
                logger.warning(
                    "No ATAC-seq track for %s in %s. Skipping.", cl, build,
                )
                continue

            logger.info(
                "  [%d/%d] Processing cell line %s (%s)...",
                cl_idx + 1, n_cl, cl, track_info["accession"],
            )

            t0 = time.monotonic()
            bw = self._open_track(track_info)
            try:
                scores, ctx = self._extract_access_features(df, bw)
                cellline_scores[:, cl_idx] = scores
                cellline_context[:, cl_idx] = ctx
            finally:
                bw.close()

            elapsed = time.monotonic() - t0
            n_valid = int(np.sum(~np.isnan(scores)))
            logger.info(
                "  [%d/%d] %s done: %d/%d positions scored in %.1fs",
                cl_idx + 1, n_cl, cl, n_valid, n, elapsed,
            )

        # Aggregate across cell lines (ignoring NaN)
        with np.errstate(all="ignore"):
            max_vals = np.nanmax(cellline_scores, axis=1)
            mean_vals = np.nanmean(cellline_scores, axis=1)
            var_vals = np.nanvar(cellline_scores, axis=1)
            context_mean_vals = np.nanmean(cellline_context, axis=1)

        # Tissue breadth: count of cell lines with signal > threshold
        breadth_vals = np.nansum(
            cellline_scores > self._cfg.signal_threshold, axis=1
        ).astype(np.float64)

        # Binary peak indicator: max signal > peak_threshold
        has_peak = (max_vals > self._cfg.peak_threshold).astype(np.float64)
        # Ensure NaN positions stay NaN (not False → 0.0)
        all_nan_mask = np.all(np.isnan(cellline_scores), axis=1)
        has_peak[all_nan_mask] = np.nan

        df = df.with_columns([
            pl.Series("atac_max_across_tissues", max_vals, dtype=pl.Float64),
            pl.Series("atac_mean_across_tissues", mean_vals, dtype=pl.Float64),
            pl.Series("atac_tissue_breadth", breadth_vals, dtype=pl.Float64),
            pl.Series("atac_variance", var_vals, dtype=pl.Float64),
            pl.Series("atac_context_mean", context_mean_vals, dtype=pl.Float64),
            pl.Series("atac_has_peak", has_peak, dtype=pl.Float64),
        ])

        return df

    # ── Core extraction ───────────────────────────────────────────────

    def _extract_access_features(
        self,
        df: pl.DataFrame,
        bw: "pyBigWig.pyBigWig",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract score and context mean for one cell line.

        Returns
        -------
        scores : np.ndarray
            Per-position signal (fold-change over control).
        context_means : np.ndarray
            Mean signal in a window around each position.
        """
        n = df.height
        w = self._cfg.window

        scores = np.full(n, np.nan)
        context_means = np.full(n, np.nan)

        bw_chroms = bw.chroms()

        for chrom in df["chrom"].unique().to_list():
            bw_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"

            if bw_chrom not in bw_chroms:
                logger.debug(
                    "Chromosome '%s' not in bigWig. Skipping.", bw_chrom,
                )
                continue

            chrom_length = bw_chroms[bw_chrom]
            chrom_mask = df["chrom"] == chrom
            chrom_indices = np.where(chrom_mask.to_numpy())[0]
            positions = df.filter(chrom_mask)["position"].to_numpy()

            if len(positions) == 0:
                continue

            # Query the full span (padded by window)
            span_start = max(0, int(positions.min()) - w)
            span_end = min(chrom_length, int(positions.max()) + w + 1)

            raw_values = bw.values(bw_chrom, span_start, span_end)
            if raw_values is None:
                continue

            span_array = np.array(raw_values, dtype=np.float64)
            span_array = np.nan_to_num(span_array, nan=0.0)

            for idx, pos in zip(chrom_indices, positions):
                center = int(pos) - span_start
                if center < 0 or center >= len(span_array):
                    continue

                # Point score
                scores[idx] = span_array[center]

                # Context mean (symmetric window)
                if w > 0:
                    ws = max(0, center - w)
                    we = min(len(span_array), center + w + 1)
                    window = span_array[ws:we]
                    if len(window) > 0:
                        context_means[idx] = np.mean(window)

        return scores, context_means

    # ── Helpers ───────────────────────────────────────────────────────

    def _resolve_build(self) -> str:
        """Resolve base_model → genomic build."""
        if self._build is not None:
            return self._build

        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        self._build = resources.build
        return self._build

    def _open_track(self, track_info: dict[str, str]) -> "pyBigWig.pyBigWig":
        """Open a bigWig track (local or remote)."""
        import pyBigWig

        # Check local cache first
        if self._cfg.cache_dir is not None:
            local_path = Path(self._cfg.cache_dir) / track_info["filename"]
            if local_path.exists():
                logger.debug("Using local bigWig: %s", local_path)
                bw = pyBigWig.open(str(local_path))
                if bw is not None:
                    return bw

        if not self._cfg.remote_fallback and self._cfg.cache_dir is not None:
            raise FileNotFoundError(
                f"Local bigWig not found: {track_info['filename']}. "
                f"Download from ENCODE: {track_info['url']}"
            )

        logger.debug(
            "Opening remote bigWig: %s (%s)",
            track_info["accession"], track_info["url"],
        )
        bw = pyBigWig.open(track_info["url"])
        if bw is None:
            raise RuntimeError(
                f"Failed to open bigWig: {track_info['url']}"
            )
        return bw

    def _fill_nan(self, df: pl.DataFrame) -> pl.DataFrame:
        """Fill all output columns with NaN (for unsupported builds)."""
        for col_name in self._compute_output_columns():
            df = df.with_columns(
                pl.lit(None).cast(pl.Float64).alias(col_name)
            )
        return df
