"""Chromatin accessibility modality — ATAC-seq + DNase-seq from ENCODE.

Extracts chromatin accessibility signals from two complementary assays:

- **ATAC-seq** (fold-change-over-control): 5 ENCODE cancer cell lines
  (K562, GM12878, HepG2, A549, IMR-90)
- **DNase-seq** (read-depth normalized signal): 5 primary tissues
  (brain cortex, heart, lung, muscle, liver)

Both assays measure nucleosome-free DNA — positions physically accessible
to regulatory factors — but use different signal normalization. ATAC-seq
reports fold-change over control; DNase-seq reports read-depth normalized
signal. The values are on different scales, so they are kept as **separate
column groups** (``atac_*`` and ``dnase_*``) within the same modality.
The meta-layer model learns their individual contributions.

Uses Strategy B (cross-tissue summary statistics) for both: max, mean,
breadth, variance, context_mean, and a binary has_peak indicator.

GRCh38 only. For GRCh37 builds, all columns are filled with NaN.

Requires ``pyBigWig`` (optional dependency).

See Also
--------
examples/features/docs/chromatin-accessibility-tutorial.md
    Full tutorial on chromatin accessibility, ATAC-seq vs DNase-seq data,
    and feature interpretation.
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


# ── ENCODE DNase-seq track registry (GRCh38 only) ────────────────────
# Read-depth normalized signal bigWig files (NOT fold-change — DNase-seq
# processing does not produce fold-change-over-control tracks on ENCODE).
#
# WHY A SEPARATE REGISTRY (not merged into ATAC_TRACK_REGISTRY):
# DNase-seq and ATAC-seq measure the same biological property (chromatin
# accessibility) but use different signal normalization:
#   - ATAC-seq: fold-change over control (values typically 0-20)
#   - DNase-seq: read-depth normalized (values typically 0-100+)
# Mixing them in the same summary statistics would be meaningless.
# They are kept as separate column groups (atac_* and dnase_*) so the
# meta-layer model can learn their separate contributions.
#
# These are PRIMARY TISSUE samples (not cancer cell lines), providing
# complementary tissue coverage to the ATAC-seq cancer cell line panel.

DNASE_TRACK_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "GRCh38": {
        "brain_cortex": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2020/11/22/"
                "a8f1a670-279c-4096-bd41-dfea58797f8c/ENCFF243QQE.bigWig"
            ),
            "accession": "ENCFF243QQE",
            "experiment": "ENCSR000EIY",
            "filename": "ENCFF243QQE.bigWig",
        },
        "heart": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2025/08/21/"
                "2221005d-7829-495b-8e59-2f99e6f91148/ENCFF885WJZ.bigWig"
            ),
            "accession": "ENCFF885WJZ",
            "experiment": "ENCSR989YAW",
            "filename": "ENCFF885WJZ.bigWig",
        },
        "lung": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2025/10/01/"
                "a199b4b3-6547-4c98-b6ff-191254c80b0d/ENCFF958NZO.bigWig"
            ),
            "accession": "ENCFF958NZO",
            "experiment": "ENCSR141IUS",
            "filename": "ENCFF958NZO.bigWig",
        },
        "muscle": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2025/10/01/"
                "fb9a13cf-3cf6-4565-937b-93eb4a184a0a/ENCFF952ARY.bigWig"
            ),
            "accession": "ENCFF952ARY",
            "experiment": "ENCSR019MYA",
            "filename": "ENCFF952ARY.bigWig",
        },
        "liver": {
            "url": (
                "https://encode-public.s3.amazonaws.com/2025/08/14/"
                "caacdb4d-3c4e-41ef-b355-001984b7f0f3/ENCFF017TVV.bigWig"
            ),
            "accession": "ENCFF017TVV",
            "experiment": "ENCSR562FNN",
            "filename": "ENCFF017TVV.bigWig",
        },
    },
    # GRCh37: No ENCODE DNase-seq signal tracks aligned to hg19.
}

DEFAULT_DNASE_CELL_LINES = ("brain_cortex", "heart", "lung", "muscle", "liver")


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
        ATAC-seq cell lines. Default: 5-cell panel (K562, GM12878,
        HepG2, A549, IMR-90). Fold-change-over-control signal.
    window : int
        Half-window (bp) for context features. Default: 150.
    aggregation : str
        'summarized' (Strategy B, default): cross-tissue summary stats.
    signal_threshold : float
        ATAC fold-change threshold for tissue_breadth. Default: 2.0.
    peak_threshold : float
        ATAC fold-change threshold for has_peak. Default: 3.0.
    dnase_cell_lines : tuple of str
        DNase-seq primary tissues. Default: brain_cortex, heart, lung,
        muscle, liver. Read-depth normalized signal (different scale).
    dnase_signal_threshold : float
        DNase read-depth threshold for tissue_breadth. Default: 5.0.
    dnase_peak_threshold : float
        DNase read-depth threshold for has_peak. Default: 10.0.
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
    # DNase-seq settings (primary tissues, read-depth normalized signal)
    dnase_cell_lines: tuple[str, ...] = DEFAULT_DNASE_CELL_LINES
    dnase_signal_threshold: float = 5.0  # read-depth scale (higher than ATAC fold-change)
    dnase_peak_threshold: float = 10.0
    cache_dir: Optional[Path] = None
    remote_fallback: bool = True


class ChromAccessModality(Modality):
    """Chromatin accessibility features from ENCODE ATAC-seq + DNase-seq.

    Extracts accessibility signals from two complementary data sources:
    - ATAC-seq (fold-change-over-control) from cancer cell lines → ``atac_*``
    - DNase-seq (read-depth normalized) from primary tissues → ``dnase_*``

    In 'summarized' mode (default), produces cross-tissue statistics
    for each source: max, mean, breadth, variance, context_mean, has_peak.

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
                "Chromatin accessibility (ATAC-seq + DNase-seq) "
                "from ENCODE across cell lines and primary tissues."
            ),
        )

    def _compute_output_columns(self) -> list[str]:
        """Compute output column names based on config."""
        if self._cfg.aggregation != "summarized":
            return []
        cols = [
            "atac_max_across_tissues",
            "atac_mean_across_tissues",
            "atac_tissue_breadth",
            "atac_variance",
            "atac_context_mean",
            "atac_has_peak",
        ]
        if self._cfg.dnase_cell_lines:
            cols.extend([
                "dnase_max_across_tissues",
                "dnase_mean_across_tissues",
                "dnase_tissue_breadth",
                "dnase_variance",
                "dnase_context_mean",
                "dnase_has_peak",
            ])
        return cols

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
            # Validate DNase cell lines
            if self._cfg.dnase_cell_lines and build in DNASE_TRACK_REGISTRY:
                available_dnase = DNASE_TRACK_REGISTRY[build]
                for cl in self._cfg.dnase_cell_lines:
                    if cl not in available_dnase:
                        errors.append(
                            f"Tissue '{cl}' not in DNase-seq registry for "
                            f"{build}. Available: {list(available_dnase.keys())}"
                        )

        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add chromatin accessibility features to the DataFrame."""
        _check_pybigwig()

        build = self._resolve_build()
        n_positions = df.height

        logger.info(
            "Chrom access modality: %d positions, build=%s, "
            "atac_cell_lines=%s, dnase_cell_lines=%s, aggregation=%s",
            n_positions, build,
            self._cfg.cell_lines, self._cfg.dnase_cell_lines,
            self._cfg.aggregation,
        )

        # Graceful degradation for unsupported builds
        if build not in ATAC_TRACK_REGISTRY:
            logger.warning(
                "Build '%s' has no ENCODE ATAC-seq/DNase-seq tracks. "
                "Filling all chrom_access columns with NaN.", build,
            )
            return self._fill_nan(df)

        # ATAC-seq (cancer cell lines, fold-change signal)
        df = self._transform_source(
            df, build,
            registry=ATAC_TRACK_REGISTRY,
            cell_lines=self._cfg.cell_lines,
            signal_threshold=self._cfg.signal_threshold,
            peak_threshold=self._cfg.peak_threshold,
            prefix="atac",
            source_label="ATAC-seq",
        )

        # DNase-seq (primary tissues, read-depth normalized signal)
        if self._cfg.dnase_cell_lines:
            if build in DNASE_TRACK_REGISTRY:
                df = self._transform_source(
                    df, build,
                    registry=DNASE_TRACK_REGISTRY,
                    cell_lines=self._cfg.dnase_cell_lines,
                    signal_threshold=self._cfg.dnase_signal_threshold,
                    peak_threshold=self._cfg.dnase_peak_threshold,
                    prefix="dnase",
                    source_label="DNase-seq",
                )
            else:
                logger.warning(
                    "Build '%s' has no DNase-seq tracks. "
                    "Filling dnase_* columns with NaN.", build,
                )
                for col_name in self._compute_output_columns():
                    if col_name.startswith("dnase_") and col_name not in df.columns:
                        df = df.with_columns(
                            pl.lit(None).cast(pl.Float64).alias(col_name)
                        )

        logger.info(
            "Chrom access modality: added %d columns",
            len(self.meta.output_columns),
        )
        return df

    # ── Summarized mode (Strategy B) ──────────────────────────────────

    def _transform_source(
        self,
        df: pl.DataFrame,
        build: str,
        *,
        registry: dict[str, dict[str, dict[str, str]]],
        cell_lines: tuple[str, ...],
        signal_threshold: float,
        peak_threshold: float,
        prefix: str,
        source_label: str,
    ) -> pl.DataFrame:
        """Cross-tissue summary statistics for one accessibility source.

        Reused for both ATAC-seq and DNase-seq — the extraction logic
        is identical; only the registry, thresholds, and column prefix
        differ.
        """
        n = df.height

        cellline_scores = np.full((n, len(cell_lines)), np.nan)
        cellline_context = np.full((n, len(cell_lines)), np.nan)

        n_cl = len(cell_lines)
        for cl_idx, cl in enumerate(cell_lines):
            track_info = registry[build].get(cl)
            if track_info is None:
                logger.warning(
                    "No %s track for %s in %s. Skipping.",
                    source_label, cl, build,
                )
                continue

            logger.info(
                "  [%s %d/%d] Processing %s (%s)...",
                prefix.upper(), cl_idx + 1, n_cl, cl, track_info["accession"],
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
                "  [%s %d/%d] %s done: %d/%d positions scored in %.1fs",
                prefix.upper(), cl_idx + 1, n_cl, cl, n_valid, n, elapsed,
            )

        # Aggregate across cell lines (ignoring NaN)
        with np.errstate(all="ignore"):
            max_vals = np.nanmax(cellline_scores, axis=1)
            mean_vals = np.nanmean(cellline_scores, axis=1)
            var_vals = np.nanvar(cellline_scores, axis=1)
            context_mean_vals = np.nanmean(cellline_context, axis=1)

        breadth_vals = np.nansum(
            cellline_scores > signal_threshold, axis=1
        ).astype(np.float64)

        has_peak = (max_vals > peak_threshold).astype(np.float64)
        all_nan_mask = np.all(np.isnan(cellline_scores), axis=1)
        has_peak[all_nan_mask] = np.nan

        df = df.with_columns([
            pl.Series(f"{prefix}_max_across_tissues", max_vals, dtype=pl.Float64),
            pl.Series(f"{prefix}_mean_across_tissues", mean_vals, dtype=pl.Float64),
            pl.Series(f"{prefix}_tissue_breadth", breadth_vals, dtype=pl.Float64),
            pl.Series(f"{prefix}_variance", var_vals, dtype=pl.Float64),
            pl.Series(f"{prefix}_context_mean", context_mean_vals, dtype=pl.Float64),
            pl.Series(f"{prefix}_has_peak", has_peak, dtype=pl.Float64),
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
