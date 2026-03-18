"""Epigenetic modality — histone modification features from ENCODE ChIP-seq.

Extracts H3K36me3 (exon body mark) and H3K4me3 (promoter mark) signals
from ENCODE fold-change-over-control bigWig tracks. Supports two
aggregation modes:

- **summarized** (default, Strategy B): Cross-tissue summary statistics
  (max, mean, breadth, variance, context_mean, exon_intron_ratio per mark).
  Captures constitutive vs tissue-specific exon usage in ~12 columns.
- **detailed** (Strategy A): Per-cell-line features for exploratory analysis.

GRCh38 only — ENCODE does not provide hg19 signal tracks. For GRCh37
builds, all columns are filled with NaN and a warning is logged.

Requires ``pyBigWig`` (optional dependency).

See Also
--------
examples/features/docs/epigenetic-marks-tutorial.md
    Full tutorial on histone marks, tissue panel, and interpretation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── ENCODE ChIP-seq track registry (GRCh38 only) ────────────────────
# Fold-change-over-control bigWig files, replicate-pooled.
# Keys: build → cell_line → mark → {url, accession, filename}
# All URLs are ENCODE S3 direct links (stable, no redirects).

EPIGENETIC_TRACK_REGISTRY: dict[str, dict[str, dict[str, dict[str, str]]]] = {
    "GRCh38": {
        "K562": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/29/"
                    "2d536994-e28f-4f50-b5d4-c0441e0e059f/ENCFF163NTH.bigWig"
                ),
                "accession": "ENCFF163NTH",
                "experiment": "ENCSR000AKR",
                "filename": "ENCFF163NTH.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2016/08/12/"
                    "e59a8bb2-02ee-451d-994d-732db7c523a0/ENCFF814IYI.bigWig"
                ),
                "accession": "ENCFF814IYI",
                "experiment": "ENCSR000AKU",
                "filename": "ENCFF814IYI.bigWig",
            },
        },
        "GM12878": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/29/"
                    "7bf3c0a1-df2b-40fd-950f-9aa735aca81a/ENCFF312MUY.bigWig"
                ),
                "accession": "ENCFF312MUY",
                "experiment": "ENCSR000AKE",
                "filename": "ENCFF312MUY.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2016/08/23/"
                    "6b4cd8db-2a4c-4023-97c4-1232aaa3ef59/ENCFF776DPQ.bigWig"
                ),
                "accession": "ENCFF776DPQ",
                "experiment": "ENCSR000AKA",
                "filename": "ENCFF776DPQ.bigWig",
            },
        },
        "H1": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2017/02/15/"
                    "4564f196-6174-4578-8ca2-c6d9deec4f96/ENCFF141YAA.bigWig"
                ),
                "accession": "ENCFF141YAA",
                "experiment": "ENCSR476KTK",
                "filename": "ENCFF141YAA.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/19/"
                    "14b72686-b8a9-48f1-81b2-5c57d157dc30/ENCFF602IAY.bigWig"
                ),
                "accession": "ENCFF602IAY",
                "experiment": "ENCSR443YAS",
                "filename": "ENCFF602IAY.bigWig",
            },
        },
        "HepG2": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/29/"
                    "9a9c2868-0d43-4f3d-bedd-81517a49dad8/ENCFF414GNH.bigWig"
                ),
                "accession": "ENCFF414GNH",
                "experiment": "ENCSR000AMB",
                "filename": "ENCFF414GNH.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/09/30/"
                    "2ca06a67-3a26-48f4-8354-fc4613c73b67/ENCFF284FVP.bigWig"
                ),
                "accession": "ENCFF284FVP",
                "experiment": "ENCSR000AMP",
                "filename": "ENCFF284FVP.bigWig",
            },
        },
        "keratinocyte": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/29/"
                    "9e0a8b55-8146-4ec1-8804-ca5e50e80ef2/ENCFF049JNX.bigWig"
                ),
                "accession": "ENCFF049JNX",
                "experiment": "ENCSR000ALM",
                "filename": "ENCFF049JNX.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/20/"
                    "72767546-aa79-4c57-8c72-68c380e41387/ENCFF632PUY.bigWig"
                ),
                "accession": "ENCFF632PUY",
                "experiment": "ENCSR970FPM",
                "filename": "ENCFF632PUY.bigWig",
            },
        },
        "A549": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/03/31/"
                    "cd595129-5ff2-4219-8afb-c803b59ec1e2/ENCFF473XIC.bigWig"
                ),
                "accession": "ENCFF473XIC",
                "experiment": "ENCSR000AUL",
                "filename": "ENCFF473XIC.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/01/"
                    "0e005a13-0b6a-4d72-acb8-ea1e7e937a3c/ENCFF242FAU.bigWig"
                ),
                "accession": "ENCFF242FAU",
                "experiment": "ENCSR944WVU",
                "filename": "ENCFF242FAU.bigWig",
            },
        },
        "MCF-7": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/04/09/"
                    "3b9141b8-e2f0-4d0b-baca-3e9489219ded/ENCFF685ECA.bigWig"
                ),
                "accession": "ENCFF685ECA",
                "experiment": "ENCSR610IYQ",
                "filename": "ENCFF685ECA.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2020/10/09/"
                    "859ccb8a-1bef-4d66-a351-1845e814ec8b/ENCFF267OQQ.bigWig"
                ),
                "accession": "ENCFF267OQQ",
                "experiment": "ENCSR985MIB",
                "filename": "ENCFF267OQQ.bigWig",
            },
        },
        "SK-N-SH": {
            "h3k36me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2016/10/17/"
                    "679d3c1e-e9f8-4815-bd38-136a85420b4c/ENCFF681BMN.bigWig"
                ),
                "accession": "ENCFF681BMN",
                "experiment": "ENCSR978CNH",
                "filename": "ENCFF681BMN.bigWig",
            },
            "h3k4me3": {
                "url": (
                    "https://encode-public.s3.amazonaws.com/2023/01/21/"
                    "8925ff0b-7233-45dc-8886-772033154655/ENCFF755FHH.bigWig"
                ),
                "accession": "ENCFF755FHH",
                "experiment": "ENCSR975GZA",
                "filename": "ENCFF755FHH.bigWig",
            },
        },
    },
    # GRCh37: No ENCODE signal tracks available for hg19.
    # The modality gracefully degrades (fills NaN, logs warning).
}

# Minimal panel: ENCODE Tier 1 + liver + skin (diverse, smaller files)
DEFAULT_CELL_LINES = ("K562", "GM12878", "H1", "HepG2", "keratinocyte")


@dataclass
class EpigeneticConfig(ModalityConfig):
    """Configuration for the epigenetic modality.

    Attributes
    ----------
    base_model : str
        Base model name for build resolution (e.g., 'openspliceai').
    marks : tuple of str
        Histone marks to extract. Default: H3K36me3 + H3K4me3.
    cell_lines : tuple of str
        Cell lines to query. Default: minimal 5-tissue panel.
    window : int
        Half-window (bp) for context features. Histone marks are
        regional (~200bp nucleosome scale), so larger windows than
        conservation are appropriate. Default: 200.
    exon_intron_window : int
        Half-window (bp) for exon-intron ratio computation at
        donor/acceptor boundaries. Default: 500.
    aggregation : str
        'summarized' (Strategy B, default): cross-tissue summary stats.
        'detailed' (Strategy A): per-cell-line features.
    breadth_threshold : float
        Minimum fold-change signal to count a tissue as "active"
        for the tissue_breadth feature. Default: 1.5.
    cache_dir : Path or None
        Local bigWig cache directory. If None, uses remote ENCODE URLs.
    remote_fallback : bool
        Fall back to remote URLs when local files not found.
    """

    base_model: str = "openspliceai"
    marks: tuple[str, ...] = ("h3k36me3", "h3k4me3")
    cell_lines: tuple[str, ...] = DEFAULT_CELL_LINES
    window: int = 200
    exon_intron_window: int = 500
    aggregation: str = "summarized"
    breadth_threshold: float = 1.5
    cache_dir: Optional[Path] = None
    remote_fallback: bool = True


def _check_pybigwig() -> None:
    """Verify pyBigWig is importable."""
    try:
        import pyBigWig  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyBigWig is required for the epigenetic modality. Install with:\n"
            "  pip install pyBigWig\n"
            "If you get ABI errors, force-reinstall:\n"
            "  pip install --no-cache-dir --force-reinstall pyBigWig"
        )


class EpigeneticModality(Modality):
    """Histone modification features from ENCODE ChIP-seq data.

    Extracts H3K36me3 and H3K4me3 fold-change-over-control signals from
    ENCODE bigWig tracks across a panel of cell types.

    In 'summarized' mode (default), produces cross-tissue statistics:
    max, mean, breadth, variance, context_mean, exon_intron_ratio per mark.

    In 'detailed' mode, produces per-cell-line scores and context features.

    GRCh38 only. For GRCh37, fills all columns with NaN.
    """

    def __init__(self, config: EpigeneticConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: EpigeneticConfig = self.config  # type: ignore[assignment]
        self._build: Optional[str] = None

    @property
    def meta(self) -> ModalityMeta:
        cols = self._compute_output_columns()
        return ModalityMeta(
            name="epigenetic",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position"}),
            optional_inputs=frozenset({"strand"}),
            description=(
                "Histone modification features (H3K36me3/H3K4me3) "
                "from ENCODE ChIP-seq across multiple cell types."
            ),
        )

    def _compute_output_columns(self) -> list[str]:
        """Compute output column names based on config."""
        cols: list[str] = []
        if self._cfg.aggregation == "summarized":
            for mark in self._cfg.marks:
                cols.extend([
                    f"{mark}_max_across_tissues",
                    f"{mark}_mean_across_tissues",
                    f"{mark}_tissue_breadth",
                    f"{mark}_variance",
                    f"{mark}_context_mean",
                    f"{mark}_exon_intron_ratio",
                ])
        else:  # detailed
            for mark in self._cfg.marks:
                for cl in self._cfg.cell_lines:
                    cl_key = cl.lower().replace("-", "")
                    cols.append(f"{mark}_{cl_key}_score")
                    if self._cfg.window > 0:
                        cols.append(f"{mark}_{cl_key}_context_mean")
        return cols

    @classmethod
    def default_config(cls) -> EpigeneticConfig:
        return EpigeneticConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        try:
            _check_pybigwig()
        except ImportError as e:
            errors.append(str(e))

        if self._cfg.aggregation not in ("summarized", "detailed"):
            errors.append(
                f"Invalid aggregation mode: '{self._cfg.aggregation}'. "
                f"Must be 'summarized' or 'detailed'."
            )

        build = self._resolve_build()
        if build not in EPIGENETIC_TRACK_REGISTRY:
            # Not a hard error — we fill NaN for unsupported builds
            logger.warning(
                "No epigenetic tracks for build '%s'. All epigenetic "
                "features will be NaN. Use GRCh38 (openspliceai) for "
                "epigenetic features.",
                build,
            )
        else:
            available_cl = EPIGENETIC_TRACK_REGISTRY[build]
            for cl in self._cfg.cell_lines:
                if cl not in available_cl:
                    errors.append(
                        f"Cell line '{cl}' not in registry for {build}. "
                        f"Available: {list(available_cl.keys())}"
                    )

        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add epigenetic features to the DataFrame."""
        _check_pybigwig()

        build = self._resolve_build()
        n_positions = df.height

        logger.info(
            "Epigenetic modality: %d positions, build=%s, marks=%s, "
            "cell_lines=%s, aggregation=%s",
            n_positions, build, self._cfg.marks,
            self._cfg.cell_lines, self._cfg.aggregation,
        )

        # Graceful degradation for unsupported builds
        if build not in EPIGENETIC_TRACK_REGISTRY:
            logger.warning(
                "Build '%s' has no ENCODE epigenetic tracks. "
                "Filling all epigenetic columns with NaN.", build,
            )
            return self._fill_nan(df)

        if self._cfg.aggregation == "summarized":
            df = self._transform_summarized(df, build)
        else:
            df = self._transform_detailed(df, build)

        logger.info(
            "Epigenetic modality: added %d columns",
            len(self.meta.output_columns),
        )
        return df

    # ── Summarized mode (Strategy B) ──────────────────────────────────

    def _transform_summarized(
        self, df: pl.DataFrame, build: str
    ) -> pl.DataFrame:
        """Cross-tissue summary statistics per mark."""
        n = df.height
        has_strand = "strand" in df.columns

        for mark in self._cfg.marks:
            # Collect per-cell-line scores at each position
            # Shape: (n_positions, n_cell_lines)
            cellline_scores = np.full(
                (n, len(self._cfg.cell_lines)), np.nan
            )
            # Context means (pooled across cell lines for final context_mean)
            cellline_context = np.full(
                (n, len(self._cfg.cell_lines)), np.nan
            )
            # Exon-intron ratios per cell line
            cellline_eir = np.full(
                (n, len(self._cfg.cell_lines)), np.nan
            )

            for cl_idx, cl in enumerate(self._cfg.cell_lines):
                track_info = EPIGENETIC_TRACK_REGISTRY[build].get(cl, {}).get(mark)
                if track_info is None:
                    logger.warning(
                        "No %s track for %s in %s. Skipping.", mark, cl, build,
                    )
                    continue

                bw = self._open_track(track_info)
                try:
                    scores, ctx, eir = self._extract_mark_features(
                        df, bw, has_strand,
                    )
                    cellline_scores[:, cl_idx] = scores
                    cellline_context[:, cl_idx] = ctx
                    cellline_eir[:, cl_idx] = eir
                finally:
                    bw.close()

            # Aggregate across cell lines (ignoring NaN)
            with np.errstate(all="ignore"):
                max_vals = np.nanmax(cellline_scores, axis=1)
                mean_vals = np.nanmean(cellline_scores, axis=1)
                var_vals = np.nanvar(cellline_scores, axis=1)
                context_mean_vals = np.nanmean(cellline_context, axis=1)
                eir_mean = np.nanmean(cellline_eir, axis=1)

            # Tissue breadth: count of cell lines with signal > threshold
            breadth_vals = np.nansum(
                cellline_scores > self._cfg.breadth_threshold, axis=1
            ).astype(np.float64)

            df = df.with_columns([
                pl.Series(f"{mark}_max_across_tissues", max_vals, dtype=pl.Float64),
                pl.Series(f"{mark}_mean_across_tissues", mean_vals, dtype=pl.Float64),
                pl.Series(f"{mark}_tissue_breadth", breadth_vals, dtype=pl.Float64),
                pl.Series(f"{mark}_variance", var_vals, dtype=pl.Float64),
                pl.Series(f"{mark}_context_mean", context_mean_vals, dtype=pl.Float64),
                pl.Series(f"{mark}_exon_intron_ratio", eir_mean, dtype=pl.Float64),
            ])

        return df

    # ── Detailed mode (Strategy A) ────────────────────────────────────

    def _transform_detailed(
        self, df: pl.DataFrame, build: str
    ) -> pl.DataFrame:
        """Per-cell-line scores and context features."""
        has_strand = "strand" in df.columns

        for mark in self._cfg.marks:
            for cl in self._cfg.cell_lines:
                cl_key = cl.lower().replace("-", "")
                track_info = EPIGENETIC_TRACK_REGISTRY[build].get(cl, {}).get(mark)
                if track_info is None:
                    logger.warning(
                        "No %s track for %s. Filling NaN.", mark, cl,
                    )
                    df = df.with_columns(
                        pl.lit(None).cast(pl.Float64).alias(f"{mark}_{cl_key}_score")
                    )
                    if self._cfg.window > 0:
                        df = df.with_columns(
                            pl.lit(None).cast(pl.Float64).alias(
                                f"{mark}_{cl_key}_context_mean"
                            )
                        )
                    continue

                bw = self._open_track(track_info)
                try:
                    scores, ctx, _ = self._extract_mark_features(
                        df, bw, has_strand,
                    )
                finally:
                    bw.close()

                df = df.with_columns(
                    pl.Series(f"{mark}_{cl_key}_score", scores, dtype=pl.Float64)
                )
                if self._cfg.window > 0:
                    df = df.with_columns(
                        pl.Series(
                            f"{mark}_{cl_key}_context_mean", ctx, dtype=pl.Float64
                        )
                    )

        return df

    # ── Core extraction ───────────────────────────────────────────────

    def _extract_mark_features(
        self,
        df: pl.DataFrame,
        bw: "pyBigWig.pyBigWig",
        has_strand: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract score, context, and exon-intron ratio for one mark+cell_line.

        Returns
        -------
        scores : np.ndarray
            Per-position signal (fold-change over control).
        context_means : np.ndarray
            Mean signal in a window around each position.
        exon_intron_ratios : np.ndarray
            Ratio of upstream to downstream signal. For + strand donors,
            upstream = exonic, downstream = intronic. Ratio > 1 means
            the exon body is marked. For unknown strand, uses symmetric window.
        """
        n = df.height
        w = self._cfg.window
        eir_w = self._cfg.exon_intron_window

        scores = np.full(n, np.nan)
        context_means = np.full(n, np.nan)
        exon_intron_ratios = np.full(n, np.nan)

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

            # Strand info for exon-intron ratio directionality
            if has_strand:
                strands = df.filter(chrom_mask)["strand"].to_list()
            else:
                strands = ["+"] * len(positions)  # default

            # Query the full span (padded by the larger window)
            max_w = max(w, eir_w)
            span_start = max(0, int(positions.min()) - max_w)
            span_end = min(chrom_length, int(positions.max()) + max_w + 1)

            raw_values = bw.values(bw_chrom, span_start, span_end)
            if raw_values is None:
                continue

            span_array = np.array(raw_values, dtype=np.float64)
            span_array = np.nan_to_num(span_array, nan=0.0)

            for idx, pos, strand in zip(chrom_indices, positions, strands):
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

                # Exon-intron ratio
                # For + strand: upstream (exonic) is left, downstream (intronic) is right
                # For - strand: reversed
                if eir_w > 0:
                    left_start = max(0, center - eir_w)
                    left_end = center
                    right_start = center + 1
                    right_end = min(len(span_array), center + eir_w + 1)

                    left_signal = span_array[left_start:left_end]
                    right_signal = span_array[right_start:right_end]

                    if len(left_signal) > 0 and len(right_signal) > 0:
                        left_mean = np.mean(left_signal)
                        right_mean = np.mean(right_signal)

                        # Assign upstream/downstream based on strand
                        if strand == "-":
                            upstream, downstream = right_mean, left_mean
                        else:
                            upstream, downstream = left_mean, right_mean

                        # Log-ratio: symmetric, bounded, standard in epigenomics
                        # Positive = exonic enrichment, negative = intronic
                        # Zero = equal signal on both sides
                        exon_intron_ratios[idx] = (
                            np.log2(upstream + 1.0) - np.log2(downstream + 1.0)
                        )

        return scores, context_means, exon_intron_ratios

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
