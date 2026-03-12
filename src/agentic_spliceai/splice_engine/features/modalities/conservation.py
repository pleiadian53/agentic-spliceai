"""Conservation modality — evolutionary constraint features.

Extracts PhyloP and PhastCons scores from UCSC bigWig tracks, providing
per-position conservation features for the meta-layer. Scores are
build-matched: GRCh38 uses 100-way vertebrate alignment, GRCh37 uses
46-way.

Requires ``pyBigWig`` (optional dependency). Supports both local bigWig
files and remote UCSC URLs with HTTP range-request streaming.

See Also
--------
examples/features/docs/conservation-scores-tutorial.md
    Full tutorial on conservation scores, build alignment, and interpretation.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, List

import numpy as np
import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


# ── Build-specific conservation track registry ───────────────────────────
# BigWig files use UCSC naming (chr prefix) regardless of Ensembl convention.

TRACK_REGISTRY: dict[str, dict[str, dict[str, str]]] = {
    "GRCh38": {
        "phylop": {
            "alignment": "100way",
            "url": (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
                "phyloP100way/hg38.phyloP100way.bw"
            ),
            "filename": "hg38.phyloP100way.bw",
        },
        "phastcons": {
            "alignment": "100way",
            "url": (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/"
                "phastCons100way/hg38.phastCons100way.bw"
            ),
            "filename": "hg38.phastCons100way.bw",
        },
    },
    "GRCh37": {
        "phylop": {
            "alignment": "46way",
            "url": (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/"
                "phyloP46way/vertebrate.phyloP46way.bw"
            ),
            "filename": "vertebrate.phyloP46way.bw",
        },
        "phastcons": {
            "alignment": "46way",
            "url": (
                "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/"
                "phastCons46way/vertebrate.phastCons46way.bw"
            ),
            "filename": "vertebrate.phastCons46way.bw",
        },
    },
}


@dataclass
class ConservationConfig(ModalityConfig):
    """Configuration for the conservation modality.

    Attributes
    ----------
    base_model : str
        Base model name for build resolution (e.g., 'openspliceai').
        Determines which bigWig tracks (100-way vs 46-way) are used.
    tracks : tuple of str
        Which conservation tracks to extract. Default: both.
    window : int
        Half-window size (bp) for context features. Total window is
        ``2 * window + 1``. Set to 0 for single-base scores only.
    cache_dir : Path or None
        Directory containing local bigWig files. If None or files not
        found, falls back to remote UCSC URLs.
    remote_fallback : bool
        If True, use remote UCSC URLs when local files are not found.
        If False, raise FileNotFoundError when local files are missing.
    """

    base_model: str = "openspliceai"
    tracks: tuple[str, ...] = ("phylop", "phastcons")
    window: int = 10
    cache_dir: Optional[Path] = None
    remote_fallback: bool = True


def _check_pybigwig() -> None:
    """Verify pyBigWig is importable."""
    try:
        import pyBigWig  # noqa: F401
    except ImportError:
        raise ImportError(
            "pyBigWig is required for the conservation modality. Install with:\n"
            "  pip install pyBigWig\n"
            "If you get ABI errors, force-reinstall:\n"
            "  pip install --no-cache-dir --force-reinstall pyBigWig"
        )


class ConservationModality(Modality):
    """Evolutionary conservation features from PhyloP and PhastCons.

    Extracts per-position conservation scores from UCSC bigWig tracks.
    Build-matched: resolves the correct track (100-way for GRCh38,
    46-way for GRCh37) based on the base_model config.

    Output columns depend on which tracks are enabled:
    - phylop_score, phastcons_score: raw scores at each position
    - *_context_mean, *_context_max, *_context_std: windowed statistics
    - conservation_contrast: how much this position stands out from context
    """

    def __init__(self, config: ConservationConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: ConservationConfig = self.config  # type: ignore[assignment]
        self._build: Optional[str] = None
        # Lazy-loaded bigWig handles, keyed by track name
        self._handles: dict[str, object] = {}

    @property
    def meta(self) -> ModalityMeta:
        cols: list[str] = []
        for track in self._cfg.tracks:
            cols.append(f"{track}_score")
            if self._cfg.window > 0:
                cols.extend([
                    f"{track}_context_mean",
                    f"{track}_context_max",
                    f"{track}_context_std",
                ])
        # Cross-track feature (only if both tracks enabled)
        if "phylop" in self._cfg.tracks and self._cfg.window > 0:
            cols.append("conservation_contrast")

        return ModalityMeta(
            name="conservation",
            version="0.1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset({"chrom", "position"}),
            description=(
                "Evolutionary conservation from multi-species alignment "
                "(PhyloP / PhastCons)."
            ),
        )

    @classmethod
    def default_config(cls) -> ConservationConfig:
        return ConservationConfig()

    def validate(self, available_columns: Set[str]) -> List[str]:
        errors = super().validate(available_columns)
        try:
            _check_pybigwig()
        except ImportError as e:
            errors.append(str(e))
        # Validate that requested tracks exist for the resolved build
        build = self._resolve_build()
        if build not in TRACK_REGISTRY:
            errors.append(
                f"No conservation tracks configured for build '{build}'. "
                f"Available: {list(TRACK_REGISTRY.keys())}"
            )
        else:
            available_tracks = TRACK_REGISTRY[build]
            for track in self._cfg.tracks:
                if track not in available_tracks:
                    errors.append(
                        f"Track '{track}' not available for build '{build}'. "
                        f"Available: {list(available_tracks.keys())}"
                    )
        return errors

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add conservation features to the DataFrame.

        Batches queries by chromosome for efficiency — opens each bigWig
        once per chromosome rather than per position.
        """
        _check_pybigwig()

        build = self._resolve_build()
        chromosomes = df["chrom"].unique().to_list()
        n_positions = df.height

        logger.info(
            "Conservation modality: %d positions across %d chromosome(s), "
            "build=%s, tracks=%s, window=%d",
            n_positions, len(chromosomes), build,
            self._cfg.tracks, self._cfg.window,
        )

        # Initialize output arrays for each feature column
        output_arrays: dict[str, np.ndarray] = {}
        for track in self._cfg.tracks:
            output_arrays[f"{track}_score"] = np.full(n_positions, np.nan)
            if self._cfg.window > 0:
                output_arrays[f"{track}_context_mean"] = np.full(n_positions, np.nan)
                output_arrays[f"{track}_context_max"] = np.full(n_positions, np.nan)
                output_arrays[f"{track}_context_std"] = np.full(n_positions, np.nan)

        # Process each track
        for track in self._cfg.tracks:
            bw = self._open_track(track, build)
            try:
                self._extract_track_features(
                    df, bw, track, output_arrays,
                )
            finally:
                bw.close()

        # Build Polars columns from numpy arrays
        new_cols = []
        for col_name, arr in output_arrays.items():
            new_cols.append(pl.Series(col_name, arr, dtype=pl.Float64))

        df = df.with_columns(new_cols)

        # Cross-track feature: conservation contrast
        if "phylop" in self._cfg.tracks and self._cfg.window > 0:
            df = df.with_columns(
                (pl.col("phylop_score") - pl.col("phylop_context_mean"))
                .alias("conservation_contrast")
            )

        logger.info("Conservation modality: added %d columns", len(self.meta.output_columns))
        return df

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_build(self) -> str:
        """Resolve base_model → genomic build."""
        if self._build is not None:
            return self._build

        from agentic_spliceai.splice_engine.resources import get_model_resources

        resources = get_model_resources(self._cfg.base_model)
        self._build = resources.build
        return self._build

    def _resolve_track_path(self, track: str, build: str) -> str:
        """Resolve a track to a local path or remote URL.

        Returns
        -------
        str
            Local file path or remote URL suitable for pyBigWig.open().
        """
        track_info = TRACK_REGISTRY[build][track]

        # Check local cache first
        if self._cfg.cache_dir is not None:
            local_path = Path(self._cfg.cache_dir) / track_info["filename"]
            if local_path.exists():
                logger.info("Using local bigWig: %s", local_path)
                return str(local_path)

        # Fall back to remote
        if not self._cfg.remote_fallback:
            raise FileNotFoundError(
                f"Local bigWig not found: {track_info['filename']}. "
                f"Set cache_dir to a directory containing the file, or "
                f"enable remote_fallback=True to use UCSC URLs.\n"
                f"Download from: {track_info['url']}"
            )

        logger.info(
            "Using remote bigWig for %s (%s %s): %s",
            track, build, track_info["alignment"], track_info["url"],
        )
        return track_info["url"]

    def _open_track(self, track: str, build: str) -> "pyBigWig.pyBigWig":
        """Open a bigWig track (local or remote)."""
        import pyBigWig

        path_or_url = self._resolve_track_path(track, build)
        bw = pyBigWig.open(path_or_url)
        if bw is None:
            raise RuntimeError(f"Failed to open bigWig: {path_or_url}")
        return bw

    def _extract_track_features(
        self,
        df: pl.DataFrame,
        bw: "pyBigWig.pyBigWig",
        track: str,
        output_arrays: dict[str, np.ndarray],
    ) -> None:
        """Extract conservation features for one track across all positions.

        Batches by chromosome: queries the full span of positions on each
        chromosome in a single bigWig call, then indexes into the result.
        """
        w = self._cfg.window
        bw_chroms = bw.chroms()

        # Process each chromosome
        for chrom in df["chrom"].unique().to_list():
            # Map our chromosome name to bigWig's naming convention
            # (UCSC bigWig always uses chr prefix)
            bw_chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"

            if bw_chrom not in bw_chroms:
                logger.warning(
                    "Chromosome '%s' (mapped to '%s') not found in %s bigWig. "
                    "Skipping %s conservation scores for this chromosome.",
                    chrom, bw_chrom, track, track,
                )
                continue

            chrom_length = bw_chroms[bw_chrom]

            # Get row indices and positions for this chromosome
            chrom_mask = df["chrom"] == chrom
            chrom_indices = np.where(chrom_mask.to_numpy())[0]
            positions = df.filter(chrom_mask)["position"].to_numpy()

            if len(positions) == 0:
                continue

            # Determine the span to query (with window padding)
            span_start = max(0, int(positions.min()) - w)
            span_end = min(chrom_length, int(positions.max()) + w + 1)

            # Single bigWig query for the entire span
            logger.debug(
                "Querying %s on %s: %d-%d (%d bp span, %d positions)",
                track, bw_chrom, span_start, span_end,
                span_end - span_start, len(positions),
            )
            raw_values = bw.values(bw_chrom, span_start, span_end)

            if raw_values is None:
                logger.warning(
                    "bigWig returned None for %s:%d-%d. Filling with NaN.",
                    bw_chrom, span_start, span_end,
                )
                continue

            # Convert to numpy, replace NaN (uncovered regions) with 0
            span_array = np.array(raw_values, dtype=np.float64)
            span_array = np.nan_to_num(span_array, nan=0.0)

            # Extract per-position features by indexing into the span
            for i, (idx, pos) in enumerate(zip(chrom_indices, positions)):
                center = int(pos) - span_start

                # Bounds check
                if center < 0 or center >= len(span_array):
                    continue

                # Core score at position
                output_arrays[f"{track}_score"][idx] = span_array[center]

                # Context window features
                if w > 0:
                    win_start = max(0, center - w)
                    win_end = min(len(span_array), center + w + 1)
                    window = span_array[win_start:win_end]

                    if len(window) > 0:
                        output_arrays[f"{track}_context_mean"][idx] = np.mean(window)
                        output_arrays[f"{track}_context_max"][idx] = np.max(window)
                        output_arrays[f"{track}_context_std"][idx] = np.std(window)

            logger.debug(
                "Extracted %s features for %d positions on %s",
                track, len(positions), chrom,
            )
