"""
Dense per-position feature extraction for sequence-level meta models.

Extracts multimodal features at **every position** in a genomic window,
returning a dense ``[L, C]`` array suitable for the sequence-level
``MetaSpliceModel`` (M*-S models).

This differs from the position-level modality pipeline (``FeaturePipeline``)
which operates on pre-sampled DataFrame rows.  Here we produce raw arrays
indexed by genomic coordinate, designed to be stacked as input channels
alongside one-hot DNA and base model scores.

Channels (default, 9 total)::

    0: phylop_score        (PhyloP conservation, BigWig)
    1: phastcons_score     (PhastCons conservation, BigWig)
    2: h3k36me3_max        (H3K36me3 histone mark, BigWig, cross-tissue max)
    3: h3k4me3_max         (H3K4me3 histone mark, BigWig, cross-tissue max)
    4: atac_max            (ATAC-seq accessibility, BigWig, cross-tissue max)
    5: dnase_max           (DNase-seq accessibility, BigWig, cross-tissue max)
    6: junction_log1p      (GTEx junction reads, sparse, 0-filled)
    7: junction_has_support (binary junction indicator, sparse)
    8: rbp_n_bound         (RBP binding count from eCLIP, sparse, 0-filled)

Usage::

    extractor = DenseFeatureExtractor(build="GRCh38")
    features = extractor.extract_window("chr17", 43044295, 43125364)
    # features.shape == (81069, 9)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel definitions
# ---------------------------------------------------------------------------

# Ordered list of output channels.  Index matches the channel dimension
# in the MetaSpliceModel's mm_features input.
CHANNEL_NAMES: List[str] = [
    "phylop_score",
    "phastcons_score",
    "h3k36me3_max",
    "h3k4me3_max",
    "atac_max",
    "dnase_max",
    "junction_log1p",
    "junction_has_support",
    "rbp_n_bound",
]

# Channels to exclude for M3-S (junction as target, not feature)
M3_EXCLUDE = {"junction_log1p", "junction_has_support"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DenseFeatureConfig:
    """Configuration for dense feature extraction.

    Parameters
    ----------
    build : str
        Genome build (``"GRCh38"`` or ``"GRCh37"``).
    exclude_channels : set of str
        Channel names to skip (e.g., junction channels for M3-S).
    bigwig_cache_dir : Path, optional
        Local directory for cached bigWig files.  If None, streams
        from remote URLs (slower, requires network).
    junction_parquet : Path, optional
        Path to aggregated junction parquet.  If None, junction
        channels are zero-filled.
    eclip_parquet : Path, optional
        Path to aggregated eCLIP peaks parquet.  If None, RBP
        channel is zero-filled.
    """

    build: str = "GRCh38"
    exclude_channels: set = field(default_factory=set)
    bigwig_cache_dir: Optional[Path] = None
    junction_parquet: Optional[Path] = None
    eclip_parquet: Optional[Path] = None


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class DenseFeatureExtractor:
    """Extract dense multimodal features for genomic windows.

    Opens bigWig handles lazily on first use and caches them for
    subsequent calls.  Sparse data sources (junction, eCLIP) are
    loaded once into memory.

    Parameters
    ----------
    config : DenseFeatureConfig
        Extraction configuration.
    """

    def __init__(self, config: Optional[DenseFeatureConfig] = None) -> None:
        if config is None:
            config = DenseFeatureConfig()
        self.config = config

        # Determine active channels
        self._channels = [
            ch for ch in CHANNEL_NAMES if ch not in config.exclude_channels
        ]
        self._channel_idx = {ch: i for i, ch in enumerate(self._channels)}

        # Lazy-loaded handles
        self._bigwig_handles: Dict[str, object] = {}
        self._junction_index: Optional[Dict] = None
        self._eclip_peaks: Optional[object] = None

        logger.info(
            "DenseFeatureExtractor: %d channels (%s excluded)",
            len(self._channels),
            config.exclude_channels or "none",
        )

    @property
    def num_channels(self) -> int:
        """Number of active output channels."""
        return len(self._channels)

    @property
    def channel_names(self) -> List[str]:
        """Ordered names of active output channels."""
        return list(self._channels)

    def extract_window(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Extract dense features for a genomic window.

        Parameters
        ----------
        chrom : str
            Chromosome name (e.g., ``"chr17"``).
        start : int
            0-based start coordinate (inclusive).
        end : int
            0-based end coordinate (exclusive).

        Returns
        -------
        np.ndarray
            Shape ``[end - start, num_channels]``, dtype float32.
            Columns correspond to :attr:`channel_names`.
        """
        L = end - start
        out = np.zeros((L, self.num_channels), dtype=np.float32)

        for ch_name in self._channels:
            idx = self._channel_idx[ch_name]
            try:
                if ch_name == "phylop_score":
                    out[:, idx] = self._query_bigwig("phylop", chrom, start, end)
                elif ch_name == "phastcons_score":
                    out[:, idx] = self._query_bigwig("phastcons", chrom, start, end)
                elif ch_name == "h3k36me3_max":
                    out[:, idx] = self._query_epigenetic_max("H3K36me3", chrom, start, end)
                elif ch_name == "h3k4me3_max":
                    out[:, idx] = self._query_epigenetic_max("H3K4me3", chrom, start, end)
                elif ch_name == "atac_max":
                    out[:, idx] = self._query_access_max("atac", chrom, start, end)
                elif ch_name == "dnase_max":
                    out[:, idx] = self._query_access_max("dnase", chrom, start, end)
                elif ch_name == "junction_log1p":
                    out[:, idx] = self._query_junction(chrom, start, end, "log1p")
                elif ch_name == "junction_has_support":
                    out[:, idx] = self._query_junction(chrom, start, end, "has_support")
                elif ch_name == "rbp_n_bound":
                    out[:, idx] = self._query_rbp(chrom, start, end)
            except Exception as e:
                logger.warning("Channel %s failed for %s:%d-%d: %s", ch_name, chrom, start, end, e)
                # Leave as zeros (graceful degradation)

        return out

    def close(self) -> None:
        """Close all open bigWig handles."""
        for handle in self._bigwig_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._bigwig_handles.clear()

    # ------------------------------------------------------------------
    # BigWig tracks (conservation)
    # ------------------------------------------------------------------

    def _query_bigwig(
        self,
        track: str,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Query a single conservation bigWig track."""
        bw = self._get_bigwig_handle(track)
        bw_chrom = self._resolve_bigwig_chrom(bw, chrom)
        if bw_chrom is None:
            return np.zeros(end - start, dtype=np.float32)

        chrom_len = bw.chroms()[bw_chrom]
        s = max(0, start)
        e = min(chrom_len, end)
        if s >= e:
            return np.zeros(end - start, dtype=np.float32)

        raw = bw.values(bw_chrom, s, e)
        arr = np.array(raw, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        # Pad if window extends beyond chromosome
        if s > start or e < end:
            padded = np.zeros(end - start, dtype=np.float32)
            padded[s - start:s - start + len(arr)] = arr
            return padded
        return arr

    def _get_bigwig_handle(self, track: str) -> object:
        """Get or open a bigWig handle for a conservation track."""
        if track not in self._bigwig_handles:
            import pyBigWig

            from .modalities.conservation import TRACK_REGISTRY

            build = self.config.build
            if build not in TRACK_REGISTRY or track not in TRACK_REGISTRY[build]:
                raise ValueError(f"No {track} track for build {build}")

            track_info = TRACK_REGISTRY[build][track]
            url = track_info["url"]

            # Prefer local cache if available
            if self.config.bigwig_cache_dir:
                local = self.config.bigwig_cache_dir / track_info["filename"]
                if local.exists():
                    url = str(local)

            logger.info("Opening bigWig: %s", url)
            self._bigwig_handles[track] = pyBigWig.open(url)

        return self._bigwig_handles[track]

    # ------------------------------------------------------------------
    # Epigenetic marks (cross-tissue max)
    # ------------------------------------------------------------------

    def _query_epigenetic_max(
        self,
        mark: str,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Query histone mark bigWigs and return cross-tissue max."""
        from .modalities.epigenetic import EPIGENETIC_TRACK_REGISTRY, DEFAULT_CELL_LINES

        build = self.config.build
        if build not in EPIGENETIC_TRACK_REGISTRY:
            return np.zeros(end - start, dtype=np.float32)

        L = end - start
        mark_lower = mark.lower()
        scores = []
        for cell_line in DEFAULT_CELL_LINES:
            if cell_line not in EPIGENETIC_TRACK_REGISTRY[build]:
                continue
            marks_dict = EPIGENETIC_TRACK_REGISTRY[build][cell_line]
            if mark_lower not in marks_dict:
                continue

            key = f"epi_{mark_lower}_{cell_line}"
            arr = self._query_bigwig_url(
                key, marks_dict[mark_lower]["url"], chrom, start, end,
            )
            scores.append(arr)

        if not scores:
            return np.zeros(L, dtype=np.float32)

        stacked = np.stack(scores, axis=0)  # [n_cell_lines, L]
        with np.errstate(all="ignore"):
            return np.nanmax(stacked, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Chromatin accessibility (cross-tissue max)
    # ------------------------------------------------------------------

    def _query_access_max(
        self,
        source: str,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Query ATAC or DNase bigWigs and return cross-tissue max."""
        from .modalities.chrom_access import ATAC_TRACK_REGISTRY, DNASE_TRACK_REGISTRY

        build = self.config.build
        registry = ATAC_TRACK_REGISTRY if source == "atac" else DNASE_TRACK_REGISTRY
        if build not in registry:
            return np.zeros(end - start, dtype=np.float32)

        L = end - start
        scores = []
        for tissue, track_info in registry[build].items():
            key = f"access_{source}_{tissue}"
            arr = self._query_bigwig_url(key, track_info["url"], chrom, start, end)
            scores.append(arr)

        if not scores:
            return np.zeros(L, dtype=np.float32)

        stacked = np.stack(scores, axis=0)
        with np.errstate(all="ignore"):
            return np.nanmax(stacked, axis=0).astype(np.float32)

    def _query_bigwig_url(
        self,
        cache_key: str,
        url: str,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Query a bigWig by URL with handle caching."""
        if cache_key not in self._bigwig_handles:
            import pyBigWig
            logger.debug("Opening bigWig: %s", url)
            self._bigwig_handles[cache_key] = pyBigWig.open(url)

        bw = self._bigwig_handles[cache_key]
        bw_chrom = self._resolve_bigwig_chrom(bw, chrom)
        if bw_chrom is None:
            return np.zeros(end - start, dtype=np.float32)

        chrom_len = bw.chroms()[bw_chrom]
        s = max(0, start)
        e = min(chrom_len, end)
        if s >= e:
            return np.zeros(end - start, dtype=np.float32)

        raw = bw.values(bw_chrom, s, e)
        arr = np.array(raw, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)

        if s > start or e < end:
            padded = np.zeros(end - start, dtype=np.float32)
            padded[s - start:s - start + len(arr)] = arr
            return padded
        return arr

    # ------------------------------------------------------------------
    # Junction reads (sparse lookup)
    # ------------------------------------------------------------------

    def _query_junction(
        self,
        chrom: str,
        start: int,
        end: int,
        field: str,
    ) -> np.ndarray:
        """Query junction features for a window (sparse, 0-filled)."""
        idx = self._get_junction_index()
        if idx is None:
            return np.zeros(end - start, dtype=np.float32)

        L = end - start
        out = np.zeros(L, dtype=np.float32)

        # Filter to this chrom + window
        chrom_data = idx.get(chrom)
        if chrom_data is None:
            return out

        for pos, data in chrom_data.items():
            if start <= pos < end:
                if field == "log1p":
                    out[pos - start] = data["log1p"]
                elif field == "has_support":
                    out[pos - start] = data["has_support"]

        return out

    def _get_junction_index(self) -> Optional[Dict]:
        """Load and cache junction index from parquet."""
        if self._junction_index is not None:
            return self._junction_index

        path = self.config.junction_parquet
        if path is None:
            # Try auto-resolve
            try:
                from agentic_spliceai.splice_engine.resources import get_model_resources
                resources = get_model_resources("openspliceai")
                registry = resources.get_registry()
                from pathlib import Path as P
                candidate = P(registry.stash) / "junction_data" / "gtex_junction_summary.parquet"
                if candidate.exists():
                    path = candidate
            except Exception:
                pass

        if path is None or not Path(path).exists():
            logger.info("No junction parquet available; junction channels will be zeros")
            self._junction_index = {}
            return self._junction_index

        import polars as pl
        logger.info("Loading junction index from %s", path)
        df = pl.read_parquet(path)

        # Build dict: chrom -> {position -> {log1p, has_support}}
        index: Dict[str, Dict[int, Dict[str, float]]] = {}
        for row in df.iter_rows(named=True):
            chrom = row.get("chrom", "")
            pos = int(row.get("position", row.get("donor_pos", 0)))
            total_reads = float(row.get("unique_reads", row.get("total_reads", 0)))

            if chrom not in index:
                index[chrom] = {}
            index[chrom][pos] = {
                "log1p": float(np.log1p(total_reads)),
                "has_support": 1.0 if total_reads > 0 else 0.0,
            }

        self._junction_index = index
        logger.info("Junction index: %d chromosomes, %d positions",
                     len(index), sum(len(v) for v in index.values()))
        return self._junction_index

    # ------------------------------------------------------------------
    # RBP eCLIP (sparse interval overlap)
    # ------------------------------------------------------------------

    def _query_rbp(
        self,
        chrom: str,
        start: int,
        end: int,
    ) -> np.ndarray:
        """Query RBP binding count for a window (sparse, 0-filled)."""
        peaks = self._get_eclip_peaks()
        if peaks is None:
            return np.zeros(end - start, dtype=np.float32)

        import polars as pl

        L = end - start
        out = np.zeros(L, dtype=np.float32)

        # Filter peaks to this chrom + window overlap
        chrom_peaks = peaks.filter(
            (pl.col("chrom") == chrom)
            & (pl.col("start") < end)
            & (pl.col("end") > start)
        )

        if chrom_peaks.height == 0:
            return out

        # For each peak, increment binding count at overlapping positions
        for row in chrom_peaks.iter_rows(named=True):
            p_start = max(int(row["start"]), start) - start
            p_end = min(int(row["end"]), end) - start
            # Count distinct RBPs (simplified: increment by 1 per peak)
            out[p_start:p_end] += 1.0

        return out

    def _get_eclip_peaks(self) -> Optional[object]:
        """Load and cache eCLIP peaks parquet."""
        if self._eclip_peaks is not None:
            return self._eclip_peaks

        path = self.config.eclip_parquet
        if path is None:
            # Try auto-resolve
            try:
                from agentic_spliceai.splice_engine.resources import get_model_resources
                resources = get_model_resources("openspliceai")
                registry = resources.get_registry()
                from pathlib import Path as P
                candidate = P(registry.stash) / "rbp_data" / "eclip_peaks.parquet"
                if candidate.exists():
                    path = candidate
            except Exception:
                pass

        if path is None or not Path(path).exists():
            logger.info("No eCLIP parquet available; RBP channel will be zeros")
            self._eclip_peaks = None
            return None

        import polars as pl
        logger.info("Loading eCLIP peaks from %s", path)
        self._eclip_peaks = pl.read_parquet(path)
        logger.info("eCLIP peaks: %d rows", self._eclip_peaks.height)
        return self._eclip_peaks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bigwig_chrom(bw: object, chrom: str) -> Optional[str]:
        """Resolve chromosome name to match bigWig convention.

        BigWig files use UCSC naming (chr prefix).  Input may or may
        not have the prefix.
        """
        chroms = bw.chroms()
        if chrom in chroms:
            return chrom
        # Try adding/removing chr prefix
        if chrom.startswith("chr"):
            bare = chrom[3:]
            if bare in chroms:
                return bare
        else:
            prefixed = f"chr{chrom}"
            if prefixed in chroms:
                return prefixed
        return None
