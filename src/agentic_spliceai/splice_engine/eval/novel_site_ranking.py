"""Primitives for ranking candidate **novel** splice sites within a gene.

Novel-site discovery (M3) asks a different question from M1/M2 refinement:

    "For this gene, which unannotated positions are the best candidates to inspect?"

That is a *within-gene ranking* problem, scored by per-gene precision@k / recall@k
against independent truth — not a genome-wide PR-AUC. This module holds the pieces
that answer it, shared by the offline evaluation harness
(``examples/meta_layer/13_evaluate_m3_novel.py``) and the Bio Lab UI's Novel Site
Explorer so both rank sites through exactly one implementation.

Score arrays are ``[L, 3]`` in meta channel order ``[donor, acceptor, neither]``,
where row ``i`` is genomic position ``gene.start + i``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

# Score-array channel order emitted by infer_full_gene / base_scores: [donor, acceptor, neither]
DONOR_IDX, ACCEPTOR_IDX = 0, 1
IDX_TO_TYPE = {DONOR_IDX: "donor", ACCEPTOR_IDX: "acceptor"}
JOIN_KEY = ["chrom", "position", "strand", "splice_type"]


# ---------------------------------------------------------------------------
# Coordinate / chrom helpers
# ---------------------------------------------------------------------------

def strip_chr(chrom: str) -> str:
    """``"chr1" -> "1"``; leave bare names untouched."""
    return chrom[3:] if chrom.startswith("chr") else chrom


def _strip_chr_expr(col: str = "chrom") -> pl.Expr:
    return pl.col(col).cast(pl.String).str.replace_all(r"^chr", "").alias("chrom")


# ---------------------------------------------------------------------------
# Truth / annotation loading
# ---------------------------------------------------------------------------

def load_sites_parquet(
    path: Path,
    *,
    keep_chroms_bare: Optional[Sequence[str]] = None,
    is_novel_only: bool = False,
    min_biosamples: int = 1,
    extra_cols: Sequence[str] = (),
) -> pl.DataFrame:
    """Load a sites parquet, normalise chrom to bare, optionally filter.

    Works for ``annotation_mask.parquet``, ``longread_truth_novel.parquet``
    (``min_biosamples`` / ``n_biosamples``), and ``disease_anchors.parquet``
    (``is_novel_only``). All share the ``chrom, position, strand, splice_type``
    key with a bare ``chrom``.
    """
    df = pl.read_parquet(path)
    df = df.with_columns(_strip_chr_expr("chrom"))
    if is_novel_only and "is_novel" in df.columns:
        df = df.filter(pl.col("is_novel"))
    if min_biosamples > 1 and "n_biosamples" in df.columns:
        df = df.filter(pl.col("n_biosamples") >= min_biosamples)
    if keep_chroms_bare is not None:
        df = df.filter(pl.col("chrom").is_in(list(keep_chroms_bare)))
    cols = [*JOIN_KEY, *[c for c in extra_cols if c in df.columns]]
    return df.select(cols)


# ---------------------------------------------------------------------------
# Gene universe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeneInterval:
    gene_id: str
    chrom: str  # bare
    start: int  # 0-based, matches build_gene_cache / pyfaidx slicing
    end: int
    strand: str


# ---------------------------------------------------------------------------
# Site lookup
# ---------------------------------------------------------------------------

class SiteIndex:
    """Fast lookup of site positions inside a gene window.

    Pre-groups a sites DataFrame into ``(chrom, strand, splice_type) -> sorted
    positions`` so per-gene lookups are ``searchsorted`` slices, not repeated
    polars filters over an 800K-row frame. Built once per truth/annotation set.

    Membership only — every non-key column is dropped. Use :class:`SiteEvidenceIndex`
    when you need the payload (e.g. ``n_biosamples``, ``mechanism``) as well.
    """

    def __init__(self, by_key: Dict[Tuple[str, str, str], np.ndarray]):
        self._by_key = by_key

    @classmethod
    def from_df(cls, df: pl.DataFrame) -> "SiteIndex":
        by_key: Dict[Tuple[str, str, str], np.ndarray] = {}
        if df.height:
            for key, sub in df.group_by(["chrom", "strand", "splice_type"]):
                k = tuple(str(x) for x in key)  # (chrom, strand, splice_type)
                by_key[k] = np.sort(sub.get_column("position").to_numpy().astype(np.int64))
        return cls(by_key)

    def positions_in(
        self, chrom: str, strand: str, splice_type: str, start: int, end: int,
    ) -> np.ndarray:
        """Positions of matching sites in ``[start, end)``."""
        arr = self._by_key.get((chrom, strand, splice_type))
        if arr is None or arr.size == 0:
            return np.empty(0, dtype=np.int64)
        lo = np.searchsorted(arr, start, side="left")
        hi = np.searchsorted(arr, end, side="left")
        return arr[lo:hi]


class SiteEvidenceIndex:
    """Position-keyed lookup that *keeps* the payload columns.

    :class:`SiteIndex` answers "is this position a known site?"; this answers
    "…and what do we know about it?" — the extra columns needed to explain a
    candidate to a user (``n_biosamples`` for long-read support, ``mechanism`` /
    ``source`` / ``support`` for a disease anchor).

    Build it from :func:`load_sites_parquet` with ``extra_cols`` populated.
    """

    def __init__(self, by_site: Dict[Tuple[str, str, str, int], Dict[str, Any]]):
        self._by_site = by_site

    @classmethod
    def from_df(cls, df: pl.DataFrame, payload_cols: Sequence[str]) -> "SiteEvidenceIndex":
        keep = [c for c in payload_cols if c in df.columns]
        by_site: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
        for row in df.iter_rows(named=True):
            key = (
                str(row["chrom"]),
                str(row["strand"]),
                str(row["splice_type"]),
                int(row["position"]),
            )
            by_site[key] = {c: row[c] for c in keep}
        return cls(by_site)

    def lookup(
        self, chrom: str, strand: str, splice_type: str, position: int,
    ) -> Optional[Dict[str, Any]]:
        """Payload for this exact site, or ``None`` if it is not in the set."""
        return self._by_site.get((chrom, strand, splice_type, int(position)))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def top_novel_candidates(
    probs_type: np.ndarray,
    gene_start: int,
    annotated_positions: np.ndarray,
    max_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Top-``max_k`` novel candidate (genomic_position, prob), highest prob first.

    Annotated positions are masked out (the post-filter) *before* ranking.
    ``argpartition`` keeps this O(L) rather than a full O(L log L) sort.
    """
    p = probs_type.astype(np.float64).copy()
    if annotated_positions.size:
        rel = annotated_positions - gene_start
        rel = rel[(rel >= 0) & (rel < p.size)]
        p[rel] = -np.inf
    k = int(min(max_k, p.size))
    if k <= 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    part = np.argpartition(-p, k - 1)[:k]
    order = part[np.argsort(-p[part])]
    order = order[np.isfinite(p[order])]  # drop masked if fewer than k survive
    return gene_start + order, p[order]


def suppress_adjacent(
    positions: np.ndarray,
    scores: np.ndarray,
    min_separation: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Non-maximum suppression over one candidate group: keep the peak, drop its shoulders.

    A single splice-site signal spans a few adjacent positions, so an un-suppressed
    top-k can spend three slots on one peak. This keeps the highest-scoring position
    and discards anything within ``min_separation`` bp of an already-kept one.

    .. important::
       Apply this **within a single ``(chrom, strand, splice_type)`` group** — i.e.
       call it once per splice type, then merge. Donors and acceptors are distinct
       biological events, so a donor must never suppress a nearby acceptor (a real
       acceptor sitting a few bp from a stronger donor would be silently dropped).
       Within one gene, chrom and strand are constant, so grouping by ``splice_type``
       is sufficient.

    Parameters
    ----------
    positions, scores
        Candidates for one group, **already sorted by descending score**
        (as returned by :func:`top_novel_candidates`).
    min_separation
        Minimum bp between kept candidates. ``<= 1`` disables suppression.

    Returns
    -------
    tuple of np.ndarray
        Filtered ``(positions, scores)``, still in descending-score order.
    """
    if positions.size == 0 or min_separation <= 1:
        return positions, scores

    kept_idx: List[int] = []
    kept_pos: List[int] = []
    for i, pos in enumerate(positions):
        if all(abs(int(pos) - kp) >= min_separation for kp in kept_pos):
            kept_idx.append(i)
            kept_pos.append(int(pos))
    sel = np.asarray(kept_idx, dtype=np.int64)
    return positions[sel], scores[sel]


# ---------------------------------------------------------------------------
# Feature-channel selection
# ---------------------------------------------------------------------------

def slice_mm_channels(mm: np.ndarray, cfg, exclude_channels: Optional[Sequence[str]] = None) -> np.ndarray:
    """Select the multimodal channels a meta model expects from a full cache array.

    Per-gene feature caches are written with **all** channels so one cache can serve
    every model; a model that was trained on a subset (M3 drops the junction channels,
    because junction support is its training *target*) must be fed the matching slice.
    ``infer_full_gene`` does no channel selection, so an unsliced array fails inside
    the model's first convolution.

    Parameters
    ----------
    mm : np.ndarray
        ``[L, C_all]`` features from the cache.
    cfg
        The model config; ``cfg.mm_channels`` is the expected channel count.
    exclude_channels : sequence of str, optional
        Channel names to drop. Defaults to ``M3_EXCLUDE`` when the model expects
        exactly the M3 channel count, otherwise nothing is dropped.

    Raises
    ------
    ValueError
        If the resulting channel count does not match ``cfg.mm_channels``. This is a
        hard error rather than a warning: a silent mismatch surfaces later as an
        opaque shape error inside the model.
    """
    # Imported lazily so this module stays free of the feature stack's heavy deps.
    from ..features.dense_feature_extractor import CHANNEL_NAMES, M3_EXCLUDE

    expected = int(getattr(cfg, "mm_channels", mm.shape[1]))
    if mm.shape[1] == expected and exclude_channels is None:
        return mm  # already the right shape (e.g. a 9-channel model on a 9-channel cache)

    drop = set(exclude_channels) if exclude_channels is not None else set(M3_EXCLUDE)
    keep = [i for i, name in enumerate(CHANNEL_NAMES) if name not in drop]
    sliced = mm[:, keep]

    if sliced.shape[1] != expected:
        raise ValueError(
            f"Channel selection produced {sliced.shape[1]} channels but the model "
            f"expects {expected}. Cache had {mm.shape[1]} "
            f"(CHANNEL_NAMES={len(CHANNEL_NAMES)}), dropped={sorted(drop)}. "
            "Check the model's exclude_channels in settings.yaml."
        )
    return sliced
