"""Position sampling for analysis sequences — splice-aware subsampling.

Reduces storage from ~271 GB (full genome) to ~1-5 GB by keeping only
splice-relevant positions and a configurable background sample.

The full genome has ~1.15 billion nucleotide positions, but only ~0.06%
have a splice probability > 0.01. Positions far from predicted splice
sites carry minimal signal for meta-layer training and can be aggressively
subsampled without information loss.

Sampling tiers (applied per chromosome):

1. **Splice sites**: All positions where donor_prob or acceptor_prob
   exceeds ``score_threshold`` — always retained.
2. **Proximity zone**: Positions within ``proximity_window`` bp of a
   retained splice site — always retained. These capture the local
   context that meta-layer models need.
3. **Background**: Remaining positions subsampled at ``background_rate``.
   A small representative sample preserves the class prior for training.

Inspired by the TN sampling strategy in meta-spliceai's
``enhanced_evaluation.py`` (tn_sample_factor, proximity, window modes).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PositionSamplingConfig:
    """Configuration for position-level sampling.

    Attributes
    ----------
    enabled : bool
        Whether to apply position sampling. If False, all positions are kept.
    early : bool
        If True, sampling is applied BEFORE feature engineering (faster —
        modalities only process sampled positions). If False, sampling
        happens after all modality features are computed. Default True.
    score_threshold : float
        Minimum donor_prob or acceptor_prob to classify as a splice site.
        Positions above this are always retained. Default 0.01.
    proximity_window : int
        Number of base pairs around each splice site to retain.
        Captures local context for meta-layer features. Default 0 (disabled).
        Set to 50-200 if downstream models need positional context around
        splice sites (e.g., CNN-based meta-layer). Costs ~15-40 GB genome.
    background_rate : float
        Fraction of remaining (non-splice, non-proximity) positions to keep.
        0.005 = 0.5% background sample. Default 0.005.
        With proximity_window=0, this is the main knob for storage vs
        representativeness: 0.005 -> ~1.5 GB, 0.001 -> ~0.4 GB genome.
    random_seed : int
        Seed for reproducible background sampling.
    donor_col : str
        Column name for donor probability scores.
    acceptor_col : str
        Column name for acceptor probability scores.
    """

    enabled: bool = True
    early: bool = True
    score_threshold: float = 0.01
    proximity_window: int = 0
    background_rate: float = 0.005
    random_seed: int = 42
    donor_col: str = "donor_prob"
    acceptor_col: str = "acceptor_prob"


def sample_positions(
    df: pl.DataFrame,
    config: Optional[PositionSamplingConfig] = None,
) -> pl.DataFrame:
    """Apply splice-aware position sampling to a chromosome DataFrame.

    Parameters
    ----------
    df : pl.DataFrame
        Feature-enriched DataFrame for a single chromosome. Must contain
        columns for donor/acceptor probabilities and ``position``.
    config : PositionSamplingConfig, optional
        Sampling configuration. If None or disabled, returns df unchanged.

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame containing splice sites, proximity zones,
        and a background sample. Sorted by position within each gene.
    """
    if config is None or not config.enabled:
        return df

    if config.donor_col not in df.columns or config.acceptor_col not in df.columns:
        logger.warning(
            "Sampling skipped: missing columns %s/%s",
            config.donor_col,
            config.acceptor_col,
        )
        return df

    n_total = df.height
    if n_total == 0:
        return df

    # --- Tier 1: Splice sites (score above threshold) ---
    splice_mask = (
        (pl.col(config.donor_col) > config.score_threshold)
        | (pl.col(config.acceptor_col) > config.score_threshold)
    )

    # --- Tier 2: Proximity zone ---
    # Get positions of splice sites per gene, then expand by window
    if config.proximity_window > 0 and "gene_id" in df.columns:
        proximity_mask = _build_proximity_mask(
            df, splice_mask, config.proximity_window
        )
    elif config.proximity_window > 0:
        # No gene_id — use global position proximity
        proximity_mask = _build_proximity_mask_global(
            df, splice_mask, config.proximity_window
        )
    else:
        proximity_mask = pl.lit(False)

    # --- Tier 3: Background sample ---
    # Positions that are neither splice nor proximity
    keep_mask = splice_mask | proximity_mask
    n_kept_so_far = df.select(keep_mask.sum()).item()
    n_remaining = n_total - n_kept_so_far

    if n_remaining > 0 and config.background_rate > 0:
        rng = np.random.default_rng(config.random_seed)
        # Create a random mask for background positions
        random_vals = pl.Series("_rand", rng.random(n_total))
        background_mask = (
            ~keep_mask
            & (random_vals < config.background_rate)
        )
        keep_mask = keep_mask | background_mask

    result = df.filter(keep_mask)

    # Sort within genes for consistent output
    sort_cols = []
    if "gene_id" in result.columns:
        sort_cols.append("gene_id")
    if "position" in result.columns:
        sort_cols.append("position")
    if sort_cols:
        result = result.sort(sort_cols)

    # Report sampling stats
    n_splice = df.select(splice_mask.sum()).item()
    n_result = result.height
    logger.info(
        "Position sampling: %d → %d (%.2f%%) | splice=%d, proximity+bg=%d",
        n_total,
        n_result,
        100 * n_result / n_total if n_total > 0 else 0,
        n_splice,
        n_result - n_splice,
    )

    return result


def _build_proximity_mask(
    df: pl.DataFrame,
    splice_mask: pl.Expr,
    window: int,
) -> pl.Expr:
    """Build a proximity mask using per-gene position windows.

    For each gene, finds splice site positions, expands them by ±window,
    and marks all positions within those windows.
    """
    # Materialize splice positions per gene
    splice_df = df.filter(splice_mask).select("gene_id", "position")

    if splice_df.height == 0:
        return pl.lit(False)

    # Create expanded windows: (gene_id, window_start, window_end)
    windows = splice_df.with_columns(
        (pl.col("position") - window).alias("_win_start"),
        (pl.col("position") + window).alias("_win_end"),
    ).select("gene_id", "_win_start", "_win_end")

    # Merge overlapping windows per gene to reduce join size
    windows = _merge_overlapping_windows(windows)

    # Mark positions within any window for their gene via inequality join
    # Polars doesn't have native inequality joins, so use a grouped approach
    proximity_positions = _expand_windows_to_positions(df, windows)

    # Create a boolean mask
    key_col = (
        df["gene_id"].cast(pl.Utf8) + ":" + df["position"].cast(pl.Utf8)
    )
    prox_keys = (
        proximity_positions["gene_id"].cast(pl.Utf8)
        + ":"
        + proximity_positions["position"].cast(pl.Utf8)
    )

    return key_col.is_in(prox_keys)


def _build_proximity_mask_global(
    df: pl.DataFrame,
    splice_mask: pl.Expr,
    window: int,
) -> pl.Expr:
    """Build proximity mask without gene grouping (fallback)."""
    splice_positions = df.filter(splice_mask)["position"].to_numpy()

    if len(splice_positions) == 0:
        return pl.lit(False)

    positions = df["position"].to_numpy()

    # Vectorized: for each position, check if any splice site is within window
    # Use sorted splice positions + searchsorted for O(n log m) instead of O(n*m)
    splice_sorted = np.sort(splice_positions)
    idx = np.searchsorted(splice_sorted, positions)

    near = np.zeros(len(positions), dtype=bool)
    # Check left neighbor
    left = np.clip(idx - 1, 0, len(splice_sorted) - 1)
    near |= np.abs(positions - splice_sorted[left]) <= window
    # Check right neighbor
    right = np.clip(idx, 0, len(splice_sorted) - 1)
    near |= np.abs(positions - splice_sorted[right]) <= window

    return pl.Series("_prox", near)


def _merge_overlapping_windows(windows: pl.DataFrame) -> pl.DataFrame:
    """Merge overlapping windows per gene to reduce expansion size."""
    merged_rows = []
    for gene_id, group in windows.group_by("gene_id"):
        starts = group["_win_start"].sort().to_list()
        ends = group["_win_end"].sort().to_list()

        merged_starts = [starts[0]]
        merged_ends = [ends[0]]

        for s, e in zip(starts[1:], ends[1:]):
            if s <= merged_ends[-1]:
                # Overlapping — extend current window
                merged_ends[-1] = max(merged_ends[-1], e)
            else:
                merged_starts.append(s)
                merged_ends.append(e)

        for s, e in zip(merged_starts, merged_ends):
            merged_rows.append({"gene_id": gene_id[0], "_win_start": s, "_win_end": e})

    if not merged_rows:
        return windows.head(0)

    return pl.DataFrame(merged_rows, schema=windows.schema)


def _expand_windows_to_positions(
    df: pl.DataFrame,
    windows: pl.DataFrame,
) -> pl.DataFrame:
    """Find positions in df that fall within any window for their gene.

    Uses a per-gene filter approach that's efficient when window count
    is small relative to total positions (typical: ~10K windows vs 17M positions).
    """
    if windows.height == 0:
        return pl.DataFrame(schema={"gene_id": pl.Utf8, "position": pl.Int64})

    result_frames = []
    for gene_id, gene_windows in windows.group_by("gene_id"):
        gene_id_val = gene_id[0]
        gene_positions = df.filter(pl.col("gene_id") == gene_id_val)

        if gene_positions.height == 0:
            continue

        # Build a combined filter for all windows of this gene
        win_filter = pl.lit(False)
        for row in gene_windows.iter_rows(named=True):
            win_filter = win_filter | (
                (pl.col("position") >= row["_win_start"])
                & (pl.col("position") <= row["_win_end"])
            )

        matched = gene_positions.filter(win_filter).select("gene_id", "position")
        if matched.height > 0:
            result_frames.append(matched)

    if not result_frames:
        return pl.DataFrame(schema={"gene_id": pl.Utf8, "position": pl.Int64})

    return pl.concat(result_frames)
