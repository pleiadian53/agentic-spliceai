"""Base score modality — derived features from raw splice site predictions.

Transforms raw per-nucleotide probabilities (donor_prob, acceptor_prob,
neither_prob) into ~40 engineered features across five groups:

1. **Context scores**: raw predicted probabilities at neighboring positions
2. **Derived probability features**: entropy, log-odds, ratios
3. **Donor gradient features**: diff, surge, peak detection
4. **Acceptor gradient features**: same structure as donor
5. **Cross-type comparative features**: donor vs acceptor ratios

All computations are vectorized Polars expressions. Context-aware features
use ``.over('gene_id')`` to prevent cross-gene leakage.
"""

import logging
from dataclasses import dataclass
from typing import List

import polars as pl

from ..modality import Modality, ModalityConfig, ModalityMeta

logger = logging.getLogger(__name__)


@dataclass
class BaseScoreConfig(ModalityConfig):
    """Configuration for the base score modality.

    Attributes
    ----------
    context_window : int
        Number of flanking positions on each side for neighbor extraction.
        Default 2 produces context_score_m2, m1, p1, p2 (4 columns).
        Set to 5 for context_score_m5..m1, p1..p5 (10 columns).
    include_gradients : bool
        Include donor/acceptor gradient features (diff, surge, peak).
    include_comparative : bool
        Include cross-type comparative features.
    epsilon : float
        Small constant to prevent division by zero and log(0).
    """

    context_window: int = 2
    include_gradients: bool = True
    include_comparative: bool = True
    epsilon: float = 1e-10


class BaseScoreModality(Modality):
    """Derive ~40 features from raw splice site prediction scores.

    Input columns: donor_prob, acceptor_prob, neither_prob, gene_id
    Output: context scores, probability features, gradient features,
            comparative features.
    """

    def __init__(self, config: BaseScoreConfig | None = None) -> None:
        super().__init__(config or self.default_config())
        self._cfg: BaseScoreConfig = self.config  # type: ignore[assignment]

    @property
    def meta(self) -> ModalityMeta:
        cols = self._build_output_column_list()
        return ModalityMeta(
            name="base_scores",
            version="1.0",
            output_columns=tuple(cols),
            required_inputs=frozenset(
                {"donor_prob", "acceptor_prob", "neither_prob", "gene_id"}
            ),
            description="Derived features from raw splice site prediction scores.",
        )

    @classmethod
    def default_config(cls) -> BaseScoreConfig:
        return BaseScoreConfig()

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add all base-score-derived features to the DataFrame."""
        eps = self._cfg.epsilon
        w = self._cfg.context_window

        # Step 0: Alias _prob → _score for FeatureSchema compatibility
        df = df.with_columns(
            pl.col("donor_prob").alias("donor_score"),
            pl.col("acceptor_prob").alias("acceptor_score"),
            pl.col("neither_prob").alias("neither_score"),
        )

        # Step 1: Composite score column (max of donor, acceptor at each position)
        df = df.with_columns(
            pl.max_horizontal("donor_prob", "acceptor_prob").alias("_splice_score")
        )

        # Step 2: Context scores via shift within gene groups
        df = self._add_context_scores(df, w)

        # Step 3: Derived probability features
        df = self._add_probability_features(df, eps)

        # Step 4: Context pattern features (derived from context scores)
        df = self._add_context_pattern_features(df, w, eps)

        # Step 5: Donor/acceptor gradient features
        if self._cfg.include_gradients:
            df = self._add_gradient_features(df, "donor_prob", "donor", w, eps)
            df = self._add_gradient_features(df, "acceptor_prob", "acceptor", w, eps)

        # Step 6: Cross-type comparative features
        if self._cfg.include_comparative:
            df = self._add_comparative_features(df, eps)

        # Drop internal helper column
        df = df.drop("_splice_score")

        return df

    # ------------------------------------------------------------------
    # Group 1: Context scores (raw neighbor probabilities)
    # ------------------------------------------------------------------

    def _add_context_scores(self, df: pl.DataFrame, w: int) -> pl.DataFrame:
        """Add context_score_m{i} and context_score_p{i} columns.

        These are the actual predicted splice-site probabilities at
        positions neighboring the query position. For a true splice site,
        scores peak sharply at that site; neighboring scores capture the
        shape of this peak profile.

        Uses shift() within gene groups to avoid cross-gene leakage.
        """
        exprs: list[pl.Expr] = []
        for i in range(1, w + 1):
            # Negative shift = look at earlier (upstream) positions
            exprs.append(
                pl.col("_splice_score")
                .shift(i)
                .over("gene_id")
                .alias(f"context_score_m{i}")
            )
            # Positive shift = look at later (downstream) positions
            exprs.append(
                pl.col("_splice_score")
                .shift(-i)
                .over("gene_id")
                .alias(f"context_score_p{i}")
            )
        return df.with_columns(exprs).fill_null(0.0)

    # ------------------------------------------------------------------
    # Group 2: Derived probability features
    # ------------------------------------------------------------------

    def _add_probability_features(
        self, df: pl.DataFrame, eps: float
    ) -> pl.DataFrame:
        """Add 7 derived probability features from raw scores."""
        donor = pl.col("donor_prob")
        acceptor = pl.col("acceptor_prob")
        neither = pl.col("neither_prob")
        max_da = pl.max_horizontal("donor_prob", "acceptor_prob")
        max_dan = pl.max_horizontal("donor_prob", "acceptor_prob", "neither_prob")

        return df.with_columns(
            # Relative donor probability
            (donor / (donor + acceptor + eps)).alias("relative_donor_probability"),
            # Splice probability (donor + acceptor vs total)
            ((donor + acceptor) / (donor + acceptor + neither + eps)).alias(
                "splice_probability"
            ),
            # Donor-acceptor difference (normalized)
            ((donor - acceptor) / (max_da + eps)).alias("donor_acceptor_diff"),
            # Splice vs neither difference (normalized)
            ((max_da - neither) / (max_dan + eps)).alias("splice_neither_diff"),
            # Log-odds: donor vs acceptor
            ((donor + eps).log() - (acceptor + eps).log()).alias(
                "donor_acceptor_logodds"
            ),
            # Log-odds: splice vs neither
            ((donor + acceptor + eps).log() - (neither + eps).log()).alias(
                "splice_neither_logodds"
            ),
            # Shannon entropy of the 3-class distribution
            (
                -(
                    donor * (donor + eps).log()
                    + acceptor * (acceptor + eps).log()
                    + neither * (neither + eps).log()
                )
            ).alias("probability_entropy"),
        )

    # ------------------------------------------------------------------
    # Group 2b: Context pattern features
    # ------------------------------------------------------------------

    def _add_context_pattern_features(
        self, df: pl.DataFrame, w: int, eps: float
    ) -> pl.DataFrame:
        """Add context_neighbor_mean, context_asymmetry, context_max."""
        ctx_cols = [f"context_score_m{i}" for i in range(1, w + 1)] + [
            f"context_score_p{i}" for i in range(1, w + 1)
        ]
        n = len(ctx_cols)

        upstream_cols = [f"context_score_m{i}" for i in range(1, w + 1)]
        downstream_cols = [f"context_score_p{i}" for i in range(1, w + 1)]

        return df.with_columns(
            # Mean of all context scores
            (pl.sum_horizontal(*ctx_cols) / n).alias("context_neighbor_mean"),
            # Asymmetry: upstream sum - downstream sum
            (
                pl.sum_horizontal(*upstream_cols)
                - pl.sum_horizontal(*downstream_cols)
            ).alias("context_asymmetry"),
            # Max across all context positions
            pl.max_horizontal(*ctx_cols).alias("context_max"),
        )

    # ------------------------------------------------------------------
    # Groups 3 & 4: Gradient features (donor or acceptor)
    # ------------------------------------------------------------------

    def _add_gradient_features(
        self,
        df: pl.DataFrame,
        score_col: str,
        prefix: str,
        w: int,
        eps: float,
    ) -> pl.DataFrame:
        """Add gradient features for a given score type (donor or acceptor).

        Parameters
        ----------
        score_col : str
            Source column ('donor_prob' or 'acceptor_prob').
        prefix : str
            Output column prefix ('donor' or 'acceptor').
        w : int
            Context window size.
        eps : float
            Division safety constant.
        """
        score = pl.col(score_col)
        exprs: list[pl.Expr] = []

        # Differential features: score - context_score at each offset
        for i in range(1, w + 1):
            exprs.append(
                (score - pl.col(f"context_score_m{i}")).alias(f"{prefix}_diff_m{i}")
            )
            exprs.append(
                (score - pl.col(f"context_score_p{i}")).alias(f"{prefix}_diff_p{i}")
            )

        # Surge ratio: score / (immediate neighbors sum)
        exprs.append(
            (score / (pl.col("context_score_m1") + pl.col("context_score_p1") + eps))
            .alias(f"{prefix}_surge_ratio")
        )

        # Local peak detection (binary)
        exprs.append(
            (
                (score > pl.col("context_score_m1"))
                & (score > pl.col("context_score_p1"))
                & (score > 1e-3)
            )
            .cast(pl.Int8)
            .alias(f"{prefix}_is_local_peak")
        )

        # Weighted context (Gaussian-like: center=0.4, ±1=0.2, ±2=0.1, ...)
        weights = self._gaussian_weights(w)
        weighted_expr = score * weights[w]  # center weight
        for i in range(1, w + 1):
            weighted_expr = weighted_expr + (
                pl.col(f"context_score_m{i}") * weights[w - i]
                + pl.col(f"context_score_p{i}") * weights[w + i]
            )
        exprs.append(weighted_expr.alias(f"{prefix}_weighted_context"))

        # Peak height ratio: score / mean(context)
        ctx_cols = [f"context_score_m{i}" for i in range(1, w + 1)] + [
            f"context_score_p{i}" for i in range(1, w + 1)
        ]
        ctx_mean = pl.sum_horizontal(*ctx_cols) / len(ctx_cols)
        exprs.append(
            (score / (ctx_mean + eps)).alias(f"{prefix}_peak_height_ratio")
        )

        # Second derivative: (score - m1) - (p1 - score) = 2*score - m1 - p1
        exprs.append(
            (
                (score - pl.col("context_score_m1"))
                - (pl.col("context_score_p1") - score)
            ).alias(f"{prefix}_second_derivative")
        )

        # Signal strength: score - mean(context)
        exprs.append(
            (score - ctx_mean).alias(f"{prefix}_signal_strength")
        )

        # Context diff ratio: score / max(context)
        ctx_max = pl.max_horizontal(*ctx_cols)
        exprs.append(
            (score / (ctx_max + eps)).alias(f"{prefix}_context_diff_ratio")
        )

        return df.with_columns(exprs)

    # ------------------------------------------------------------------
    # Group 5: Cross-type comparative features
    # ------------------------------------------------------------------

    def _add_comparative_features(
        self, df: pl.DataFrame, eps: float
    ) -> pl.DataFrame:
        """Add cross-type (donor vs acceptor) comparative features."""
        return df.with_columns(
            # Peak height ratio: donor / acceptor
            (
                pl.col("donor_peak_height_ratio")
                / (pl.col("acceptor_peak_height_ratio") + eps)
            ).alias("donor_acceptor_peak_ratio"),
            # Signal strength difference
            (
                pl.col("donor_signal_strength")
                - pl.col("acceptor_signal_strength")
            ).alias("type_signal_difference"),
            # Score difference ratio (normalized)
            (
                (pl.col("donor_prob") - pl.col("acceptor_prob"))
                / (pl.col("donor_prob") + pl.col("acceptor_prob") + eps)
            ).alias("score_difference_ratio"),
            # Signal strength ratio
            (
                pl.col("donor_signal_strength")
                / (pl.col("acceptor_signal_strength") + eps)
            ).alias("signal_strength_ratio"),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gaussian_weights(self, w: int) -> list[float]:
        """Generate symmetric Gaussian-like weights for 2*w+1 positions.

        For w=2: [0.1, 0.2, 0.4, 0.2, 0.1] (matches meta-spliceai).
        For other w: approximate Gaussian with center=0.4, normalized to sum=1.
        """
        if w == 2:
            return [0.1, 0.2, 0.4, 0.2, 0.1]

        import math

        sigma = w / 2.0
        raw = [math.exp(-0.5 * ((i - w) / sigma) ** 2) for i in range(2 * w + 1)]
        total = sum(raw)
        return [r / total for r in raw]

    def _build_output_column_list(self) -> List[str]:
        """Build the full list of output columns based on config."""
        w = self._cfg.context_window
        cols: List[str] = []

        # Score aliases (prob → score for FeatureSchema compatibility)
        cols.extend(["donor_score", "acceptor_score", "neither_score"])

        # Context scores
        for i in range(1, w + 1):
            cols.append(f"context_score_m{i}")
            cols.append(f"context_score_p{i}")

        # Probability features
        cols.extend([
            "relative_donor_probability",
            "splice_probability",
            "donor_acceptor_diff",
            "splice_neither_diff",
            "donor_acceptor_logodds",
            "splice_neither_logodds",
            "probability_entropy",
        ])

        # Context pattern features
        cols.extend(["context_neighbor_mean", "context_asymmetry", "context_max"])

        # Gradient features
        if self._cfg.include_gradients:
            for prefix in ("donor", "acceptor"):
                for i in range(1, w + 1):
                    cols.append(f"{prefix}_diff_m{i}")
                    cols.append(f"{prefix}_diff_p{i}")
                cols.extend([
                    f"{prefix}_surge_ratio",
                    f"{prefix}_is_local_peak",
                    f"{prefix}_weighted_context",
                    f"{prefix}_peak_height_ratio",
                    f"{prefix}_second_derivative",
                    f"{prefix}_signal_strength",
                    f"{prefix}_context_diff_ratio",
                ])

        # Comparative features
        if self._cfg.include_comparative:
            cols.extend([
                "donor_acceptor_peak_ratio",
                "type_signal_difference",
                "score_difference_ratio",
                "signal_strength_ratio",
            ])

        return cols
