"""Probability calibration metrics and reliability curves.

Provides tools to assess whether predicted probabilities match
observed frequencies. Essential for validating meta-layer outputs
and comparing base model vs recalibrated scores.

A model is **well-calibrated** if, among all positions where it
predicts P(donor) = 0.8, approximately 80% are truly donor sites.

Metrics
-------
- **ECE** (Expected Calibration Error): Weighted average of
  |predicted - observed| across probability bins.
- **MCE** (Maximum Calibration Error): Worst-case bin error.
- **Brier Score Decomposition**: Separates overall error into
  calibration, refinement, and uncertainty components.
- **Reliability Curve**: Per-bin (predicted, observed) pairs for
  plotting reliability diagrams.

Example
-------
>>> import numpy as np
>>> from agentic_spliceai.splice_engine.eval.calibration import (
...     compute_ece, compute_brier_decomposition, reliability_curve,
... )
>>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
>>> y_prob = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.6, 0.4])
>>> compute_ece(y_true, y_prob, n_bins=5)
0.05
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Core metrics ─────────────────────────────────────────────────────


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Partitions predictions into equal-width probability bins and
    computes the weighted average absolute difference between
    predicted confidence and observed accuracy.

    .. math::
        ECE = \\sum_{b=1}^{B} \\frac{n_b}{N} |\\text{acc}(b) - \\text{conf}(b)|

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 or 1), shape ``[N]``.
    y_prob : np.ndarray
        Predicted probabilities in ``[0, 1]``, shape ``[N]``.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        ECE in ``[0, 1]``. Lower is better. Perfect calibration = 0.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_prob)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        ece += (n_bin / n_total) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_mce(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Maximum Calibration Error (MCE).

    The worst-case absolute calibration error across all bins. High
    MCE indicates the model is severely miscalibrated in at least
    one probability range.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 or 1), shape ``[N]``.
    y_prob : np.ndarray
        Predicted probabilities in ``[0, 1]``, shape ``[N]``.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float
        MCE in ``[0, 1]``. Lower is better.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    mce = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_confidence = y_prob[mask].mean()
        bin_accuracy = y_true[mask].mean()
        mce = max(mce, abs(bin_accuracy - bin_confidence))

    return float(mce)


@dataclass
class BrierDecomposition:
    """Brier score with calibration-refinement decomposition.

    The Brier score decomposes as::

        Brier = Calibration - Refinement + Uncertainty

    Attributes
    ----------
    brier : float
        Overall Brier score (mean squared error of probabilities).
    calibration : float
        Calibration component — how well predicted probabilities
        match observed frequencies within each bin. Lower = better.
    refinement : float
        Refinement component — how spread out the per-bin observed
        frequencies are. Higher = better (model uses more of the
        probability range). A model predicting 0.5 for everything
        has zero refinement.
    uncertainty : float
        Base rate entropy, ``p * (1 - p)`` where ``p`` is the overall
        positive rate. Constant for a given dataset.
    """

    brier: float
    calibration: float
    refinement: float
    uncertainty: float


def compute_brier_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Compute Brier score with calibration-refinement decomposition.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 or 1), shape ``[N]``.
    y_prob : np.ndarray
        Predicted probabilities in ``[0, 1]``, shape ``[N]``.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    BrierDecomposition
        Named decomposition with brier, calibration, refinement,
        uncertainty fields.
    """
    brier = float(np.mean((y_prob - y_true) ** 2))
    p_bar = y_true.mean()
    uncertainty = float(p_bar * (1 - p_bar))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration = 0.0
    refinement = 0.0
    n_total = len(y_prob)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        calibration += (n_bin / n_total) * (bin_acc - bin_conf) ** 2
        refinement += (n_bin / n_total) * bin_acc * (1 - bin_acc)

    return BrierDecomposition(
        brier=brier,
        calibration=float(calibration),
        refinement=float(refinement),
        uncertainty=uncertainty,
    )


# ── Reliability curve ────────────────────────────────────────────────


@dataclass
class ReliabilityCurve:
    """Reliability curve data for plotting.

    Each entry represents one probability bin with its mean predicted
    probability, observed positive frequency, and sample count.

    Attributes
    ----------
    bin_centers : list of float
        Center of each bin.
    predicted : list of float
        Mean predicted probability per bin.
    observed : list of float
        Observed positive frequency per bin.
    counts : list of int
        Number of samples in each bin.
    """

    bin_centers: list[float] = field(default_factory=list)
    predicted: list[float] = field(default_factory=list)
    observed: list[float] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)


def reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> ReliabilityCurve:
    """Compute reliability curve data.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 or 1), shape ``[N]``.
    y_prob : np.ndarray
        Predicted probabilities in ``[0, 1]``, shape ``[N]``.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    ReliabilityCurve
        Per-bin statistics for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rc = ReliabilityCurve()

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi if hi == 1.0 else y_prob < hi)
        n_bin = mask.sum()
        if n_bin == 0:
            continue
        rc.bin_centers.append(float((lo + hi) / 2))
        rc.predicted.append(float(y_prob[mask].mean()))
        rc.observed.append(float(y_true[mask].mean()))
        rc.counts.append(int(n_bin))

    return rc


# ── Comparison helper ────────────────────────────────────────────────


@dataclass
class CalibrationComparison:
    """Side-by-side calibration results for two models.

    Attributes
    ----------
    class_name : str
        Which class this comparison is for (e.g., 'donor').
    base_ece : float
        ECE of the base model.
    meta_ece : float
        ECE of the meta-layer model.
    base_mce : float
        MCE of the base model.
    meta_mce : float
        MCE of the meta-layer model.
    base_brier : BrierDecomposition
        Brier decomposition for the base model.
    meta_brier : BrierDecomposition
        Brier decomposition for the meta-layer model.
    base_reliability : ReliabilityCurve
        Reliability curve for the base model.
    meta_reliability : ReliabilityCurve
        Reliability curve for the meta-layer model.
    """

    class_name: str
    base_ece: float
    meta_ece: float
    base_mce: float
    meta_mce: float
    base_brier: BrierDecomposition
    meta_brier: BrierDecomposition
    base_reliability: ReliabilityCurve
    meta_reliability: ReliabilityCurve

    @property
    def ece_delta(self) -> float:
        """Meta ECE - Base ECE. Negative = meta is better."""
        return self.meta_ece - self.base_ece

    @property
    def brier_delta(self) -> float:
        """Meta Brier - Base Brier. Negative = meta is better."""
        return self.meta_brier.brier - self.base_brier.brier


def compare_calibration(
    y_true: np.ndarray,
    base_prob: np.ndarray,
    meta_prob: np.ndarray,
    class_name: str = "",
    n_bins: int = 10,
) -> CalibrationComparison:
    """Compare calibration between two models.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth (0 or 1).
    base_prob : np.ndarray
        Predicted probabilities from the base model.
    meta_prob : np.ndarray
        Predicted probabilities from the meta-layer model.
    class_name : str
        Label for this comparison (e.g., 'donor').
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    CalibrationComparison
        Side-by-side metrics and reliability curves.
    """
    return CalibrationComparison(
        class_name=class_name,
        base_ece=compute_ece(y_true, base_prob, n_bins),
        meta_ece=compute_ece(y_true, meta_prob, n_bins),
        base_mce=compute_mce(y_true, base_prob, n_bins),
        meta_mce=compute_mce(y_true, meta_prob, n_bins),
        base_brier=compute_brier_decomposition(y_true, base_prob, n_bins),
        meta_brier=compute_brier_decomposition(y_true, meta_prob, n_bins),
        base_reliability=reliability_curve(y_true, base_prob, n_bins),
        meta_reliability=reliability_curve(y_true, meta_prob, n_bins),
    )


def print_calibration_comparison(
    comparisons: list[CalibrationComparison],
) -> None:
    """Print a formatted calibration comparison table.

    Parameters
    ----------
    comparisons : list of CalibrationComparison
        One per class (e.g., donor, acceptor, neither).
    """
    print(f"\n  {'Metric':<14s} {'Class':<12s} {'Base':>10s} {'Meta':>10s} "
          f"{'Delta':>10s}  {'Winner'}")
    print(f"  {'-' * 14} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 10}  {'-' * 6}")

    for c in comparisons:
        _winner = lambda d: "Meta" if d < -1e-8 else "Base" if d > 1e-8 else "Tie"

        d_ece = c.ece_delta
        d_mce = c.meta_mce - c.base_mce
        d_brier = c.brier_delta
        d_cal = c.meta_brier.calibration - c.base_brier.calibration

        print(f"  {'ECE':<14s} {c.class_name:<12s} "
              f"{c.base_ece:10.6f} {c.meta_ece:10.6f} "
              f"{d_ece:+10.6f}  {_winner(d_ece)}")
        print(f"  {'MCE':<14s} {c.class_name:<12s} "
              f"{c.base_mce:10.6f} {c.meta_mce:10.6f} "
              f"{d_mce:+10.6f}  {_winner(d_mce)}")
        print(f"  {'Brier':<14s} {c.class_name:<12s} "
              f"{c.base_brier.brier:10.6f} {c.meta_brier.brier:10.6f} "
              f"{d_brier:+10.6f}  {_winner(d_brier)}")
        print(f"  {'  calibration':<14s} {c.class_name:<12s} "
              f"{c.base_brier.calibration:10.6f} {c.meta_brier.calibration:10.6f} "
              f"{d_cal:+10.6f}  {_winner(d_cal)}")
        print(f"  {'  refinement':<14s} {c.class_name:<12s} "
              f"{c.base_brier.refinement:10.6f} {c.meta_brier.refinement:10.6f}")
        print(f"  {'  uncertainty':<14s} {c.class_name:<12s} "
              f"{c.base_brier.uncertainty:10.6f}")
        print()
