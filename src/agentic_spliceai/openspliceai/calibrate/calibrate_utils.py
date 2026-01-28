"""
Filename: calibrate_utils.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Utility functions for calibrating models.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

def compute_calibration_curve(labels, probs, n_bins=10, strategy='quantile'):
    prob_true, prob_pred = calibration_curve(labels, probs, n_bins=n_bins, strategy=strategy)
    if strategy == 'quantile':
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(probs, quantiles)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bin_edges) - 1
    bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
    return prob_true, prob_pred, bin_counts


def compute_confidence_intervals(prob_true, bin_counts, z=1.96):
    ci_lower, ci_upper = [], []
    for p, n in zip(prob_true, bin_counts):
        std_error = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        delta = z * std_error
        ci_lower.append(max(p - delta, 0))
        ci_upper.append(min(p + delta, 1))
    return np.array(ci_lower), np.array(ci_upper)


def reverse_softmax(probs):
    return np.log(np.clip(probs, 1e-8, 1.0))


def save_calibration_data(output_dir, class_name, flanking_size, prob_true, prob_pred, bin_counts, suffix):
    np.savez(
        f"{output_dir}/calibration_data_{class_name}_{suffix}_{flanking_size}nt.npz",
        prob_true=prob_true,
        prob_pred=prob_pred,
        bin_counts=bin_counts
    )


def calculate_brier_scores(labels, probs, probs_scaled):
    return (
        [brier_score_loss((labels == i).astype(int), probs[:, i]) for i in range(3)],
        [brier_score_loss((labels == i).astype(int), probs_scaled[:, i]) for i in range(3)]
    )