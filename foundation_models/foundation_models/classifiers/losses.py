"""
Loss functions for splice site classification.

Provides:
- ``FocalLoss`` — focal loss for multi-class classification with extreme
  class imbalance (splice sites are ~0.01% of nucleotides).
- ``compute_class_weights`` — inverse-frequency class weights from labels.
- ``compute_class_weights_from_counts`` — same, from pre-computed counts.
- ``ECELoss`` — Expected Calibration Error for post-hoc calibration evaluation.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples (high p_t) and focuses learning on hard
    examples.  With gamma=0 this reduces to standard weighted cross-entropy.

    Reference: Lin et al. (2017), "Focal Loss for Dense Object Detection".

    Args:
        gamma: Focusing parameter (default: 2.0).  Higher values increase
            focus on hard examples.
        alpha: Per-class weights as a tensor of shape ``[num_classes]``.
            If None, all classes weighted equally.
        reduction: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Raw predictions ``[batch, num_classes, seq_len]`` or
                ``[batch, num_classes]``.
            targets: Integer class labels ``[batch, seq_len]`` or ``[batch]``.

        Returns:
            Scalar loss (if reduction != "none").
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)  # [batch, C, ...]

        # Gather the probability of the true class
        # targets shape: [batch, ...] → [batch, 1, ...]
        targets_unsqueeze = targets.unsqueeze(1).long()  # [batch, 1, ...]
        p_t = probs.gather(1, targets_unsqueeze).squeeze(1)  # [batch, ...]

        # Compute focal weight
        focal_weight = (1.0 - p_t) ** self.gamma

        # Cross-entropy for the true class
        ce = -torch.log(p_t + 1e-8)

        # Apply per-class alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets.long()]  # [batch, ...]
            loss = alpha_t * focal_weight * ce
        else:
            loss = focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int = 3,
    max_weight: float = 1000.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from label array.

    Args:
        labels: Integer label array (any shape, flattened internally).
        num_classes: Number of classes.
        max_weight: Cap on maximum weight to prevent instability.

    Returns:
        Tensor of shape ``[num_classes]`` with weights.
    """
    flat = labels.flatten()
    counts = np.bincount(flat.astype(np.int64), minlength=num_classes)
    # Avoid division by zero for absent classes
    counts = np.maximum(counts, 1)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = np.minimum(weights, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def compute_class_weights_from_counts(
    counts: np.ndarray,
    num_classes: int = 3,
    max_weight: float = 1000.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from pre-computed class counts.

    Same formula as :func:`compute_class_weights` but accepts counts directly,
    avoiding the need to load all labels into memory.

    Args:
        counts: Array of shape ``[num_classes]`` with per-class sample counts.
        num_classes: Number of classes.
        max_weight: Cap on maximum weight to prevent instability.

    Returns:
        Tensor of shape ``[num_classes]`` with weights.
    """
    counts = np.maximum(np.asarray(counts, dtype=np.float64), 1)
    total = counts.sum()
    weights = total / (num_classes * counts)
    weights = np.minimum(weights, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


class ECELoss(nn.Module):
    """Expected Calibration Error.

    Bins predictions by confidence and measures the gap between average
    confidence and accuracy in each bin.  A perfectly calibrated model
    has ECE = 0.

    Ported from OpenSpliceAI ``calibrate/temperature_scaling.py``.

    Reference: Naeini et al. (2015), "Obtaining Well Calibrated Probabilities
    Using Bayesian Binning into Quantiles".

    Args:
        n_bins: Number of equal-width confidence bins (default: 15).
    """

    def __init__(self, n_bins: int = 15) -> None:
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.register_buffer("bin_lowers", bin_boundaries[:-1])
        self.register_buffer("bin_uppers", bin_boundaries[1:])

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ECE from logits and integer labels.

        Args:
            logits: ``[N, num_classes]`` raw logits (pre-softmax).
            labels: ``[N]`` integer class labels.

        Returns:
            Scalar ECE in [0, 1].
        """
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece.squeeze()
