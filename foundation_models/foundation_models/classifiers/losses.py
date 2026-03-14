"""
Loss functions for splice site classification.

Provides:
- ``FocalLoss`` — focal loss for multi-class classification with extreme
  class imbalance (splice sites are ~0.01% of nucleotides).
- ``compute_class_weights`` — inverse-frequency class weights from labels.
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
