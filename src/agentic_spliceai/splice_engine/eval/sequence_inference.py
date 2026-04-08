"""Sliding-window inference for sequence-level meta-splice models.

Provides ``infer_full_gene()`` — the standard way to run a trained
MetaSpliceModel on a full gene, handling padding, overlap averaging,
and variable-length sequences.

Also provides ``apply_temperature_blend()`` for post-hoc temperature
scaling of raw logits with residual blending.
"""

from __future__ import annotations

import numpy as np


def apply_temperature_blend(
    logits: np.ndarray,
    base_scores: np.ndarray,
    temperature: "float | np.ndarray" = 1.0,
    blend_alpha: float = 0.5,
    blend_mode: str = "logit",
) -> np.ndarray:
    """Apply temperature scaling to logits, then residual blend with base scores.

    Supports two blend modes:

    - ``"logit"`` (default): Product-of-experts in logit space.
      ``softmax((alpha * logits + (1-alpha) * log(base_scores)) / T)``
    - ``"probability"``: Legacy probability-space blend (no double-softmax).
      ``alpha * softmax(logits / T) + (1-alpha) * base_scores``

    Supports both scalar temperature (single T for all classes) and
    class-wise temperature (one T per class, matching OpenSpliceAI).

    Parameters
    ----------
    logits : np.ndarray
        Raw model logits ``[L, C]``.  For ``blend_mode="logit"`` these
        are **blended** logits (already include base signal + learned
        temperature from model forward).
    base_scores : np.ndarray
        Base model probabilities ``[L, C]``.
    temperature : float or np.ndarray
        Post-hoc temperature parameter(s).  Scalar applies to all classes.
        Array of shape ``[C]`` applies per-class.
        Values >1 soften, <1 sharpen.  Default 1.0 (no change).
    blend_alpha : float
        Residual blend weight.  Only used in ``"probability"`` mode;
        in ``"logit"`` mode, blending is already part of the model logits.
    blend_mode : str
        ``"logit"`` (default) or ``"probability"``.

    Returns
    -------
    np.ndarray
        Calibrated probabilities ``[L, C]``.
    """
    from scipy.special import softmax

    temperature = np.asarray(temperature, dtype=np.float64)

    if blend_mode == "logit":
        # Logits are already blended by model.forward(return_logits=True).
        # Apply post-hoc temperature only (often T=1.0, i.e. no-op).
        return softmax(logits / temperature, axis=-1)

    # Legacy probability-space blend (for old checkpoints).
    # base_scores are already probabilities — use directly, no double-softmax.
    scaled_logits = logits / temperature
    meta_probs = softmax(scaled_logits, axis=-1)
    return blend_alpha * meta_probs + (1.0 - blend_alpha) * base_scores


def infer_full_gene(
    model,
    gene_data: dict,
    window_size: int = 5001,
    context_padding: int = 400,
    device=None,
    return_logits: bool = False,
) -> np.ndarray:
    """Run sliding window inference on a full gene, return [L, 3] probs.

    Parameters
    ----------
    model : MetaSpliceModel
        Trained model in eval mode.
    gene_data : dict
        Keys: ``sequence`` (str), ``base_scores`` ([L, 3]),
        ``mm_features`` ([L, C]).
    window_size : int
        Output window size (default 5001).
    context_padding : int
        Extra DNA context for the sequence CNN (default 400).
    device : torch.device, optional
        Inference device (default: cpu).
    return_logits : bool
        If True, return raw logits instead of probabilities.  Used for
        temperature scaling calibration.

    Returns
    -------
    np.ndarray
        Shape ``[L, 3]`` — per-position (donor, acceptor, neither) probs,
        or raw logits if ``return_logits=True``.
    """
    import torch
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _one_hot_encode,
    )

    if device is None:
        device = torch.device("cpu")

    sequence = gene_data["sequence"]
    base_scores = gene_data["base_scores"]  # [L, 3]
    mm_features = gene_data["mm_features"]  # [L, C]
    gene_len = len(sequence)
    W = window_size
    ctx = context_padding

    if gene_len <= W:
        # Small gene: single window with padding
        seq_onehot = _one_hot_encode(sequence)  # [4, L]
        total_len = W + ctx
        padded_seq = np.zeros((4, total_len), dtype=np.float32)
        offset = (total_len - seq_onehot.shape[1]) // 2
        padded_seq[:, offset:offset + seq_onehot.shape[1]] = seq_onehot

        padded_base = np.full((W, 3), 1.0 / 3, dtype=np.float32)
        padded_base[:gene_len] = base_scores[:gene_len]

        padded_mm = np.zeros((mm_features.shape[1], W), dtype=np.float32)
        padded_mm[:, :gene_len] = mm_features[:gene_len].T

        with torch.no_grad():
            seq_t = torch.from_numpy(padded_seq).unsqueeze(0).to(device)
            base_t = torch.from_numpy(padded_base).unsqueeze(0).to(device)
            mm_t = torch.from_numpy(padded_mm).unsqueeze(0).to(device)
            out = model(seq_t, base_t, mm_t, return_logits=return_logits)
            return out[0, :gene_len].cpu().numpy()

    # Sliding window with overlap
    stride = W // 2  # 50% overlap
    accum = np.zeros((gene_len, 3), dtype=np.float64)
    counts = np.zeros(gene_len, dtype=np.float64)

    starts = list(range(0, gene_len - W + 1, stride))
    if starts[-1] + W < gene_len:
        starts.append(gene_len - W)  # ensure last positions are covered

    for out_start in starts:
        out_end = out_start + W

        # Sequence with context
        seq_start = max(0, out_start - ctx // 2)
        seq_end = min(gene_len, out_end + (ctx - ctx // 2))
        seq_onehot = _one_hot_encode(sequence[seq_start:seq_end])

        total_len = W + ctx
        if seq_onehot.shape[1] < total_len:
            padded = np.zeros((4, total_len), dtype=np.float32)
            off = (total_len - seq_onehot.shape[1]) // 2
            padded[:, off:off + seq_onehot.shape[1]] = seq_onehot
            seq_onehot = padded

        bs = base_scores[out_start:out_end]  # [W, 3]
        mm = mm_features[out_start:out_end].T.copy()  # [C, W]

        with torch.no_grad():
            seq_t = torch.from_numpy(seq_onehot).unsqueeze(0).to(device)
            base_t = torch.from_numpy(bs).unsqueeze(0).to(device)
            mm_t = torch.from_numpy(mm).unsqueeze(0).to(device)
            out = model(seq_t, base_t, mm_t, return_logits=return_logits)
            window_out = out[0].cpu().numpy()

        accum[out_start:out_end] += window_out
        counts[out_start:out_end] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    return (accum / counts[:, None]).astype(np.float32)
