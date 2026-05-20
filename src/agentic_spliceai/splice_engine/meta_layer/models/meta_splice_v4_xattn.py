"""Sequence-level meta-splice model with cross-attention fusion (Stage 3 / v4_xattn).

Variant of :class:`MetaSpliceModel` (v3) where the ``cat → 1×1 conv`` fusion
stage is replaced with cross-attention: sequence features (query) attend to
the merged base+multimodal signal features (key/value). Each output position
gets a position-specific weighted combination of the signal channels — the
selectivity v3 lacks (its 1×1 conv applies the same channel weighting at
every position).

Architecture (differences from v3 highlighted with >>>)::

    Stream A: sequence [B, 4, L_seq] → dilated CNN → [B, H, L]
    Stream B: base+mm [B, 3+C, L]    → dilated CNN → [B, H, L]
    >>> Cross-attention: fused = seq_feat + CrossAttn(seq_feat, signal_feat, signal_feat)
    fusion CNN blocks → [B, H, L]
    output head + logit-space blend with base_scores → [B, L, num_classes]

Hypothesis being tested for M2-S: position-wise selective modality weighting
(an alt splice site near a cryptic donor up-weights junction support; a
deep-intron position up-weights conservation) outperforms v3's cat-fusion,
which has no positional selectivity.

The sequence encoder is still a dilated CNN, so ``cfg.receptive_field`` uses
the same formula as v3 — :func:`_compute_receptive_field`. The cross-attention
operates *post-encoder*, so it doesn't affect the input context_padding budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse v3 building blocks so the per-stream encoders + RF math stay in lockstep.
from .meta_splice_model_v3 import (
    ResidualDilatedBlock,
    _build_encoder,
    _compute_receptive_field,
    _get_activation,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MetaSpliceXAttnConfig:
    """Config for the cross-attention-fusion meta-splice model (v4_xattn).

    Mirrors :class:`MetaSpliceConfig` field-for-field with one addition:
    ``n_heads`` for the cross-attention layer.
    """

    variant: Literal["M1-S", "M2-S", "M3-S", "M4-S"] = "M1-S"

    # CNN streams (same defaults as v3)
    hidden_dim: int = 32
    seq_n_blocks: int = 8
    seq_dilations: List[int] = field(
        default_factory=lambda: [1, 1, 1, 1, 4, 4, 4, 4]
    )
    mm_n_blocks: int = 4
    mm_dilations: List[int] = field(
        default_factory=lambda: [1, 1, 4, 4]
    )
    fusion_n_blocks: int = 4
    fusion_dilations: List[int] = field(
        default_factory=lambda: [1, 1, 4, 4]
    )
    kernel_size: int = 11
    dropout: float = 0.1
    activation: Literal["relu", "gelu", "selu"] = "gelu"

    # Cross-attention fusion (new vs v3)
    n_heads: int = 4
    # Local windowed attention. When None (default), attention is global —
    # every output position attends to all input positions. When set to an
    # int W, each position attends only to positions within [i - W//2, i + W//2].
    # For splice-site prediction the canonical biology is local (~100 bp:
    # donor/acceptor consensus + branchpoint + polypyrimidine tract), and
    # global attention dilutes the rare positive class with the abundant
    # "neither" majority. Recommended window: 128–256.
    attention_window_size: Optional[int] = None

    # Base-score blending (same as v3)
    use_residual_blend: bool = True
    blend_mode: Literal["probability", "logit"] = "logit"
    use_blend_temperature: bool = True
    merge_base_scores: bool = True  # v4_xattn requires this; 3-stream not implemented

    # Channels
    mm_channels: int = 9
    num_classes: int = 3

    # Window
    window_size: int = 5001
    # Explicit override; if None, defaults to self.receptive_field.
    context_padding: Optional[int] = None

    @property
    def receptive_field(self) -> int:
        """Sequence encoder RF (unchanged from v3 — still a dilated CNN)."""
        return _compute_receptive_field(self.seq_dilations, self.kernel_size)

    @property
    def effective_context_padding(self) -> int:
        """Resolved context padding: explicit override else receptive_field."""
        return self.context_padding if self.context_padding is not None else self.receptive_field

    @property
    def total_input_length(self) -> int:
        return self.window_size + self.effective_context_padding


# ---------------------------------------------------------------------------
# Cross-attention fusion module
# ---------------------------------------------------------------------------


class CrossAttentionFusion(nn.Module):
    """Replace ``cat → 1×1 conv`` fusion with one cross-attention layer.

    Sequence features query the merged base+multimodal signal features.
    Pre-norm residual (more stable than post-norm).

    Memory: O(L²) per head per sample for global attention. For L=5001,
    H=64, n_heads=4, B=4 on A40 → ~1.6 GB. When ``window_size`` is set,
    the attention mask restricts each position to ±window_size//2
    neighbors — PyTorch's SDPA still allocates the full attention buffer
    but skips computation on masked positions. (Switch to an unfold-based
    custom kernel if memory becomes the bottleneck.)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
            )
        self.window_size = window_size
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # Cache the band-diagonal mask by sequence length (one per process).
        self._mask_cache: dict = {}

    def _get_local_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        """Boolean band-diagonal mask. True = position is masked OUT.

        Each row i has False (allowed) for columns j in [i - W//2, i + W//2];
        all others True (forbidden). Returns shape [L, L].
        """
        key = (seq_len, device, dtype)
        if key in self._mask_cache:
            return self._mask_cache[key]
        half = self.window_size // 2
        idx = torch.arange(seq_len, device=device)
        diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [L, L]
        mask = diff > half
        self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        seq_feat: torch.Tensor,     # [B, H, L]
        signal_feat: torch.Tensor,  # [B, H, L]
    ) -> torch.Tensor:
        q = seq_feat.permute(0, 2, 1)       # [B, L, H]
        kv = signal_feat.permute(0, 2, 1)   # [B, L, H]

        attn_mask = None
        if self.window_size is not None:
            attn_mask = self._get_local_mask(q.size(1), q.device, q.dtype)

        attn_out, _ = self.cross_attn(
            query=self.norm_q(q),
            key=self.norm_kv(kv),
            value=self.norm_kv(kv),
            attn_mask=attn_mask,
            need_weights=False,
        )
        fused = q + self.dropout(attn_out)  # residual

        return fused.permute(0, 2, 1).contiguous()  # [B, H, L]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MetaSpliceXAttnModel(nn.Module):
    """Sequence-level meta-splice model with cross-attention fusion.

    Drop-in replacement for :class:`MetaSpliceModel` (v3): same I/O shapes,
    same blend protocol, same training-time logits behavior. Different
    ``state_dict`` keys, so checkpoints are not interchangeable.
    """

    def __init__(self, config: Optional[MetaSpliceXAttnConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = MetaSpliceXAttnConfig()
        if not config.merge_base_scores:
            raise NotImplementedError(
                "v4_xattn requires merge_base_scores=True (2-stream design). "
                "For 3-stream layouts, use the v3 model."
            )

        self.config = config
        H = config.hidden_dim
        act = config.activation

        # Stream A — sequence (same as v3)
        self.seq_encoder = _build_encoder(
            in_channels=4,
            hidden_dim=H,
            n_blocks=config.seq_n_blocks,
            dilations=config.seq_dilations,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=act,
        )

        # Stream B — base scores + multimodal merged (same as v3 in merged mode)
        signal_channels = config.mm_channels + 3
        self.signal_encoder = _build_encoder(
            in_channels=signal_channels,
            hidden_dim=H,
            n_blocks=config.mm_n_blocks,
            dilations=config.mm_dilations,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=act,
        )

        # Cross-attention fusion (new vs v3)
        self.cross_attn_fusion = CrossAttentionFusion(
            hidden_dim=H,
            n_heads=config.n_heads,
            dropout=config.dropout,
            window_size=config.attention_window_size,
        )

        # Post-fusion CNN refinement (same as v3)
        self.fusion_blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(
                    H,
                    config.kernel_size,
                    dilation=config.fusion_dilations[i]
                    if i < len(config.fusion_dilations)
                    else 1,
                    dropout=config.dropout,
                    activation=act,
                )
                for i in range(config.fusion_n_blocks)
            ]
        )

        # Output head (same as v3)
        self.output_head = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, config.num_classes),
        )

        # Blend params (same as v3)
        if config.use_residual_blend:
            self.blend_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        if getattr(config, "use_blend_temperature", False):
            self.blend_temperature = nn.Parameter(torch.ones(config.num_classes))

        self._log_model_info()

    def _log_model_info(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "MetaSpliceXAttnModel (%s): %.1fK params, H=%d, n_heads=%d, "
            "mm_channels=%d, blend=%s%s",
            self.config.variant,
            n_params / 1000,
            self.config.hidden_dim,
            self.config.n_heads,
            self.config.mm_channels,
            self.config.blend_mode,
            "+classT" if hasattr(self, "blend_temperature") else "",
        )

    @property
    def receptive_field(self) -> int:
        """Delegates to ``config.receptive_field`` (CNN-based; xattn is post-encoder)."""
        return self.config.receptive_field

    def forward(
        self,
        sequence: torch.Tensor,     # [B, 4, L_seq]
        base_scores: torch.Tensor,  # [B, L, 3]
        mm_features: torch.Tensor,  # [B, C_mm, L]
        return_logits: bool = False,
    ) -> torch.Tensor:
        L = base_scores.shape[1]

        # Stream A — sequence, center-cropped to L if encoder produced longer
        seq_feat = self.seq_encoder(sequence)
        if seq_feat.shape[2] > L:
            excess = seq_feat.shape[2] - L
            left = excess // 2
            seq_feat = seq_feat[:, :, left:left + L]

        # Stream B — merge base scores into multimodal stream
        base_ch = base_scores.permute(0, 2, 1).contiguous()           # [B, 3, L]
        signal_input = torch.cat([base_ch, mm_features], dim=1)       # [B, 3+C, L]
        signal_feat = self.signal_encoder(signal_input)               # [B, H, L]

        # Cross-attention fusion (the v4_xattn change)
        fused = self.cross_attn_fusion(seq_feat, signal_feat)         # [B, H, L]

        # Post-fusion CNN refinement
        fused = self.fusion_blocks(fused)                             # [B, H, L]

        # Pointwise output head
        fused_lh = fused.permute(0, 2, 1).contiguous()                # [B, L, H]
        logits = self.output_head(fused_lh)                           # [B, L, C]

        # Residual blend with base scores. Diverges from v3 on temperature
        # placement: v3 applies T to the blended logits (T divides both meta
        # AND base contributions), which is fine when the meta-layer is small
        # and well-calibrated. v4_xattn's higher capacity overfits more easily,
        # which drives T up — and at v3's placement, that softens the
        # well-calibrated base scores too, washing them out. Here T calibrates
        # only the meta-layer; the base scores pass through with their
        # original calibration. (See dev/sessions/ M2-S v4_xattn post-mortem.)
        if self.config.use_residual_blend:
            alpha = torch.sigmoid(self.blend_alpha)

            if self.config.blend_mode == "logit":
                base_logits = torch.log(base_scores.clamp(min=1e-8))
                if hasattr(self, "blend_temperature"):
                    T = self.blend_temperature.clamp(min=0.05, max=5.0)
                    meta_part = logits / T
                else:
                    meta_part = logits
                blended = alpha * meta_part + (1.0 - alpha) * base_logits
                if self.training or return_logits:
                    return blended
                return F.softmax(blended, dim=-1)

            # Legacy probability-space blend
            if self.training or return_logits:
                return logits
            meta_probs = F.softmax(logits, dim=-1)
            return alpha * meta_probs + (1.0 - alpha) * base_scores

        # No blend
        if self.training or return_logits:
            return logits
        return F.softmax(logits, dim=-1)
