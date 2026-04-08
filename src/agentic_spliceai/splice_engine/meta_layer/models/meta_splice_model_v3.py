"""
Sequence-level multimodal meta-layer for splice site prediction.

Third generation (v3) of the meta-splice model series.  Previous versions:
  - v1: Position-level classification (``meta_splice_model.py``)
  - v2: Sequence-to-sequence with base scores only (``meta_splice_model_v2.py``)

This version adds a multimodal feature stream (conservation, epigenetic,
chromatin, junction, RBP) and operates in a two-pass refinement mode:

    Pass 1 (frozen): base model → base_scores [L, 3]
    Pass 2 (trainable): sequence + base_scores + multimodal → refined [L, 3]

Architecture::

    Stream A: DNA sequence [B, 4, L]       → dilated 1D CNN  → [B, H, L]
    Stream B: Base model scores [B, L, 3]  → per-position MLP → [B, H, L]
    Stream C: Multimodal features [B, C, L] → 1D CNN          → [B, H, L]
                            |
                    cat → [B, 3H, L] → fusion CNN → [B, H, L]
                            |
                    output head → logits [B, L, 3]
                            |
                softmax((α × logits + (1-α) × log(base_probs)) / T)
                            |
                        output [B, L, 3]

Output follows the same ``[L, 3]`` per-nucleotide protocol as SpliceAI
and OpenSpliceAI, enabling direct delta computation for variant analysis.

Model naming convention:
    M1-S / M2-S / M3-S / M4-S  (S = sequence-level, practical)
    M1-P / M2-P / M3-P         (P = position-level, proof-of-concept)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class MetaSpliceConfig:
    """Configuration for the sequence-level multimodal meta-splice model.

    Supports M1-S through M4-S variants via ``variant`` and
    ``mm_channels`` / ``num_classes``.
    """

    # Model variant
    variant: Literal["M1-S", "M2-S", "M3-S", "M4-S"] = "M1-S"

    # Architecture
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
    use_residual_blend: bool = True
    blend_mode: Literal["probability", "logit"] = "logit"
    use_blend_temperature: bool = True

    # Base score handling: if True, base scores (3 channels) are
    # concatenated into the multimodal stream as additional channels
    # (2-stream design).  If False, they get a dedicated MLP encoder
    # (3-stream design, legacy).
    merge_base_scores: bool = True

    # Multimodal input channels (9 by default, 7 for M3-S).
    # When merge_base_scores=True, the model internally adds 3 for
    # the base score channels — callers should NOT include them here.
    mm_channels: int = 9
    num_classes: int = 3

    # Training
    window_size: int = 5001
    context_padding: int = 400  # dilated CNN receptive field

    @property
    def total_input_length(self) -> int:
        """Sequence input length including context padding."""
        return self.window_size + self.context_padding

    @classmethod
    def m1(cls, **kwargs) -> MetaSpliceConfig:
        """M1-S: canonical splice site classification, all modalities."""
        return cls(variant="M1-S", mm_channels=9, num_classes=3, **kwargs)

    @classmethod
    def m3(cls, **kwargs) -> MetaSpliceConfig:
        """M3-S: novel site prediction, junction excluded (7 channels)."""
        return cls(variant="M3-S", mm_channels=7, num_classes=2, **kwargs)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _get_activation(name: str) -> nn.Module:
    """Return activation module by name."""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "selu":
        return nn.SELU()
    raise ValueError(f"Unknown activation: {name!r}. Choose from: relu, gelu, selu")


class ResidualDilatedBlock(nn.Module):
    """Residual 1D convolution block with configurable dilation and activation.

    Default layout: ``BN → act → DilConv → BN → act → DilConv → + residual``

    When ``activation="selu"``, BatchNorm is replaced with identity (SELU
    is self-normalizing) and AlphaDropout is used instead of standard Dropout.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 11,
        dilation: int = 1,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2
        use_selu = activation == "selu"

        # SELU is self-normalizing — skip BatchNorm to preserve its properties
        self.norm1 = nn.Identity() if use_selu else nn.BatchNorm1d(channels)
        self.norm2 = nn.Identity() if use_selu else nn.BatchNorm1d(channels)

        self.act1 = _get_activation(activation)
        self.act2 = _get_activation(activation)

        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )

        # SELU requires AlphaDropout to maintain self-normalizing properties
        self.dropout = nn.AlphaDropout(dropout) if use_selu else nn.Dropout(dropout)

        # LeCun normal init for SELU (required for self-normalization)
        if use_selu:
            nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="linear")
            nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act1(self.norm1(x))
        out = self.dropout(self.conv1(out))
        out = self.act2(self.norm2(out))
        out = self.dropout(self.conv2(out))
        return out + residual


def _build_encoder(
    in_channels: int,
    hidden_dim: int,
    n_blocks: int,
    dilations: List[int],
    kernel_size: int,
    dropout: float,
    activation: str = "relu",
) -> nn.Sequential:
    """Build a 1D CNN encoder: project → N × ResidualDilatedBlock."""
    layers: List[nn.Module] = [nn.Conv1d(in_channels, hidden_dim, kernel_size=1)]
    for i in range(n_blocks):
        d = dilations[i] if i < len(dilations) else 1
        layers.append(
            ResidualDilatedBlock(
                hidden_dim, kernel_size, dilation=d,
                dropout=dropout, activation=activation,
            )
        )
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MetaSpliceModel(nn.Module):
    """Sequence-level multimodal meta-layer for splice site prediction.

    Produces ``[B, L, 3]`` per-nucleotide predictions matching the base
    model protocol, enabling direct delta computation for variant analysis.

    Parameters
    ----------
    config : MetaSpliceConfig
        Model configuration.

    Examples
    --------
    >>> cfg = MetaSpliceConfig.m1()
    >>> model = MetaSpliceModel(cfg)
    >>> seq = torch.randn(1, 4, 5401)          # one-hot DNA + context
    >>> base = torch.randn(1, 5001, 3)          # base model scores
    >>> mm = torch.randn(1, 9, 5001)            # multimodal features
    >>> out = model(seq, base, mm)
    >>> print(out.shape)  # [1, 5001, 3]
    """

    def __init__(self, config: Optional[MetaSpliceConfig] = None) -> None:
        super().__init__()
        if config is None:
            config = MetaSpliceConfig()
        self.config = config
        H = config.hidden_dim

        act = config.activation

        # Stream A: Sequence encoder (dilated 1D CNN)
        self.seq_encoder = _build_encoder(
            in_channels=4,
            hidden_dim=H,
            n_blocks=config.seq_n_blocks,
            dilations=config.seq_dilations,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=act,
        )

        # Stream B: Per-position signal encoder (1D CNN)
        # In merged mode (default): base scores (3 ch) + multimodal (C ch)
        # In legacy mode: multimodal only (base scores get a separate MLP)
        if config.merge_base_scores:
            signal_channels = config.mm_channels + 3
            self.score_encoder = None
        else:
            signal_channels = config.mm_channels
            self.score_encoder = nn.Sequential(
                nn.Linear(3, H),
                _get_activation(act),
                nn.Dropout(config.dropout),
                nn.Linear(H, H),
            )

        self.signal_encoder = _build_encoder(
            in_channels=signal_channels,
            hidden_dim=H,
            n_blocks=config.mm_n_blocks,
            dilations=config.mm_dilations,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            activation=act,
        )

        # Fusion: 2 or 3 streams → single representation
        n_streams = 2 if config.merge_base_scores else 3
        self.fusion_proj = nn.Conv1d(n_streams * H, H, kernel_size=1)
        self.fusion_blocks = nn.Sequential(
            *[
                ResidualDilatedBlock(
                    H, config.kernel_size,
                    dilation=config.fusion_dilations[i]
                    if i < len(config.fusion_dilations)
                    else 1,
                    dropout=config.dropout,
                    activation=act,
                )
                for i in range(config.fusion_n_blocks)
            ]
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(H, H),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(H, config.num_classes),
        )

        # Learnable residual blend weight (initialized at 0.5)
        if config.use_residual_blend:
            self.blend_alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        if getattr(config, "use_blend_temperature", False):
            self.blend_temperature = nn.Parameter(torch.ones(config.num_classes))

        self._log_model_info()

    def _log_model_info(self) -> None:
        n_params = sum(p.numel() for p in self.parameters())
        blend_mode = getattr(self.config, "blend_mode", "probability")
        has_temp = hasattr(self, "blend_temperature")
        logger.info(
            "MetaSpliceModel (%s): %.1fK params, H=%d, mm_channels=%d, "
            "blend=%s%s",
            self.config.variant,
            n_params / 1000,
            self.config.hidden_dim,
            self.config.mm_channels,
            blend_mode,
            "+classT" if has_temp else "",
        )

    def forward(
        self,
        sequence: torch.Tensor,
        base_scores: torch.Tensor,
        mm_features: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        sequence : torch.Tensor
            One-hot encoded DNA ``[B, 4, L_seq]`` where
            ``L_seq = window_size + context_padding``.
        base_scores : torch.Tensor
            Base model predictions ``[B, L, 3]`` (L = window_size).
        mm_features : torch.Tensor
            Dense multimodal features ``[B, C_mm, L]``.
        return_logits : bool
            If True, return raw logits ``[B, L, 3]`` instead of
            probabilities.  Used for post-hoc temperature scaling
            calibration.

        Returns
        -------
        torch.Tensor
            Refined predictions ``[B, L, 3]`` with softmax constraint,
            or raw logits if ``return_logits=True``.
        """
        L = base_scores.shape[1]

        # Stream A: encode sequence (may be longer due to context padding)
        seq_feat = self.seq_encoder(sequence)  # [B, H, L_seq]
        # Crop to match output length (center crop)
        if seq_feat.shape[2] > L:
            excess = seq_feat.shape[2] - L
            left = excess // 2
            seq_feat = seq_feat[:, :, left:left + L]  # [B, H, L]

        # Stream B: per-position signals
        if self.config.merge_base_scores:
            # Merge base scores into multimodal stream as 3 extra channels
            base_ch = base_scores.permute(0, 2, 1).contiguous()  # [B, 3, L]
            signal_input = torch.cat([base_ch, mm_features], dim=1)  # [B, 3+C, L]
            signal_feat = self.signal_encoder(signal_input)  # [B, H, L]
            fused = torch.cat([seq_feat, signal_feat], dim=1)  # [B, 2H, L]
        else:
            # Legacy 3-stream: separate MLP for base scores
            score_feat = self.score_encoder(base_scores)  # [B, L, H]
            score_feat = score_feat.permute(0, 2, 1).contiguous()  # [B, H, L]
            mm_feat = self.signal_encoder(mm_features)  # [B, H, L]
            fused = torch.cat([seq_feat, score_feat, mm_feat], dim=1)  # [B, 3H, L]

        # Fusion
        fused = self.fusion_proj(fused)  # [B, H, L]
        fused = self.fusion_blocks(fused)  # [B, H, L]

        # Output head (pointwise)
        fused = fused.permute(0, 2, 1).contiguous()  # [B, L, H]
        logits = self.output_head(fused)  # [B, L, num_classes]

        # Residual blend: combine meta logits with base model signal
        blend_mode = getattr(self.config, "blend_mode", "probability")

        if self.config.use_residual_blend:
            alpha = torch.sigmoid(self.blend_alpha)

            if blend_mode == "logit":
                # Product-of-experts: blend in logit space before softmax.
                # base_scores are cached probabilities → convert to logits.
                base_logits = torch.log(base_scores.clamp(min=1e-8))
                blended = alpha * logits + (1.0 - alpha) * base_logits

                # Class-wise learned temperature (subsumes post-hoc calibration)
                if hasattr(self, "blend_temperature"):
                    T = self.blend_temperature.clamp(min=0.05, max=5.0)
                    blended = blended / T  # [B, L, C] / [C] broadcasts

                if self.training or return_logits:
                    return blended
                return F.softmax(blended, dim=-1)

            else:
                # Legacy probability-space mode (for old checkpoints).
                # base_scores are already probabilities — use directly.
                if self.training or return_logits:
                    return logits
                meta_probs = F.softmax(logits, dim=-1)
                return alpha * meta_probs + (1.0 - alpha) * base_scores

        # No blend
        if self.training or return_logits:
            return logits
        return F.softmax(logits, dim=-1)

    def predict_with_delta(
        self,
        ref_sequence: torch.Tensor,
        alt_sequence: torch.Tensor,
        ref_base_scores: torch.Tensor,
        alt_base_scores: torch.Tensor,
        ref_mm_features: torch.Tensor,
        alt_mm_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict and compute delta scores for variant analysis (M4-S).

        Returns
        -------
        (ref_probs, alt_probs, delta)
            Each ``[B, L, 3]``.  ``delta = alt_probs - ref_probs``.
        """
        ref_probs = self.forward(ref_sequence, ref_base_scores, ref_mm_features)
        alt_probs = self.forward(alt_sequence, alt_base_scores, alt_mm_features)
        return ref_probs, alt_probs, alt_probs - ref_probs

    @property
    def receptive_field(self) -> int:
        """Approximate receptive field of the sequence encoder in bp."""
        k = self.config.kernel_size
        return 2 * sum(d * (k - 1) for d in self.config.seq_dilations)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_meta_splice_model(
    config: Optional[MetaSpliceConfig] = None,
    **kwargs,
) -> MetaSpliceModel:
    """Create a MetaSpliceModel instance.

    Parameters
    ----------
    config : MetaSpliceConfig, optional
        Full configuration.  If None, creates default M1-S config.
    **kwargs
        Overrides passed to ``MetaSpliceConfig()`` if ``config`` is None.
    """
    if config is None:
        config = MetaSpliceConfig(**kwargs)
    return MetaSpliceModel(config)
