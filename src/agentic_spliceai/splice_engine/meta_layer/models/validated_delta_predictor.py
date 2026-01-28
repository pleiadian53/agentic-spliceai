"""
Validated Delta Predictor: Single-Pass Delta Prediction with Ground Truth Targets.

This module implements a single-pass delta predictor that uses SpliceVarDB
to validate/filter training targets, addressing the core limitation of
paired prediction (where base model deltas may be inaccurate).

Key Design Principles:
- Input: alt_seq + variant_info (NO reference sequence needed at inference)
- Target: SpliceVarDB-validated delta (ground truth filtering)
- Inference: Single forward pass (efficient)

Target Derivation Strategy:
- Splice-altering variants: Use base model delta (trusted - SpliceVarDB confirms effect)
- Normal variants: Zero delta (ground truth - no effect, even if base model disagrees)
- Low-frequency/Conflicting: Exclude from training (uncertain labels)

This approach outperforms paired prediction because it doesn't learn from
potentially incorrect base model predictions for non-splice-altering variants.

Usage:
    from agentic_spliceai.splice_engine.meta_layer.models import (
        ValidatedDeltaPredictor,
        create_validated_delta_predictor
    )
    
    model = ValidatedDeltaPredictor(hidden_dim=128)
    delta = model(alt_seq, ref_base, alt_base)  # [B, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GatedCNNEncoder(nn.Module):
    """
    Gated CNN encoder for DNA sequences.
    
    Uses dilated convolutions with gating for efficient
    long-range dependency modeling.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initial projection from one-hot [4] to hidden_dim
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Dilated residual blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, 1, 2, ...
            self.blocks.append(
                GatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to global features.
        
        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Global features [B, hidden_dim]
        """
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)
        
        x = self.pool(x).squeeze(-1)  # [B, H]
        
        return x


class GatedResidualBlock(nn.Module):
    """Gated residual block with dilated convolution."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            channels, channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Gated convolution
        out = self.conv(x)  # [B, 2*C, L]
        out, gate = out.chunk(2, dim=1)
        out = out * torch.sigmoid(gate)
        
        # Normalize (requires permute for LayerNorm)
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        return out + residual


class ValidatedDeltaPredictor(nn.Module):
    """
    Single-pass delta predictor with SpliceVarDB-validated targets.
    
    This model predicts splice site delta scores from a single alternate
    sequence (with variant embedded) plus variant information, without
    requiring both reference and alternate sequences at inference.
    
    Architecture:
        alt_seq ──→ [Gated CNN] ──→ seq_features [B, H]
        ref_base ──┐                      ↓
        alt_base ──┴→ [Embed] ──→ var_features [B, H]
                                          ↓
                              [Fusion + Delta Head]
                                          ↓
                                  Δ = [Δ_donor, Δ_acceptor, Δ_neither]
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for all layers
    n_layers : int
        Number of encoder layers
    dropout : float
        Dropout probability
    include_base_scores : bool
        If True, also takes base model scores as input (for residual learning)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        dropout: float = 0.1,
        include_base_scores: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.include_base_scores = include_base_scores
        
        # Sequence encoder
        self.encoder = GatedCNNEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Variant embedding: ref_base[4] + alt_base[4] → hidden_dim
        var_input_dim = 8
        if include_base_scores:
            var_input_dim += 3  # base model scores [donor, acceptor, neither]
        
        self.variant_embed = nn.Sequential(
            nn.Linear(var_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Delta prediction head
        # Predicts: [Δ_donor, Δ_acceptor, Δ_neither]
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
        )
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        base_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict delta scores.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            Alternate sequence (with variant) [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        base_scores : torch.Tensor, optional
            Base model scores [B, 3] (donor, acceptor, neither)
        
        Returns
        -------
        torch.Tensor
            Delta scores [B, 3] (Δ_donor, Δ_acceptor, Δ_neither)
        """
        # Encode sequence
        seq_features = self.encoder(alt_seq)  # [B, H]
        
        # Encode variant info
        if self.include_base_scores and base_scores is not None:
            var_info = torch.cat([ref_base, alt_base, base_scores], dim=-1)
        else:
            var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]
        
        var_features = self.variant_embed(var_info)  # [B, H]
        
        # Fuse and predict delta
        combined = torch.cat([seq_features, var_features], dim=-1)  # [B, 2H]
        delta = self.delta_head(combined)  # [B, 3]
        
        return delta
    
    def predict_with_interpretation(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        base_scores: Optional[torch.Tensor] = None,
        threshold: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        Predict delta with interpretation.
        
        Returns
        -------
        Dict with:
            - 'delta': [B, 3] raw delta values
            - 'effects': List of effect descriptions
            - 'max_effect': Maximum absolute delta
        """
        delta = self.forward(alt_seq, ref_base, alt_base, base_scores)
        
        # Interpret
        effects = []
        for i in range(delta.shape[0]):
            d = delta[i].detach().cpu().numpy()
            sample_effects = []
            if d[0] > threshold:
                sample_effects.append(f'Donor gain (+{d[0]:.3f})')
            if d[0] < -threshold:
                sample_effects.append(f'Donor loss ({d[0]:.3f})')
            if d[1] > threshold:
                sample_effects.append(f'Acceptor gain (+{d[1]:.3f})')
            if d[1] < -threshold:
                sample_effects.append(f'Acceptor loss ({d[1]:.3f})')
            if not sample_effects:
                sample_effects.append('No significant effect')
            effects.append(sample_effects)
        
        return {
            'delta': delta,
            'effects': effects,
            'max_effect': delta.abs().max(dim=-1)[0]
        }


class ValidatedDeltaPredictorWithAttention(nn.Module):
    """
    Validated delta predictor with position attention for interpretability.
    
    Adds an attention mechanism that shows which positions in the
    sequence are most important for the delta prediction.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sequence encoder (per-position)
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Dilated blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)
            self.blocks.append(
                GatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=15,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Position attention
        self.position_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Variant embedding
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights.
        
        Returns
        -------
        Tuple of:
            - delta: [B, 3]
            - attention: [B, L] position attention weights
        """
        # Per-position encoding
        x = self.embed(alt_seq)  # [B, H, L]
        for block in self.blocks:
            x = block(x)
        
        x = x.permute(0, 2, 1)  # [B, L, H]
        
        # Position attention
        attn_logits = self.position_attention(x).squeeze(-1)  # [B, L]
        attention = F.softmax(attn_logits, dim=-1)  # [B, L]
        
        # Attention-weighted pooling
        seq_features = torch.einsum('bl,blh->bh', attention, x)  # [B, H]
        
        # Variant features
        var_info = torch.cat([ref_base, alt_base], dim=-1)
        var_features = self.variant_embed(var_info)  # [B, H]
        
        # Delta prediction
        combined = torch.cat([seq_features, var_features], dim=-1)
        delta = self.delta_head(combined)
        
        return delta, attention


def create_validated_delta_predictor(
    variant: str = 'basic',
    hidden_dim: int = 128,
    n_layers: int = 6,
    dropout: float = 0.1,
    include_base_scores: bool = False
) -> nn.Module:
    """
    Factory function for validated delta predictors.
    
    Parameters
    ----------
    variant : str
        'basic' - Standard predictor
        'attention' - With position attention for interpretability
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    dropout : float
        Dropout rate
    include_base_scores : bool
        Whether to include base model scores as input
    
    Returns
    -------
    nn.Module
        Validated delta predictor
    """
    if variant == 'basic':
        return ValidatedDeltaPredictor(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            include_base_scores=include_base_scores
        )
    elif variant == 'attention':
        return ValidatedDeltaPredictorWithAttention(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Use: basic, attention")


def one_hot_base(base: str) -> np.ndarray:
    """Convert single base to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = mapping.get(base.upper(), 0)
    onehot = np.zeros(4, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot


def one_hot_seq(seq: str) -> np.ndarray:
    """Convert sequence to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    oh = np.zeros((4, len(seq)), dtype=np.float32)
    for i, n in enumerate(seq.upper()):
        oh[mapping.get(n, 0), i] = 1.0
    return oh












