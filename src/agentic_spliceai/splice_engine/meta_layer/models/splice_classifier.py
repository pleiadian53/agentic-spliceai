"""
Splice-Effect Classifiers for Variant Analysis.

Single-pass models that predict whether a variant affects splicing,
using SpliceVarDB classifications as ground truth labels.

Key Features:
- Input: alt_sequence + variant_info (NO reference sequence needed)
- Label: SpliceVarDB classification (ground truth, not base model deltas)
- Output: P(splice-altering) or effect type distribution
- Inference: Single forward pass (efficient)

Models:
- SpliceInducingClassifier: Binary "Is this variant splice-altering?"
- EffectTypeClassifier: Multi-class effect type (gain/loss, donor/acceptor)
- UnifiedSpliceClassifier: Multi-task with position attention

Usage:
    from agentic_spliceai.splice_engine.meta_layer.models import (
        SpliceInducingClassifier,
        EffectTypeClassifier,
        create_splice_classifier
    )
    
    # Binary: Is this variant splice-altering?
    model = SpliceInducingClassifier(hidden_dim=128)
    p_splice = model(alt_seq, ref_base, alt_base)  # [B, 1]
    
    # Multi-class: What type of effect?
    model = EffectTypeClassifier(hidden_dim=128, num_classes=5)
    effect_logits = model(alt_seq, ref_base, alt_base)  # [B, 5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
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
        
        # Initial projection
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Dilated residual blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)
            self.blocks.append(
                GatedResidualBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
    
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
        x = self.out_proj(x)
        
        return x
    
    def encode_per_position(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to per-position features.
        
        Returns
        -------
        torch.Tensor
            Per-position features [B, L, hidden_dim]
        """
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)
        
        return x.permute(0, 2, 1)  # [B, L, H]


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
        
        # Normalize
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        return out + residual


class SpliceInducingClassifier(nn.Module):
    """
    Binary classifier: Does this variant affect splicing?
    
    Architecture:
        alt_seq ──→ [Gated CNN] ──→ seq_features [B, H]
        ref_base ──┐                      ↓
        alt_base ──┴→ [Embed] ──→ var_features [B, H]
                                          ↓
                              [Fusion + Classifier]
                                          ↓
                              P(splice-altering) [B, 1]
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Sequence encoder
        self.encoder = GatedCNNEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Variant embedding: ref_base[4] + alt_base[4] → hidden_dim
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict probability of splice-altering.
        
        Parameters
        ----------
        alt_seq : torch.Tensor
            Alternate sequence (with variant) [B, 4, L]
        ref_base : torch.Tensor
            Reference base one-hot [B, 4]
        alt_base : torch.Tensor
            Alternate base one-hot [B, 4]
        
        Returns
        -------
        torch.Tensor
            P(splice-altering) [B, 1]
        """
        # Encode sequence
        seq_features = self.encoder(alt_seq)  # [B, H]
        
        # Encode variant info
        var_info = torch.cat([ref_base, alt_base], dim=-1)  # [B, 8]
        var_features = self.variant_embed(var_info)  # [B, H]
        
        # Fuse and classify
        combined = torch.cat([seq_features, var_features], dim=-1)  # [B, 2H]
        logits = self.classifier(combined)  # [B, 1]
        
        return torch.sigmoid(logits)
    
    def predict_proba(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """Get probability of splice-altering."""
        return self.forward(alt_seq, ref_base, alt_base)
    
    def predict(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """Get binary prediction."""
        proba = self.forward(alt_seq, ref_base, alt_base)
        return (proba > threshold).float()


class EffectTypeClassifier(nn.Module):
    """
    Multi-class classifier: What type of splicing effect?
    
    Classes (from SpliceVarDB):
      0: No effect (Normal)
      1: Donor gain
      2: Donor loss
      3: Acceptor gain
      4: Acceptor loss
      5: Complex (multiple effects)
    """
    
    EFFECT_TYPES = [
        'Normal',
        'Donor_gain',
        'Donor_loss',
        'Acceptor_gain',
        'Acceptor_loss',
        'Complex'
    ]
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_classes: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Sequence encoder
        self.encoder = GatedCNNEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Variant embedding
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Multi-class classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict effect type logits.
        
        Returns
        -------
        torch.Tensor
            Effect type logits [B, num_classes]
        """
        seq_features = self.encoder(alt_seq)
        
        var_info = torch.cat([ref_base, alt_base], dim=-1)
        var_features = self.variant_embed(var_info)
        
        combined = torch.cat([seq_features, var_features], dim=-1)
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(alt_seq, ref_base, alt_base)
        return F.softmax(logits, dim=-1)
    
    def predict(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(alt_seq, ref_base, alt_base)
        return logits.argmax(dim=-1)


class UnifiedSpliceClassifier(nn.Module):
    """
    Multi-task classifier combining binary + effect type + position attention.
    
    Outputs:
      1. P(splice-altering): Binary probability [B, 1]
      2. Effect type logits: [B, num_classes]
      3. Position attention: [B, L] (where in the sequence is most important)
    
    The position attention provides interpretability without requiring
    explicit position labels.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_effect_classes: int = 6,
        n_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Shared encoder
        self.encoder = GatedCNNEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
        
        # Variant embedding
        self.variant_embed = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position attention (learned)
        self.position_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Task heads
        self.binary_head = nn.Linear(hidden_dim * 2, 1)
        self.effect_head = nn.Linear(hidden_dim * 2, num_effect_classes)
    
    def forward(
        self,
        alt_seq: torch.Tensor,
        ref_base: torch.Tensor,
        alt_base: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with all outputs.
        
        Returns
        -------
        Dict with keys:
            - 'p_splice_altering': [B, 1]
            - 'effect_logits': [B, num_classes]
            - 'position_attention': [B, L]
            - 'attended_features': [B, H]
        """
        # Per-position encoding
        per_pos_features = self.encoder.encode_per_position(alt_seq)  # [B, L, H]
        
        # Compute position attention
        attn_logits = self.position_attention(per_pos_features).squeeze(-1)  # [B, L]
        position_attn = F.softmax(attn_logits, dim=-1)  # [B, L]
        
        # Attention-weighted features
        attended = torch.einsum('bl,blh->bh', position_attn, per_pos_features)  # [B, H]
        
        # Variant features
        var_info = torch.cat([ref_base, alt_base], dim=-1)
        var_features = self.variant_embed(var_info)  # [B, H]
        
        # Combined features
        combined = torch.cat([attended, var_features], dim=-1)  # [B, 2H]
        
        # Task outputs
        binary_logits = self.binary_head(combined)  # [B, 1]
        effect_logits = self.effect_head(combined)  # [B, num_classes]
        
        return {
            'p_splice_altering': torch.sigmoid(binary_logits),
            'effect_logits': effect_logits,
            'position_attention': position_attn,
            'attended_features': attended
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        is_splice_altering: torch.Tensor,
        effect_type: Optional[torch.Tensor] = None,
        binary_weight: float = 1.0,
        effect_weight: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Parameters
        ----------
        is_splice_altering : torch.Tensor
            Binary labels [B] (1 = splice-altering)
        effect_type : torch.Tensor, optional
            Effect type labels [B] (class indices)
        """
        # Binary loss (always computed)
        binary_loss = F.binary_cross_entropy(
            outputs['p_splice_altering'].squeeze(-1),
            is_splice_altering.float()
        )
        
        total_loss = binary_weight * binary_loss
        losses = {'binary': binary_loss}
        
        # Effect type loss (if labels provided)
        if effect_type is not None:
            effect_loss = F.cross_entropy(
                outputs['effect_logits'],
                effect_type.long()
            )
            total_loss = total_loss + effect_weight * effect_loss
            losses['effect'] = effect_loss
        
        losses['total'] = total_loss
        return losses


def create_splice_classifier(
    task: str = 'binary',
    hidden_dim: int = 128,
    n_layers: int = 6,
    num_classes: int = 6,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function for splice classifiers.
    
    Parameters
    ----------
    task : str
        One of: 'binary', 'effect', 'unified'
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    num_classes : int
        Number of effect classes (for 'effect' and 'unified')
    
    Returns
    -------
    nn.Module
        Splice classifier model
    """
    if task == 'binary':
        return SpliceInducingClassifier(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout
        )
    elif task == 'effect':
        return EffectTypeClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            n_layers=n_layers,
            dropout=dropout
        )
    elif task == 'unified':
        return UnifiedSpliceClassifier(
            hidden_dim=hidden_dim,
            num_effect_classes=num_classes,
            n_layers=n_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown task: {task}. Use: binary, effect, unified")


def one_hot_base(base: str) -> np.ndarray:
    """Convert single base to one-hot encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    idx = mapping.get(base.upper(), 0)
    onehot = np.zeros(4, dtype=np.float32)
    onehot[idx] = 1.0
    return onehot












