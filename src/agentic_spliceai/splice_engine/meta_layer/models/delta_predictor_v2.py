"""
Delta Predictor V2: Per-Position Delta Prediction.

Outputs delta scores for EVERY position in the variant context window,
enabling detection of:
- Suppressed splice sites (negative delta at existing sites)
- Induced new splice sites (positive delta at non-splice positions)
- Other perturbations across the window

Output Format:
    [B, L, 2] where L = context window length (e.g., 101 for ±50nt)
    Each position gets [Δ_donor, Δ_acceptor]

This matches the base model's ability to output per-position scores,
enabling fair comparison of delta tensors.

Comparison Strategy:
    base_delta = base_model(alt) - base_model(ref)  # [L, 2]
    meta_delta = meta_model(ref_seq, alt_seq)       # [L, 2]
    
    # Compare:
    # - Max absolute delta (any position)
    # - Delta at specific positions (known splice sites)
    # - Detection rates with thresholds

Ported from: meta_spliceai/splice_engine/meta_layer/models/delta_predictor_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class DeltaPredictorV2(nn.Module):
    """
    Per-position delta predictor using sequence-to-sequence architecture.
    
    Unlike V1 which outputs a single [2] delta, V2 outputs [L, 2] deltas
    for every position in the context window.
    
    Architecture:
        ref_seq ──→ [Shared Encoder] ──→ ref_features [B, L, H]
        alt_seq ──→ [Shared Encoder] ──→ alt_features [B, L, H]
                                              ↓
                               diff = alt_features - ref_features
                                              ↓
                                    [Per-Position Head]
                                              ↓
                                   Output: [B, L, 2]
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_conv_layers : int
        Number of convolutional layers
    kernel_size : int
        Convolution kernel size
    dropout : float
        Dropout probability
    
    Examples
    --------
    >>> model = DeltaPredictorV2(hidden_dim=64)
    >>> ref_seq = torch.randn(32, 4, 101)  # [B, 4, L]
    >>> alt_seq = torch.randn(32, 4, 101)
    >>> delta = model(ref_seq, alt_seq)
    >>> print(delta.shape)  # [32, 101, 2]
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        n_conv_layers: int = 4,
        kernel_size: int = 11,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Shared sequence encoder (outputs per-position features)
        self.encoder = Seq2SeqEncoder(
            input_channels=4,
            hidden_dim=hidden_dim,
            n_layers=n_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Per-position delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [Δ_donor, Δ_acceptor] per position
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        ref_seq : torch.Tensor
            Reference sequence [B, 4, L]
        alt_seq : torch.Tensor
            Alternate sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Per-position delta scores [B, L, 2]
        """
        # Encode both sequences (shared weights)
        ref_features = self.encoder(ref_seq)  # [B, L, H]
        alt_features = self.encoder(alt_seq)  # [B, L, H]
        
        # Compute difference at each position
        diff_features = alt_features - ref_features  # [B, L, H]
        
        # Predict delta at each position
        delta = self.delta_head(diff_features)  # [B, L, 2]
        
        return delta
    
    def predict_with_summary(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Predict with summary statistics.
        
        Returns
        -------
        dict
            - 'delta': Full [B, L, 2] tensor
            - 'max_delta': [B, 2] max absolute delta per sample
            - 'max_positions': [B, 2] positions of max delta
            - 'center_delta': [B, 2] delta at center position
        """
        delta = self.forward(ref_seq, alt_seq)  # [B, L, 2]
        
        B, L, _ = delta.shape
        center = L // 2
        
        # Max absolute delta
        abs_delta = delta.abs()
        max_delta_donor, max_pos_donor = abs_delta[:, :, 0].max(dim=1)
        max_delta_acceptor, max_pos_acceptor = abs_delta[:, :, 1].max(dim=1)
        
        return {
            'delta': delta,
            'max_delta': torch.stack([max_delta_donor, max_delta_acceptor], dim=1),
            'max_positions': torch.stack([max_pos_donor, max_pos_acceptor], dim=1),
            'center_delta': delta[:, center, :]
        }


class Seq2SeqEncoder(nn.Module):
    """
    Sequence-to-sequence encoder that preserves spatial dimensions.
    
    Unlike pooling encoders, this outputs features for each position.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 64,
        n_layers: int = 4,
        kernel_size: int = 11,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, hidden_dim, kernel_size=1)
        
        # Residual conv blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            self.blocks.append(
                ResidualConvBlock(
                    channels=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=1,  # Could use dilated convs
                    dropout=dropout
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to per-position features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Features [B, L, H]
        """
        # Project input
        x = self.input_proj(x)  # [B, H, L]
        
        # Apply conv blocks
        for block in self.blocks:
            x = block(x)  # [B, H, L]
        
        # Transpose to [B, L, H]
        x = x.permute(0, 2, 1)
        
        return x


class ResidualConvBlock(nn.Module):
    """Residual 1D convolution block."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 11,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)
        
        out = out + residual
        out = F.relu(out)
        
        return out


class DeltaComparisonMetrics:
    """
    Metrics for comparing base model and meta model delta tensors.
    
    Handles the comparison of [L, 2] delta tensors.
    """
    
    @staticmethod
    def compare_deltas(
        base_delta: np.ndarray,
        meta_delta: np.ndarray,
        threshold: float = 0.1
    ) -> Dict:
        """
        Compare base model and meta model delta tensors.
        
        Parameters
        ----------
        base_delta : np.ndarray
            Base model deltas [L, 2]
        meta_delta : np.ndarray
            Meta model deltas [L, 2]
        threshold : float
            Detection threshold
        
        Returns
        -------
        dict
            Comparison metrics
        """
        # Ensure same shape
        L = min(len(base_delta), len(meta_delta))
        base_delta = base_delta[:L]
        meta_delta = meta_delta[:L]
        
        # Max absolute deltas
        base_max_donor = np.abs(base_delta[:, 0]).max()
        base_max_acceptor = np.abs(base_delta[:, 1]).max()
        meta_max_donor = np.abs(meta_delta[:, 0]).max()
        meta_max_acceptor = np.abs(meta_delta[:, 1]).max()
        
        # Positions of max deltas
        base_pos_donor = np.abs(base_delta[:, 0]).argmax()
        base_pos_acceptor = np.abs(base_delta[:, 1]).argmax()
        meta_pos_donor = np.abs(meta_delta[:, 0]).argmax()
        meta_pos_acceptor = np.abs(meta_delta[:, 1]).argmax()
        
        # Detection (any position above threshold)
        base_detected = (np.abs(base_delta).max() > threshold)
        meta_detected = (np.abs(meta_delta).max() > threshold)
        
        # Correlation
        from scipy.stats import pearsonr
        corr_donor, _ = pearsonr(base_delta[:, 0], meta_delta[:, 0])
        corr_acceptor, _ = pearsonr(base_delta[:, 1], meta_delta[:, 1])
        
        # Effect types
        effects = DeltaComparisonMetrics._classify_effects(
            base_delta, meta_delta, threshold
        )
        
        return {
            'base_max_donor': float(base_max_donor),
            'base_max_acceptor': float(base_max_acceptor),
            'meta_max_donor': float(meta_max_donor),
            'meta_max_acceptor': float(meta_max_acceptor),
            'base_detected': base_detected,
            'meta_detected': meta_detected,
            'correlation_donor': float(corr_donor),
            'correlation_acceptor': float(corr_acceptor),
            'position_match_donor': (base_pos_donor == meta_pos_donor),
            'position_match_acceptor': (base_pos_acceptor == meta_pos_acceptor),
            'effects': effects
        }
    
    @staticmethod
    def _classify_effects(
        base_delta: np.ndarray,
        meta_delta: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        Classify the type of splice effect.
        
        Returns counts of:
        - suppressed: Negative delta (splice site weakened)
        - induced: Positive delta (new splice site created)
        - unchanged: Small delta
        """
        def count_effects(delta):
            suppressed = (delta < -threshold).sum()
            induced = (delta > threshold).sum()
            unchanged = ((np.abs(delta) <= threshold)).sum()
            return {'suppressed': int(suppressed), 'induced': int(induced), 'unchanged': int(unchanged)}
        
        return {
            'base_donor': count_effects(base_delta[:, 0]),
            'base_acceptor': count_effects(base_delta[:, 1]),
            'meta_donor': count_effects(meta_delta[:, 0]),
            'meta_acceptor': count_effects(meta_delta[:, 1])
        }


def create_delta_predictor_v2(
    hidden_dim: int = 64,
    n_layers: int = 4,
    dropout: float = 0.1
) -> DeltaPredictorV2:
    """Create a DeltaPredictorV2 model."""
    return DeltaPredictorV2(
        hidden_dim=hidden_dim,
        n_conv_layers=n_layers,
        dropout=dropout
    )









