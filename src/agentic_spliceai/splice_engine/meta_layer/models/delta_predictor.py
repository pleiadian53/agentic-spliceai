"""
Delta Predictor: Siamese Network for Variant Effect Prediction.

Predicts splice site score changes (deltas) caused by genetic variants.
Uses a Siamese architecture with shared encoder for reference and alternate sequences.

Key Design:
- Shared encoder ensures model learns the DIFFERENCE, not just sequences
- Direct delta output enables comparison with base model
- Variant-aware training on SpliceVarDB data

Architecture:
    ref_seq ──→ [Shared Encoder] ──→ ref_embed ─┐
                                                 ├──→ diff = alt - ref ──→ [Delta Head] ──→ [Δ_donor, Δ_acceptor]
    alt_seq ──→ [Shared Encoder] ──→ alt_embed ─┘

Usage:
    from agentic_spliceai.splice_engine.meta_layer.models.delta_predictor import DeltaPredictor
    
    model = DeltaPredictor(hidden_dim=256)
    
    # Forward pass
    delta = model(ref_seq, alt_seq)  # [B, 2] = [Δ_donor, Δ_acceptor]

Ported from: meta_spliceai/splice_engine/meta_layer/models/delta_predictor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class DeltaPredictor(nn.Module):
    """
    Siamese network for predicting splice site score changes.
    
    Takes reference and alternate sequences as input, outputs predicted
    delta scores [Δ_donor, Δ_acceptor].
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for encoder and delta head
    n_conv_layers : int
        Number of convolutional layers in encoder
    kernel_size : int
        Convolution kernel size
    dropout : float
        Dropout probability
    use_attention : bool
        Use self-attention in encoder
    
    Examples
    --------
    >>> model = DeltaPredictor(hidden_dim=256)
    >>> ref_seq = torch.randn(32, 4, 501)  # [B, 4, L]
    >>> alt_seq = torch.randn(32, 4, 501)
    >>> delta = model(ref_seq, alt_seq)
    >>> print(delta.shape)  # [32, 2]
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        n_conv_layers: int = 4,
        kernel_size: int = 11,
        dropout: float = 0.1,
        use_attention: bool = False
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Shared sequence encoder
        self.encoder = SequenceEncoder(
            input_channels=4,
            hidden_dim=hidden_dim,
            n_layers=n_conv_layers,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Optional: Self-attention layer
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Delta prediction head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # [Δ_donor, Δ_acceptor]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
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
            Reference sequence [B, 4, L] (one-hot encoded)
        alt_seq : torch.Tensor
            Alternate sequence [B, 4, L] (one-hot encoded)
        
        Returns
        -------
        torch.Tensor
            Delta scores [B, 2] = [Δ_donor, Δ_acceptor]
        """
        # Encode both sequences with SHARED encoder
        ref_embed = self.encoder(ref_seq)  # [B, hidden_dim]
        alt_embed = self.encoder(alt_seq)  # [B, hidden_dim]
        
        # Compute difference embedding
        diff_embed = alt_embed - ref_embed  # [B, hidden_dim]
        
        # Predict delta scores
        delta = self.delta_head(diff_embed)  # [B, 2]
        
        return delta
    
    def predict_with_embeddings(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with intermediate embeddings (for analysis).
        
        Returns
        -------
        Tuple
            (delta, ref_embed, alt_embed, diff_embed)
        """
        ref_embed = self.encoder(ref_seq)
        alt_embed = self.encoder(alt_seq)
        diff_embed = alt_embed - ref_embed
        delta = self.delta_head(diff_embed)
        
        return delta, ref_embed, alt_embed, diff_embed


class SequenceEncoder(nn.Module):
    """
    CNN-based sequence encoder.
    
    Encodes a DNA sequence to a fixed-size embedding.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        kernel_size: int = 11,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initial projection
        self.input_proj = nn.Conv1d(input_channels, hidden_dim // 2, kernel_size=1)
        
        # Convolutional layers with increasing dilation
        layers = []
        current_dim = hidden_dim // 2
        
        for i in range(n_layers):
            out_dim = hidden_dim if i == n_layers - 1 else hidden_dim // 2
            dilation = 2 ** i  # Exponentially increasing dilation
            padding = (kernel_size - 1) * dilation // 2
            
            layers.append(nn.Conv1d(
                current_dim, out_dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            current_dim = out_dim
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global pooling + linear projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence to embedding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Sequence embedding [B, hidden_dim]
        """
        # Initial projection
        x = self.input_proj(x)  # [B, hidden/2, L]
        
        # Convolutional layers
        x = self.conv_layers(x)  # [B, hidden, L]
        
        # Global average pooling
        x = self.global_pool(x)  # [B, hidden, 1]
        x = x.squeeze(-1)  # [B, hidden]
        
        # Output projection
        x = self.output_proj(x)  # [B, hidden]
        
        return x


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss for variant delta prediction.
    
    Applies higher weight to splice-altering variants.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Parameters
        ----------
        predicted : torch.Tensor
            Predicted deltas [B, 2]
        target : torch.Tensor
            Target deltas [B, 2]
        weights : torch.Tensor
            Sample weights [B]
        
        Returns
        -------
        torch.Tensor
            Weighted MSE loss (scalar)
        """
        # Compute per-sample MSE
        mse = F.mse_loss(predicted, target, reduction='none')  # [B, 2]
        mse = mse.mean(dim=1)  # [B]
        
        # Apply weights
        weighted_mse = mse * weights
        
        # Average
        return weighted_mse.mean()


class DeltaPredictorWithClassifier(nn.Module):
    """
    Extended delta predictor with classification head.
    
    Multi-task model that predicts both:
    1. Delta scores (regression)
    2. Variant classification (Splice-altering vs Normal)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        n_conv_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Base delta predictor
        self.delta_predictor = DeltaPredictor(
            hidden_dim=hidden_dim,
            n_conv_layers=n_conv_layers,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # [Normal, Splice-altering]
        )
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns
        -------
        Tuple
            (delta [B, 2], class_logits [B, 2])
        """
        # Get delta and embeddings
        delta, ref_embed, alt_embed, diff_embed = \
            self.delta_predictor.predict_with_embeddings(ref_seq, alt_seq)
        
        # Classify based on difference embedding
        class_logits = self.classifier(diff_embed)
        
        return delta, class_logits


def create_delta_predictor(
    hidden_dim: int = 256,
    n_layers: int = 4,
    dropout: float = 0.1,
    with_classifier: bool = False
) -> nn.Module:
    """
    Create a delta predictor model.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of conv layers
    dropout : float
        Dropout rate
    with_classifier : bool
        Include classification head
    
    Returns
    -------
    nn.Module
        Delta predictor model
    """
    if with_classifier:
        return DeltaPredictorWithClassifier(
            hidden_dim=hidden_dim,
            n_conv_layers=n_layers,
            dropout=dropout
        )
    else:
        return DeltaPredictor(
            hidden_dim=hidden_dim,
            n_conv_layers=n_layers,
            dropout=dropout
        )









