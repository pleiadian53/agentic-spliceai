"""
MetaSpliceModel V2: Sequence-to-Sequence Architecture

This version outputs [L, 3] per-nucleotide predictions, matching the base model format.
The meta-layer recalibrates/refines base model predictions while maintaining
the same output format for direct comparison.

Output Format (same as base model):
    [L, 3] where:
    - L = sequence length (after context trimming)
    - 3 = [donor_prob, acceptor_prob, neither_prob]

This enables:
    base_delta = base_model(alt) - base_model(ref)   # [L, 3]
    meta_delta = meta_model(alt) - meta_model(ref)   # [L, 3] ← Same format!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MetaSpliceModelV2(nn.Module):
    """
    Sequence-to-Sequence Meta-Layer for Splice Site Prediction.
    
    Takes base model predictions as input and outputs refined predictions
    at EVERY position, matching the base model's [L, 3] output format.
    
    Architecture:
        1. Base model scores [L, 3] as input features
        2. Sequence encoder (1D CNN) processes DNA context
        3. Feature fusion at each position
        4. Output refined [L, 3] predictions
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for CNN channels
    n_conv_layers : int
        Number of convolutional layers
    kernel_size : int
        Convolution kernel size (should be odd)
    dropout : float
        Dropout probability
    
    Examples
    --------
    >>> model = MetaSpliceModelV2(hidden_dim=32)
    >>> seq = torch.randn(1, 4, 10000)
    >>> base_scores = torch.randn(1, 10000, 3)
    >>> output = model(seq, base_scores)
    >>> print(output.shape)  # [1, 10000, 3]
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        n_conv_layers: int = 4,
        kernel_size: int = 11,
        dropout: float = 0.1,
        use_residual: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        
        # Sequence encoder: 1D CNN
        self.seq_conv_in = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Stacked convolutions with residual connections
        self.conv_layers = nn.ModuleList()
        for i in range(n_conv_layers):
            self.conv_layers.append(
                ResidualConvBlock(
                    hidden_dim, 
                    kernel_size=kernel_size,
                    dilation=1,
                    dropout=dropout
                )
            )
        
        # Base score encoder
        self.score_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Fusion: combine sequence features and score features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output head: [B, L, hidden_dim] -> [B, L, 3]
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3)
        )
        
        # Optional: learnable residual weight for base scores
        if use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self, 
        sequence: torch.Tensor, 
        base_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        sequence : torch.Tensor
            One-hot encoded DNA sequence [B, 4, L]
        base_scores : torch.Tensor
            Base model predictions [B, L, 3]
        
        Returns
        -------
        torch.Tensor
            Refined predictions [B, L, 3] (same format as base model)
        """
        B, _, L = sequence.shape
        
        # Encode sequence
        seq_features = self.seq_conv_in(sequence)  # [B, hidden, L]
        
        for conv_layer in self.conv_layers:
            seq_features = conv_layer(seq_features)  # [B, hidden, L]
        
        # Transpose for linear layers: [B, L, hidden]
        seq_features = seq_features.permute(0, 2, 1)
        
        # Encode base scores: [B, L, 3] -> [B, L, hidden]
        score_features = self.score_encoder(base_scores)
        
        # Fuse features: [B, L, hidden*2] -> [B, L, hidden]
        combined = torch.cat([seq_features, score_features], dim=-1)
        fused = self.fusion(combined)
        
        # Output: [B, L, 3]
        logits = self.output_head(fused)
        
        # Optional residual connection with base scores
        if self.use_residual:
            meta_probs = F.softmax(logits, dim=-1)
            base_probs = F.softmax(base_scores, dim=-1)
            blended = self.residual_weight * meta_probs + (1 - self.residual_weight) * base_probs
            return blended
        else:
            return F.softmax(logits, dim=-1)
    
    def predict_with_delta(
        self,
        ref_sequence: torch.Tensor,
        alt_sequence: torch.Tensor,
        ref_base_scores: torch.Tensor,
        alt_base_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict and compute delta scores for variant analysis.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - ref_probs: Reference predictions [B, L, 3]
            - alt_probs: Alternate predictions [B, L, 3]
            - delta: alt_probs - ref_probs [B, L, 3]
        """
        ref_probs = self.forward(ref_sequence, ref_base_scores)
        alt_probs = self.forward(alt_sequence, alt_base_scores)
        delta = alt_probs - ref_probs
        
        return ref_probs, alt_probs, delta


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
        
        out = out + residual  # Residual connection
        out = F.relu(out)
        
        return out


def create_meta_model_v2(
    hidden_dim: int = 32,
    n_layers: int = 4,
    dropout: float = 0.1
) -> MetaSpliceModelV2:
    """
    Create a MetaSpliceModelV2 instance.
    
    This is a lightweight model designed to refine base model predictions
    while outputting the same [L, 3] format.
    """
    return MetaSpliceModelV2(
        hidden_dim=hidden_dim,
        n_conv_layers=n_layers,
        dropout=dropout,
        use_residual=True
    )












