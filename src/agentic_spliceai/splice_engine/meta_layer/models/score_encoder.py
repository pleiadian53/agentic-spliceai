"""
Score encoder for base model features.

Encodes the 50+ numeric features from base model predictions
into a fixed-dimensional embedding.

Ported from: meta_spliceai/splice_engine/meta_layer/models/score_encoder.py
"""

import torch
import torch.nn as nn
from typing import Optional


class ScoreEncoder(nn.Module):
    """
    MLP encoder for base model score features.
    
    Transforms the numeric features (base scores, context scores,
    derived features) into a fixed-dimensional embedding.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (typically 50+).
    hidden_dim : int
        Hidden dimension and output dimension.
    num_layers : int
        Number of hidden layers.
    dropout : float
        Dropout probability.
    use_batch_norm : bool
        Whether to use batch normalization.
    
    Examples
    --------
    >>> encoder = ScoreEncoder(input_dim=50, hidden_dim=256)
    >>> features = torch.randn(32, 50)  # [batch, features]
    >>> embeddings = encoder(features)   # [batch, 256]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        self.encoder = nn.Sequential(*layers)
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode score features.
        
        Parameters
        ----------
        features : torch.Tensor
            Input features of shape [batch, input_dim].
        
        Returns
        -------
        torch.Tensor
            Encoded features of shape [batch, hidden_dim].
        """
        encoded = self.encoder(features)
        return self.output_norm(encoded)


class AttentiveScoreEncoder(nn.Module):
    """
    Score encoder with self-attention over feature groups.
    
    Groups features into categories (base scores, context, etc.)
    and applies attention to learn feature importance.
    
    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dim : int
        Hidden/output dimension.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout probability.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project each feature to hidden_dim
        self.feature_proj = nn.Linear(1, hidden_dim)
        
        # Self-attention over features
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Pooling
        self.pool = nn.Linear(input_dim, 1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features with attention.
        
        Parameters
        ----------
        features : torch.Tensor
            Input of shape [batch, input_dim].
        
        Returns
        -------
        torch.Tensor
            Output of shape [batch, hidden_dim].
        """
        batch_size = features.shape[0]
        
        # Expand features: [batch, input_dim] -> [batch, input_dim, 1]
        x = features.unsqueeze(-1)
        
        # Project: [batch, input_dim, hidden_dim]
        x = self.feature_proj(x)
        
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        # Pool over features: [batch, input_dim, hidden_dim] -> [batch, hidden_dim]
        # Use attention-weighted pooling
        weights = self.pool(x.transpose(1, 2)).squeeze(-1)  # [batch, hidden_dim]
        x = x.mean(dim=1)  # Simple mean pooling
        
        return x









