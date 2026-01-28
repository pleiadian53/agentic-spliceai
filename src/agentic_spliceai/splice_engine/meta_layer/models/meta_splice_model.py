"""
Main multimodal meta-layer model.

Combines sequence and score encodings for splice site prediction.

Ported from: meta_spliceai/splice_engine/meta_layer/models/meta_splice_model.py
"""

import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequence_encoder import SequenceEncoderFactory, CNNEncoder
from .score_encoder import ScoreEncoder

logger = logging.getLogger(__name__)


class MetaSpliceModel(nn.Module):
    """
    Multimodal meta-layer model for splice site recalibration.
    
    Combines:
    1. Sequence embeddings (from DNA language model or CNN)
    2. Score embeddings (from base model features)
    
    Using cross-modal fusion to produce recalibrated predictions.
    
    Parameters
    ----------
    sequence_encoder : str
        Type of sequence encoder: 'hyenadna', 'cnn', 'none'.
    num_score_features : int
        Number of base model score features.
    hidden_dim : int
        Hidden dimension for all components.
    num_classes : int
        Number of output classes (3: donor, acceptor, neither).
    dropout : float
        Dropout probability.
    fusion_type : str
        Fusion method: 'concat', 'add', 'cross_attention'.
    sequence_encoder_config : dict, optional
        Additional config for sequence encoder.
    
    Examples
    --------
    >>> model = MetaSpliceModel(
    ...     sequence_encoder='cnn',
    ...     num_score_features=50,
    ...     hidden_dim=256
    ... )
    >>> 
    >>> sequence = torch.randn(32, 4, 501)  # One-hot encoded
    >>> features = torch.randn(32, 50)       # Score features
    >>> logits = model(sequence, features)   # [32, 3]
    """
    
    def __init__(
        self,
        sequence_encoder: str = 'cnn',
        num_score_features: int = 50,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1,
        fusion_type: str = 'concat',
        sequence_encoder_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # Sequence encoder
        seq_config = sequence_encoder_config or {}
        self.seq_encoder = SequenceEncoderFactory.create(
            sequence_encoder,
            output_dim=hidden_dim,
            **seq_config
        )
        
        # Score encoder
        self.score_encoder = ScoreEncoder(
            input_dim=num_score_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Fusion layer
        if fusion_type == 'concat':
            fusion_dim = hidden_dim * 2
            self.fusion = nn.Identity()
        elif fusion_type == 'add':
            fusion_dim = hidden_dim
            self.fusion = nn.Identity()
        elif fusion_type == 'cross_attention':
            fusion_dim = hidden_dim * 2
            self.fusion = CrossAttentionFusion(hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        logger.info(f"MetaSpliceModel initialized:")
        logger.info(f"  Sequence encoder: {sequence_encoder}")
        logger.info(f"  Score features: {num_score_features}")
        logger.info(f"  Hidden dim: {hidden_dim}")
        logger.info(f"  Fusion: {fusion_type}")
    
    def forward(
        self,
        sequence: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        sequence : torch.Tensor
            DNA sequence (one-hot or tokens) [batch, ...].
        features : torch.Tensor
            Score features [batch, num_features].
        
        Returns
        -------
        torch.Tensor
            Logits [batch, num_classes].
        """
        # Encode sequence
        seq_emb = self.seq_encoder(sequence)  # [batch, hidden_dim]
        
        # Encode scores
        score_emb = self.score_encoder(features)  # [batch, hidden_dim]
        
        # Fuse
        if self.fusion_type == 'concat':
            fused = torch.cat([seq_emb, score_emb], dim=-1)
        elif self.fusion_type == 'add':
            fused = seq_emb + score_emb
        elif self.fusion_type == 'cross_attention':
            fused = self.fusion(seq_emb, score_emb)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def predict_proba(
        self,
        sequence: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Returns
        -------
        torch.Tensor
            Probabilities [batch, num_classes].
        """
        logits = self.forward(sequence, features)
        return F.softmax(logits, dim=-1)
    
    def get_embeddings(
        self,
        sequence: torch.Tensor,
        features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get intermediate embeddings for analysis.
        
        Returns
        -------
        tuple
            (seq_emb, score_emb, fused_emb)
        """
        seq_emb = self.seq_encoder(sequence)
        score_emb = self.score_encoder(features)
        
        if self.fusion_type == 'concat':
            fused = torch.cat([seq_emb, score_emb], dim=-1)
        elif self.fusion_type == 'add':
            fused = seq_emb + score_emb
        elif self.fusion_type == 'cross_attention':
            fused = self.fusion(seq_emb, score_emb)
        
        return seq_emb, score_emb, fused


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion for sequence and score embeddings.
    
    Allows the model to learn which aspects of the sequence
    are relevant given the score features, and vice versa.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        # Sequence attends to scores
        self.seq_to_score_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Scores attend to sequence
        self.score_to_seq_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        seq_emb: torch.Tensor,
        score_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse embeddings via cross-attention.
        
        Parameters
        ----------
        seq_emb : torch.Tensor
            Sequence embeddings [batch, hidden_dim].
        score_emb : torch.Tensor
            Score embeddings [batch, hidden_dim].
        
        Returns
        -------
        torch.Tensor
            Fused embeddings [batch, hidden_dim * 2].
        """
        # Add sequence dimension for attention
        seq_emb = seq_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        score_emb = score_emb.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention
        seq_enhanced, _ = self.seq_to_score_attn(seq_emb, score_emb, score_emb)
        score_enhanced, _ = self.score_to_seq_attn(score_emb, seq_emb, seq_emb)
        
        # Residual + norm
        seq_out = self.norm1(seq_emb + seq_enhanced).squeeze(1)
        score_out = self.norm2(score_emb + score_enhanced).squeeze(1)
        
        # Concatenate
        return torch.cat([seq_out, score_out], dim=-1)


class ScoreOnlyModel(nn.Module):
    """
    Score-only baseline model (no sequence input).
    
    Used for ablation studies to measure the contribution
    of sequence information.
    """
    
    def __init__(
        self,
        num_score_features: int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = ScoreEncoder(
            input_dim=num_score_features,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass (ignores sequence)."""
        emb = self.encoder(features)
        return self.classifier(emb)












