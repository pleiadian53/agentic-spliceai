"""
HyenaDNA-based Delta Predictor for Variant Effect Prediction.

Uses HyenaDNA pre-trained encoder for DNA sequence understanding,
which should provide much better representations than training from scratch.

HyenaDNA Benefits:
- Pre-trained on large DNA corpus
- Understands genomic patterns
- O(n) complexity (efficient for long sequences)
- Strong performance on genomic benchmarks

Usage:
    from agentic_spliceai.splice_engine.meta_layer.models.hyenadna_delta_predictor import (
        HyenaDNADeltaPredictor, create_hyenadna_delta_predictor
    )
    
    model = create_hyenadna_delta_predictor()
    delta = model(ref_seq, alt_seq)  # [B, L, 2]

Note: Requires HyenaDNA to be installed:
    pip install hyena-dna
    # or
    git clone https://github.com/HazyResearch/hyena-dna

Ported from: meta_spliceai/splice_engine/meta_layer/models/hyenadna_delta_predictor.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


class HyenaDNADeltaPredictor(nn.Module):
    """
    Delta predictor using HyenaDNA as the sequence encoder.
    
    Architecture:
        ref_seq ──→ [HyenaDNA Encoder] ──→ ref_features [B, L, H]
        alt_seq ──→ [HyenaDNA Encoder] ──→ alt_features [B, L, H]
                                              ↓
                               diff = alt_features - ref_features
                                              ↓
                                    [Per-Position Head]
                                              ↓
                                   Output: [B, L, 2]
    
    Parameters
    ----------
    model_name : str
        HyenaDNA model variant: 'hyenadna-tiny-1k', 'hyenadna-small-32k', etc.
    hidden_dim : int
        Hidden dimension for delta head
    freeze_encoder : bool
        Whether to freeze HyenaDNA weights
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        model_name: str = 'hyenadna-tiny-1k-seqlen',
        hidden_dim: int = 128,
        freeze_encoder: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        
        # Load HyenaDNA encoder
        self.encoder, self.encoder_dim = self._load_hyenadna(model_name)
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("HyenaDNA encoder frozen")
        
        # Projection from HyenaDNA dim to hidden_dim
        self.proj = nn.Linear(self.encoder_dim, hidden_dim)
        
        # Per-position delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [Δ_donor, Δ_acceptor]
        )
        
        logger.info(f"HyenaDNADeltaPredictor initialized with {model_name}")
    
    def _load_hyenadna(self, model_name: str):
        """Load HyenaDNA model."""
        try:
            # Try importing HyenaDNA
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # HuggingFace model path
            hf_path = f"LongSafari/{model_name}"
            
            logger.info(f"Loading HyenaDNA from {hf_path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True
            )
            
            # Get embedding dimension
            if hasattr(model.config, 'd_model'):
                embed_dim = model.config.d_model
            elif hasattr(model.config, 'hidden_size'):
                embed_dim = model.config.hidden_size
            else:
                embed_dim = 256  # Default for tiny model
            
            return model, embed_dim
            
        except ImportError:
            logger.warning("HyenaDNA not available, using fallback CNN encoder")
            return self._create_fallback_encoder(), 256
        except Exception as e:
            logger.warning(f"Could not load HyenaDNA: {e}. Using fallback.")
            return self._create_fallback_encoder(), 256
    
    def _create_fallback_encoder(self):
        """Create a CNN fallback if HyenaDNA is not available."""
        return nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.ReLU()
        )
    
    def encode_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode a DNA sequence using HyenaDNA.
        
        Parameters
        ----------
        seq : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Sequence features [B, L, H]
        """
        # Check if using fallback CNN
        if isinstance(self.encoder, nn.Sequential):
            # CNN fallback: [B, 4, L] -> [B, 256, L] -> [B, L, 256]
            features = self.encoder(seq)
            features = features.permute(0, 2, 1)
        else:
            # HyenaDNA: needs token IDs
            # Convert one-hot to token IDs (A=0, C=1, G=2, T=3)
            token_ids = seq.argmax(dim=1)  # [B, L]
            
            with torch.set_grad_enabled(not self.freeze_encoder):
                outputs = self.encoder(token_ids, output_hidden_states=True)
                
                # Get last hidden state
                if hasattr(outputs, 'hidden_states'):
                    features = outputs.hidden_states[-1]  # [B, L, H]
                else:
                    features = outputs.logits  # Fallback
        
        return features
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict per-position delta scores.
        
        Parameters
        ----------
        ref_seq : torch.Tensor
            Reference sequence [B, 4, L]
        alt_seq : torch.Tensor
            Alternate sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Delta scores [B, L, 2]
        """
        # Encode both sequences
        ref_features = self.encode_sequence(ref_seq)  # [B, L, H_enc]
        alt_features = self.encode_sequence(alt_seq)  # [B, L, H_enc]
        
        # Project to hidden dim
        ref_proj = self.proj(ref_features)  # [B, L, hidden]
        alt_proj = self.proj(alt_features)  # [B, L, hidden]
        
        # Compute difference
        diff = alt_proj - ref_proj  # [B, L, hidden]
        
        # Predict deltas
        delta = self.delta_head(diff)  # [B, L, 2]
        
        return delta


class SimpleCNNDeltaPredictor(nn.Module):
    """
    Simplified CNN-based delta predictor as a baseline.
    
    Uses a deeper, more carefully designed CNN without HyenaDNA.
    This is the BEST PERFORMING model from Phase 2 experiments (r=0.38).
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
        
        # Initial embedding
        self.embed = nn.Conv1d(4, hidden_dim, kernel_size=1)
        
        # Residual blocks with increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 3)  # 1, 2, 4, 1, 2, 4
            self.blocks.append(
                DilatedResidualBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence to features."""
        x = self.embed(x)  # [B, H, L]
        
        for block in self.blocks:
            x = block(x)
        
        return x.permute(0, 2, 1)  # [B, L, H]
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        ref_feat = self.encode(ref_seq)  # [B, L, H]
        alt_feat = self.encode(alt_seq)  # [B, L, H]
        
        diff = alt_feat - ref_feat
        delta = self.delta_head(diff)  # [B, L, 2]
        
        return delta


class DilatedResidualBlock(nn.Module):
    """Residual block with dilated convolutions."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 15,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # First conv
        out = self.conv1(x)  # [B, C, L]
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm1(out)
        out = F.gelu(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        # Second conv
        out = self.conv2(out)
        out = out.permute(0, 2, 1)
        out = self.norm2(out)
        out = self.dropout(out)
        
        # Gating
        gate = self.gate(out)
        out = out * gate
        
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        # Residual connection
        return out + residual


def create_hyenadna_delta_predictor(
    model_name: str = 'hyenadna-tiny-1k-seqlen',
    hidden_dim: int = 128,
    freeze_encoder: bool = True,
    use_fallback: bool = False
) -> nn.Module:
    """
    Create a delta predictor with HyenaDNA encoder.
    
    Parameters
    ----------
    model_name : str
        HyenaDNA model name
    hidden_dim : int
        Hidden dimension
    freeze_encoder : bool
        Freeze HyenaDNA weights
    use_fallback : bool
        Force use of CNN fallback
    
    Returns
    -------
    nn.Module
        Delta predictor model
    """
    if use_fallback:
        logger.info("Using SimpleCNN fallback (HyenaDNA not requested)")
        return SimpleCNNDeltaPredictor(hidden_dim=hidden_dim)
    
    return HyenaDNADeltaPredictor(
        model_name=model_name,
        hidden_dim=hidden_dim,
        freeze_encoder=freeze_encoder
    )









