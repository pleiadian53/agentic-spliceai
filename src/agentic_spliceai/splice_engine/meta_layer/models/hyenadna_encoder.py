"""
HyenaDNA Encoder Integration for Delta Prediction.

HyenaDNA is a State Space Model-based DNA language model with:
- Pre-trained on human genome
- O(n) memory and speed (efficient for long sequences)
- Strong performance on genomic benchmarks

This module provides:
1. HyenaDNA encoder wrapper (GPU with sufficient VRAM)
2. Lightweight fallback (M1 Mac / CPU)
3. Configuration for different compute environments

Model variants (from HuggingFace LongSafari):
- hyenadna-tiny-1k-seqlen:   ~1.6M params, 1024 context
- hyenadna-small-32k-seqlen: ~7M params, 32k context
- hyenadna-medium-160k-seqlen: ~42M params, 160k context
- hyenadna-medium-450k-seqlen: ~42M params, 450k context
- hyenadna-large-1m-seqlen:  ~171M params, 1M context

Usage:
    # Local M1 Mac (fallback to lightweight CNN)
    encoder = create_encoder(use_hyenadna=False)
    
    # GPU server (load HyenaDNA)
    encoder = create_encoder(use_hyenadna=True, model_name='hyenadna-tiny-1k-seqlen')

Ported from: meta_spliceai/splice_engine/meta_layer/models/hyenadna_encoder.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class HyenaDNAEncoder(nn.Module):
    """
    HyenaDNA-based sequence encoder.
    
    Loads pre-trained HyenaDNA from HuggingFace and provides
    per-position embeddings for DNA sequences.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model identifier
    freeze : bool
        Whether to freeze pre-trained weights
    device : str, optional
        Target device
    max_length : int
        Maximum sequence length
    """
    
    MODELS = {
        'tiny': 'LongSafari/hyenadna-tiny-1k-seqlen',
        'small': 'LongSafari/hyenadna-small-32k-seqlen', 
        'medium': 'LongSafari/hyenadna-medium-160k-seqlen',
        'large': 'LongSafari/hyenadna-large-1m-seqlen'
    }
    
    def __init__(
        self,
        model_name: str = 'tiny',
        freeze: bool = True,
        device: Optional[str] = None,
        max_length: int = 1024
    ):
        super().__init__()
        
        self.freeze = freeze
        self.max_length = max_length
        
        # Resolve model name
        if model_name in self.MODELS:
            hf_path = self.MODELS[model_name]
        else:
            hf_path = model_name
        
        self.model_name = model_name
        self.hf_path = hf_path
        
        # Load model
        self.model, self.embed_dim = self._load_model(hf_path)
        
        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Move to device
        if device:
            self.to(device)
        
        logger.info(f"HyenaDNA encoder loaded: {model_name} (dim={self.embed_dim})")
    
    def _load_model(self, hf_path: str) -> Tuple[Optional[nn.Module], int]:
        """Load HyenaDNA model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM
            
            logger.info(f"Loading HyenaDNA from {hf_path}")
            
            model = AutoModelForCausalLM.from_pretrained(
                hf_path,
                trust_remote_code=True,
                torch_dtype=torch.float16  # Use half precision for memory
            )
            
            # Get embedding dimension
            if hasattr(model.config, 'd_model'):
                embed_dim = model.config.d_model
            elif hasattr(model.config, 'hidden_size'):
                embed_dim = model.config.hidden_size
            else:
                # Default dims for known models
                dims = {'tiny': 128, 'small': 256, 'medium': 256, 'large': 256}
                embed_dim = dims.get(self.model_name, 256)
            
            return model, embed_dim
            
        except ImportError as e:
            logger.error(f"HyenaDNA requires: pip install transformers")
            raise ImportError(
                "HyenaDNA requires transformers library. "
                "Install with: pip install transformers"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load HyenaDNA: {e}")
            raise
    
    def _seq_to_tokens(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Convert one-hot encoded sequence to token IDs.
        
        Parameters
        ----------
        seq : torch.Tensor
            One-hot encoded [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Token IDs [B, L]
        """
        # One-hot to indices: A=0, C=1, G=2, T=3
        return seq.argmax(dim=1).long()
    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode DNA sequence.
        
        Parameters
        ----------
        seq : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Per-position embeddings [B, L, embed_dim]
        """
        # Convert to token IDs
        token_ids = self._seq_to_tokens(seq)  # [B, L]
        
        # Truncate if needed
        if token_ids.shape[1] > self.max_length:
            token_ids = token_ids[:, :self.max_length]
        
        # Forward through HyenaDNA
        with torch.set_grad_enabled(not self.freeze):
            outputs = self.model(
                token_ids,
                output_hidden_states=True
            )
            
            # Get last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                embeddings = outputs.hidden_states[-1]  # [B, L, H]
            else:
                # Fallback
                embeddings = outputs.logits  # [B, L, vocab]
        
        return embeddings


class LightweightEncoder(nn.Module):
    """
    Lightweight CNN encoder for M1 Mac / CPU environments.
    
    Uses dilated convolutions with gating for efficient
    long-range dependency modeling.
    
    Parameters
    ----------
    embed_dim : int
        Output embedding dimension
    n_layers : int
        Number of convolutional layers
    kernel_size : int
        Convolution kernel size
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        n_layers: int = 6,
        kernel_size: int = 15,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Initial projection
        self.embed = nn.Conv1d(4, embed_dim, kernel_size=1)
        
        # Dilated residual blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8, ...
            self.blocks.append(
                GatedResidualBlock(
                    channels=embed_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
    
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Encode DNA sequence.
        
        Parameters
        ----------
        seq : torch.Tensor
            One-hot encoded sequence [B, 4, L]
        
        Returns
        -------
        torch.Tensor
            Per-position embeddings [B, L, embed_dim]
        """
        # Initial embedding
        x = self.embed(seq)  # [B, embed_dim, L]
        
        # Apply blocks
        for block in self.blocks:
            x = block(x)
        
        # Transpose to [B, L, embed_dim]
        return x.permute(0, 2, 1)


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
        
        self.conv1 = nn.Conv1d(
            channels, channels * 2,  # 2x for gating
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding
        )
        
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Gated convolution
        out = self.conv1(x)  # [B, 2*C, L]
        out, gate = out.chunk(2, dim=1)  # Each [B, C, L]
        out = out * torch.sigmoid(gate)
        
        # Normalize
        out = out.permute(0, 2, 1)  # [B, L, C]
        out = self.norm(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)  # [B, C, L]
        
        # Residual
        return out + residual


class DeltaPredictorWithEncoder(nn.Module):
    """
    Delta predictor using a configurable encoder.
    
    Works with either HyenaDNA or lightweight CNN encoder.
    
    Parameters
    ----------
    encoder : nn.Module
        Sequence encoder
    hidden_dim : int
        Hidden dimension for delta head
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = encoder
        
        # Get encoder output dimension
        if hasattr(encoder, 'embed_dim'):
            encoder_dim = encoder.embed_dim
        else:
            encoder_dim = hidden_dim
        
        # Projection if needed
        if encoder_dim != hidden_dim:
            self.proj = nn.Linear(encoder_dim, hidden_dim)
        else:
            self.proj = nn.Identity()
        
        # Per-position delta head
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # [Δ_donor, Δ_acceptor]
        )
    
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
        ref_embed = self.encoder(ref_seq)  # [B, L, H_enc]
        alt_embed = self.encoder(alt_seq)  # [B, L, H_enc]
        
        # Project
        ref_proj = self.proj(ref_embed)  # [B, L, hidden]
        alt_proj = self.proj(alt_embed)
        
        # Difference
        diff = alt_proj - ref_proj
        
        # Predict deltas
        delta = self.delta_head(diff)  # [B, L, 2]
        
        return delta


def create_encoder(
    use_hyenadna: bool = False,
    model_name: str = 'tiny',
    hidden_dim: int = 128,
    freeze: bool = True,
    device: Optional[str] = None
) -> nn.Module:
    """
    Create sequence encoder based on compute environment.
    
    Parameters
    ----------
    use_hyenadna : bool
        Whether to use HyenaDNA (requires GPU + sufficient VRAM)
    model_name : str
        HyenaDNA model variant (tiny/small/medium/large)
    hidden_dim : int
        Hidden dimension for lightweight encoder
    freeze : bool
        Freeze HyenaDNA weights
    device : str, optional
        Target device
    
    Returns
    -------
    nn.Module
        Sequence encoder
    """
    if use_hyenadna:
        return HyenaDNAEncoder(
            model_name=model_name,
            freeze=freeze,
            device=device
        )
    else:
        return LightweightEncoder(
            embed_dim=hidden_dim,
            n_layers=6,
            kernel_size=15,
            dropout=0.1
        )


def create_delta_predictor(
    use_hyenadna: bool = False,
    model_name: str = 'tiny',
    hidden_dim: int = 128,
    freeze_encoder: bool = True,
    device: Optional[str] = None
) -> DeltaPredictorWithEncoder:
    """
    Create delta predictor with appropriate encoder.
    
    For M1 Mac:
        model = create_delta_predictor(use_hyenadna=False)
    
    For GPU server:
        model = create_delta_predictor(use_hyenadna=True, model_name='small')
    
    Parameters
    ----------
    use_hyenadna : bool
        Use HyenaDNA encoder
    model_name : str
        HyenaDNA variant
    hidden_dim : int
        Hidden dimension
    freeze_encoder : bool
        Freeze encoder weights
    device : str, optional
        Target device
    
    Returns
    -------
    DeltaPredictorWithEncoder
        Delta prediction model
    """
    encoder = create_encoder(
        use_hyenadna=use_hyenadna,
        model_name=model_name,
        hidden_dim=hidden_dim,
        freeze=freeze_encoder,
        device=device
    )
    
    return DeltaPredictorWithEncoder(
        encoder=encoder,
        hidden_dim=hidden_dim,
        dropout=0.1
    )












