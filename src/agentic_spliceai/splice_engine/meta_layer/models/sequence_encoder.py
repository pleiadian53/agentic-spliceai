"""
Sequence encoders for DNA sequences.

Provides multiple encoder options:
- HyenaDNA: State-of-the-art DNA language model (requires GPU)
- CNN: Lightweight multi-scale convolutional encoder (CPU-friendly)
- Identity: No encoding (for ablation studies)

Ported from: meta_spliceai/splice_engine/meta_layer/models/sequence_encoder.py
"""

import logging
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SequenceEncoderFactory:
    """
    Factory for creating sequence encoders.
    
    Supports multiple encoder types with a unified interface.
    
    Examples
    --------
    >>> encoder = SequenceEncoderFactory.create('cnn', output_dim=256)
    >>> sequence = torch.randn(32, 4, 501)  # [batch, 4, length]
    >>> embeddings = encoder(sequence)       # [batch, 256]
    """
    
    ENCODERS = {
        'hyenadna': 'HyenaDNAEncoder',
        'cnn': 'CNNEncoder',
        'none': 'IdentityEncoder'
    }
    
    @classmethod
    def create(
        cls,
        encoder_type: str,
        output_dim: int = 256,
        **kwargs
    ) -> nn.Module:
        """
        Create a sequence encoder.
        
        Parameters
        ----------
        encoder_type : str
            Type of encoder: 'hyenadna', 'cnn', 'none'.
        output_dim : int
            Output embedding dimension.
        **kwargs
            Additional encoder-specific parameters.
        
        Returns
        -------
        nn.Module
            Sequence encoder module.
        """
        encoder_type = encoder_type.lower()
        
        if encoder_type == 'hyenadna':
            return HyenaDNAEncoder(output_dim=output_dim, **kwargs)
        elif encoder_type == 'cnn':
            return CNNEncoder(output_dim=output_dim, **kwargs)
        elif encoder_type == 'none':
            return IdentityEncoder(output_dim=output_dim)
        else:
            raise ValueError(
                f"Unknown encoder type: {encoder_type}. "
                f"Available: {list(cls.ENCODERS.keys())}"
            )


class CNNEncoder(nn.Module):
    """
    Multi-scale CNN encoder for DNA sequences.
    
    Uses multiple kernel sizes to capture motifs at different scales.
    Lightweight and CPU-friendly, suitable for M1 MacBook.
    
    Parameters
    ----------
    input_channels : int
        Number of input channels (4 for one-hot DNA).
    output_dim : int
        Output embedding dimension.
    kernel_sizes : list of int
        Convolution kernel sizes.
    num_filters : int
        Number of filters per kernel size.
    dropout : float
        Dropout probability.
    
    Examples
    --------
    >>> encoder = CNNEncoder(output_dim=256)
    >>> sequence = torch.randn(32, 4, 501)  # One-hot encoded
    >>> embeddings = encoder(sequence)       # [32, 256]
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_dim: int = 256,
        kernel_sizes: List[int] = None,
        num_filters: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 11, 15]  # Multi-scale
        
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters
        self.output_dim = output_dim
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, num_filters, k, padding=k//2),
                nn.BatchNorm1d(num_filters),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        
        # Second layer convolutions
        total_filters = num_filters * len(kernel_sizes)
        self.conv2 = nn.Sequential(
            nn.Conv1d(total_filters, total_filters, 3, padding=1),
            nn.BatchNorm1d(total_filters),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Global pooling + projection
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(total_filters, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode DNA sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            One-hot encoded sequences [batch, 4, length].
        
        Returns
        -------
        torch.Tensor
            Sequence embeddings [batch, output_dim].
        """
        # Apply multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]
        
        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)  # [batch, total_filters, length]
        
        # Second conv layer
        x = self.conv2(x)
        
        # Global max pooling
        x = self.pool(x).squeeze(-1)  # [batch, total_filters]
        
        # Project to output dimension
        return self.projection(x)


class HyenaDNAEncoder(nn.Module):
    """
    HyenaDNA-based sequence encoder.
    
    Uses the HyenaDNA foundation model for DNA sequence understanding.
    State space model architecture enables O(n) memory for long sequences.
    
    Note: Requires the `transformers` library and downloads model weights
    on first use.
    
    Parameters
    ----------
    model_size : str
        Model size: 'tiny', 'small', 'medium', 'large'.
    output_dim : int
        Output embedding dimension.
    pretrained : bool
        Whether to load pretrained weights.
    freeze : bool
        Whether to freeze pretrained weights.
    max_length : int
        Maximum sequence length.
    
    Examples
    --------
    >>> encoder = HyenaDNAEncoder(model_size='small', output_dim=256)
    >>> tokens = torch.randint(0, 4, (32, 501))  # Tokenized sequence
    >>> embeddings = encoder(tokens)             # [32, 256]
    """
    
    # HyenaDNA checkpoint mapping
    CHECKPOINTS = {
        'tiny': 'LongSafari/hyenadna-tiny-1k-seqlen',
        'small': 'LongSafari/hyenadna-small-32k-seqlen',
        'medium': 'LongSafari/hyenadna-medium-160k-seqlen',
        'large': 'LongSafari/hyenadna-large-1m-seqlen',
    }
    
    HIDDEN_DIMS = {
        'tiny': 128,
        'small': 256,
        'medium': 512,
        'large': 1024,
    }
    
    def __init__(
        self,
        model_size: str = 'small',
        output_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        max_length: int = 512
    ):
        super().__init__()
        
        self.model_size = model_size
        self.output_dim = output_dim
        self.max_length = max_length
        self.pretrained = pretrained
        self.freeze = freeze
        
        # Get hidden dimension
        if model_size not in self.HIDDEN_DIMS:
            raise ValueError(
                f"Unknown model size: {model_size}. "
                f"Available: {list(self.HIDDEN_DIMS.keys())}"
            )
        self.hidden_dim = self.HIDDEN_DIMS[model_size]
        
        # Load backbone (lazy initialization)
        self._backbone = None
        self._tokenizer = None
        
        # Projection head (always trainable)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
        # Fallback CNN for when HyenaDNA is not available
        self._fallback_cnn = None
    
    def _load_backbone(self):
        """Lazy-load the HyenaDNA backbone."""
        if self._backbone is not None:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            checkpoint = self.CHECKPOINTS[self.model_size]
            logger.info(f"Loading HyenaDNA: {checkpoint}")
            
            self._backbone = AutoModel.from_pretrained(
                checkpoint,
                trust_remote_code=True
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                checkpoint,
                trust_remote_code=True
            )
            
            if self.freeze:
                for param in self._backbone.parameters():
                    param.requires_grad = False
                logger.info("HyenaDNA backbone frozen")
            
            logger.info(f"Loaded HyenaDNA {self.model_size}")
            
        except ImportError:
            logger.warning(
                "transformers not available, using CNN fallback. "
                "Install with: pip install transformers"
            )
            self._use_fallback()
        except Exception as e:
            logger.warning(f"Could not load HyenaDNA: {e}. Using CNN fallback.")
            self._use_fallback()
    
    def _use_fallback(self):
        """Initialize fallback CNN encoder."""
        if self._fallback_cnn is None:
            self._fallback_cnn = CNNEncoder(
                output_dim=self.hidden_dim,
                num_filters=self.hidden_dim // 4
            )
            logger.info("Using CNN fallback encoder")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences.
        
        Parameters
        ----------
        x : torch.Tensor
            For HyenaDNA: token IDs [batch, length].
            For CNN fallback: one-hot [batch, 4, length].
        
        Returns
        -------
        torch.Tensor
            Sequence embeddings [batch, output_dim].
        """
        # Lazy load backbone
        self._load_backbone()
        
        # Use fallback CNN if HyenaDNA not available
        if self._fallback_cnn is not None:
            # Convert token IDs to one-hot if needed
            if x.dim() == 2:
                x = F.one_hot(x.long(), num_classes=4).float()
                x = x.transpose(1, 2)  # [batch, 4, length]
            hidden = self._fallback_cnn(x)
        else:
            # Use HyenaDNA
            outputs = self._backbone(x)
            
            # Get last hidden state
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
            else:
                hidden = outputs
            
            # Mean pooling over sequence length
            hidden = hidden.mean(dim=1)  # [batch, hidden_dim]
        
        # Project to output dimension
        return self.projection(hidden)
    
    def tokenize(self, sequences: List[str]) -> torch.Tensor:
        """
        Tokenize DNA sequences.
        
        Parameters
        ----------
        sequences : list of str
            DNA sequences to tokenize.
        
        Returns
        -------
        torch.Tensor
            Token IDs [batch, length].
        """
        self._load_backbone()
        
        if self._tokenizer is not None:
            encoded = self._tokenizer(
                sequences,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            return encoded['input_ids']
        else:
            # Fallback: simple encoding
            mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
            tokens = []
            for seq in sequences:
                seq = seq.upper()[:self.max_length]
                token_ids = [mapping.get(base, 0) for base in seq]
                # Pad
                token_ids = token_ids + [0] * (self.max_length - len(token_ids))
                tokens.append(token_ids)
            return torch.tensor(tokens)


class IdentityEncoder(nn.Module):
    """
    Identity encoder (no sequence encoding).
    
    Used for ablation studies to evaluate the contribution
    of sequence information.
    
    Parameters
    ----------
    output_dim : int
        Output dimension (returns zeros).
    """
    
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return zero embeddings."""
        batch_size = x.shape[0]
        return torch.zeros(batch_size, self.output_dim, device=x.device)









