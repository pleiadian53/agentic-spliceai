"""
Evo2 Model Wrapper

Provides a clean interface to Arc Institute's Evo2 models.
"""

import torch
from typing import Optional, Union, List
from pathlib import Path

from foundation_models.evo2.config import Evo2Config
from foundation_models.utils.quantization import (
    get_bnb_quantization_config,
    apply_quantization,
    print_quantization_summary,
)


class Evo2Model:
    """
    Wrapper for Evo2 foundation model.
    
    Handles model loading, quantization, and device placement.
    Provides methods for encoding sequences and extracting embeddings.
    
    Example:
        >>> config = Evo2Config.for_local_mac()
        >>> model = Evo2Model(config)
        >>> embeddings = model.encode("ATCGATCG...")
        >>> print(embeddings.shape)  # [seq_len, hidden_dim]
    """
    
    def __init__(self, config: Optional[Evo2Config] = None):
        """
        Initialize Evo2 model.
        
        Args:
            config: Evo2Config instance. If None, uses default config.
        """
        self.config = config or Evo2Config()
        self.model = None
        self.tokenizer = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load Evo2 model from HuggingFace."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: pip install transformers"
            )
        
        print(f"Loading Evo2 {self.config.model_size} model...")
        print(f"  Model: {self.config.model_id}")
        print(f"  Device: {self.config.effective_device}")
        print(f"  Quantize: {self.config.quantize} ({self.config.quantization_bits}-bit)")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )
        
        # Configure model loading
        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_id,
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
        }

        device = self.config.effective_device

        # Quantization via bitsandbytes (CUDA only) or deferred to post-load
        if self.config.quantize and device == "cuda":
            bnb_kwargs = get_bnb_quantization_config(self.config.quantization_bits)
            load_kwargs.update(bnb_kwargs)
            load_kwargs["device_map"] = "auto"
        else:
            # Set dtype and device; native quantization applied post-load for MPS/CPU
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            load_kwargs["torch_dtype"] = dtype_map.get(
                self.config.torch_dtype, torch.float32
            )
            load_kwargs["device_map"] = device
        
        # Load model
        try:
            self.model = AutoModel.from_pretrained(**load_kwargs)
        except Exception as e:
            print(f"\nError loading Evo2 model: {e}")
            print("\nTroubleshooting:")
            print("1. Check if model weights are downloaded")
            print("2. For M1 Mac, ensure quantize=True")
            print("3. For GPU, check CUDA availability")
            print("4. Try smaller model_size='7b' first")
            raise
        
        # Apply native quantization for MPS/CPU after loading
        if self.config.quantize and device != "cuda":
            self.model = apply_quantization(
                self.model, device, self.config.quantization_bits
            )

        # Enable gradient checkpointing if requested (for fine-tuning)
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Set to eval mode by default (frozen backbone)
        self.model.eval()

        print(f"✓ Evo2 {self.config.model_size} loaded successfully!")
        print_quantization_summary(self.model, device)
    
    def _count_parameters(self) -> int:
        """Count total model parameters."""
        return sum(p.numel() for p in self.model.parameters())
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension of model."""
        # Evo2 uses different hidden dims for 7B and 40B
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            return self.model.config.d_model
        else:
            # Default fallback (check Evo2 architecture for actual values)
            return 2560 if self.config.model_size == "7b" else 5120
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.model.parameters()).device
    
    def encode(
        self,
        sequences: Union[str, List[str]],
        return_embeddings: bool = True,
        layer: int = -1,
    ) -> torch.Tensor:
        """
        Encode DNA sequences to embeddings.
        
        Args:
            sequences: DNA sequence(s) to encode
            return_embeddings: If True, return embeddings. If False, return logits.
            layer: Which layer to extract embeddings from (-1 = last layer)
        
        Returns:
            embeddings: Tensor of shape [seq_len, hidden_dim] or [batch, seq_len, hidden_dim]
        """
        # Convert to list if single sequence
        if isinstance(sequences, str):
            sequences = [sequences]
            single_input = True
        else:
            single_input = False
        
        # Tokenize
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=return_embeddings)
        
        if return_embeddings:
            # Extract embeddings from specified layer
            embeddings = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]
            
            # Remove padding if single input
            if single_input:
                # Get attention mask to identify non-padding tokens
                mask = inputs['attention_mask'][0].bool()
                embeddings = embeddings[0][mask]  # [seq_len, hidden_dim]
            
            return embeddings
        else:
            # Return logits for likelihood-based prediction
            logits = outputs.logits
            return logits
    
    def get_likelihood(self, sequence: str) -> float:
        """
        Get sequence likelihood (for zero-shot variant effect prediction).
        
        Args:
            sequence: DNA sequence
        
        Returns:
            log_likelihood: Log probability of sequence under the model
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
        
        # Convert loss to log-likelihood (model uses cross-entropy)
        log_likelihood = -loss
        return log_likelihood
    
    def compute_delta_likelihood(self, ref_seq: str, alt_seq: str) -> float:
        """
        Compute delta log-likelihood for variant effect prediction.
        
        Args:
            ref_seq: Reference sequence
            alt_seq: Alternate sequence (with variant)
        
        Returns:
            delta: Change in log-likelihood (alt - ref)
                   Negative delta = variant predicted to be deleterious
        """
        ref_ll = self.get_likelihood(ref_seq)
        alt_ll = self.get_likelihood(alt_seq)
        delta = alt_ll - ref_ll
        return delta
    
    def __repr__(self) -> str:
        return (
            f"Evo2Model(size={self.config.model_size}, "
            f"device={self.device}, "
            f"quantize={self.config.quantize})"
        )


def load_evo2_model(
    model_size: str = "7b",
    quantize: bool = False,
    device: str = "auto",
    **kwargs
) -> Evo2Model:
    """
    Convenience function to load Evo2 model.
    
    Args:
        model_size: "7b" or "40b"
        quantize: Whether to quantize (recommended for M1 Mac)
        device: Device to load on ("auto", "cuda", "cpu", "mps")
        **kwargs: Additional config parameters
    
    Returns:
        model: Evo2Model instance
    
    Example:
        >>> # On M1 Mac
        >>> model = load_evo2_model(model_size="7b", quantize=True)
        
        >>> # On GPU pod
        >>> model = load_evo2_model(model_size="40b", device="cuda")
    """
    config = Evo2Config(
        model_size=model_size,
        quantize=quantize,
        device=device,
        **kwargs
    )
    return Evo2Model(config)
