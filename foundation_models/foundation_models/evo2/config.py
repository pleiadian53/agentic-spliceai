"""
Configuration for Evo2 models.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Evo2Config:
    """
    Configuration for Evo2 foundation model.
    
    Attributes:
        model_size: Size of Evo2 model ("7b" or "40b")
        quantize: Whether to quantize model (INT8/INT4)
        quantization_bits: Bits for quantization (4 or 8)
        device: Device to load model on ("auto", "cuda", "cpu", "mps")
        torch_dtype: PyTorch dtype (None = auto, "float16", "bfloat16", "float32")
        trust_remote_code: Trust remote code from HuggingFace
        max_length: Maximum sequence length to process
        batch_size: Batch size for inference
        cache_dir: Directory to cache model weights
    """
    
    # Model selection
    model_size: Literal["7b", "40b"] = "7b"
    
    # Quantization (for M1 Mac or memory-constrained GPUs)
    quantize: bool = False
    quantization_bits: Literal[4, 8] = 8
    
    # Device and precision
    device: Literal["auto", "cuda", "cpu", "mps"] = "auto"
    torch_dtype: Optional[Literal["float16", "bfloat16", "float32"]] = None
    
    # HuggingFace settings
    trust_remote_code: bool = True
    cache_dir: Optional[str] = None
    
    # Inference settings
    max_length: int = 32768  # 32kb default (full 1M possible but slow)
    batch_size: int = 1
    
    # Performance optimization
    use_flash_attention: bool = False  # Requires flash-attn package
    gradient_checkpointing: bool = False  # For fine-tuning
    
    # Model paths (override for local weights)
    model_name_or_path: Optional[str] = None
    
    def __post_init__(self):
        """Validate and set defaults."""
        
        # Set model name if not provided
        if self.model_name_or_path is None:
            self.model_name_or_path = f"arcinstitute/evo2-{self.model_size}"
        
        # Auto-detect device if needed
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Auto-set dtype if not provided
        if self.torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = "float16"  # FP16 on GPU
            elif self.device == "mps":
                self.torch_dtype = "float32"  # MPS doesn't support FP16 well
            else:
                self.torch_dtype = "float32"
        
        # Validate quantization
        if self.quantize and self.quantization_bits not in [4, 8]:
            raise ValueError(f"quantization_bits must be 4 or 8, got {self.quantization_bits}")
        
        # Validate max_length
        if self.max_length > 1_000_000:
            import warnings
            warnings.warn(
                f"max_length={self.max_length} is very large. "
                "This will be very slow and may cause OOM errors."
            )
    
    @property
    def model_id(self) -> str:
        """Get HuggingFace model ID."""
        return self.model_name_or_path
    
    @property
    def effective_device(self) -> str:
        """Get effective device after auto-detection."""
        return self.device
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_size": self.model_size,
            "quantize": self.quantize,
            "quantization_bits": self.quantization_bits,
            "device": self.device,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "use_flash_attention": self.use_flash_attention,
            "gradient_checkpointing": self.gradient_checkpointing,
            "model_name_or_path": self.model_name_or_path,
        }
    
    @classmethod
    def for_local_mac(cls, **kwargs) -> "Evo2Config":
        """
        Preset for MacBook Pro M1/M2/M3.
        
        Uses 7B model with INT8 quantization and MPS device.
        """
        defaults = {
            "model_size": "7b",
            "quantize": True,
            "quantization_bits": 8,
            "device": "mps",
            "max_length": 32768,  # Practical limit on M1
            "batch_size": 1,
        }
        defaults.update(kwargs)
        return cls(**defaults)
    
    @classmethod
    def for_gpu_pod(cls, model_size: Literal["7b", "40b"] = "40b", **kwargs) -> "Evo2Config":
        """
        Preset for GPU pod (A40, A100, H100).
        
        Uses 40B model with FP16 and CUDA device by default.
        """
        defaults = {
            "model_size": model_size,
            "quantize": False if model_size == "7b" else True,  # Quantize 40B to fit
            "quantization_bits": 8,
            "device": "cuda",
            "torch_dtype": "float16",
            "max_length": 131072,  # 128kb (full 1M possible but very slow)
            "batch_size": 4 if model_size == "40b" else 8,
        }
        defaults.update(kwargs)
        return cls(**defaults)


# Preset configurations
LOCAL_MAC_CONFIG = Evo2Config.for_local_mac()
GPU_POD_CONFIG_7B = Evo2Config.for_gpu_pod(model_size="7b")
GPU_POD_CONFIG_40B = Evo2Config.for_gpu_pod(model_size="40b")
