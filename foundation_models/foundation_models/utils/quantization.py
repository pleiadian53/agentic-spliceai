"""
Quantization helpers for Evo2 and other large genomic foundation models.

Handles INT8 / INT4 quantization on both CUDA (bitsandbytes) and
Apple Silicon MPS (torch.quantization).  Also provides device capability
queries so callers don't have to probe torch internals.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_optimal_device() -> str:
    """Return the best available device string: 'cuda', 'mps', or 'cpu'."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_optimal_dtype(device: str) -> torch.dtype:
    """Return the preferred floating-point dtype for *device*.

    - CUDA  → float16  (fast, well-supported)
    - MPS   → float32  (MPS has limited float16 op coverage)
    - CPU   → float32
    """
    if device == "cuda":
        return torch.float16
    return torch.float32


def get_device_memory_gb(device: str = "cuda") -> float:
    """Return available VRAM / RAM in GB for *device* (best-effort)."""
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024 ** 3)
    if device == "mps" and torch.backends.mps.is_available():
        # MPS shares system RAM; psutil gives a rough estimate
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            return 0.0
    return 0.0


def recommend_quantization(device: str, model_size: str) -> tuple[bool, int]:
    """Suggest whether to quantize and at how many bits.

    Parameters
    ----------
    device:
        'cuda', 'mps', or 'cpu'
    model_size:
        '7b' or '40b'

    Returns
    -------
    (should_quantize, bits) – e.g. (True, 8) or (False, 8)
    """
    mem_gb = get_device_memory_gb(device)
    size_b = int(model_size.replace("b", ""))

    # Rough estimate: FP16 needs ~2 bytes/param, INT8 ~1 byte/param
    fp16_gb = size_b * 2
    int8_gb = size_b * 1
    int4_gb = size_b * 0.5

    if device == "mps":
        # MPS cannot use bitsandbytes; fall back to native torch quantization
        return mem_gb > 0 and int8_gb > mem_gb, 8

    if device == "cpu":
        return True, 8  # always quantize on CPU to be safe

    # CUDA path
    if mem_gb == 0 or fp16_gb <= mem_gb * 0.85:
        return False, 8   # fits in FP16 comfortably
    if int8_gb <= mem_gb * 0.85:
        return True, 8
    if int4_gb <= mem_gb * 0.85:
        return True, 4
    warnings.warn(
        f"Model {model_size} may not fit in {mem_gb:.1f} GB VRAM even at INT4."
    )
    return True, 4


# ---------------------------------------------------------------------------
# CUDA: bitsandbytes-based quantization (INT8 / INT4)
# ---------------------------------------------------------------------------

def _check_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


def get_bnb_quantization_config(bits: int = 8) -> dict:
    """Build a ``transformers``-compatible BitsAndBytes loading config dict.

    Raises
    ------
    ImportError
        If ``bitsandbytes`` is not installed.
    """
    if not _check_bitsandbytes():
        raise ImportError(
            "bitsandbytes is required for GPU quantization.\n"
            "Install with: pip install bitsandbytes"
        )

    from transformers import BitsAndBytesConfig

    if bits == 8:
        return {
            "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        }
    elif bits == 4:
        return {
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",  # NF4 preserves outliers better
            ),
        }
    else:
        raise ValueError(f"Unsupported bits={bits}; must be 4 or 8.")


# ---------------------------------------------------------------------------
# MPS / CPU: native PyTorch dynamic quantization
# ---------------------------------------------------------------------------

def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    layer_types: Optional[tuple] = None,
) -> nn.Module:
    """Apply dynamic quantization (INT8) to *model* in-place.

    Suitable for MPS and CPU where bitsandbytes is unavailable.

    Parameters
    ----------
    model:
        The loaded PyTorch model.
    dtype:
        Quantization dtype (default: ``torch.qint8``).
    layer_types:
        Which layer classes to quantize. Defaults to ``(nn.Linear,)``.

    Returns
    -------
    Quantized model.
    """
    if layer_types is None:
        layer_types = (nn.Linear,)

    model = torch.quantization.quantize_dynamic(
        model,
        set(layer_types),
        dtype=dtype,
    )
    return model


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def apply_quantization(
    model: nn.Module,
    device: str,
    bits: int = 8,
) -> nn.Module:
    """Apply the appropriate quantization strategy for *device* and *bits*.

    On CUDA this is a no-op (bitsandbytes is applied during model loading via
    ``get_bnb_quantization_config``).  On MPS/CPU it applies dynamic INT8
    quantization post-load.

    Parameters
    ----------
    model:
        Already-loaded model (FP32 on MPS/CPU, or pre-quantized on CUDA).
    device:
        Target device string.
    bits:
        Quantization width (4 or 8).  Only INT8 is supported on MPS/CPU.

    Returns
    -------
    Quantized model.
    """
    if device == "cuda":
        # bitsandbytes handles this at load time; nothing to do here
        return model

    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")

    if bits == 4:
        warnings.warn(
            "INT4 via native PyTorch is not yet well-supported on MPS/CPU. "
            "Falling back to INT8."
        )

    return quantize_model_dynamic(model)


# ---------------------------------------------------------------------------
# Memory-usage estimation
# ---------------------------------------------------------------------------

def estimate_model_memory_gb(model: nn.Module) -> float:
    """Estimate peak memory usage of *model* in GB (parameters only)."""
    total_bytes = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    )
    return total_bytes / (1024 ** 3)


def print_quantization_summary(model: nn.Module, device: str) -> None:
    """Print a brief summary of the model's quantization state."""
    mem_gb = estimate_model_memory_gb(model)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  Device         : {device}")
    print(f"  Parameters     : {total_params / 1e9:.2f}B")
    print(f"  Memory (params): {mem_gb:.2f} GB")

    # Detect quantization
    has_bnb = any(
        "bitsandbytes" in type(m).__module__ for m in model.modules()
    )
    has_dynamic = any(
        hasattr(m, "_packed_params") for m in model.modules()
    )

    if has_bnb:
        print("  Quantization   : bitsandbytes (CUDA)")
    elif has_dynamic:
        print("  Quantization   : PyTorch dynamic INT8")
    else:
        print("  Quantization   : none (full precision)")
