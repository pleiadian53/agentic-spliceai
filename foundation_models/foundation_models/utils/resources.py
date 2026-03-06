"""
Resource requirement estimation for foundation model workflows.

Checks whether a given task (embedding extraction, classifier training,
LoRA fine-tuning) is feasible on the current or a specified hardware
profile, before wasting time on an OOM crash.
"""

import logging
from typing import Dict, Literal, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known model specifications
# ---------------------------------------------------------------------------

MODEL_SPECS: Dict[str, dict] = {
    "evo2-7b": {
        "params_b": 7,
        "hidden_dim": 4096,
        "fp16_gb": 14.0,
        "int8_gb": 7.0,
        "int4_gb": 3.5,
    },
    "evo2-40b": {
        "params_b": 40,
        "hidden_dim": 8192,
        "fp16_gb": 80.0,
        "int8_gb": 40.0,
        "int4_gb": 20.0,
    },
}

# ---------------------------------------------------------------------------
# Hardware profiles (for simulation / planning)
# ---------------------------------------------------------------------------

HARDWARE_PROFILES: Dict[str, dict] = {
    "m1-16gb": {"device": "mps", "memory_gb": 16, "label": "Apple Silicon 16 GB"},
    "m1-32gb": {"device": "mps", "memory_gb": 32, "label": "Apple Silicon 32 GB"},
    "m1-64gb": {"device": "mps", "memory_gb": 64, "label": "Apple Silicon 64 GB"},
    "a40-48gb": {"device": "cuda", "memory_gb": 48, "label": "NVIDIA A40 48 GB"},
    "a100-40gb": {"device": "cuda", "memory_gb": 40, "label": "NVIDIA A100 40 GB"},
    "a100-80gb": {"device": "cuda", "memory_gb": 80, "label": "NVIDIA A100 80 GB"},
    "h100-80gb": {"device": "cuda", "memory_gb": 80, "label": "NVIDIA H100 80 GB"},
}

# Safety margin — reserve 15% of VRAM for OS/framework overhead
_MEMORY_SAFETY_FACTOR = 0.85


def check_current_hardware() -> dict:
    """Detect current device, available memory, and return a hardware summary.

    Returns:
        Dict with keys: device, memory_gb, label, profile (closest match or None).
    """
    from .quantization import get_optimal_device, get_device_memory_gb

    device = get_optimal_device()
    memory_gb = get_device_memory_gb(device)

    # Find closest named profile
    profile = None
    for name, hw in HARDWARE_PROFILES.items():
        if hw["device"] == device and abs(hw["memory_gb"] - memory_gb) < 4:
            profile = name
            break

    return {
        "device": device,
        "memory_gb": round(memory_gb, 1),
        "label": profile or f"{device.upper()} {memory_gb:.0f} GB",
        "profile": profile,
    }


def _resolve_hardware(hardware: Optional[str] = None) -> dict:
    """Resolve hardware spec — detect current or look up named profile."""
    if hardware is None:
        return check_current_hardware()
    if hardware in HARDWARE_PROFILES:
        hw = HARDWARE_PROFILES[hardware].copy()
        hw["profile"] = hardware
        return hw
    raise ValueError(
        f"Unknown hardware profile: {hardware!r}. "
        f"Available: {', '.join(HARDWARE_PROFILES)}"
    )


def estimate_embedding_extraction(
    model_size: Literal["7b", "40b"] = "7b",
    n_genes: int = 100,
    avg_seq_len: int = 30_000,
    hardware: Optional[str] = None,
) -> dict:
    """Estimate feasibility and resources for Evo2 embedding extraction.

    Args:
        model_size: Evo2 model size ("7b" or "40b").
        n_genes: Number of genes to process.
        avg_seq_len: Average gene sequence length in bp.
        hardware: Hardware profile name (None = auto-detect).

    Returns:
        Dict with: feasible, model_memory_gb, output_hdf5_gb,
        quantization, notes.
    """
    hw = _resolve_hardware(hardware)
    spec = MODEL_SPECS[f"evo2-{model_size}"]
    usable_gb = hw["memory_gb"] * _MEMORY_SAFETY_FACTOR

    # Determine best quantization for this hardware
    if hw["device"] == "cuda" and usable_gb >= spec["fp16_gb"]:
        model_gb = spec["fp16_gb"]
        quant = "none (FP16)"
    elif usable_gb >= spec["int8_gb"]:
        model_gb = spec["int8_gb"]
        quant = "INT8"
    elif usable_gb >= spec["int4_gb"]:
        model_gb = spec["int4_gb"]
        quant = "INT4"
    else:
        model_gb = spec["int4_gb"]
        quant = "INT4 (insufficient)"

    feasible = model_gb <= usable_gb

    # Output size estimate: n_genes * avg_seq_len * hidden_dim * 4 bytes (float32)
    output_gb = (n_genes * avg_seq_len * spec["hidden_dim"] * 4) / (1024**3)

    notes = []
    if not feasible:
        notes.append(
            f"Evo2 {model_size} requires {model_gb:.1f} GB but only "
            f"{usable_gb:.1f} GB available on {hw['label']}"
        )
        # Suggest alternatives
        for name, profile in HARDWARE_PROFILES.items():
            if profile["memory_gb"] * _MEMORY_SAFETY_FACTOR >= spec["int8_gb"]:
                notes.append(f"  Try: --hardware {name} ({profile['label']})")
                break
    else:
        notes.append(f"Evo2 {model_size} ({quant}): {model_gb:.1f} GB of {usable_gb:.1f} GB available")

    return {
        "feasible": feasible,
        "model_memory_gb": round(model_gb, 1),
        "output_hdf5_gb": round(output_gb, 1),
        "quantization": quant,
        "hardware": hw["label"],
        "notes": notes,
    }


def estimate_classifier_training(
    n_windows: int = 500,
    window_size: int = 1024,
    hidden_dim: int = 2560,
    hardware: Optional[str] = None,
) -> dict:
    """Estimate feasibility and resources for ExonClassifier training.

    The classifier itself is tiny (< 1 MB). Memory is dominated by
    the training tensor: n_windows * window_size * hidden_dim * 4 bytes.

    Args:
        n_windows: Total number of training + validation windows.
        window_size: Window size in bp.
        hidden_dim: Embedding dimension.
        hardware: Hardware profile name (None = auto-detect).

    Returns:
        Dict with: feasible, training_memory_gb, notes.
    """
    hw = _resolve_hardware(hardware)
    usable_gb = hw["memory_gb"] * _MEMORY_SAFETY_FACTOR

    # Training tensor memory (float32)
    tensor_gb = (n_windows * window_size * hidden_dim * 4) / (1024**3)
    # Rough overhead: gradients + optimizer states ~ 3x model (but model is tiny)
    # Main overhead is the data tensors themselves
    total_gb = tensor_gb * 1.2  # 20% overhead for framework buffers

    feasible = total_gb <= usable_gb

    notes = []
    if not feasible:
        # Suggest reducing windows or hidden_dim
        max_windows = int(usable_gb / 1.2 / (window_size * hidden_dim * 4 / (1024**3)))
        notes.append(
            f"Training data requires {total_gb:.1f} GB but only "
            f"{usable_gb:.1f} GB available"
        )
        notes.append(f"  Reduce to --max-windows {max_windows} or use smaller hidden_dim")
    else:
        notes.append(f"Training data: {total_gb:.1f} GB of {usable_gb:.1f} GB available")

    return {
        "feasible": feasible,
        "training_memory_gb": round(total_gb, 1),
        "hardware": hw["label"],
        "notes": notes,
    }


def estimate_lora_finetuning(
    model_size: Literal["7b", "40b"] = "7b",
    lora_rank: int = 8,
    hardware: Optional[str] = None,
) -> dict:
    """Estimate feasibility and resources for LoRA fine-tuning.

    Args:
        model_size: Evo2 model size.
        lora_rank: LoRA rank (8, 16, 64).
        hardware: Hardware profile name (None = auto-detect).

    Returns:
        Dict with: feasible, vram_required_gb, recommended_deepspeed, notes.
    """
    hw = _resolve_hardware(hardware)
    spec = MODEL_SPECS[f"evo2-{model_size}"]
    usable_gb = hw["memory_gb"] * _MEMORY_SAFETY_FACTOR

    # LoRA VRAM estimates (from deepspeed_training.md)
    if model_size == "7b":
        vram_gb = 16.0 + (lora_rank - 8) * 0.06  # ~16 GB base + rank scaling
        deepspeed = "ZeRO-2, FP16"
        min_hw = "a40-48gb"
    else:  # 40b
        vram_gb = 85.0 + (lora_rank - 8) * 0.08
        deepspeed = "ZeRO-3 + CPU offload, BF16"
        min_hw = "a100-80gb"

    feasible = vram_gb <= usable_gb

    notes = []
    if not feasible:
        notes.append(
            f"LoRA fine-tuning Evo2 {model_size} (rank={lora_rank}) requires "
            f"~{vram_gb:.0f} GB VRAM"
        )
        notes.append(f"  Minimum hardware: {HARDWARE_PROFILES[min_hw]['label']}")
        if hw["device"] != "cuda":
            notes.append("  LoRA fine-tuning requires CUDA (not supported on MPS/CPU)")
    else:
        notes.append(
            f"LoRA rank={lora_rank}: ~{vram_gb:.0f} GB of {usable_gb:.0f} GB available"
        )
        notes.append(f"  Recommended: {deepspeed}")

    return {
        "feasible": feasible,
        "vram_required_gb": round(vram_gb, 1),
        "recommended_deepspeed": deepspeed,
        "hardware": hw["label"],
        "notes": notes,
    }


def print_feasibility_report(hardware: Optional[str] = None) -> None:
    """Print a comprehensive feasibility report for all tasks.

    Args:
        hardware: Hardware profile name (None = auto-detect).
    """
    hw = _resolve_hardware(hardware)
    print(f"Hardware: {hw['label']} ({hw['device']}, {hw['memory_gb']} GB)")
    print()

    tasks = [
        ("Evo2 7B Embedding Extraction", estimate_embedding_extraction("7b", hardware=hardware)),
        ("Evo2 40B Embedding Extraction", estimate_embedding_extraction("40b", hardware=hardware)),
        ("Classifier Training (500 windows)", estimate_classifier_training(500, hardware=hardware)),
        ("Classifier Training (5000 windows)", estimate_classifier_training(5000, hardware=hardware)),
        ("LoRA Fine-Tuning Evo2 7B (r=8)", estimate_lora_finetuning("7b", 8, hardware=hardware)),
        ("LoRA Fine-Tuning Evo2 40B (r=8)", estimate_lora_finetuning("40b", 8, hardware=hardware)),
    ]

    print(f"{'Task':<40} {'Feasible':<10} {'Memory':<12} {'Notes'}")
    print("-" * 100)

    for name, result in tasks:
        status = "YES" if result["feasible"] else "NO"
        mem_key = next(
            (k for k in ("model_memory_gb", "training_memory_gb", "vram_required_gb") if k in result),
            None,
        )
        mem_str = f"{result[mem_key]:.1f} GB" if mem_key else "—"
        note = result["notes"][0] if result["notes"] else ""
        print(f"  {name:<38} {status:<10} {mem_str:<12} {note}")

    print()
