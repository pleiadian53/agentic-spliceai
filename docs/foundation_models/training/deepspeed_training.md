# Efficient Training with DeepSpeed and ZeRO

**Purpose**: Guide for using DeepSpeed ZeRO to train large genomic foundation
model adapters (LoRA, classification heads) on RunPods GPU instances, with
graceful fallback to standard PyTorch on M1 Mac.

**Audience**: Developers running training experiments on both local M1 and
remote GPU pods.

**Status**: Design — implementation stub ready in `adapters/`

---

## Overview

DeepSpeed is a deep learning optimization library from Microsoft that provides
several complementary benefits for large model training:

| Feature | What it does | Why we need it |
|---|---|---|
| **ZeRO Stage 1** | Partition optimizer states across GPUs | Reduces GPU memory by ~4× |
| **ZeRO Stage 2** | + Partition gradients | Reduces GPU memory by ~8× |
| **ZeRO Stage 3** | + Partition parameters | Enables models too large for a single GPU |
| **Mixed precision (FP16/BF16)** | Half-precision forward/backward | 2× memory, 2–3× speed |
| **Gradient checkpointing** | Recompute activations instead of storing | Large batch size on limited VRAM |
| **CPU offloading** | Move optimizer state/params to CPU RAM | Train 40B models on 48 GB GPU |

For our use case (LoRA fine-tuning of Evo2, or training the ExonClassifier):

- **M1 Mac (local)**: DeepSpeed does not support MPS.  We use standard PyTorch
  with INT8 quantization and gradient checkpointing.
- **RunPods A40/A100**: ZeRO Stage 2 or 3 with FP16 is the recommended setup.

---

## When to Use What

```
Training scenario                      Recommended strategy
─────────────────────────────────────────────────────────────────────────────
ExonClassifier on Evo2 embeddings
  (embeddings pre-computed, small model)
  → Local M1                          Standard PyTorch, batch_size=32
  → A40 48 GB                         Standard PyTorch, batch_size=256

LoRA fine-tuning of Evo2 7B
  → Local M1                          NOT feasible (too slow, 8 GB model)
  → A40 48 GB                         DeepSpeed ZeRO-2, FP16, lr=1e-4
  → A100 80 GB                        DeepSpeed ZeRO-2 or ZeRO-3, BF16

LoRA fine-tuning of Evo2 40B
  → A100 80 GB × 1                    DeepSpeed ZeRO-3 + CPU offload
  → A100 80 GB × 4 (multi-GPU)        DeepSpeed ZeRO-3, DDP
```

---

## Installation

DeepSpeed is not included in the base environment; install it on the GPU pod:

```bash
# On RunPods pod (after SSH in)
pip install deepspeed

# Verify CUDA ops can be built
ds_report

# Install with optional CUDA extensions (faster kernels)
DS_BUILD_OPS=1 pip install deepspeed
```

On M1 Mac (for development/testing without actual training):

```bash
# DeepSpeed installs but MPS backend is not supported
pip install deepspeed  # CPU-only ops only
```

---

## ZeRO Configuration Reference

DeepSpeed is configured via a JSON file.  Here are three ready-to-use configs.

### Config A: ZeRO Stage 2 — A40 48 GB (Evo2 7B LoRA)

Save as `foundation_models/configs/deepspeed_zero2.json`:

```json
{
  "train_batch_size": 16,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 4,

  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 100,
      "total_num_steps": 2000
    }
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 50,
  "wall_clock_breakdown": false
}
```

### Config B: ZeRO Stage 3 + CPU offload — A100 80 GB (Evo2 40B LoRA)

Save as `foundation_models/configs/deepspeed_zero3_offload.json`:

```json
{
  "train_batch_size": 8,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 8,

  "bf16": {
    "enabled": true
  },

  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "gather_16bit_weights_on_model_save": true
  },

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 5e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },

  "gradient_clipping": 1.0,
  "steps_per_print": 20
}
```

### Config C: No-op (M1 Mac / CPU development)

Save as `foundation_models/configs/deepspeed_noop.json`:

```json
{
  "train_batch_size": 4,
  "train_micro_batch_size_per_gpu": 1,
  "gradient_accumulation_steps": 4,
  "steps_per_print": 10,
  "fp16": { "enabled": false },
  "zero_optimization": { "stage": 0 }
}
```

---

## Integrating DeepSpeed into the Training Loop

### Device-aware trainer

The following pattern selects the right training strategy based on available
hardware and falls back gracefully:

```python
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional

def get_training_strategy(device: str) -> str:
    """Return 'deepspeed', 'standard', or 'cpu'."""
    if device == "mps" or device == "cpu":
        return "standard"
    try:
        import deepspeed  # noqa: F401
        return "deepspeed"
    except ImportError:
        return "standard"


def load_deepspeed_config(config_path: str | Path) -> dict:
    import json
    with open(config_path) as f:
        return json.load(f)


class AdaptiveTrainer:
    """Training wrapper that selects DeepSpeed or standard PyTorch automatically.

    Example
    -------
    >>> trainer = AdaptiveTrainer(
    ...     model=classifier,
    ...     device="cuda",
    ...     ds_config="foundation_models/configs/deepspeed_zero2.json",
    ... )
    >>> trainer.train(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        ds_config: Optional[str | Path] = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
    ):
        self.device = device
        self.strategy = get_training_strategy(device)
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

        if self.strategy == "deepspeed":
            self._init_deepspeed(ds_config)
        else:
            self._init_standard()

        print(f"Training strategy: {self.strategy} on {device}")

    def _init_deepspeed(self, config_path: Optional[str | Path]):
        import deepspeed

        if config_path is None:
            # Auto-select config based on available VRAM
            from foundation_models.utils.quantization import get_device_memory_gb
            mem_gb = get_device_memory_gb("cuda")
            if mem_gb >= 60:
                config_path = "foundation_models/configs/deepspeed_zero3_offload.json"
            else:
                config_path = "foundation_models/configs/deepspeed_zero2.json"
            print(f"  Auto-selected config: {config_path} ({mem_gb:.0f} GB VRAM)")

        ds_config = load_deepspeed_config(config_path)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            config=ds_config,
        )

    def _init_standard(self):
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.scheduler = None

    def train_step(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:
        """Single training step. Returns loss value."""
        from torch.nn import BCEWithLogitsLoss

        if self.strategy == "deepspeed":
            embeddings = embeddings.to(self.engine.device)
            labels = labels.to(self.engine.device)
            logits = self.engine(embeddings)
            # Handle class imbalance
            pos_weight = (labels == 0).sum() / (labels == 1).sum().clamp(min=1)
            loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
            self.engine.backward(loss)
            self.engine.step()
        else:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(embeddings)
            pos_weight = (labels == 0).sum() / (labels == 1).sum().clamp(min=1)
            loss = BCEWithLogitsLoss(pos_weight=pos_weight)(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def save_checkpoint(self, output_dir: str | Path, tag: str = "best"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.strategy == "deepspeed":
            self.engine.save_checkpoint(str(output_dir), tag=tag)
        else:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "tag": tag,
                },
                output_dir / f"checkpoint_{tag}.pt",
            )

    def load_checkpoint(self, checkpoint_dir: str | Path, tag: str = "best"):
        if self.strategy == "deepspeed":
            self.engine.load_checkpoint(str(checkpoint_dir), tag=tag)
        else:
            ckpt = torch.load(
                Path(checkpoint_dir) / f"checkpoint_{tag}.pt",
                map_location=self.device,
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
```

---

## LoRA Fine-Tuning of Evo2 Backbone

For LoRA specifically, we freeze all Evo2 parameters and inject small trainable
rank-decomposition matrices into the attention layers.  DeepSpeed ZeRO is
essential here because the frozen backbone still occupies VRAM even though its
gradients are zero — ZeRO Stage 3 can offload those frozen params to CPU.

### Setting up LoRA with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType

def wrap_evo2_with_lora(
    evo2_model,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> "PeftModel":
    """Inject LoRA adapters into Evo2's attention layers.

    Parameters
    ----------
    evo2_model:
        The base Evo2Model (or its HuggingFace backbone).
    r:
        LoRA rank.  Higher = more capacity, more parameters.
        Typical range: 8–64.
    lora_alpha:
        Scaling factor.  Often set to 2*r.
    lora_dropout:
        Dropout in LoRA layers.
    target_modules:
        Which layer names to insert LoRA into.  None → auto-detect
        (looks for 'q_proj', 'v_proj', 'k_proj', 'out_proj').

    Returns
    -------
    PEFT-wrapped model with only LoRA params trainable.
    """
    if target_modules is None:
        # Common attention projection names; Evo2 uses Mamba layers
        # Inspect model to confirm actual names:
        #   for name, _ in evo2_model.named_modules(): print(name)
        target_modules = ["q_proj", "v_proj", "k_proj", "out_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )

    model = get_peft_model(evo2_model, lora_config)
    model.print_trainable_parameters()
    # Expected output (7B, r=16):
    #   trainable params: ~8M || all params: 7B || trainable%: 0.11%
    return model
```

### Memory estimates for LoRA on Evo2

| Model | LoRA rank | Trainable params | VRAM (ZeRO-2, FP16) |
|---|---|---|---|
| Evo2 7B | r=8 | ~4M | ~16 GB |
| Evo2 7B | r=16 | ~8M | ~16 GB |
| Evo2 7B | r=64 | ~32M | ~17 GB |
| Evo2 40B | r=8 | ~20M | ~85 GB (ZeRO-3 + offload) |
| Evo2 40B | r=16 | ~40M | ~86 GB (ZeRO-3 + offload) |

Evo2 40B requires at least one A100 80 GB with ZeRO-3 CPU offload, or four
A40 48 GB GPUs with ZeRO-3.

---

## Launching Multi-GPU Training on RunPods

RunPods supports multi-GPU pods (2×/4× A100).  Use DeepSpeed's launcher:

```bash
# Single GPU (most common for Evo2 7B)
deepspeed --num_gpus=1 train_exon_classifier.py \
    --deepspeed foundation_models/configs/deepspeed_zero2.json \
    --model_size 7b \
    --chromosomes 1,2,3,4,5 \
    --output_dir checkpoints/exon_classifier_v1

# Multi-GPU (4× A100, Evo2 40B LoRA)
deepspeed --num_gpus=4 train_lora.py \
    --deepspeed foundation_models/configs/deepspeed_zero3_offload.json \
    --model_size 40b \
    --lora_rank 16 \
    --output_dir checkpoints/evo2_40b_lora_v1
```

Or via `torchrun` (compatible with HuggingFace Trainer):

```bash
torchrun --nproc_per_node=4 train_lora.py \
    --use_deepspeed \
    --deepspeed_config foundation_models/configs/deepspeed_zero3_offload.json
```

---

## Gradient Checkpointing (Memory-Speed Trade-off)

Without gradient checkpointing, all layer activations are stored during the
forward pass for use in the backward pass.  For a 7B model this can be
10–20 GB of activation memory.

Gradient checkpointing recomputes activations during the backward pass instead
of storing them, at a ~30% slowdown but ~50% memory reduction:

```python
# Enable on the Evo2 backbone (set in Evo2Config)
from foundation_models.evo2.config import Evo2Config

config = Evo2Config.for_gpu_pod(model_size="7b")
config.gradient_checkpointing = True   # passed to model.gradient_checkpointing_enable()

# Or enable manually after loading:
evo2_model.model.gradient_checkpointing_enable()
```

Rule of thumb:
- Always enable for LoRA fine-tuning (you're paying the backward cost anyway)
- Disable for embedding extraction (inference only, no backward needed)

---

## Complete RunPods Setup Script

```bash
#!/bin/bash
# setup_runpod.sh — Run once after pod creation

set -e

# 1. Clone repo (or it's already at /workspace)
cd /workspace
git clone https://github.com/pleiadian53/agentic-spliceai.git || true
cd agentic-spliceai

# 2. Install packages
pip install -e .
pip install -e ./foundation_models
pip install deepspeed peft bitsandbytes accelerate

# 3. Build DeepSpeed CUDA ops (takes ~5 min, speeds up training)
DS_BUILD_OPS=1 pip install deepspeed --upgrade

# 4. Verify
ds_report
python -c "import deepspeed; print(f'DeepSpeed {deepspeed.__version__} OK')"
python -c "from foundation_models.evo2 import Evo2Embedder; print('foundation_models OK')"

# 5. Download Evo2 weights (cached to /workspace/models)
HF_HOME=/workspace/models python -c "
from foundation_models.evo2.config import Evo2Config
from transformers import AutoModel
config = Evo2Config.for_gpu_pod(model_size='7b')
print(f'Downloading {config.model_id}...')
AutoModel.from_pretrained(config.model_id, trust_remote_code=True, cache_dir='/workspace/models')
print('Done.')
"

echo "Setup complete!"
```

---

## Monitoring Training

```bash
# Real-time GPU utilization
watch -n1 nvidia-smi

# DeepSpeed logs loss and throughput automatically
# Look for lines like:
#   [2026-02-22 10:15:00,123] [INFO] steps=50, loss=0.3421, lr=1.0e-04, ...

# TensorBoard (if enabled in ds_config)
tensorboard --logdir runs/ --port 6006
```

---

## Troubleshooting

### OOM (Out of Memory) on A40 48 GB with Evo2 7B

1. Enable gradient checkpointing
2. Reduce `train_micro_batch_size_per_gpu` to 1
3. Increase `gradient_accumulation_steps` to keep effective batch size constant
4. Switch from ZeRO Stage 2 to Stage 3

### NaN Loss with FP16

Evo2 uses Mamba (SSM) layers which can overflow in FP16.  Switch to BF16:

```json
"fp16": { "enabled": false },
"bf16": { "enabled": true }
```

BF16 requires Ampere GPU (A100, A30, RTX 3090+).  For older GPUs (V100), use
FP16 with a lower `initial_scale_power` (e.g. 8 or 4).

### DeepSpeed not found on M1 Mac

Expected — MPS backend is not supported.  The `AdaptiveTrainer` class above
automatically falls back to standard PyTorch when DeepSpeed is unavailable.

---

## Summary: Recommended Configurations by Hardware

| Hardware | Model | Strategy | Config |
|---|---|---|---|
| M1 Mac 16 GB | ExonClassifier (small) | Standard PyTorch | — |
| M1 Mac 16 GB | Evo2 7B embeddings | INT8 quantized inference | `Evo2Config.for_local_mac()` |
| A40 48 GB × 1 | ExonClassifier | Standard PyTorch | — |
| A40 48 GB × 1 | Evo2 7B LoRA | DeepSpeed ZeRO-2, FP16 | `deepspeed_zero2.json` |
| A100 80 GB × 1 | Evo2 7B LoRA | DeepSpeed ZeRO-2, BF16 | `deepspeed_zero2.json` |
| A100 80 GB × 1 | Evo2 40B LoRA | DeepSpeed ZeRO-3 + offload | `deepspeed_zero3_offload.json` |
| A100 80 GB × 4 | Evo2 40B LoRA | DeepSpeed ZeRO-3 DDP | `deepspeed_zero3_offload.json` |

---

**Last Updated**: February 22, 2026
**Status**: Design complete — configs ready; training scripts to be scaffolded next
