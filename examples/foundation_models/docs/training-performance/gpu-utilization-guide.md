# GPU and VRAM Utilization Guide

**Related**: [large-scale-training-guide.md](../large-scale-training-guide.md) · [training-io-optimization.md](../training-io-optimization.md)

---

## Overview

Two metrics appear on GPU dashboards (RunPod, nvidia-smi) that are often confused:

| Metric | What it measures | Typical bottleneck |
|--------|-----------------|-------------------|
| **GPU utilization** | % of time the GPU SM cores are active | I/O starvation — GPU waits for data |
| **VRAM utilization** | % of GPU memory in use | Model size + batch activations |

They are **independent**. A tiny classifier on a large GPU can have low VRAM (2%) *and* low GPU compute (33%) simultaneously. Understanding which one is the limiting factor points to the right fix.

---

## Part 1 — GPU Compute Utilization

### The pipeline view

Every training step is a pipeline with two stages:

```
[CPU: load batch from disk] → [GPU: forward + backward]
        I/O stage                   compute stage
```

GPU utilization ≈ compute stage / (compute stage + I/O stage).

If I/O takes 500 ms and compute takes 5 ms, GPU utilization = 5/505 = **1%** — even though the GPU is doing its job correctly.

### Bottleneck types

**I/O-bound** (most common for embedding-based pipelines):

- Symptom: GPU util low, CPU at 100%, GPU idle most of the time
- Cause: serial data loading (`num_workers=0`), slow storage (network volumes), compressed data formats
- Diagnosis: time a single `__getitem__` call; if it's >10 ms, I/O is the bottleneck

**Batch-size-bound** (after fixing I/O):

- Symptom: GPU util plateau despite fast storage and multiple workers
- Cause: batches arrive fast enough, but each batch is too small to saturate GPU cores
- Diagnosis: VRAM well below capacity; increasing batch size improves GPU util proportionally

**Compute-bound** (rare for small heads, common for large encoder fine-tuning):

- Symptom: GPU util near 100%, VRAM near capacity
- Cause: model is large relative to GPU — this is the ideal steady state

### General fixes for low GPU utilization

| Fix | Effect | When to apply |
|-----|--------|--------------|
| `num_workers=2–4` in DataLoader | Prefetch overlaps I/O with compute | Always with GPU training |
| `multiprocessing_context="spawn"` | Avoids fork-unsafe libraries (Polars, h5py) | Linux + Polars/h5py |
| Larger batch size | More GPU work per data delivery | VRAM well below capacity |
| Uncompressed data formats | Eliminates decompression CPU cost | gzip/lz4 HDF5 datasets |
| Local NVMe vs network volume | 5–20× faster random reads | Cloud pods with volume mounts |
| `pin_memory=True` | Faster host→device transfer | CUDA training |
| `persistent_workers=True` | Avoid worker re-spawn overhead per epoch | With `num_workers > 0` |

### Fork-safety: the hidden `num_workers=0` trap

PyTorch DataLoader workers are created by forking the main process. Any library that
has initialized C-level state (thread pools, file handles, memory allocators) before the
fork will be corrupted in the child process.

Common offenders in genomics pipelines:

| Library | Problem | Fix |
|---------|---------|-----|
| `h5py` | File handles not valid after fork | Lazy-open per worker PID |
| `Polars` | Arrow thread pool deadlocks | Use `multiprocessing_context="spawn"` |
| `pyfaidx` | File handle state | Lazy-open (`self._fasta = None` at init) |

The **pid-keyed lazy open** pattern for HDF5 datasets:

```python
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self._paths = paths
        self._handles = {}          # pid → [h5py.File, ...]
        # DO NOT open files here

    def _get_handles(self):
        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = [h5py.File(p, "r") for p in self._paths]
        return self._handles[pid]

    def __getitem__(self, idx):
        f = self._get_handles()[self._shard_idx(idx)]
        ...
```

Each forked worker has a unique PID and opens its own handles on first access.
No state is shared across processes.

---

## Part 2 — VRAM Utilization

### What consumes VRAM

For a single training step with batch size B and sequence length L:

```
VRAM = model_params × dtype_bytes
     + activations × B × L    ← grows with batch and sequence
     + gradients (= model_params × dtype_bytes, for trainable params only)
     + optimizer state (AdamW: 2× gradients for m, v)
     + framework overhead (~200–500 MB)
```

For **frozen-head** pipelines (07a): only the classifier head (1M params) is trainable.
For **fine-tuning** (08): encoder layers added, activations stored for backprop through deeper layers.

### VRAM reference table (A40, 48 GB)

| Pipeline | Model | Trainable | VRAM (batch=64) | VRAM (batch=256) |
|----------|-------|-----------|-----------------|------------------|
| Frozen head (07a) | dilated_cnn head | ~1M | ~0.3 GB (0.6%) | ~0.6 GB (1.2%) |
| Fine-tune last_n=2 (08) | SpliceBERT + head | ~11.7M | ~4 GB (8%) | ~14 GB (29%) |
| Fine-tune full (08) | SpliceBERT + head | ~20.5M | ~6 GB (12%) | OOM |
| Fine-tune LoRA r=8 (08) | SpliceBERT + head | ~2.5M | ~3 GB (6%) | ~10 GB (21%) |

### When low VRAM is expected vs a problem

**Expected low VRAM** — frozen-head pipeline:

The embedding model is frozen; only the classifier head is in VRAM. On an A40, a
dilated CNN head with hidden_dim=128 uses ~0.3–0.6 GB regardless of batch size.
This is architecturally correct — the large GPU is being used for its compute speed,
not its memory capacity. Low VRAM is not a problem here.

**Expected low VRAM** that can be improved — increase batch size:

If VRAM is low and GPU utilization is also low (both <10%), the GPU has headroom in
both dimensions. Increasing batch size uses more VRAM and more compute per step.

**Unexpectedly low VRAM** — possible issues:

- Model moved to CPU accidentally (`device` resolved to `"cpu"` at runtime)
- Mixed precision not enabled (FP32 uses 2× the VRAM of BF16)
- Model loaded but not used in the training loop

### Increasing VRAM utilization

1. **Increase batch size** — most direct lever; limited by GPU memory
2. **Enable mixed precision** — halves activation memory, allows 2× larger batches
3. **Gradient accumulation** — simulates larger batches without more VRAM
4. **Fine-tune more layers** — deeper backprop stores more activations

```python
# Mixed precision (Ampere+ GPUs: A40, A100, H100)
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(sequences)
    loss = criterion(logits, labels)
```

---

## Part 3 — Case Study: Genome-Scale Splice Site Predictor (07a)

This section traces the real optimization journey for
`07a_direct_shard_splice_predictor.py` on an A40 (48 GB VRAM, 48 GB RAM).

### Baseline: 0% GPU, 100% CPU

**Symptom** (initial run):
```
Epoch 22/100: batch 72/729 (38 s), loss=0.0016
```
- 729 batches/epoch × 0.53 s/batch = **~7 min/epoch**
- RunPod dashboard: CPU 100%, GPU 0%

`torch.cuda.is_available()` returned `True` — CUDA was present. The GPU was computing
each batch in ~3 ms and then sitting idle for ~527 ms waiting for data.

**Root cause**: `ShardedWindowDataset` opened all shard file handles at `__init__`:

```python
# BEFORE — fork-unsafe
for path in shard_paths:
    f = h5py.File(path, "r")      # opened at init
    self.shards.append(f)          # stored in self
```

Because `h5py` handles cannot survive `os.fork()`, `DataLoader` was forced to
`num_workers=0`. All 64 HDF5 reads per batch ran serially on the main thread:
64 reads × 8 ms = **512 ms of CPU work** → GPU idle 99.4% of the time.

### Fix 1: Fork-safe ShardedWindowDataset + `num_workers=2`

Changed `ShardedWindowDataset` to lazy per-PID handle opening:

```python
# AFTER — fork-safe
def __init__(self, shard_paths):
    self._shard_paths = [str(p) for p in shard_paths]
    self._handles = {}              # pid → handles, populated lazily

def _get_handles(self):
    pid = os.getpid()
    if pid not in self._handles:
        self._handles[pid] = [h5py.File(p, "r") for p in self._shard_paths]
    return self._handles[pid]
```

DataLoader updated:

```python
_nw = 2 if (use_shards and cuda_available) else 0
DataLoader(..., num_workers=_nw, pin_memory=cuda_available,
           persistent_workers=True, multiprocessing_context="spawn")
```

**Result**: GPU 0% → **13%** with batch=64.

Workers now prefetch batch N+1 and N+2 while the GPU processes batch N:

```
Worker 0: ──[load N+1]────────[load N+3]──────
Worker 1: ──────[load N+2]────────[load N+4]──
GPU:      ──[N]──[N+1]──[N+2]──[N+3]──[N+4]──
```

GPU utilization ceiling: 3 ms / (3 ms + ~20 ms data arrival) ≈ 13%.

### Fix 2: Larger batch size

With VRAM at only 2% (0.3 GB of 48 GB), there was large headroom.
Increasing from batch=64 to batch=256:

| Batch | Batches/epoch | GPU ms/batch | I/O ms/batch | GPU util | Epoch time |
|-------|--------------|--------------|--------------|---------|------------|
| 64 | 729 | 3 ms | ~20 ms | 13% | ~7 min → ~4 min |
| 256 | 183 | 12 ms | ~20 ms | **33%** | **~51 s** |
| 512 | 92 | 24 ms | ~20 ms | ~55% | ~40 s |

At batch=256, one epoch takes **~51 seconds** — an **8× speedup** from the baseline.

### Remaining constraints

- **VRAM (2–5%)**: structurally low for frozen-head pipeline. Acceptable.
  Only fine-tuning (08) will fill VRAM meaningfully.
- **Network volume I/O (40% network)**: shards are on a mounted network volume
  (`/workspace/`), not local NVMe. This sets a hard floor on data latency regardless
  of `num_workers`. For maximum GPU utilization, copy shards to local `/tmp` before training.
- **RAM (56–98%)**: OS page cache fills with shard data (17 GB of shards).
  Normal Linux behavior — pages evict under pressure. Watch for OOM if >95% sustained.

### Polars fork warning

After enabling `num_workers=2`, a Polars warning appeared:

```
RuntimeWarning: Using fork() can cause Polars to deadlock in the child process.
```

Polars initializes an Arrow thread pool in Phase 1. Forking after that corrupts
the pool in child processes. Fix: `multiprocessing_context="spawn"` in DataLoader.

The warning did not cause failures in this run because Polars is unused in worker
processes (Phase 1 finishes before workers are spawned), but `spawn` is the
correct setting.

### Summary of changes and their effects

```
Initial state:     num_workers=0, batch=64  →  0% GPU, 7 min/epoch
After fork fix:    num_workers=2, batch=64  →  13% GPU, ~4 min/epoch
After batch fix:   num_workers=2, batch=256 →  33% GPU, ~51 s/epoch
```

---

## Part 4 — Recommendations by Pipeline Type

### Frozen-head embedding pipeline (07a)

```python
# Optimal DataLoader settings for A40 with sharded HDF5
DataLoader(
    ShardedWindowDataset(shard_paths),   # fork-safe (pid-keyed lazy open)
    batch_size=256,                       # fill GPU compute; VRAM has headroom
    shuffle=True,
    num_workers=2,                        # prefetch overlap
    pin_memory=True,
    persistent_workers=True,
    multiprocessing_context="spawn",      # avoid Polars/h5py fork deadlock
)
```

Expected: 30–40% GPU util on A40 with network volume. 50–70% with local NVMe.

To push further: copy shards to `/tmp/` at job start (fast local NVMe):
```bash
cp /workspace/output/.../shards/*.h5 /tmp/shards/
```

### Fine-tuning pipeline (08)

Fine-tuning (SpliceBERT encoder + head) is naturally compute-bound:
- Each batch runs forward+backward through the encoder (~300 ms)
- Data loading (`SequenceWindowDataset` from FASTA) is negligible (~8 ms)
- VRAM: 8–29% depending on batch size and strategy
- GPU util: 80–95% expected without any special DataLoader tuning

`num_workers=0` is fine here — the GPU is always the bottleneck.
Enable mixed precision for 2× VRAM efficiency:

```python
scaler = torch.cuda.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(sequences)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Evo2 (large model, future)

Evo2 7B (hidden_dim=4096) produces embeddings 8× larger than SpliceBERT.
Batch=64 embeddings alone: 64 × 512 × 4096 × 4 = **512 MB/batch**.
VRAM will be the binding constraint, not compute.

- Use `batch_size=16` or `batch_size=8`
- Mixed precision (BF16) is mandatory — Evo2 overflows in FP16
- `num_workers=1` sufficient (GPU is slower than I/O at large batches)
- Consider gradient checkpointing for fine-tuning
