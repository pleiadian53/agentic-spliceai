# Evo2 GPU Memory: OOM During Model Loading

## The Problem

Evo2 7B crashes with `torch.OutOfMemoryError` on A40 (48 GB) during **model construction**
-- before any inference begins. The error occurs while allocating `nn.Linear` weights inside
`StripedHyena` blocks, typically at block 2 of 32.

### Error Signature

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 176.00 MiB.
GPU 0 has a total capacity of 44.43 GiB of which 10.50 MiB is free.
Process 9578 has 42.94 GiB memory in use.
```

Stack trace leads through:
```
Evo2(checkpoint)
  -> StripedHyena(global_config)
    -> ParallelGatedConvBlock(config, layer_idx)
      -> ParallelGatedMLP(config, layer_idx)
        -> nn.Linear(...)       # <-- OOM here
          -> torch.empty(...)
```

## Root Cause

The Evo2 library's `StripedHyena` model constructs all layers on CUDA in **float32** by default.
During loading, both the checkpoint and the model architecture coexist in GPU memory:

| Component | Size (float32) | Size (bfloat16) |
|-----------|---------------|-----------------|
| Model weights (7B params) | ~28 GB | ~14 GB |
| Checkpoint in GPU memory during loading | ~14 GB | ~14 GB |
| **Total during construction** | **~42 GB** | **~28 GB** |
| A40 VRAM | 48 GB | 48 GB |
| Remaining for activations/inference | ~6 GB (not enough) | ~20 GB (comfortable) |

In float32, the model barely fits during loading (~42 GB), leaving no headroom for the final
blocks to allocate. The OOM occurs at block 2/32 because the checkpoint (~14 GB) is already
resident in GPU memory when the model architecture starts allocating.

## The Fix

Set `torch.set_default_dtype(torch.bfloat16)` before calling `Evo2(checkpoint)`:

```python
# In foundation_models/foundation_models/evo2/model.py
prev_dtype = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
try:
    self._evo2 = Evo2(checkpoint)
finally:
    torch.set_default_dtype(prev_dtype)
```

This forces `nn.Linear` (and all other layers) to allocate in bfloat16, halving the model
allocation from ~28 GB to ~14 GB. Combined with the checkpoint (~14 GB), total peak memory
during loading is ~28 GB -- well within the A40's 48 GB.

### Why bfloat16 is safe

- **Ampere+ GPUs** (A40, A100, H100) have native bfloat16 ALUs -- no performance penalty
- **Evo2's own checkpoint** is stored in bfloat16 (13.8 GB download for 7B params)
- **No precision loss**: the model was trained in bfloat16/mixed precision
- The sparse classifier already converts to float32 for numpy: `.cpu().float().numpy()`

### What does NOT work

- `torch.cuda.amp.autocast(dtype=torch.bfloat16)` -- autocast only affects operations,
  not `torch.empty()` calls during layer construction
- Loading to CPU first -- the evo2 library hardcodes CUDA device
- Gradient checkpointing -- irrelevant during model construction (no forward pass yet)

## GPU Memory Budget

After loading in bfloat16, the memory budget for Evo2 7B:

| GPU | VRAM | Model (bf16) | Available for inference | Status |
|-----|------|-------------|----------------------|--------|
| A40 | 48 GB | ~14 GB | ~34 GB | Works well |
| A100 | 80 GB | ~14 GB | ~66 GB | Comfortable |
| H100 | 80 GB | ~14 GB | ~66 GB | Comfortable (+ FP8 native) |
| RTX 4090 | 24 GB | ~14 GB | ~10 GB | Tight, may OOM on long sequences |
| RTX A5000 | 24 GB | ~14 GB | ~10 GB | Tight, may OOM on long sequences |

### Inference memory per sequence

For a context window of 8,192 bp (sparse exon classifier):
- Input tokens: negligible
- Activations through 28 blocks: ~2-4 GB (single sequence, no grad)
- Total per inference: ~16-18 GB on A40 (well within budget)

For longer sequences (e.g., 32K bp dense extraction):
- Activations scale linearly with sequence length
- Budget ~8-16 GB for 32K bp, ~32-64 GB for 131K bp
- Use `torch.cuda.empty_cache()` between sequences to reclaim activation memory

## Related Issues

- **FP8 patch**: GPUs with compute capability < 8.9 (A40=8.6, A100=8.0) need the
  `fp8_autocast` monkey-patch. See `model.py:_patch_fp8_if_needed()`.
- **Embedding dtype**: Embeddings from `model.encode()` are returned in bfloat16.
  Always use `.cpu().float().numpy()` to convert for downstream processing.
- **Storage challenge**: Dense per-nucleotide extraction produces ~400-500 GB.
  See `embedding-storage-challenge.md` for solutions.

## Verified On

- RunPod A40 (48 GB), CUDA 12.8, PyTorch 2.6, evo2 package (2026-03-12)
- Checkpoint: `evo2_7b` (13.8 GB download, bfloat16)
