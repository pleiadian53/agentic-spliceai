# Training I/O Optimization for Foundation Model Classifiers

How to achieve fast training I/O when training splice site classifiers on
foundation model embeddings cached in HDF5 files.

## Problem

Foundation model embeddings are cached as per-gene HDF5 files during
Phase 1 (extraction).  When training a classifier (Phase 2), the
`DataLoader` reads windows from these files on every batch.

With **gzip compression** (the original default), each window read requires
decompression.  For genome-scale data this dominates training time:

| Metric | chr22 SpliceBERT on A40 |
|--------|------------------------|
| Windows | 45,404 |
| Batches/epoch (batch=16) | 2,838 |
| Time per batch | ~3.3 s (gzip I/O) |
| **Epoch time** | **~2.7 hours** |
| GPU utilization | < 5% |

The GPU sits idle waiting for I/O.

## Disk Space Tradeoffs

| Format | Compression | Read speed | chr22 size | Full genome |
|--------|------------|------------|-----------|-------------|
| gzip (level 4) | ~10x | Slow (~3s/batch) | ~5 GB | ~175 GB |
| Uncompressed | 1x | Fast (~0.1s/batch) | ~49 GB | ~1.7 TB |
| LZ4 | ~2x | Fast (~0.15s/batch) | ~25 GB | ~850 GB |

**Per-gene HDF5 files** are the canonical cache (one file per gene, created
in Phase 1).  They can use any compression.

**Shard files** are a derived, training-optimized format.  Always written
uncompressed for maximum read speed.

## Three-Tier I/O Strategy

```
Phase 1: Extract          Phase 1.5: Shard (optional)      Phase 2: Train
──────────────────        ──────────────────────────        ──────────────
Per-gene HDF5             Shuffled shard files              DataLoader
(gzip or uncompressed)    (uncompressed, contiguous)        (streaming)
  cache/chr22/              shards/                         batch_size=64+
    ENSG00001.h5              train_shard_000.h5
    ENSG00002.h5              train_shard_001.h5
    ...                       val_shard_000.h5
```

### Tier 1: Per-gene HDF5 (always created)

- One file per gene: `cache/{chrom}/{gene_id}.h5`
- Datasets: `embeddings [n_windows, W, H]`, `labels [n_windows, W]`
- Supports `--resume` (skip genes with existing cache)
- Default: uncompressed (fast reads, larger disk)

### Tier 2: Shard files (optional, `--shard` flag)

- Reads per-gene files (any compression), shuffles windows, packs into
  large contiguous shard files (50K windows per shard, uncompressed)
- One-time cost: ~5 min for chr22, ~30 min for full genome
- 10-50x faster random-access reads vs per-gene gzip

### Tier 3: DataLoader (automatic)

- `HDF5WindowDataset` reads from per-gene files (LRU handle cache)
- `ShardedWindowDataset` reads from shard files (all handles open)
- Both expose `.class_counts` for focal loss class weights

## Usage

### Quick run (single chromosome, no shards)

```bash
python 07_genome_scale_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 \
    -o /tmp/07-chr22/
```

### Fast training with shards

```bash
# Extract + shard + train in one command
python 07_genome_scale_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 \
    --shard -o /tmp/07-chr22/

# Or: shard existing gzip cache (skip re-extraction)
python 07_genome_scale_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 \
    --resume --shard -o /tmp/07-chr22/
```

### Batch size selection

| GPU | SpliceBERT (512-dim) | Evo2 (4096-dim) |
|-----|---------------------|-----------------|
| A40 (48 GB) | batch=128 | batch=64 |
| A100 (80 GB) | batch=256 | batch=128 |
| M1 Mac (16 GB) | batch=16 | batch=4 |

Use `--batch-size N` to override the default (64).

## Expected Performance

| Config | Batches/epoch | Time/batch | Epoch time |
|--------|--------------|------------|------------|
| gzip, batch=16 | 2,838 | ~3.3 s | ~2.7 hr |
| Uncompressed, batch=64 | 710 | ~0.1 s | ~1.2 min |
| Sharded, batch=64 | 710 | ~0.05 s | ~35 s |
| Sharded, batch=128 | 355 | ~0.03 s | ~10 s |

**Full genome estimate** (sharded, batch=128): ~5 s/epoch.

## Reusable Package

The I/O classes are available as a standalone package:

```python
from foundation_models.data import (
    HDF5WindowDataset,      # Per-gene HDF5, LRU handle cache
    ShardedWindowDataset,   # Pre-packed shard files
    repack_into_shards,     # Per-gene → shard conversion
)
```

Package location: `foundation_models/foundation_models/data/`

### HDF5WindowDataset

Reads from per-gene HDF5 files.  Uses an LRU cache of open file handles
(default 64) to avoid open/close overhead.

```python
dataset = HDF5WindowDataset(manifest, gene_set)
print(dataset.class_counts)  # [n_neither, n_acceptor, n_donor]
loader = DataLoader(dataset, batch_size=64, shuffle=True)
# ... training loop ...
dataset.close()
```

### ShardedWindowDataset

Reads from pre-packed shard files.  All handles kept open (typically < 20
files).

```python
shard_paths = sorted(shard_dir.glob("train_shard_*.h5"))
dataset = ShardedWindowDataset(shard_paths)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
# ... training loop ...
dataset.close()
```

### repack_into_shards

Converts per-gene HDF5 → shuffled shard files.

```python
shard_paths = repack_into_shards(
    manifest, train_genes, output_dir, "train",
    windows_per_shard=50_000, seed=42,
)
```
