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

## Two Pipeline Variants

### 07_genome_scale_splice_predictor.py (per-gene cache)

Original pipeline: extracts embeddings to per-gene HDF5 files, optionally
repacks into shards for fast training.  Good for single-chromosome runs
where disk space isn't tight.

```
Phase 1: Extract          Phase 1.5: Shard (--shard)       Phase 2: Train
──────────────────        ──────────────────────────        ──────────────
Per-gene HDF5             Shuffled shard files              DataLoader
(gzip compressed)         (uncompressed, contiguous)        (streaming)
  cache/chr22/              shards/                         batch_size=64+
    ENSG00001.h5              train_shard_000.h5
    ENSG00002.h5              train_shard_001.h5
    ...                       val_shard_000.h5
```

**Disk problem**: gzip only achieves ~1.4x compression on float32 embeddings
(not the 10x expected).  chr1 alone produces ~180 GB of "compressed" cache —
exceeding most pod volumes.

### 07a_direct_shard_splice_predictor.py (recommended)

Direct-to-shard pipeline: never creates per-gene HDF5 files.  Embeddings
are extracted gene-by-gene, accumulated in a memory buffer (~4 GB), and
flushed directly to shard files when the buffer fills.

```
Phase 1: Extract → Direct to Shard         Phase 2: Train
─────────────────────────────────           ──────────────
For each gene:                              DataLoader
  FASTA → model.encode() → emb             (ShardedWindowDataset)
  build_splice_labels() → lbl              batch_size=64+
  Append to memory buffer [4 GB]
  When buffer full → flush to shard file
  (no per-gene files on disk!)
```

**Disk usage**: Only shard files exist.  Train shards for 4 chromosomes
with SpliceBERT: ~15-20 GB total (vs ~250 GB+ with per-gene cache).

## Usage

### Quick run (single chromosome)

```bash
# Per-gene cache (07) — simple, works for small runs
python 07_genome_scale_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 \
    -o /tmp/07-chr22/

# Direct-to-shard (07a) — recommended
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 \
    -o /tmp/07a-chr22/
```

### Multi-chromosome with train/test split

```bash
# 07a handles multi-chrom without disk issues
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 1,2,3,22 \
    -o /workspace/output/splice_classifier/splicebert-multi-chrom/
```

### Resume from existing shards

```bash
# Skip Phase 1 if shards already exist
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 1,2,3,22 \
    --resume -o /workspace/output/splice_classifier/splicebert-multi-chrom/
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
    SequenceWindowDataset,  # Raw DNA + labels from FASTA (for fine-tuning)
    repack_into_shards,     # Per-gene → shard conversion
)
```

Package location: `foundation_models/foundation_models/data/`

### ShardedWindowDataset (recommended for frozen-embedding training)

Reads from pre-packed shard files.  All handles kept open (typically < 20
files).  Used by both `07_*.py` (with `--shard`) and `07a_*.py` (always).

```python
shard_paths = sorted(shard_dir.glob("train*shard_*.h5"))
dataset = ShardedWindowDataset(shard_paths)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
# ... training loop ...
dataset.close()
```

### HDF5WindowDataset (per-gene fallback)

Reads from per-gene HDF5 files.  Uses an LRU cache of open file handles
(default 64) to avoid open/close overhead.  Used by `07_*.py` without
`--shard`.

```python
dataset = HDF5WindowDataset(manifest, gene_set)
print(dataset.class_counts)  # [n_neither, n_acceptor, n_donor]
loader = DataLoader(dataset, batch_size=64, shuffle=True)
# ... training loop ...
dataset.close()
```

### SequenceWindowDataset (for fine-tuning)

Reads raw DNA strings + labels from FASTA.  ~500x smaller than embedding
datasets (512 bytes per window vs 1 MB).  Used by `08_foundation_model_finetuning.py`.

```python
dataset = SequenceWindowDataset(
    gene_entries, gene_set, fasta_path, splice_sites_df,
    window_size=512, step_size=256,
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
for dna_strings, labels in loader:
    logits = model(list(dna_strings))  # Foundation model tokenizes internally
    ...
```

### repack_into_shards

Converts per-gene HDF5 → shuffled shard files.  Only needed with `07_*.py`;
`07a_*.py` writes shards directly during extraction.

```python
shard_paths = repack_into_shards(
    manifest, train_genes, output_dir, "train", seed=42,
)
```
