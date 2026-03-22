# Large-Scale ML Training on Genomic Data

A practical guide to training classifiers on genome-scale datasets, from data
organization to GPU utilization.  Uses foundation model splice site prediction
as the running example, but the principles apply broadly to any large-scale
supervised learning on sequential data.

## Table of Contents

1. [The Scaling Problem](#the-scaling-problem)
2. [Data Layout: Why Storage Format Matters](#data-layout-why-storage-format-matters)
3. [Data Sharding: Pre-Shuffled Sequential Reads](#data-sharding-pre-shuffled-sequential-reads)
4. [Streaming vs Bulk Loading](#streaming-vs-bulk-loading)
5. [Batch Size and GPU Utilization](#batch-size-and-gpu-utilization)
6. [Compression Tradeoffs](#compression-tradeoffs)
7. [Class Imbalance at Scale](#class-imbalance-at-scale)
8. [Probability Calibration](#probability-calibration)
9. [Putting It Together: The Full Pipeline](#putting-it-together-the-full-pipeline)
10. [Appendix: Quick Reference](#appendix-quick-reference)

---

## The Scaling Problem

Training a classifier on 10 genes takes seconds.  Training on a full genome
(~19,000 genes, ~500K windowed samples) takes days — if you do it naively.
The bottleneck is almost never the GPU; it's how you feed data to it.

A typical genome-scale training setup:

| Component | 10-gene run | chr22 (410 genes) | Full genome |
|-----------|------------|-------------------|-------------|
| Training windows | 13 | 45,404 | ~500,000 |
| Data size (SpliceBERT) | 13 MB | 47 GB | ~500 GB |
| Data size (Evo2) | 100 MB | 375 GB | ~4 TB |
| Naive epoch time (A40) | 2 s | 2.7 hr | ~4 days |
| Optimized epoch time | 2 s | ~1 min | ~5 min |

The difference between 4 days and 5 minutes per epoch is entirely about I/O
strategy.

## Data Layout: Why Storage Format Matters

### The naive approach: one file per sample

```
cache/
  chr22/
    ENSG00000015475.h5    # TP53: 150 windows, gzip-compressed
    ENSG00000141510.h5    # BRCA1: 200 windows, gzip-compressed
    ...                   # 409 files
```

Each training batch requires:
1. Pick 64 random windows (from random genes)
2. Open 64 HDF5 files (or reuse cached handles)
3. Seek to the correct window index
4. Decompress gzip chunk
5. Copy to CPU tensor
6. Transfer to GPU

Steps 2-4 dominate.  On a modern GPU (A40, A100), the forward + backward pass
for a 1M-parameter classifier takes <10 ms.  The HDF5 gzip read takes >200 ms.
**The GPU is idle 95% of the time.**

### The root cause: random I/O on compressed data

Storage devices are optimized for sequential reads.  An SSD can read 3 GB/s
sequentially but only ~50 MB/s for small random reads.  When you add gzip
decompression (CPU-bound, ~200 MB/s), each random access becomes expensive.

**Key insight**: Training needs random sample order (SGD), but storage needs
sequential access.  These are fundamentally at odds — unless you reorganize
the data.

## Data Sharding: Pre-Shuffled Sequential Reads

Sharding resolves the tension between random training order and sequential I/O
by pre-shuffling the data at preparation time.

### How it works

```
Step 1: Collect all windows across all genes
Step 2: Shuffle the window order (random seed for reproducibility)
Step 3: Pack shuffled windows into large contiguous shard files
Step 4: Train by reading shards sequentially
```

```
Per-gene files (canonical)          Shard files (training-optimized)
─────────────────────────           ──────────────────────────────
cache/chr22/                        shards/
  gene_A.h5 [w0,w1,w2]      ──→      train_shard_000.h5 [w7,w2,w15,w4,w9,...]
  gene_B.h5 [w3,w4,w5]               train_shard_001.h5 [w1,w11,w3,w8,w6,...]
  gene_C.h5 [w6,w7,w8]               val_shard_000.h5   [w5,w0,w10,...]
  ...
```

Within each shard, windows are stored contiguously and uncompressed.  Reading
a batch means reading a sequential chunk of a single file — exactly what SSDs
are optimized for.

### Industry practice

This is the standard approach for large-scale ML training:

| Framework | Shard format | Used by |
|-----------|-------------|---------|
| TFRecord | Protocol buffer shards | Google, DeepMind, Enformer |
| WebDataset | TAR file shards | LAION, OpenCLIP, Stable Diffusion |
| Mosaic StreamingDataset | MDS binary shards | MosaicML, Databricks |
| HuggingFace Datasets | Apache Arrow files | Most NLP training |
| Our approach | HDF5 shard files | Genomic foundation models |

The format varies, but the principle is the same: **shuffle once, read
sequentially many times**.

### Is the shuffle good enough?

A common concern: if you shuffle once and read in the same order every epoch,
won't the model overfit to the ordering?

In practice, this doesn't matter because:

1. **The DataLoader adds batch-level shuffling**: PyTorch `DataLoader(shuffle=True)`
   shuffles the index before each epoch, so within-shard window order varies.
2. **Shard order is randomized per epoch**: With multiple shards, the DataLoader
   visits shards in different orders each epoch.
3. **Empirically equivalent**: Training curves with pre-shuffled shards match
   those with true random access.  The original SpliceAI paper used TFRecords
   (pre-shuffled shards) for all their published results.

For extra randomness, you can reshuffle and re-shard periodically (e.g., every
10 epochs), but this is rarely necessary.

## Streaming vs Bulk Loading

### Bulk loading (don't do this)

```python
# Loads ALL data into RAM before training — OOMs at scale
train_emb = np.concatenate([f["embeddings"][:] for f in all_files])  # 50 GB!
train_lbl = np.concatenate([f["labels"][:] for f in all_files])
dataset = TensorDataset(torch.tensor(train_emb), torch.tensor(train_lbl))
```

This works for 10 genes (13 MB).  It fails for chr22 (47 GB) and is
impossible for full genome (500 GB).

### Streaming (correct approach)

```python
# Reads one batch at a time — constant memory regardless of dataset size
dataset = ShardedWindowDataset(shard_paths)    # Opens file handles, no data read
loader = DataLoader(dataset, batch_size=64)    # Lazy iterator
for batch_emb, batch_lbl in loader:            # Reads 64 windows per iteration
    batch_emb = batch_emb.to(device)           # CPU → GPU transfer
    logits = model(batch_emb)                  # Forward pass
    ...
```

Peak memory is proportional to batch size, not dataset size:

| Approach | chr22 peak RAM | Full genome peak RAM |
|----------|---------------|---------------------|
| Bulk loading | ~50 GB | ~500 GB (impossible) |
| Streaming (per-gene) | ~200 MB | ~200 MB |
| Streaming (sharded) | ~200 MB | ~200 MB |

### The DataLoader as a decoupling layer

PyTorch's `DataLoader` abstracts the storage format from the training loop.
The model doesn't know (or care) whether data comes from per-gene files,
shards, or a database.  This lets you swap I/O strategies without changing
training code:

```python
# Same training loop works with any dataset
if use_shards:
    dataset = ShardedWindowDataset(shard_paths)
else:
    dataset = HDF5WindowDataset(manifest, gene_set)

loader = DataLoader(dataset, batch_size=64, shuffle=True)
classifier.fit_streaming(loader, ...)
```

## Batch Size and GPU Utilization

### Why batch size matters for I/O

Each batch requires one I/O round trip.  Larger batches amortize I/O overhead:

| Batch size | Batches/epoch (45K windows) | I/O round trips |
|-----------|---------------------------|----------------|
| 16 | 2,838 | 2,838 |
| 64 | 710 | 710 |
| 128 | 355 | 355 |
| 256 | 178 | 178 |

With sharded I/O (~0.05 s/batch), the difference between batch=16 and
batch=256 is 142 s vs 9 s per epoch.

### How large can you go?

The limit is GPU VRAM.  Each batch needs memory for:
- Input tensor: `batch × window_size × hidden_dim × 4 bytes`
- Model parameters + gradients: fixed (small for classifier heads)
- Activations for backprop: proportional to batch size

| GPU (VRAM) | SpliceBERT (512-dim) | Evo2 (4096-dim) |
|------------|---------------------|-----------------|
| M1 Mac (16 GB) | batch=16 | batch=4 |
| A40 (48 GB) | batch=128 | batch=64 |
| A100 (80 GB) | batch=256 | batch=128 |
| H100 (80 GB) | batch=256 | batch=128 |

### Learning rate scaling

When increasing batch size, scale the learning rate proportionally:

```
lr_scaled = lr_base * (batch_size / batch_base)
```

Example: if `lr=0.001` works for `batch=16`, use `lr=0.004` for `batch=64`.
This keeps the effective gradient step size similar.  Some practitioners use
a warmup schedule (linear ramp over the first 1-5 epochs) to stabilize large-batch
training.

For our classifier head (~1M params), this is less critical than for large
models, but still good practice.

## Compression Tradeoffs

### The compression spectrum

| Format | Ratio | Decompress speed | Random access | Best for |
|--------|-------|-----------------|---------------|----------|
| None | 1x | N/A (direct read) | Fast | Training (shards) |
| LZ4 | ~2x | ~4 GB/s | Fast | Large datasets, limited disk |
| Zstandard | ~3-4x | ~1 GB/s | Moderate | Cold storage, archival |
| gzip | ~10x | ~200 MB/s | Slow | Archival, network transfer |

**Rule of thumb**: Use no compression or LZ4 for anything that's read during
training.  Use gzip only for archival or network transfer.

### Genomic embedding compression ratios

Foundation model embeddings are floating-point matrices — they don't compress
well.  The "10x" gzip ratio above is optimistic; real compression on float32
embeddings is typically 2-5x.

| Data type | gzip ratio | Why |
|-----------|-----------|-----|
| DNA sequences (text) | 10-20x | Highly repetitive (4 chars) |
| Splice labels (int8) | 50-100x | 99.9% zeros |
| Embeddings (float32) | 2-5x | Low redundancy, high entropy |

This means the disk savings from compressing embeddings are modest, while the
I/O cost is severe.  **Don't compress embeddings for training.**

## Class Imbalance at Scale

Splice sites are ~0.01-0.1% of nucleotide positions.  At genome scale, this
means millions of "neither" positions for every splice site.

### Focal loss with class weights

Standard cross-entropy treats all examples equally.  For extreme imbalance,
focal loss down-weights easy examples and focuses on hard ones:

```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

With `gamma=2.0`, a correctly-classified position with `p=0.95` gets a loss
weight of `(1 - 0.95)^2 = 0.0025` — 400x less than a hard example with `p=0.5`.

Class weights (`alpha`) further compensate for frequency imbalance:

```
alpha = total_samples / (num_classes * class_count)
```

For our data: `neither=0.3, acceptor=1000, donor=1000`.

### Computing class weights without loading all labels

At scale, you can't load all labels into RAM to count classes.  The streaming
datasets scan labels at init time and expose `.class_counts`:

```python
dataset = HDF5WindowDataset(manifest, train_genes)
weights = compute_class_weights_from_counts(dataset.class_counts)
# weights: [0.3, 1000.0, 1000.0]
```

This reads only labels (tiny: int8) without touching embeddings (large: float32).

## Probability Calibration

### The calibration problem

Focal loss + extreme class weights produce good **ranking** (AUROC, AUPRC) but
poor **calibration** (predicted probabilities don't match true frequencies).
A prediction of `P(donor) = 0.3` might actually correspond to a 0.001% true
positive rate.

### Temperature scaling

Post-hoc temperature scaling fixes calibration without changing model weights.
A learnable temperature vector divides logits before softmax:

```python
# Before softmax
scaled_logits = logits / temperature  # temperature: [T_neither, T_acceptor, T_donor]
probs = softmax(scaled_logits)
```

- `T > 1` **softens** predictions (reduces overconfidence)
- `T < 1` **sharpens** predictions (increases confidence)

Optimized on the validation set by minimizing NLL loss, typically converges
in <1000 steps.

### When to calibrate

Calibration matters when you need interpretable probabilities — clinical
reporting, evidence fusion with other modalities, threshold selection.
It doesn't affect ranking metrics (AUROC stays the same).

**Pipeline placement**: After training, before evaluation.

```
Phase 2: Train → Phase 2.5: Calibrate → Phase 3: Evaluate
```

### Measuring calibration: ECE

Expected Calibration Error (ECE) bins predictions by confidence and measures
the gap between predicted confidence and actual accuracy:

```
ECE = sum over bins: |accuracy_in_bin - avg_confidence_in_bin| * fraction_in_bin
```

ECE = 0 means perfect calibration.  Our results: ECE 0.019 → 0.0003 after
temperature scaling.

## Putting It Together: The Full Pipeline

Two pipeline variants exist.  **07a (direct-to-shard)** is recommended for
multi-chromosome and full-genome runs because it avoids the per-gene cache
that can fill disk.

### Pipeline A: Per-gene cache + optional sharding (07_*.py)

```
Phase 1: Extract & Cache              Phase 1.5: Shard (--shard)
─────────────────────────              ──────────────────────────
For each gene:                         Shuffle all windows
  FASTA → model.encode() → emb        Pack into contiguous shards
  build_splice_labels() → lbl          (4 GB per shard, uncompressed)
  Save to per-gene HDF5 (gzip)

Phase 2: Train                         Phase 2.5: Calibrate
──────────────────                     ─────────────────────
DataLoader(ShardedWindowDataset)       Collect val logits
  → batch to GPU                       Optimize temperature vector
  → forward + focal loss               (Adam, ~500 steps)
  → backward + AdamW                   Save temperature.pt
  → early stopping on val AUPRC

Phase 3: Evaluate
─────────────────
Load per-gene embeddings
predict() with temperature scaling
Report AUROC, AUPRC, F1, ECE
```

**Limitation**: gzip-compressed float32 embeddings achieve only ~1.4x
compression.  chr1 alone produces ~180 GB of cache.  Multi-chromosome runs
quickly exhaust pod storage.

### Pipeline B: Direct-to-shard (07a_*.py, recommended)

```
Phase 1: Extract → Direct to Shard
─────────────────────────────────
Pre-compute split (metadata-only gene scan)
For each gene (one at a time):
  FASTA → model.encode() → emb
  build_splice_labels() → lbl
  Window → append to memory buffer [4 GB]
  When buffer full → flush to shard file
  (NO per-gene files on disk!)
Test genes: skip embedding (Phase 3 runs live)

Phase 2: Train                         Phase 2.5: Calibrate
──────────────────                     ─────────────────────
DataLoader(ShardedWindowDataset)       (same as Pipeline A)
  → batch to GPU
  → forward + focal loss
  → backward + AdamW

Phase 3: Evaluate
─────────────────
Run model LIVE on test genes (no cached embeddings)
predict() with temperature scaling
Report AUROC, AUPRC, F1, ECE
```

**Disk budget**: Only shard files + test gene live inference.  4 chromosomes
with SpliceBERT: ~15-20 GB total (vs ~250 GB+ with per-gene cache).

### Command-line usage

```bash
# Quick test (10 genes, synthetic data)
python 07a_direct_shard_splice_predictor.py --mock --n-genes 10 -o /tmp/test/

# Single chromosome
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 22 -o /tmp/chr22/

# Multi-chromosome with train/test split (recommended)
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 1,2,3,22 \
    -o /workspace/output/splice_classifier/splicebert-multi-chrom/

# Full genome
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes all \
    --batch-size 128 -o /workspace/output/genome-splicebert/

# Resume from existing shards (skip Phase 1)
python 07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert --chromosomes 1,2,3,22 \
    --resume -o /workspace/output/splice_classifier/splicebert-multi-chrom/
```

## Appendix: Quick Reference

### Memory budget rules

| Resource | Budget | Why |
|----------|--------|-----|
| Shard buffer (RAM) | 4 GB default | Fits on any pod, auto-computed from embedding dims |
| Training batch (VRAM) | batch_size * W * H * 4 | Forward + backward fits |
| Validation accumulation (RAM) | N_val * 3 * 4 (probs only) | Only probs + labels, not embeddings |
| Calibration (RAM/VRAM) | N_val * 3 * 4 (logits) | Flat logits for temperature optimization |

### Performance expectations (A40, SpliceBERT)

| Stage | chr22 | 4 chroms | Full genome |
|-------|-------|----------|-------------|
| Extraction (Phase 1) | ~7 min | ~1 hr | ~4 hr |
| Sharding (07: separate step) | ~3 min | ~10 min | ~30 min |
| Direct-to-shard (07a: included in Phase 1) | — | — | — |
| Training epoch (sharded, batch=64) | ~1 min | ~3 min | ~5 min |
| Calibration | ~5 s | ~10 s | ~30 s |
| Evaluation (live inference) | ~2 min | ~5 min | ~20 min |

### Disk usage comparison (SpliceBERT, 4 chromosomes)

| Pipeline | Per-gene cache | Shards | Total |
|----------|---------------|--------|-------|
| 07 (no shard) | ~250 GB (gzip) | — | ~250 GB |
| 07 + shard | ~250 GB + ~20 GB | ~20 GB | ~270 GB |
| **07a (direct)** | **0** | **~20 GB** | **~20 GB** |

### Reusable package

```python
from foundation_models.data import (
    HDF5WindowDataset,      # Per-gene HDF5, LRU handle cache
    ShardedWindowDataset,   # Pre-packed shard files
    SequenceWindowDataset,  # Raw DNA + labels (for fine-tuning, 08_*.py)
    repack_into_shards,     # Per-gene → shard conversion
)

from foundation_models.classifiers.losses import (
    FocalLoss,                         # gamma + alpha class weights
    ECELoss,                           # Expected Calibration Error
    compute_class_weights_from_counts, # Weights from label counts
)

from foundation_models.classifiers.splice_classifier import SpliceClassifier
# .fit_streaming()  — train from DataLoaders
# .calibrate()      — post-hoc temperature scaling
# .predict()        — auto-applies temperature if calibrated

from foundation_models.classifiers.finetune import SpliceFineTuneModel
# End-to-end fine-tuning wrapper (08_*.py)
# Strategies: last_n, lora, full
```

### Further reading

- Lin et al. (2017) — [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- Naeini et al. (2015) — [Obtaining Well Calibrated Probabilities Using Bayesian Binning into Quantiles](https://people.cs.pitt.edu/~milos/research/2015/BBQ.pdf)
- Guo et al. (2017) — [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599)
- Jaganathan et al. (2019) — [Predicting Splicing from Primary Sequence with Deep Learning](https://doi.org/10.1016/j.cell.2018.12.015) (SpliceAI)
