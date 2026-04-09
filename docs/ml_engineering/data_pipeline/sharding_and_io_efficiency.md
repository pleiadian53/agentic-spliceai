# Sharding and I/O Efficiency for Genomic Deep Learning

A practical guide to why and how training data is packed into shards,
with concrete examples from the agentic-spliceai codebase.

See also: [I/O Bottlenecks and DataLoader Tuning](io_bottlenecks_dataloader.md)
for the complementary guide on diagnosing and fixing data pipeline stalls.

---

## 1. The Problem: Per-Gene Files Don't Scale

During meta-layer training, we need data from ~12,000 genes.  The naive
storage format is one `.npz` file per gene:

```
gene_cache/train/
  gene-BRCA1.npz    (850 KB)
  gene-TP53.npz     (120 KB)
  gene-MYBPC3.npz   (340 KB)
  ...
  gene-TTN.npz      (12 MB)    ← largest human gene
```

Each file contains the gene's sequence, base scores, multimodal features,
and labels.  Reading one file is fast.  But training samples a different
random gene for every single window, which means:

- **12,500 batches/epoch** (100K samples / batch 8)
- **100,000 file opens/epoch** (one gene file per sample)
- Each `np.load()` call: ~0.5-2ms file open + seek + read + decompress

On a local NVMe SSD this is tolerable (~100-200ms overhead per epoch).
On a network volume (RunPod, cloud NFS) each file open adds 5-30ms of
latency, turning 100K file opens into **8-50 minutes of pure I/O wait
per epoch** — while the GPU sits idle.

### The numbers

| Storage | Per-file latency | 100K files/epoch | GPU utilization |
|---------|-----------------|------------------|-----------------|
| Local NVMe | 0.5 ms | ~50s | ~40% |
| Network volume | 10 ms | ~17 min | ~7% |
| Object storage (S3) | 30 ms | ~50 min | ~2% |

The GPU can process a batch in ~10ms.  At 10ms I/O per file, data
loading is 100x slower than compute.

---

## 2. The Solution: Pack Genes into Shards

Instead of one file per gene, pack many genes into a small number of
large files.  Each shard is a **single file** containing hundreds of
genes, organized for efficient random access.

### Before (per-gene NPZ)

```
gene_cache/train/
  gene-BRCA1.npz       ← 1 file open per access
  gene-TP53.npz
  ... (12,000 files)
```

100K samples/epoch = 100K file opens.

### After (per-chromosome HDF5 shards)

```
gene_cache/train_shards/
  shard_chr2.h5         ← contains ~900 genes
  shard_chr4.h5         ← contains ~750 genes
  shard_chr6.h5
  ... (12 shard files for training chromosomes)
```

12 file handles opened **once**, then kept open.
100K samples/epoch = 100K HDF5 dataset reads within already-open files.
Each read is a seek + slice — no file-open overhead.

---

## 3. HDF5 Shard Structure

Each shard is an HDF5 file with one group per gene:

```
shard_chr2.h5
├── gene-BRCA2/
│   ├── sequence      [gene_len]     uint8    (DNA as bytes)
│   ├── base_scores   [gene_len, 3]  float32  (donor, acceptor, neither)
│   ├── mm_features   [gene_len, 9]  float32  (multimodal channels)
│   └── labels        [gene_len]     int64    (0=donor, 1=acceptor, 2=neither)
├── gene-MSH2/
│   ├── ...
└── gene-EPCAM/
    ├── ...
```

HDF5 supports chunked storage and random slicing, so reading a 5001-bp
window from a 100K-bp gene is fast — HDF5 only reads the relevant chunks,
not the whole dataset.

### Building shards (in the training pipeline)

```python
# From 07_train_sequence_model.py
python -u examples/meta_layer/07_train_sequence_model.py \
    --use-shards \              # enable shard packing
    --cache-dir /path/to/gene_cache \
    ...
```

The training script first builds the per-gene NPZ cache (if not present),
then packs genes into per-chromosome HDF5 shards.  Subsequent runs skip
building and use the shards directly.

### The packing code

```python
# From shard_packing.py (simplified)
def repack_genes_into_shards(gene_entries, output_dir):
    # Group genes by chromosome
    by_chrom = defaultdict(list)
    for entry in gene_entries:
        by_chrom[entry.chrom].append(entry)

    for chrom, entries in by_chrom.items():
        shard_path = output_dir / f"shard_{chrom}.h5"
        with h5py.File(shard_path, "w") as f:
            for entry in entries:
                gene = np.load(entry.npz_path)
                grp = f.create_group(entry.gene_id)
                for key in ["sequence", "base_scores", "mm_features", "labels"]:
                    grp.create_dataset(key, data=gene[key])
```

---

## 4. Dataset Access Patterns

### Without shards: `SequenceLevelDataset`

```python
def __getitem__(self, idx):
    gene_idx = self._rng.choice(self._valid_genes)
    entry = self.gene_index[gene_idx]
    gene = np.load(entry.npz_path)      # ← file open every call
    # ... extract window from gene ...
```

Every `__getitem__` call opens a new file.

### With shards: `ShardedSequenceLevelDataset`

```python
def __getitem__(self, idx):
    gene_idx = self._rng.choice(self._valid_genes)
    entry = self.gene_index[gene_idx]
    handle = self._get_handle(entry.shard_path)  # ← cached, opened once
    grp = handle[entry.gene_id]
    # Slice directly from HDF5 — no full gene load
    base_scores = grp["base_scores"][out_start:out_end]
    # ...
```

File handles are cached per-process (fork-safe, keyed by process ID).
A single HDF5 handle stays open for the life of the DataLoader worker.

### Fork-safety note

HDF5 file handles cannot survive a `fork()`.  The sharded dataset uses
lazy handle initialization (opened on first access per worker process)
and the DataLoader must use `multiprocessing_context="spawn"`:

```python
DataLoader(
    dataset,
    num_workers=4,
    multiprocessing_context="spawn",  # required for HDF5
    persistent_workers=True,
)
```

See [I/O Bottlenecks](io_bottlenecks_dataloader.md#the-hdf5-fork-safety-gotcha)
for the full explanation.

---

## 5. Measured Impact

From the M1-S training run on an A40 pod:

| Configuration | GPU utilization | Time per epoch |
|--------------|-----------------|----------------|
| Per-gene NPZ, `num_workers=0` | ~7% | ~45 min |
| Per-gene NPZ, `num_workers=4` | ~15% | ~25 min |
| **HDF5 shards, `num_workers=4`** | **~40%** | **~8 min** |

The jump from 7% to 40% GPU utilization came from two changes:
sharding (eliminated file-open overhead) and multi-worker prefetch
(overlapped I/O with GPU compute).

The remaining 60% gap to full GPU utilization is due to the small model
size (367K params) — the forward/backward pass is so fast that even
optimized I/O can't keep up perfectly.  Larger models would see higher
utilization with the same I/O pipeline.

---

## 6. Alternative Shard Formats

HDF5 is not the only option.  Here's how common formats compare for
genomic training data:

| Format | Random access | Compression | Multi-worker safe | Best for |
|--------|:---:|:---:|:---:|---|
| **HDF5** | Yes (chunked) | Optional (gzip, lzf) | With `spawn` | Structured arrays, sliceable |
| **Parquet** | Column-level | Snappy/Zstd | Yes | Tabular data, Polars/Pandas |
| **WebDataset (.tar)** | No (sequential) | Optional | Yes | Images, streaming-only |
| **TFRecord** | No (sequential) | Optional | Yes | TensorFlow pipelines |
| **Zarr** | Yes (chunked) | Blosc/Zstd | Yes | Large arrays, cloud-native |
| **LMDB** | Yes (key-value) | No | Yes | Small records, fast random access |
| **SQLite** | Yes (indexed) | No | Read-only concurrent | Metadata + small blobs |

For genomic sequence-level data with variable-length genes and
multi-channel features, **HDF5** and **Zarr** are the strongest fits
because they support efficient slicing of contiguous array regions.

---

## 7. When to Shard

**Shard when**:
- Training accesses many individual files per epoch (>1000)
- GPU utilization is below 30% during training
- Data lives on network storage (not local NVMe)

**Don't bother sharding when**:
- Dataset fits in memory (just load it all upfront)
- Sequential access only (streaming formats like WebDataset work better)
- Files are already large (>100 MB each) — the per-file overhead is amortized

---

## 8. Summary

The core principle: **minimize the number of file-open operations during
training**.  Every file open pays a latency tax (0.5ms local, 10-30ms
network).  Sharding amortizes this cost across thousands of samples by
keeping a small number of large files open.

```
                Individual files              Shards
                ────────────────              ──────
Epoch start:    open 0 files                  open 12 shard files
Per sample:     open 1 file (0.5-30ms)        seek within open file (~0.01ms)
100K samples:   100K file opens               100K seeks (1000x faster)
```

The training pipeline in `07_train_sequence_model.py` supports both modes
via `--use-shards`.  On local NVMe, per-gene NPZ files are fast enough.
On GPU pods with network volumes, shards are essential.
