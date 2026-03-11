# Embedding Storage Challenge: Per-Nucleotide Foundation Model Representations

## The Problem

Genomic foundation models like Evo2 produce **per-nucleotide embeddings** with
high-dimensional hidden states. Storing these naively for genome-scale datasets
quickly exceeds practical disk and memory limits.

### Concrete Numbers (Evo2 7b, MANE ~19K genes)

| Metric | Value |
|--------|-------|
| Hidden dimension | 4,096 (float32) |
| Bytes per nucleotide | 16 KB |
| Average human gene length | ~50 kb |
| Largest genes (e.g., DMD, RBFOX1) | 1–2.2 Mb |
| Total gene nucleotides (~19K genes) | ~2.4 billion |
| **Uncompressed storage** | **~37 TB** |
| **gzip-compressed HDF5 (observed)** | **~400–500 GB** |

### Observed: chr1 alone → ~50 GB (compressed)

During our first full extraction on an A40 (RunPod), chr1 (1,994 genes) consumed
**~38 GB and growing** before we terminated the run. Extrapolating across all 24
chromosomes yields **~400-500 GB** — far exceeding a typical 150 GB RunPod volume.

## Why It's So Large

```
1 nucleotide × 4,096 floats × 4 bytes = 16,384 bytes = 16 KB
1 gene (50 kb avg) × 16 KB = 800 MB uncompressed per gene
19,000 genes × 800 MB ≈ 15 TB uncompressed → ~400 GB gzip
```

gzip achieves ~30-40x compression on embedding data (structured float32), but the
raw scale is still enormous.

## Solutions (State of the Art)

### 1. Streaming Extract-and-Train (Recommended for our use case)

**Approach**: Process one chromosome at a time — extract embeddings, train/accumulate
gradients, then discard the embeddings before moving to the next chromosome.

```
for chrom in chr1..chrY:
    embeddings = extract(chrom)          # ~20-50 GB temporarily
    train_one_epoch(model, embeddings)   # accumulate gradients
    del embeddings                       # free disk
```

**Pros**: Bounded disk usage (~50 GB peak for largest chromosome), works on any volume.
**Cons**: Multi-epoch training requires re-extracting or cycling through chromosomes.
Can cache embeddings on fast local NVMe if available.

**Status**: Implementing in `06_extract_and_train.py`.

### 2. Dimensionality Reduction Before Storage

**Approach**: Reduce the 4,096-dim embedding to a lower-dimensional representation
before writing to disk.

| Method | Output dim | Compression | Quality |
|--------|-----------|-------------|---------|
| PCA | 128–512 | 8–32x | Good for linear probes |
| Random projection | 256–512 | 8–16x | Preserves distances (JL lemma) |
| Learned projection | 64–256 | 16–64x | Best if task-specific |
| Autoencoder | 128–256 | 16–32x | Nonlinear, preserves more info |

**Storage with PCA-256**: 19K genes × 50kb × 256 × 4 bytes ≈ **24 GB** (manageable)

**Cons**: Information loss (may hurt performance for fine-grained splice site tasks).
PCA must be fit on a representative subset first.

### 3. Float16 / BFloat16 Storage

**Approach**: Store embeddings in half precision instead of float32.

**Storage**: Halves to ~200-250 GB (still too large for a 150 GB volume).
**Quality**: Minimal loss — foundation model outputs are BFloat16 natively on Ampere+.
**Implementation**: `emb.half().numpy()` → HDF5 with `dtype=np.float16`.

Best combined with another method (e.g., PCA-256 + float16 → **~12 GB**).

### 4. Sparse / Selective Storage

**Approach**: Only store embeddings for positions of interest (splice sites, exon
boundaries, ±N flanking nucleotides).

| Strategy | Positions stored | Reduction factor |
|----------|-----------------|------------------|
| Splice sites ± 50 nt | ~200 per gene | ~250x |
| Exon regions only | ~30% of gene | 3x |
| Peaks > threshold | varies | 10–100x |

**Pros**: Massive reduction, directly aligned with downstream task.
**Cons**: Requires knowing positions of interest upfront (chicken-and-egg for novel
splice site discovery).

### 5. Chunked/Windowed Training (No Full-Gene Storage)

**Approach**: Extract embeddings for fixed-size windows (e.g., 1024 nt), train
immediately, discard. Never store full-gene embeddings.

```python
for gene in genes:
    for window in sliding_windows(gene.sequence, size=1024, stride=512):
        emb = model.encode(window)        # 1024 × 4096 = 16 MB
        loss = classifier(emb, labels)
        loss.backward()
    optimizer.step()
```

**Pros**: Constant memory (~16 MB per window), no disk storage needed.
**Cons**: Loses long-range context across windows. Requires model inference during
training (much slower than pre-extracted embeddings).

### 6. Hierarchical / Progressive Extraction

**Approach**: Start with low-resolution (mean-pooled per exon/intron), identify
regions of interest, then extract full-resolution only for those regions.

```
Phase 1: Mean-pool per 1kb window → ~50 positions/gene → ~3 GB total
Phase 2: Full resolution for top 5% most interesting regions → ~20 GB
```

**Pros**: Efficient exploration of the embedding space.
**Cons**: Two-phase pipeline adds complexity.

## Our Strategy

For the Evo2-based exon classifier, we implement **Option 1 (Streaming)** as the
primary approach:

1. Extract embeddings for one chromosome
2. Train classifier on that chromosome's data
3. Delete chromosome embeddings
4. Repeat for all chromosomes
5. Evaluate on held-out chromosomes

Future improvements:
- Add PCA reduction (Option 2) for multi-epoch training scenarios
- Float16 storage (Option 3) as a simple default
- Selective extraction (Option 4) for splice-site-specific tasks

## References

- Johnson-Lindenstrauss lemma (random projections): [Wikipedia](https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma)
- Evo2 paper: Brixi et al. (2025), "Genome-scale foundation models for biology"
- HDF5 compression benchmarks: [h5py docs](https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline)
- SpliceAI chromosome splitting: Jaganathan et al. (2019) — train on even chroms, test on odd
