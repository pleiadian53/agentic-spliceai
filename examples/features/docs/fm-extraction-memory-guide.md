# Foundation Model Extraction: Memory Management Guide

## The Two Memory Problems

Extracting per-position embeddings from large foundation models (Evo2 7B, 4096-dim)
creates two distinct resource pressures that require different solutions:

1. **CPU RAM exhaustion (OOM)** during extraction — the model produces a dense
   embedding matrix for each gene, and large genes can exceed system memory
2. **Disk storage explosion** when saving embeddings — storing raw vectors for
   millions of positions requires tens of gigabytes

These problems are related but not identical. Solving storage (via position
sampling) doesn't automatically solve OOM (which depends on per-gene memory),
and solving OOM (via position-targeted extraction) doesn't automatically solve
storage (which depends on total positions across all genes).

```
Gene sequence (500 KB)
    │
    ├─ Model inference (GPU)     ← bounded by max_context (~32K chunks)
    │
    ├─ Stitch full embedding     ← OOM RISK: 500K × 4096 × 4 bytes = 8 GB per strand
    │   (gene_len × hidden_dim)    With dual-strand: 2 × 8 GB = 16 GB peak
    │
    ├─ Index sampled positions   ← tiny: ~500 positions × 4096 × 4 bytes = 8 MB
    │
    └─ Write to disk             ← storage: ~500 positions × 8 scalar features = 4 KB
```

## Problem 1: CPU RAM OOM During Extraction

### What Happens

The foundation model processes gene sequences in chunks (e.g., 32K nucleotides
at a time for Evo2). Each chunk produces a `(chunk_len, hidden_dim)` embedding
on GPU, which is moved to CPU. The **stitching step** reassembles all chunks
into a single `(gene_len, hidden_dim)` numpy array on CPU.

For large genes with dual-strand extraction (forward + reverse complement):

| Gene Length | Single Strand | Dual Strand (peak) | Example Genes |
|-------------|--------------|-------------------|---------------|
| 50 KB | 800 MB | 1.6 GB | Typical gene |
| 200 KB | 3.2 GB | 6.4 GB | BRCA2, TP53 region |
| 500 KB | 8 GB | 16 GB | SYN3, MYT1L |
| 700 KB | 11.2 GB | 22.4 GB | TTC28, LARGE1 |
| 1 MB | 16 GB | 32 GB | RBFOX1 (largest) |

An A40 pod typically has 62 GB system RAM. Model weights primarily reside in
**VRAM** (~41 GB for Evo2 7B including CUDA overhead), but PyTorch loading
creates temporary CPU-side buffers during checkpoint deserialization, and some
residual references (~2-5 GB) may persist until garbage collection. Combined
with Python overhead (~1-2 GB), pyfaidx index, and Polars DataFrames, available
system RAM for numpy embedding arrays is roughly ~50-55 GB.

A 700 KB gene with dual-strand stitching (22.4 GB peak for the numpy arrays)
fits within that budget in isolation. But if the previous gene's arrays haven't
been fully freed by the garbage collector before the next gene starts, the
cumulative pressure can exceed available RAM.

The first A40 run (with `max_gene_length=1_000_000`) hit 100% system memory
and the pod became unresponsive — the OOM killer couldn't reclaim memory fast
enough because the numpy array allocation was a single atomic operation.

### The OOM Sequence

```
System RAM budget: ~55 GB available (62 GB total - residual model buffers - Python)
Model weights: primarily in VRAM (~41 GB), not competing for system RAM

1. Gene MYT1L (542 KB) starts processing
2. Forward strand: 17 chunks × 32K → each chunk encoded on GPU, moved to CPU
   → stitch all chunks into (542K, 4096) numpy array = 8.4 GB system RAM
3. Reverse complement: 17 more chunks → stitch to (542K, 4096) = 8.4 GB system RAM
4. Both arrays + flip + average coexist: peak ~25 GB system RAM
5. Gene completes → arrays should be freed, but GC timing is unpredictable
6. Next large gene starts before GC fully collects → 25 + 8.4 = 33.4 GB
7. Several large genes in sequence → cumulative pressure exceeds ~55 GB available
8. Pod becomes unresponsive (OOM killer or swap thrash)
```

### Solution: Position-Targeted Extraction

Instead of stitching the full gene embedding, process chunks sequentially and
only keep embeddings at sampled positions:

```
Gene sequence (500 KB) → 17 chunks of 32K
    │
    Chunk 1: encode → (32K, 4096) on CPU
    │   ├─ 12 sampled positions in this chunk's range → keep 12 rows
    │   └─ discard chunk embedding (free 512 MB)
    │
    Chunk 2: encode → (32K, 4096) on CPU
    │   ├─ 8 sampled positions → keep 8 rows
    │   └─ discard (free 512 MB)
    │
    ... (15 more chunks)
    │
    Result: (500, 4096) = 8 MB  ← only sampled positions kept
```

Peak CPU RAM: **one chunk** (~512 MB) instead of the full gene (~8 GB).

This is implemented in `_extract_at_positions_single_strand()` in
`07_streaming_fm_scalars.py`. It uses the chunking metadata (`chunk.keep_start`,
`chunk.keep_end`, `chunk.global_start`) to determine which sampled positions
fall within each chunk's authoritative range (after overlap trimming).

For the gradient feature (`fm_local_gradient`), we also request the +-1 genomic
neighbors of each sampled position. This adds at most 2 extra positions per
sample — negligible overhead:

```python
# Expand indices to include ±1 neighbors for gradient computation
neighbor_indices = np.unique(np.clip(
    np.concatenate([local_indices - 1, local_indices, local_indices + 1]),
    0, len(gene_seq) - 1,
))
```

### Memory Comparison

| Approach | Peak RAM (500 KB gene, dual-strand) | Genes Supported |
|----------|-------------------------------------|-----------------|
| Full-gene stitching | ~16 GB | Up to ~500 KB safely |
| **Position-targeted** | **~1 GB** | **Any gene length** |

With position-targeted extraction, `max_gene_length` can be raised to 2 MB,
covering 99.9%+ of human genes (only RBFOX1 at 1.7 MB is close to the limit).

### Additional In-Place Memory Optimization

For dual-strand averaging, the in-place pattern avoids holding 4 arrays
simultaneously:

```python
# Bad: 4 arrays coexist (emb_fwd, emb_rc, emb_rc_flipped, result)
emb_rc_flipped = emb_rc[::-1].copy()
result = (emb_fwd + emb_rc_flipped) / 2.0  # peak: 4 × size

# Good: in-place operations, max 2 arrays
emb_rc = emb_rc[::-1].copy()   # flip in-place (reuses emb_rc memory)
emb_fwd += emb_rc              # add in-place
del emb_rc                     # free immediately
emb_fwd *= 0.5                 # divide in-place
return emb_fwd                 # peak: 2 × size
```

This is less relevant with position-targeted extraction (the arrays are tiny),
but the pattern is good practice for any large-array processing.

## Problem 2: Disk Storage for Raw Embeddings

### What Happens

If you save raw per-position embeddings to disk:

| Foundation Model | Hidden Dim | ~2M Positions | ~3B Positions (full genome) |
|-----------------|------------|---------------|----------------------------|
| SpliceBERT | 512 | ~4 GB | ~6 TB |
| Evo2 7B | 4096 | ~32 GB | ~48 TB |

Even with only sampled positions (~2M), Evo2 embeddings are 32 GB.

### Solution: Position Sampling + Streaming Scalar Extraction

Two complementary strategies eliminate storage pressure:

**Position sampling** (`sampling.py`) reduces which positions need embeddings:

- Full genome: ~3 billion positions → impossible to store
- After sampling: ~2 million positions (0.06%) → feasible but still large
- Three tiers: splice sites (always kept) + proximity zone + background

**Streaming extraction** (`07_streaming_fm_scalars.py`) avoids storing raw
embeddings entirely by computing scalar features on the fly:

```
Raw embedding (4096 floats per position)  →  8 scalar features per position
32 GB for 2M positions                    →  65 MB for 2M positions
```

The 8 features (PCA components + norm + gradient) are computed during extraction
and written directly to per-chromosome parquet files. Raw embeddings are never
saved to disk.

### How They Work Together

```
Full genome: ~3B positions × 4096 dim = 48 TB (impossible)
                │
                ▼ Position sampling (0.06% retention)
                │
Sampled: ~2M positions × 4096 dim = 32 GB (large but feasible)
                │
                ▼ Streaming scalar extraction (on GPU pod)
                │
Scalars: ~2M positions × 8 features = 65 MB (trivial)
                │
                ▼ rsync to local machine
                │
Local: 65 MB of fm_scalars_{chrom}.parquet files
```

Position sampling is primarily a **storage optimization** — it determines *which*
positions get embeddings. Streaming extraction is primarily a **memory optimization**
(both CPU RAM and disk) — it determines *how* embeddings are processed.

Neither alone is sufficient:
- Sampling without streaming: 32 GB of raw embeddings on disk
- Streaming without sampling: 3B model forward passes (years of GPU time)
- Both together: 65 MB output, ~100 hrs GPU time, ~1 GB peak RAM

## Practical Lessons

### 1. Test on chr22 First

Chr22 is the smallest autosome (~423 genes). A full extraction takes ~2 hours
with dual-strand Evo2 7B. If it completes without OOM, the approach is viable.

### 2. Monitor System RAM, Not Just GPU

GPU OOM errors are immediate and obvious. System RAM exhaustion is silent —
the pod becomes unresponsive, SSH drops, and the process is killed by the OOM
killer. Monitor via:

```bash
# On the pod
watch -n 5 'free -h | head -2; nvidia-smi --query-gpu=memory.used --format=csv,noheader'

# From local machine
ssh pod "ps aux | grep 07_streaming | awk '{printf \"RSS: %.1f GB\n\", \$6/1024/1024}'"
```

### 3. Gene Length Distribution Matters

Most human genes are small (<100 KB), but a few are enormous:

| Length | Count | Examples |
|--------|-------|---------|
| <50 KB | ~15,000 | Most genes |
| 50-200 KB | ~3,500 | BRCA2, TP53 |
| 200-500 KB | ~400 | SYN3, MYT1L, NBAS |
| 500 KB-1 MB | ~60 | TTC28, LARGE1 |
| >1 MB | ~20 | RBFOX1, DMD, CNTNAP2 |

The OOM risk comes from the tail — the ~80 genes larger than 500 KB. Position-
targeted extraction eliminates this risk entirely because peak RAM depends on
chunk size (32K × 4096 × 4 = 512 MB), not gene length.

### 4. Dual-Strand Doubles Everything

For causal models (Evo2, HyenaDNA), dual-strand extraction means:
- 2x GPU inference time
- 2x peak CPU RAM (with full-gene stitching; 1x with position-targeted)
- Same disk storage (only averaged embeddings are saved)

Budget accordingly: Evo2 7B dual-strand on A40 processes ~1 gene/min for
average-sized genes.

### 5. Pod Selection

| Concern | Minimum | Recommended |
|---------|---------|-------------|
| GPU VRAM (Evo2 7B) | 24 GB | 48 GB (A40) |
| System RAM | 62 GB | 94+ GB |
| Disk | 10 GB | 50 GB (network volume) |

The L4 (24 GB VRAM, 94 GB RAM, $0.32/hr) is the best value if Evo2 fits in
24 GB. The A40 (48 GB VRAM, 62+ GB RAM, $0.44/hr) is the safe choice.

## See Also

- [`fm-embeddings-tutorial.md`](fm-embeddings-tutorial.md) — Full modality tutorial
- [`07_streaming_fm_scalars.py`](../../07_streaming_fm_scalars.py) — Extraction script
- [`sampling.py`](../../../src/agentic_spliceai/splice_engine/features/sampling.py) — Position sampling logic
