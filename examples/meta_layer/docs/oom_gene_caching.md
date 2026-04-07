# OOM in Gene Caching: Diagnosis and Fix

## The Problem

During M1-S training on an A40 pod (62 GB system RAM), the gene caching
phase exhausted all system memory and made the pod unresponsive:

```
System memory: 100% utilization
SSH: connection timeout
Result: OOM kill, all processes lost
```

## Root Cause

`build_gene_cache()` loaded all 12,390 training genes into RAM as
`GeneCacheEntry` objects, each containing:

```python
@dataclass
class GeneCacheEntry:
    sequence: str           # gene DNA string (~50 KB avg)
    base_scores: np.ndarray # [gene_len, 3] float32 (~600 KB avg)
    mm_features: np.ndarray # [gene_len, 9] float32 (~1.8 MB avg)
    labels: np.ndarray      # [gene_len] int64 (~400 KB avg)
```

Per gene: ~2.8 MB average.  For 12,390 genes: **~35 GB** in gene cache
alone.  Add the junction index (558K entries), eCLIP peaks (937K rows),
bigWig file handles, prediction parquet scans, and Python overhead — total
exceeded the pod's 62 GB.

The concurrent Ensembl prediction run (which we launched simultaneously)
added ~10 GB more, pushing past the limit.

## The Fix: Disk-Backed Gene Cache

Replaced in-memory `List[GeneCacheEntry]` with a **disk-backed cache**:

1. **Caching phase**: Each gene is saved as a compressed `.npz` file
   (~200 KB avg, much smaller than in-memory due to compression).
   Arrays are freed immediately after writing.

2. **Index in RAM**: Only lightweight `GeneIndexEntry` metadata is kept
   in memory (~1 KB per gene = ~12 MB for 12K genes):
   ```python
   @dataclass
   class GeneIndexEntry:
       gene_id: str
       npz_path: Path
       length: int
       n_splice_sites: int
       splice_positions: np.ndarray  # small array, typically <100 ints
   ```

3. **Training phase**: `__getitem__` loads one gene's `.npz` from disk,
   extracts a window, returns tensors.  Peak memory: ~3 MB (one gene).

### Memory comparison

| | Before (in-memory) | After (disk-backed) |
|---|---|---|
| Gene cache RAM | ~35 GB (12K genes) | ~12 MB (index only) |
| Peak during caching | ~35 GB (accumulates) | ~3 MB (one gene) |
| Peak during training | ~35 GB + model + batch | ~50 MB (model + batch + 1 gene) |
| Disk usage | 0 | ~25 GB (.npz files) |
| Resume on crash | Start over | Skips existing .npz files |

### I/O overhead

Loading one `.npz` per `__getitem__` call adds ~1-2 ms of disk I/O.
With batch_size=8 and `num_workers=0`, this adds ~10 ms per batch.
On NVMe (pod) or SSD (laptop), negligible compared to the forward pass.

For `num_workers > 0`, multiple workers read `.npz` files in parallel —
the disk-backed approach scales well with DataLoader workers.

## Usage

```bash
# Gene cache saved to persistent volume (survives pod restart)
python 07_train_sequence_model.py \
    --cache-dir /runpod-volume/output/meta_layer/gene_cache \
    ...

# Resume after interruption (skips existing .npz files)
python 07_train_sequence_model.py \
    --cache-dir /runpod-volume/output/meta_layer/gene_cache \
    ...
```

## Prevention

For future workflows that cache per-gene data:

1. **Never accumulate gene-sized arrays in a list** — write to disk and
   keep only metadata in RAM
2. **Use compressed `.npz`** — 5-10x smaller than in-memory arrays
3. **Make caching resume-safe** — check for existing files before
   recomputing
4. **Run one heavy process at a time** — don't launch both M1-S training
   and Ensembl predictions simultaneously on the same pod

---

## OOM #2: Evaluation Script Bulk Loading (2026-04-06)

### The Problem

Running `08_evaluate_sequence_model.py` with `--build-cache` for 5,522
test genes on an A40 pod (503 GB system RAM) caused OOM.  The pod hit
100% memory and restarted.

### Root Cause: Three Accumulation Points

The evaluation script had three compounding memory hazards:

1. **Bulk .npz loading** (lines 679-690): loaded ALL gene data into a
   `gene_data_map` dict before starting inference.  5,522 genes × ~5 MB
   each ≈ **28-35 GB**.

2. **Inference accumulation** (lines 707-726): stored `meta_probs`,
   `base_probs`, and `labels` arrays per gene in three separate dicts +
   three growing lists.  **3 parallel copies ≈ 90-105 GB**.

3. **Metrics concatenation** (lines 734-736): `np.concatenate()` on all
   accumulated arrays created a 4th copy.  **Peak ≈ 120-140 GB**.

Combined with the `build_gene_cache()` phase (base score parquets +
22 bigWig handles + feature extraction), total memory exceeded 503 GB.

### The Fix: Streaming Evaluation Architecture

Replaced bulk load-infer-accumulate with a **streaming per-gene loop**:

```python
evaluator = StreamingEvaluator()  # ~5 MB total, regardless of gene count

for gene_id in test_genes:
    data = _load_gene_npz(npz_path)           # load ONE gene
    meta_probs = infer_full_gene(model, data)  # infer ONE gene
    evaluator.update(meta_probs, data["base_scores"], data["labels"], gene_id)
    del data, meta_probs                       # free IMMEDIATELY
```

`StreamingEvaluator` accumulates only:
- **Scalar counters**: TP/FP/FN/n_correct per class (~100 bytes)
- **Splice-site probs for PR-AUC**: all splice positions (~100K) + 1%
  subsample of "neither" (~200K) ≈ **1.6 MB**
- **Top-k accumulators**: one float per gene per k ≈ **50 KB**

PR-AUC for donor/acceptor is **exact** (all true positives preserved).
Neither PR-AUC uses 1% subsample (always ~1.0, negligible error).

### Memory comparison

| | Before (bulk) | After (streaming) |
|---|---|---|
| Gene loading | ~35 GB (all in RAM) | ~5 MB (one gene) |
| Inference results | ~105 GB (3 copies) | ~5 MB (counters) |
| Metrics computation | +35 GB (concatenation) | 0 (from counters) |
| **Total evaluation peak** | **~175 GB** | **~10 MB** |

### Additional hardening: `build_gene_cache()`

- `gc.collect()` after every chromosome and every 200 genes within a
  chromosome, reclaiming Polars/Arrow/numpy buffers
- Memory logging (`[MEM]` probes) at chromosome boundaries
- Progress logging every 50 genes (was 100)

### Files changed

- `examples/meta_layer/08_evaluate_sequence_model.py` — streaming loop
- `src/.../eval/streaming_metrics.py` — `StreamingEvaluator` (NEW)
- `src/.../eval/sequence_inference.py` — shared `infer_full_gene()` (NEW)
- `examples/meta_layer/09_evaluate_alternative_sites.py` — M2a eval (NEW)
- `src/.../meta_layer/data/sequence_level_dataset.py` — gc + memory logging

---

## Related

- Memory monitor utility: `src/.../utils/memory_monitor.py`
- Ephemeral workflow (same pattern for feature engineering):
  `examples/features/06a_ephemeral_genome_workflow.py`
- FM extraction OOM fix: `examples/features/docs/fm-extraction-memory-guide.md`
- Base layer OOM history: `examples/base_layer/docs/genome_scale_memory_management.md`

## Timeline

- **2026-04-03**: OOM #1 during M1-S training on A40 pod (sky-3e26-pleiadian53)
- **2026-04-03**: Disk-backed cache implemented and tested locally
- **2026-04-06**: OOM #2 during evaluation on A40 pod (sky-e74e-pleiadian53)
- **2026-04-06**: Streaming evaluation architecture implemented
