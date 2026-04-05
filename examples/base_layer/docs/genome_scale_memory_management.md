# Genome-Scale Prediction: Memory Management and OOM Prevention

## The Problem

Running `05_genome_precomputation.py --all` on the full genome loads tens
of thousands of gene sequences, runs a deep learning model on each, and
accumulates the results.  On MANE (~19K genes) this is manageable.  On
Ensembl (~57K genes with alternative transcripts), the same code can
exhaust 503 GB of RAM and crash the pod.

This document explains where memory goes, what we changed, and how to
run genome-scale precomputation safely.

---

## Where Memory Goes

The prediction pipeline has three memory consumers that compound:

### 1. Gene Sequence Extraction (~5-15 GB per chromosome)

`_prepare_gene_data()` loads a GTF, extracts gene annotations, then
reads the DNA sequence for every gene on the requested chromosomes
into a single Polars DataFrame.  Each gene's `sequence` column holds
the full DNA string (10K-280K characters).

```
chr1: 5,682 Ensembl genes × avg 50K bp ≈ 280M characters ≈ 1-2 GB strings
All 24 chromosomes: ~57K genes × avg 50K bp ≈ 15-20 GB strings
```

With Polars overhead (string buffers, validity bitmaps), the actual
cost is 2-3x the raw string size.

### 2. Chunk Accumulation (~5-10 GB per chromosome)

The `PredictionWorkflow` processes genes in chunks of 500.  After each
chunk produces predictions (a DataFrame with position, donor_prob,
acceptor_prob, neither_prob columns), the DataFrame was appended to
a Python list:

```python
chunk_frames.append(chunk_predictions_df)  # keeps ALL chunks in RAM
```

For chr1 with 12 chunks producing ~10M positions each, this
accumulates 120M rows of float32 data: `120M × 4 cols × 4 bytes ≈ 2 GB`.
Not huge for one chromosome, but the problem is compounding.

### 3. Final Aggregation (~10-20 GB spike)

After all chunks finish, the workflow does:

```python
all_predictions = pl.concat(chunk_frames)  # doubles the memory
```

This creates a second copy of all predictions.  Then it writes the
aggregated file to disk, meaning three copies exist momentarily:
the individual chunks, the concat result, and the disk write buffer.

### The Multiplier: Running `--all`

Before our fix, `--all` meant one single `workflow.run(chromosomes=ALL)`.
This loaded gene sequences for ALL 57K Ensembl genes at once, ran ALL
chunks, accumulated ALL predictions, then concat'd everything.
Peak memory: gene sequences (20 GB) + chunk accumulation (15 GB) +
final concat (15 GB) + OS overhead + model weights = **easily 100+ GB**.

When combined with a second process (the M1-S evaluation), the pod
ran out of 503 GB.

---

## The Fix: Three-Level Memory Bounding

### Level 1: Per-Chromosome Loop (script level)

The `05_genome_precomputation.py` script now processes one chromosome
at a time in a loop:

```python
for chrom in chromosomes:
    # Resume: skip if parquet already exists
    parquet_path = output_dir / f"predictions_{chrom}.parquet"
    if args.resume and parquet_path.exists():
        continue

    workflow = PredictionWorkflow(config)
    result = workflow.run(chromosomes=[chrom])

    # Save per-chromosome parquet
    result.predictions.write_parquet(parquet_path)

    # Free everything before next chromosome
    del workflow, result
    gc.collect()
```

**Memory bound**: one chromosome's genes + one chromosome's predictions.
For the largest chromosome (chr1, 5,682 Ensembl genes), this is
~10-15 GB.

**Crash recovery**: each chromosome writes its own parquet immediately.
If the process dies at chr7, restarting with `--resume` skips chr1-6.

### Level 2: Streaming Mode (workflow level)

Inside `PredictionWorkflow.run()`, production mode now uses **streaming**:
chunk predictions are saved to disk but NOT accumulated in memory.

```python
streaming = cfg.mode == "production" and cfg.save_predictions

# After each chunk:
if not streaming:
    chunk_frames.append(chunk_predictions_df)  # accumulate
else:
    del chunk_predictions_df  # free immediately
```

**Memory bound**: one chunk (~500 genes) at a time, regardless of how
many chunks the chromosome has.

### Level 3: Per-Chromosome Parquet (output format)

Instead of one monolithic `predictions.tsv` for the entire genome,
the output is now per-chromosome parquet files:

```
precomputed/
  predictions_chr1.parquet
  predictions_chr2.parquet
  ...
  predictions_chrY.parquet
```

This is the same format the meta-layer expects — `_load_base_scores()`
and `_load_chrom_base_scores()` both look for `predictions_{chrom}.parquet`.

---

## How to Run Safely

### Standard MANE precomputation (~19K genes)

Memory requirement: ~10-15 GB peak.  Safe on any machine.

```bash
python 05_genome_precomputation.py --all --chunk-size 500
```

### Ensembl precomputation (~57K genes)

Memory requirement: ~60-80 GB peak (one chromosome at a time,
auto chunk-size=100).

```bash
# chunk-size auto-selects 100 for Ensembl (vs 500 for MANE)
python 05_genome_precomputation.py --all --annotation-source ensembl
```

### Resuming after a crash

If the process dies mid-run, simply restart with `--resume`:

```bash
python 05_genome_precomputation.py --all --annotation-source ensembl \
    --chunk-size 500 --resume
```

Completed chromosomes (those with existing parquet files) are skipped.
The process resumes from the first missing chromosome.

### Running on a GPU pod

Always run heavyweight precomputation **alone** — don't run a second
process simultaneously.  Even with 503 GB, two genome-scale processes
will compete for memory unpredictably.

```bash
cd ~/sky_workdir
nohup python -u examples/base_layer/05_genome_precomputation.py \
    --all --annotation-source ensembl --resume \
    > /runpod-volume/output/ensembl_precompute/precompute.log 2>&1 &
```

Monitor memory via the built-in `[MEM]` anchors in the log:

```bash
# Memory probes at each chromosome boundary
grep MEM /runpod-volume/output/ensembl_precompute/precompute.log

# Or watch overall system memory
watch -n 10 'free -h | head -2'
```

### Reducing chunk size for constrained machines

On a 16 GB laptop, override the auto chunk size:

```bash
python 05_genome_precomputation.py --chromosomes chr22 \
    --annotation-source ensembl --chunk-size 50
```

---

## Incident Log

### OOM #1 (2026-04-03, pod sky-dcea)

**What happened**: Ran M1-S gene caching + Ensembl `--all` precomputation
simultaneously.  Gene caching loaded 12K genes' worth of BigWig + parquet
data; Ensembl precomputation loaded 57K gene sequences + accumulated
predictions.  Combined: >503 GB.

**Root cause**: Two heavyweight processes competing for memory.

**Fix**: Per-chromosome loop in script, run processes sequentially.

### OOM #2 (2026-04-04, pod sky-037e)

**What happened**: Same Ensembl `--all` precomputation without the
per-chromosome fix.  All 57K genes' sequences loaded at once.

**Root cause**: `_prepare_gene_data()` loaded all genes upfront.

**Fix**: Per-chromosome loop bounds gene loading to one chromosome.

### OOM #3 (2026-04-05, pod sky-6124)

**What happened**: Ensembl precomputation (per-chromosome, fixed) +
M1-S evaluation (639 genes, sliding window) running simultaneously.
Memory was stable at 107 GB initially, then exceeded 503 GB.

**Root cause**: The evaluation script accumulated all gene predictions
in lists (`all_meta_probs`, `all_base_probs`).  For 639 genes with
sliding window inference, this grew to tens of GB.  Combined with the
precompute process: OOM.

**Fix**: (1) Run one process at a time.  (2) Streaming mode in
`PredictionWorkflow` — don't accumulate chunks in memory.

### OOM #4 (2026-04-05, pod sky-9cc0)

**What happened**: Ensembl precomputation running alone with
per-chromosome loop + streaming mode.  OOM'd during chr1 (first
chromosome that wasn't resumed).  Memory at 100% with 0% GPU
utilization — crash happened during data processing, not inference.

**Root cause**: Per-position Python dict overhead in
`predict_splice_sites_for_genes()` (core.py lines 267-283).  The
function creates a `merged_results` defaultdict keyed by
`(gene_id, position)`.  Each entry is a dict with three Python lists.
At ~450 bytes per entry:

```
chr1: 168M total positions across 5,682 genes
Even one chunk of 500 genes: 500 × avg 30K bp = 15M entries
15M entries × 450 bytes ≈ 7 GB per chunk
```

With chunk_size=500, `_predictions_dict_to_dataframe()` then creates
another 15M-entry list of dicts for the Polars conversion — doubling
the peak.  Combined with `gene_df` holding all 5,682 gene sequences
in memory, OS buffer cache, and model weights: >503 GB.

**Fix**: Three changes:

1. **Auto chunk-size reduction**: Ensembl defaults to 100 genes/chunk
   instead of 500.  Reduces per-chunk peak from ~15 GB to ~3 GB.

2. **Memory monitor anchors**: `_log_mem()` probes RSS at each
   chromosome boundary (before workflow, after workflow, after gc).
   The log shows exactly where memory grows:
   ```
   [MEM] start: 0.6 GB RSS
   [MEM] chr3 before workflow: 0.6 GB RSS
   [MEM] chr3 after workflow: 12.4 GB RSS
   [MEM] chr3 after gc: 2.1 GB RSS
   ```

3. **Streaming mode in PredictionWorkflow**: Chunk DataFrames are freed
   after saving to disk (`del chunk_predictions_df`), not accumulated
   in `chunk_frames` list.

**Outcome**: With chunk_size=100, chr3 (3,337 genes) processes at
61 GB RSS on 503 GB pod.  chr1 and chr2 resumed from previous parquets.

### The Hidden Cost: Per-Position Python Dicts (Root Cause Analysis)

The core prediction function `predict_splice_sites_for_genes()` in
`base_layer/prediction/core.py` uses a Python-level per-position
data structure:

```python
# Line 214: one dict entry per (gene_id, position) pair
merged_results = defaultdict(lambda: {
    'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []
})

# Lines 267-283: append probabilities position by position
for i, (donor_p, acceptor_p, neither_p) in enumerate(...):
    pos_key = (gene_id, absolute_position)
    merged_results[pos_key]['donor_prob'].append(float(donor_p))
    ...
```

This per-position dict approach is clean and handles overlapping
predictions correctly (averaging across blocks), but it creates
enormous Python overhead for genome-scale runs.  Each dict entry
costs ~450 bytes (tuple key + dict + 3 lists + float objects),
compared to ~12 bytes for the actual data (3 × float32).

**A 37x overhead factor.**

The long-term fix is to replace this with numpy arrays (pre-allocated
per gene, accumulate with array indexing instead of dict lookups).
For now, the chunk-size reduction keeps peak memory manageable.

---

## Design Principles

1. **Process one chromosome at a time**.  The largest chromosome (chr1)
   has ~5.7K Ensembl genes.  That's the natural unit of work.

2. **Save incrementally**.  Each chromosome writes its own parquet.
   Crashes lose at most one chromosome's work.

3. **Free aggressively**.  `del` large DataFrames + `gc.collect()`
   between chromosomes.  Python's garbage collector doesn't always
   reclaim memory promptly.

4. **Don't accumulate in lists**.  If chunk results are saved to disk,
   there's no reason to also keep them in a Python list.

5. **One heavyweight process at a time**.  Even with 503 GB, running
   two genome-scale processes simultaneously is unreliable.  OS buffer
   cache, Polars internals, and CUDA allocator memory are hard to
   predict and control.

---

## Related Documentation

- [OOM in M1-S gene caching](../../meta_layer/docs/oom_gene_caching.md) —
  disk-backed `.npz` cache for meta-layer training
- [I/O bottlenecks and DataLoader tuning](../../../docs/ml_engineering/data_pipeline/io_bottlenecks_dataloader.md) —
  num_workers, pin_memory, shard packing
