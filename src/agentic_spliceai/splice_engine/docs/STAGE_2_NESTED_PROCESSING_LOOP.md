# Stage 2, Step 4: Nested Processing Loop Analysis

**Purpose**: Understand the three-level nested loop structure in `splice_prediction_workflow.py`  
**Date**: December 6, 2025  
**Lines**: 500-1005

---

## Overview

The workflow uses a **three-level nested loop** to process the entire genome efficiently while managing memory constraints:

```
Level 1: CHROMOSOME (outer)     → Data locality, streaming
Level 2: CHUNK (middle)         → Checkpointing, disk I/O
Level 3: MINI-BATCH (inner)     → Memory optimization, GPU batching
```

---

## Why This Nested Structure?

### The Memory Problem

Processing the human genome involves:
- **~20,000 protein-coding genes**
- **~200,000 transcripts**
- **~500,000 splice sites**
- **~3 billion nucleotides**

Loading everything into memory would require **100+ GB RAM**. The nested structure solves this:

| Level | Size | Purpose | Memory Impact |
|-------|------|---------|---------------|
| **Chromosome** | 1-22, X, Y | Data locality | Load one chr at a time |
| **Chunk** | 500 genes | Checkpointing | Save progress, resume |
| **Mini-batch** | 50 genes | Peak memory | GPU batch size |

### Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────┐
│  CHROMOSOME LEVEL: "What data do I need to load?"                   │
│  ├── Load sequences for ONE chromosome at a time                    │
│  └── Stream from disk, don't hold entire genome                     │
├─────────────────────────────────────────────────────────────────────┤
│  CHUNK LEVEL: "How do I checkpoint progress?"                       │
│  ├── Process 500 genes, save to disk                                │
│  ├── If interrupted, resume from last saved chunk                   │
│  └── Each chunk = one output file                                   │
├─────────────────────────────────────────────────────────────────────┤
│  MINI-BATCH LEVEL: "How do I fit in GPU memory?"                    │
│  ├── Process 50 genes at a time through model                       │
│  ├── Free memory after each mini-batch                              │
│  └── Accumulate results, consolidate at chunk end                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Loop Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WORKFLOW START                                     │
│                    (Data preparation complete)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LEVEL 1: CHROMOSOME LOOP                                                    ┃
┃  for chr_ in chromosomes:  # ['1', '2', ..., '22', 'X', 'Y']                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        ├──│ 1.1 LOAD SEQUENCES                                               │
        │  │     • scan_chromosome_sequence() → LazyFrame                     │
        │  │     • OR filter pre-loaded sequences                             │
        │  │     • Count genes: n_genes = lazy_seq_df.n_unique()              │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LEVEL 2: CHUNK LOOP                                                         ┃
┃  for chunk_start in range(0, n_genes, chunk_size):  # chunk_size=500        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        ├──│ 2.1 CHECKPOINT CHECK                                             │
        │  │     • Check if chunk artifact file exists                        │
        │  │     • If exists → SKIP this chunk (resume support)               │
        │  │     • File: analysis_sequences_{chr}_chunk_{start}_{end}.tsv     │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        ├──│ 2.2 MATERIALIZE CHUNK                                            │
        │  │     • seq_chunk = lazy_seq_df.slice(start, size).collect()       │
        │  │     • Apply target_genes filter if specified                     │
        │  │     • Adjust chunk_size based on memory pressure                 │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        ├──│ 2.3 INITIALIZE ACCUMULATORS                                      │
        │  │     • all_mini_batch_predictions = {}                            │
        │  │     • all_mini_batch_errors = []                                 │
        │  │     • all_mini_batch_positions = []                              │
        │  │     • all_mini_batch_sequences = []                              │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        ▼
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  LEVEL 3: MINI-BATCH LOOP                                                    ┃
┃  for mini_batch_idx in range(n_mini_batches):  # MINI_BATCH_SIZE=50         ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        │  │ 3.1 EXTRACT MINI-BATCH                                           │
        ├──│     • seq_mini_batch = seq_chunk[start:end]                      │
        │  │     • Typically 50 genes per mini-batch                          │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        │  │ 3.2 PREDICT (Section 8.1.1)                                      │
        ├──│     ┌─────────────────────────────────────────────────────────┐  │
        │  │     │ predictions_mini = predict_splice_sites_for_genes(     │  │
        │  │     │     seq_mini_batch,                                     │  │
        │  │     │     models=models,        # 5 Keras models              │  │
        │  │     │     context=10_000,       # ±10kb context               │  │
        │  │     │     efficient_output=True                               │  │
        │  │     │ )                                                       │  │
        │  │     └─────────────────────────────────────────────────────────┘  │
        │  │     • Returns: {gene_id: {donor_prob, acceptor_prob, ...}}       │
        │  │     • Accumulate: all_mini_batch_predictions.update()            │
        │  │     • Track in gene_manifest                                     │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        │  │ 3.3 EVALUATE (Section 8.1.2)                                     │
        ├──│     ┌─────────────────────────────────────────────────────────┐  │
        │  │     │ error_df, positions_df =                                │  │
        │  │     │     enhanced_process_predictions_with_all_scores(       │  │
        │  │     │         predictions=predictions_mini,                   │  │
        │  │     │         ss_annotations_df=ss_annotations_df,            │  │
        │  │     │         threshold=config.threshold,                     │  │
        │  │     │         add_derived_features=True,  # ~58 features      │  │
        │  │     │         collect_tn=True             # True negatives    │  │
        │  │     │     )                                                   │  │
        │  │     └─────────────────────────────────────────────────────────┘  │
        │  │     • Compares predictions to known splice sites                 │
        │  │     • Generates error analysis + position-level features         │
        │  │     • Accumulate: all_mini_batch_errors.append()                 │
        │  │     • Accumulate: all_mini_batch_positions.append()              │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        │  │ 3.4 EXTRACT SEQUENCES (Section 8.1.3)                            │
        ├──│     ┌─────────────────────────────────────────────────────────┐  │
        │  │     │ df_seq_mini = extract_analysis_sequences(               │  │
        │  │     │     seq_mini_batch,                                     │  │
        │  │     │     positions_df_mini,                                  │  │
        │  │     │     window_size=250,      # ±250bp around splice site   │  │
        │  │     │     position_id_mode=position_id_mode                   │  │
        │  │     │ )                                                       │  │
        │  │     └─────────────────────────────────────────────────────────┘  │
        │  │     • Extracts nucleotide sequences around each splice site      │
        │  │     • Accumulate: all_mini_batch_sequences.append()              │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        │  ┌──────────────────────────────────────────────────────────────────┐
        │  │ 3.5 FREE MEMORY                                                  │
        ├──│     • del seq_mini_batch, predictions_mini                       │
        │  │     • del error_df_mini, positions_df_mini, df_seq_mini          │
        │  │     • gc.collect()  # Force garbage collection                   │
        │  └──────────────────────────────────────────────────────────────────┘
        │
        └──────────────────────────────────────────────────────────────────────
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ BACK TO CHUNK LEVEL: CONSOLIDATE (Section 8.1.4)                     │
        │     • predictions = all_mini_batch_predictions                       │
        │     • error_df_chunk = pl.concat(all_mini_batch_errors)              │
        │     • positions_df_chunk = pl.concat(all_mini_batch_positions)       │
        │     • df_seq = pl.concat(all_mini_batch_sequences)                   │
        │     • del all_mini_batch_* containers                                │
        └──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌──────────────────────────────────────────────────────────────────────┐
        │ 2.4 POST-PROCESSING (Section 8.2)                                    │
        │     • Check for duplicates                                           │
        │     • Deduplicate if needed                                          │
        │     • Validate positions_df_chunk                                    │
        └──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌──────────────────────────────────────────────────────────────────────┐
        │ 2.5 SAVE CHUNK ARTIFACTS (Section 8.4)                               │
        │     • data_handler.save_analysis_sequences(df_seq, ...)              │
        │     • data_handler.save_error_analysis(error_df_chunk, ...)          │
        │     • data_handler.save_splice_positions(positions_df_chunk, ...)    │
        │     • del df_seq  # Free memory                                      │
        └──────────────────────────────────────────────────────────────────────┘
                                    │
        ┌──────────────────────────────────────────────────────────────────────┐
        │ 2.6 ACCUMULATE GLOBALS (Section 8.5)                                 │
        │     • full_error_df = align_and_append(full_error_df, error_df_chunk)│
        │     • full_positions_df = align_and_append(full_positions_df, ...)   │
        └──────────────────────────────────────────────────────────────────────┘
                                    │
        └──────────────────────────────────────────────────────────────────────
                                    │
                                    ▼
        ┌──────────────────────────────────────────────────────────────────────┐
        │ BACK TO CHROMOSOME LEVEL                                             │
        │     • processed_chroms.append(chr_)                                  │
        │     • Continue to next chromosome                                    │
        └──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FINAL AGGREGATION (Section 9)                         │
│     • Save full_positions_df to aggregated file                              │
│     • Save full_error_df to aggregated file                                  │
│     • Generate summary statistics                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Analysis by Level

### Level 1: Chromosome Loop (Lines 550-1005)

**Purpose**: Data locality and streaming

```python
for chr_ in tqdm(chromosomes, desc="Processing chromosomes"):
    chr_ = str(chr_)
```

**What Happens**:

| Step | Operation | Code Location | Description |
|------|-----------|---------------|-------------|
| 1.1 | Load sequences | Lines 553-589 | `scan_chromosome_sequence()` or filter pre-loaded |
| 1.2 | Count genes | Line 594 | `n_genes = lazy_seq_df.n_unique()` |
| 1.3 | Enter chunk loop | Line 602 | Process genes in chunks |
| 1.4 | Mark complete | Line 1004 | `processed_chroms.append(chr_)` |

**Key Design Decisions**:

1. **Lazy Loading**: Uses `LazyFrame` to avoid loading entire chromosome into memory
2. **Pre-load Optimization**: For small gene sets (≤1000), pre-loads all sequences
3. **Skip Empty**: Skips chromosomes with no target genes

```python
# Memory-efficient: Stream from disk
lazy_seq_df = scan_chromosome_sequence(
    seq_result=seq_result,
    chromosome=chr_,
    format=seq_format
)

# OR for small gene sets: Use pre-loaded
chrom_df = preloaded_df.filter(pl.col('chrom') == chr_)
```

---

### Level 2: Chunk Loop (Lines 602-1004)

**Purpose**: Checkpointing and disk I/O

```python
default_chunk_size = 500 if not test_mode else 50
chunk_size = default_chunk_size

for chunk_start in range(0, n_genes, chunk_size):
    chunk_end = min(chunk_start + chunk_size, n_genes)
```

**What Happens**:

| Step | Operation | Code Location | Description |
|------|-----------|---------------|-------------|
| 2.1 | Checkpoint check | Lines 606-620 | Skip if artifact exists |
| 2.2 | Adjust chunk size | Line 623 | `adjust_chunk_size()` based on memory |
| 2.3 | Materialize chunk | Line 626 | `seq_chunk = lazy_seq_df.slice().collect()` |
| 2.4 | Filter target genes | Lines 628-639 | Optional gene filtering |
| 2.5 | Enter mini-batch loop | Line 680 | Process in smaller batches |
| 2.6 | Consolidate results | Lines 800-823 | Combine mini-batch outputs |
| 2.7 | Post-process | Lines 893-929 | Deduplicate, validate |
| 2.8 | Save artifacts | Lines 937-971 | Write to disk |
| 2.9 | Accumulate globals | Lines 976-990 | Append to full DataFrames |

**Checkpoint Mechanism**:

```python
# Check if this chunk was already processed
chunk_artifact_file = os.path.join(
    data_handler.meta_dir,
    f"analysis_sequences_{chr_}_chunk_{chunk_start+1}_{chunk_end}.tsv"
)

if os.path.exists(chunk_artifact_file):
    print(f"[checkpoint] chr{chr_} chunk {chunk_start+1}-{chunk_end} already exists - SKIPPING")
    continue  # Resume from next chunk
```

**Output Files Per Chunk**:

```
meta_dir/
├── analysis_sequences_{chr}_chunk_{start}_{end}.tsv    # Sequences
├── error_analysis_{chr}_chunk_{start}_{end}.tsv        # Error metrics
└── splice_positions_{chr}_chunk_{start}_{end}.tsv      # Position features
```

---

### Level 3: Mini-Batch Loop (Lines 680-798)

**Purpose**: Memory optimization and GPU batching

```python
MINI_BATCH_SIZE = config.mini_batch_size if hasattr(config, 'mini_batch_size') else 50

for mini_batch_idx in range(n_mini_batches):
    mini_batch_start = mini_batch_idx * MINI_BATCH_SIZE
    mini_batch_end = min(mini_batch_start + MINI_BATCH_SIZE, n_genes_in_chunk)
```

**What Happens**:

| Step | Section | Code Location | Description |
|------|---------|---------------|-------------|
| 3.1 | Extract | Line 686 | `seq_mini_batch = seq_chunk[start:end]` |
| 3.2 | **PREDICT** | Lines 695-741 | `predict_splice_sites_for_genes()` |
| 3.3 | **EVALUATE** | Lines 742-759 | `enhanced_process_predictions_with_all_scores()` |
| 3.4 | **SEQUENCES** | Lines 767-785 | `extract_analysis_sequences()` |
| 3.5 | Free memory | Lines 787-798 | `del` + `gc.collect()` |

---

## Where Key Operations Happen

### 1. Where Does PREDICTION Happen?

**Location**: Lines 695-741 (Section 8.1.1)

```python
# Inside mini-batch loop
if action == "predict":
    predictions_mini = predict_splice_sites_for_genes(
        seq_mini_batch,           # 50 genes
        models=models,            # 5 Keras models (ensemble)
        context=10_000,           # ±10kb context window
        efficient_output=True,    # Memory-efficient output
        show_gene_progress=False  # Disable inner progress bar
    )
```

**Returns**:
```python
{
    'ENSG00000123456': {
        'donor_prob': [0.01, 0.02, ...],      # Per-nucleotide donor probability
        'acceptor_prob': [0.01, 0.03, ...],   # Per-nucleotide acceptor probability
        'neither_prob': [0.98, 0.95, ...],    # Per-nucleotide neither probability
        'positions': [1, 2, 3, ...]           # Relative positions
    },
    ...
}
```

---

### 2. Where Does EVALUATION Happen?

**Location**: Lines 742-759 (Section 8.1.2)

```python
# Inside mini-batch loop, after prediction
if predictions_mini:
    error_df_mini, positions_df_mini = enhanced_process_predictions_with_all_scores(
        predictions=predictions_mini,
        ss_annotations_df=ss_annotations_df,  # Known splice sites
        threshold=config.threshold,           # Detection threshold
        consensus_window=config.consensus_window,
        error_window=config.error_window,
        analyze_position_offsets=True,
        collect_tn=True,                      # Collect true negatives
        no_tn_sampling=no_tn_sampling,
        predicted_delta_correction=True,
        splice_site_adjustments=adjustment_dict,
        add_derived_features=True             # Generate ~58 derived features
    )
```

**Returns**:
- `error_df_mini`: Error analysis per gene (TP, FP, FN counts)
- `positions_df_mini`: Position-level features (~58 columns)

---

### 3. Where Does SEQUENCE EXTRACTION Happen?

**Location**: Lines 767-785 (Section 8.1.3)

```python
# Inside mini-batch loop, after evaluation
if positions_df_mini.height > 0:
    df_seq_mini = extract_analysis_sequences(
        seq_mini_batch,           # Gene sequences
        positions_df_mini,        # Splice site positions
        include_empty_entries=True,
        essential_columns_only=False,
        drop_transcript_id=False,
        window_size=250,          # ±250bp around splice site
        position_id_mode=position_id_mode,
        preserve_transcript_list=True
    )
```

**Returns**: DataFrame with nucleotide sequences around each splice site

---

### 4. Where Is MEMORY FREED?

**Location 1**: End of mini-batch (Lines 787-798)

```python
# Free mini-batch memory immediately
del seq_mini_batch, predictions_mini
if 'error_df_mini' in locals():
    del error_df_mini
if 'positions_df_mini' in locals():
    del positions_df_mini
if 'df_seq_mini' in locals():
    del df_seq_mini

# Force garbage collection after each mini-batch
import gc
gc.collect()
```

**Location 2**: After saving chunk (Line 952)

```python
# Explicitly free memory for this chunk
del df_seq
```

**Location 3**: After consolidation (Line 823)

```python
# Clean up accumulation containers
del all_mini_batch_predictions, all_mini_batch_errors
del all_mini_batch_positions, all_mini_batch_sequences
```

---

## Memory Flow Diagram

```
                    MEMORY USAGE OVER TIME
    ▲
    │
    │     ┌─────┐     ┌─────┐     ┌─────┐
    │     │ MB1 │     │ MB2 │     │ MB3 │   ... Mini-batches
    │     │     │     │     │     │     │
    │  ───┴─────┴─────┴─────┴─────┴─────┴───────────────────
    │     ↑     ↓     ↑     ↓     ↑     ↓
    │   load  free  load  free  load  free
    │
    │  ════════════════════════════════════════════════════
    │  │              CHUNK ACCUMULATION                  │
    │  │  (all_mini_batch_* containers grow)              │
    │  ════════════════════════════════════════════════════
    │                                                    ↓
    │                                              CONSOLIDATE
    │                                              & SAVE
    │                                                    ↓
    │  ────────────────────────────────────────────────────
    │                    FREE CHUNK MEMORY
    │
    └──────────────────────────────────────────────────────► Time
         │←── Mini-batch ──→│←── Mini-batch ──→│
         │←──────────── Chunk ─────────────────→│
```

---

## Summary Table

| Level | Loop Variable | Size | Key Operations | Memory Strategy |
|-------|--------------|------|----------------|-----------------|
| **Chromosome** | `chr_` | 24 | Load sequences | Stream per-chr |
| **Chunk** | `chunk_start` | 500 genes | Checkpoint, save | Save to disk |
| **Mini-batch** | `mini_batch_idx` | 50 genes | Predict, evaluate | Free after each |

---

## Key Takeaways for Porting

### Essential to Port

1. ✅ **Chromosome loop** - Data locality pattern
2. ✅ **Chunk checkpointing** - Resume capability
3. ✅ **Mini-batch processing** - Memory management
4. ✅ **Three core operations**:
   - `predict_splice_sites_for_genes()`
   - `enhanced_process_predictions_with_all_scores()`
   - `extract_analysis_sequences()`

### Can Simplify

- ⚠️ **Pre-load optimization** - Only needed for small gene sets
- ⚠️ **Memory monitoring** - Nice-to-have, not essential
- ⚠️ **Nucleotide scores capture** - Optional feature

### Critical Functions to Port

```python
# From run_spliceai_workflow.py
predict_splice_sites_for_genes()

# From enhanced_workflow.py
enhanced_process_predictions_with_all_scores()

# From sequence_utils.py
extract_analysis_sequences()

# From data_handler.py
save_analysis_sequences()
save_error_analysis()
save_splice_positions()
```

---

**Next**: Section 5 - Aggregation and Save (Lines 1018-1202)
