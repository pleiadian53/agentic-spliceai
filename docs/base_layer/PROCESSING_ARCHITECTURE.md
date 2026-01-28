# SpliceAI Base Layer: Processing Architecture

This document explains the core processing architecture of the Meta-SpliceAI base layer, which `agentic-spliceai` integrates for splice site prediction. Understanding this architecture is essential for:

- Configuring prediction workflows
- Optimizing memory usage for large-scale analyses
- Troubleshooting and debugging
- Extending the system with custom features

---

## Table of Contents

1. [Overview](#overview)
2. [Data Preparation Pipeline](#data-preparation-pipeline)
3. [Three-Level Processing Loop](#three-level-processing-loop)
4. [Memory Management Strategy](#memory-management-strategy)
5. [Checkpoint and Resume](#checkpoint-and-resume)
6. [Output Files](#output-files)
7. [Configuration Options](#configuration-options)

---

## Overview

The base layer processes genomic data through a carefully designed pipeline that balances **accuracy**, **memory efficiency**, and **fault tolerance**. The architecture handles the challenge of processing ~20,000 genes across 24 chromosomes while keeping memory usage manageable.

### Key Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Data Locality** | Process one chromosome at a time |
| **Fault Tolerance** | Checkpoint after every 500 genes |
| **Memory Efficiency** | Process in mini-batches of 50 genes |
| **Streaming** | Use lazy loading to avoid loading entire genome |

### High-Level Flow

```
Input Files                    Processing                      Output Files
─────────────                  ──────────                      ────────────
GTF annotations ──┐
                  ├──► Data Preparation ──► Nested Loop ──► Splice predictions
FASTA genome ─────┤                         (3 levels)       Error analysis
                  │                                          Position features
SpliceAI models ──┘                                          Sequence contexts
```

---

## Data Preparation Pipeline

Before prediction begins, six preparation functions load and validate all required genomic resources.

### Preparation Steps

```
Step 1: prepare_gene_annotations()
        └── Extract transcript annotations from GTF file
        └── Output: annotations_all_transcripts.tsv

Step 2: prepare_splice_site_annotations()
        └── Extract known splice sites (donors and acceptors)
        └── Output: splice_sites_enhanced.tsv

Step 3: prepare_genomic_sequences()
        └── Extract gene sequences from FASTA
        └── Output: gene_sequences_chr{N}.parquet (per chromosome)

Step 4: handle_overlapping_genes()
        └── Identify genes with overlapping regions
        └── Output: overlapping_gene_counts.tsv

Step 5: determine_target_chromosomes()
        └── Determine which chromosomes to process
        └── Can infer from target genes if specified

Step 6: load_spliceai_models()
        └── Load 5 pre-trained Keras models (ensemble)
        └── Models predict donor/acceptor/neither probabilities
```

### Function Details

#### 1. prepare_gene_annotations()

Extracts transcript-level annotations from GTF/GFF3 files.

**Key Parameters:**
- `local_dir`: Directory for output files (build-specific)
- `gtf_file`: Path to GTF annotation file
- `do_extract`: Whether to extract from GTF (vs. load existing)
- `target_chromosomes`: Optional filter for specific chromosomes

**Output Schema:**
| Column | Type | Description |
|--------|------|-------------|
| chrom | str | Chromosome (1-22, X, Y) |
| start | int | Feature start position |
| end | int | Feature end position |
| strand | str | Strand (+ or -) |
| feature | str | Feature type (exon, CDS, UTR) |
| gene_id | str | Ensembl gene ID |
| transcript_id | str | Ensembl transcript ID |

#### 2. prepare_splice_site_annotations()

Extracts known splice site positions from gene annotations.

**Key Parameters:**
- `gene_annotations_df`: Pre-loaded annotations for filtering
- `consensus_window`: Window size for consensus calling (default: 2)
- `fasta_file`: Required for OpenSpliceAI fallback mode

**Output Schema:**
| Column | Type | Description |
|--------|------|-------------|
| chrom | str | Chromosome |
| position | int | Splice site position |
| strand | str | Strand |
| splice_type | str | 'donor' or 'acceptor' |
| gene_id | str | Gene identifier |

#### 3. prepare_genomic_sequences()

Extracts nucleotide sequences for each gene from the reference genome.

**Key Parameters:**
- `mode`: 'gene' (full gene) or 'transcript' (per-transcript)
- `seq_type`: 'full' (start to end) or 'minmax' (exon boundaries)
- `seq_format`: Output format ('parquet' recommended)
- `single_sequence_file`: Combine all chromosomes into one file

**Memory Consideration:** Sequences are stored per-chromosome to enable streaming during prediction.

#### 4. handle_overlapping_genes()

Identifies genes with overlapping genomic regions, which can cause ambiguous splice site assignments.

**Output:** DataFrame with overlap counts per gene, useful for filtering or flagging ambiguous predictions.

#### 5. determine_target_chromosomes()

Intelligently determines which chromosomes to process based on:
1. Explicitly specified chromosomes (highest priority)
2. Chromosomes containing target genes (if specified)
3. All standard chromosomes (default fallback)

#### 6. load_spliceai_models()

Loads the pre-trained SpliceAI ensemble (5 Keras models). Each model independently predicts splice site probabilities, and predictions are averaged for robustness.

---

## Three-Level Processing Loop

The core prediction workflow uses a nested loop structure to efficiently process the genome while managing memory.

### Loop Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│  LEVEL 1: CHROMOSOME LOOP                                           │
│  for chromosome in ['1', '2', ..., '22', 'X', 'Y']:                │
│      • Load sequences for this chromosome only                      │
│      • Stream from disk (lazy loading)                              │
├─────────────────────────────────────────────────────────────────────┤
│  LEVEL 2: CHUNK LOOP (500 genes per chunk)                         │
│  for chunk in range(0, n_genes, 500):                              │
│      • Check if chunk already processed (checkpoint)                │
│      • Materialize chunk from lazy frame                            │
│      • Save results to disk after processing                        │
├─────────────────────────────────────────────────────────────────────┤
│  LEVEL 3: MINI-BATCH LOOP (50 genes per batch)                     │
│  for mini_batch in range(0, chunk_size, 50):                       │
│      • Run SpliceAI prediction                                      │
│      • Evaluate against known splice sites                          │
│      • Extract sequence contexts                                    │
│      • Free memory immediately                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### What Happens at Each Level

#### Level 1: Chromosome (Data Locality)

**Purpose:** Load only one chromosome's data at a time to avoid memory overflow.

```python
for chr_ in chromosomes:
    # Load sequences for this chromosome only
    lazy_seq_df = scan_chromosome_sequence(chromosome=chr_)
    
    # Count genes on this chromosome
    n_genes = lazy_seq_df.select(pl.col("gene_id").n_unique()).collect().item()
```

**Key Operations:**
- Lazy loading via Polars `LazyFrame`
- Schema standardization
- Gene counting

#### Level 2: Chunk (Checkpointing)

**Purpose:** Enable resume capability and manage disk I/O.

```python
chunk_size = 500  # genes per chunk

for chunk_start in range(0, n_genes, chunk_size):
    # Check if already processed
    if os.path.exists(chunk_artifact_file):
        continue  # Skip - already done
    
    # Materialize this chunk
    seq_chunk = lazy_seq_df.slice(chunk_start, chunk_size).collect()
    
    # ... process mini-batches ...
    
    # Save chunk results to disk
    data_handler.save_analysis_sequences(df_seq, chunk_start, chunk_end)
    data_handler.save_splice_positions(positions_df, chunk_start, chunk_end)
```

**Key Operations:**
- Checkpoint checking (resume support)
- Chunk materialization
- Result consolidation
- Disk I/O (save artifacts)

#### Level 3: Mini-Batch (Memory Optimization)

**Purpose:** Keep peak memory usage low by processing small batches.

```python
MINI_BATCH_SIZE = 50  # genes per mini-batch

for mini_batch_idx in range(n_mini_batches):
    # Extract mini-batch
    seq_mini_batch = seq_chunk[start:end]
    
    # 1. PREDICT: Run SpliceAI models
    predictions = predict_splice_sites_for_genes(
        seq_mini_batch,
        models=models,
        context=10_000  # ±10kb context window
    )
    
    # 2. EVALUATE: Compare to known splice sites
    error_df, positions_df = enhanced_process_predictions_with_all_scores(
        predictions=predictions,
        ss_annotations_df=ss_annotations_df,
        add_derived_features=True  # Generate ~58 features
    )
    
    # 3. EXTRACT: Get sequence contexts
    df_seq = extract_analysis_sequences(
        seq_mini_batch,
        positions_df,
        window_size=250  # ±250bp around splice site
    )
    
    # 4. FREE MEMORY
    del seq_mini_batch, predictions
    gc.collect()
```

**Key Operations:**
- SpliceAI prediction (GPU-intensive)
- Evaluation against ground truth
- Sequence extraction
- Aggressive memory cleanup

---

## Memory Management Strategy

The architecture employs several strategies to keep memory usage bounded:

### Strategy 1: Streaming (Chromosome Level)

```
Instead of:  Load entire genome (100+ GB)
We do:       Load one chromosome at a time (2-8 GB)
```

### Strategy 2: Lazy Loading (Chunk Level)

```
Instead of:  Load all genes into memory
We do:       Use LazyFrame, materialize only current chunk
```

### Strategy 3: Immediate Cleanup (Mini-Batch Level)

```python
# After each mini-batch:
del seq_mini_batch, predictions_mini
del error_df_mini, positions_df_mini, df_seq_mini
gc.collect()  # Force garbage collection
```

### Memory Usage Pattern

```
Memory
  ▲
  │     ┌───┐     ┌───┐     ┌───┐
  │     │MB1│     │MB2│     │MB3│    Mini-batches
  │     │   │     │   │     │   │
  │  ───┴───┴─────┴───┴─────┴───┴───
  │     ↑   ↓     ↑   ↓     ↑   ↓
  │   load free load free load free
  │
  │  ══════════════════════════════
  │  │    Chunk accumulation      │
  │  ══════════════════════════════
  │                              ↓
  │                         Save & Free
  └──────────────────────────────────► Time
```

### Configuring Memory Usage

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 500 | Larger = fewer disk writes, more memory |
| `mini_batch_size` | 50 | Larger = faster GPU utilization, more memory |
| `MAX_GENES_FOR_PRELOAD` | 1000 | Threshold for pre-loading vs. streaming |

---

## Checkpoint and Resume

The workflow automatically saves progress and can resume from interruptions.

### How Checkpointing Works

1. **Before processing each chunk**, check if output file exists:
   ```python
   chunk_file = f"analysis_sequences_{chr}_chunk_{start}_{end}.tsv"
   if os.path.exists(chunk_file):
       print(f"Chunk already exists - SKIPPING")
       continue
   ```

2. **After processing each chunk**, save results:
   ```python
   data_handler.save_analysis_sequences(df_seq, chr, chunk_start, chunk_end)
   data_handler.save_error_analysis(error_df, chr, chunk_start, chunk_end)
   data_handler.save_splice_positions(positions_df, chr, chunk_start, chunk_end)
   ```

### Resume Behavior

If the workflow is interrupted (crash, timeout, manual stop):

1. **Restart the workflow** with the same parameters
2. **Completed chunks are automatically skipped**
3. **Processing resumes from the first incomplete chunk**

### Example Resume Scenario

```
First run (interrupted at chr3):
  chr1: ✓ Complete (all chunks saved)
  chr2: ✓ Complete (all chunks saved)
  chr3: ✗ Interrupted at chunk 501-1000

Second run (automatic resume):
  chr1: SKIP (all chunks exist)
  chr2: SKIP (all chunks exist)
  chr3: SKIP chunks 1-500 (exists)
        PROCESS chunks 501+ (resume here)
```

---

## Output Files

The workflow generates several types of output files:

### Per-Chunk Files (Intermediate)

```
output_dir/
├── meta/
│   ├── analysis_sequences_1_chunk_1_500.tsv
│   ├── analysis_sequences_1_chunk_501_1000.tsv
│   ├── error_analysis_1_chunk_1_500.tsv
│   ├── splice_positions_1_chunk_1_500.tsv
│   └── ...
```

### Aggregated Files (Final)

```
output_dir/
├── splice_positions_enhanced_aggregated.tsv    # All positions
├── error_analysis_aggregated.tsv               # All error metrics
└── analysis_sequences_aggregated.tsv           # All sequences
```

### File Contents

| File | Contents | Use Case |
|------|----------|----------|
| `splice_positions_*.tsv` | Position-level features (~58 columns) | Meta-model training |
| `error_analysis_*.tsv` | Per-gene error metrics (TP, FP, FN) | Quality assessment |
| `analysis_sequences_*.tsv` | Nucleotide sequences around splice sites | Sequence analysis |

---

## Configuration Options

### SpliceAIConfig Parameters

```python
from agentic_spliceai import SpliceAIConfig

config = SpliceAIConfig(
    # Core parameters
    threshold=0.5,              # Detection threshold
    consensus_window=2,         # Splice site consensus window
    error_window=10,            # Error analysis window
    
    # Memory optimization
    mini_batch_size=50,         # Genes per mini-batch
    chunk_size=500,             # Genes per chunk
    
    # Output control
    save_nucleotide_scores=False,  # Save per-nucleotide scores (large!)
    
    # Paths
    local_dir="./data/grch38",  # Build-specific directory
    gtf_file="./annotations.gtf",
    fasta_file="./genome.fa"
)
```

### Workflow Parameters

```python
results = run_enhanced_splice_prediction_workflow(
    config=config,
    
    # Filtering
    target_genes=['BRCA1', 'BRCA2'],  # Specific genes (optional)
    target_chromosomes=['17', '13'],   # Specific chromosomes (optional)
    
    # Behavior
    verbosity=1,                # 0=quiet, 1=normal, 2=debug
    no_final_aggregate=False,   # Skip final aggregation
    no_tn_sampling=False,       # Include all true negatives
    
    # Position ID format
    position_id_mode='genomic'  # 'genomic' or 'relative'
)
```

---

## Summary

The Meta-SpliceAI base layer processing architecture provides:

| Feature | Benefit |
|---------|---------|
| **Three-level nesting** | Balances memory, speed, and fault tolerance |
| **Lazy loading** | Handles genome-scale data without memory overflow |
| **Checkpointing** | Resume from interruptions without data loss |
| **Mini-batching** | Efficient GPU utilization with bounded memory |
| **Streaming output** | Results available as processing progresses |

This architecture enables processing of the entire human genome (~20,000 genes) on machines with as little as 16GB RAM, while providing robust fault tolerance for long-running analyses.

---

## See Also

- [Base Layer Integration Summary](./BASE_LAYER_INTEGRATION_SUMMARY.md)
- [Feature Set Documentation](./BASE_LAYER_FEATURE_SET.md)
- [API Reference](../api/splice_engine.md)
