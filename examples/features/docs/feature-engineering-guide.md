# Feature Engineering Guide

Reference for the multimodal feature engineering pipeline in `examples/features/`.
Covers the YAML-driven workflow, all 7 registered modalities, position sampling,
and practical recipes for running genome-scale pipelines.

## YAML-Driven Workflow

The production workflow is `06_multimodal_genome_workflow.py`, driven by YAML configs
in `configs/`. This replaced the earlier per-script approach (scripts 01-05 remain
for single-gene exploration and learning).

```bash
# Full-stack feature generation (7 modalities, 86 columns)
python 06_multimodal_genome_workflow.py \
    --config configs/full_stack.yaml \
    --chromosomes chr19 chr20 chr21 chr22 \
    --resume

# Default config (3 modalities, 50 columns — fast, no external data)
python 06_multimodal_genome_workflow.py \
    --config configs/default.yaml \
    --chromosomes chr22
```

### Available Configs

| Config | Modalities | Columns | Use Case |
|--------|-----------|---------|----------|
| `default.yaml` | base_scores, annotation, genomic | 50 | Quick runs, no external data dependencies |
| `full_stack.yaml` | All 7 modalities | 86 | Meta-layer training data (recommended) |
| `isoform_discovery.yaml` | 6 (lower threshold, wider sampling) | 74+ | Alternative splice site detection |
| `meta_m3_novel.yaml` | 6 (junction excluded) | 74 | M3 model: junction as held-out label |

### Key Flags

| Flag | Description |
|------|-------------|
| `--config PATH` | YAML config file (required) |
| `--chromosomes chr1 chr2 ...` | Chromosomes to process (`all` for 1-22, X, Y) |
| `--resume` | Skip chromosomes with existing output |
| `--force` | Delete existing output, regenerate everything |
| `--output-dir PATH` | Override output directory |

## Example Scripts — Progressive Storyline

The numbered scripts build on each other for learning purposes:

| Script | Focus | Modalities | Scope |
|--------|-------|-----------|-------|
| `01_base_score_features.py` | Base score features only | base_scores | Single gene |
| `02_annotation_and_genomic.py` | Standard 3-modality set | base_scores, annotation, genomic | Single gene |
| `03_configurable_modalities.py` | Per-modality config overrides | base_scores (configured) | Single gene |
| `04_genome_scale_workflow.py` | Legacy production workflow | base_scores, annotation, genomic | Chromosomes |
| `05_multimodal_exploration.py` | Experimental signal exploration | conservation, epigenetic | Single gene |
| **`06_multimodal_genome_workflow.py`** | **YAML-driven production workflow** | **All 7 (configurable)** | **Chromosomes** |

**Convention**: Script 06 is the production entry point. Scripts 01-05 are for
exploration and learning. The config loader in 06 dynamically resolves modality
classes from `FeaturePipeline._REGISTRY`.

## Modality Catalog

All 7 modalities are implemented and registered in `splice_engine/features/modalities/`.

### Model Output

| Modality | Columns | External Data | Description |
|----------|---------|--------------|-------------|
| **base_scores** | 43 | None | Probabilities, gradients, peaks, context windows, comparative features |

### Genomic Annotation

| Modality | Columns | External Data | Description |
|----------|---------|--------------|-------------|
| **annotation** | 3 | GTF (auto-resolved) | splice_type (label), transcript_id, transcript_count |
| **genomic** | 4 | Gene boundaries + FASTA | relative_gene_position, distances, gc_content |

### External Data

| Modality | Columns | External Data | Description |
|----------|---------|--------------|-------------|
| **sequence** | 3 | Reference FASTA | DNA window string (default ±500bp = 1001nt) |
| **conservation** | 9 | UCSC PhyloP/PhastCons bigWig | Evolutionary constraint (point + context stats) |
| **epigenetic** | 12 | ENCODE ChIP-seq bigWig | H3K36me3 + H3K4me3 across 8 cell lines (Strategy B summarized) |
| **junction** | 12 | GTEx v8 / STAR SJ.out.tab | RNA-seq junction read evidence (353K junctions, 54 tissues) |

**Total**: 86 feature columns in the full-stack configuration.

For detailed per-column documentation, see
[`docs/multimodal_feature_engineering/feature_catalog.md`](../../../docs/multimodal_feature_engineering/feature_catalog.md).

### Per-Modality Tutorials

| Tutorial | Modality | Location |
|----------|----------|----------|
| Conservation scores | conservation | `docs/conservation-scores-tutorial.md` |
| Epigenetic marks | epigenetic | `docs/epigenetic-marks-tutorial.md` |
| Junction reads | junction | `docs/junction-reads-tutorial.md` |

## Position Sampling

### Why sample?

The full genome has ~3.1 billion nucleotide positions, but only ~0.06%
have a splice probability above 0.01. Storing feature-enriched parquet
files for all positions is impractical.

### Early vs Late Sampling

| Mode | When | Speed | Use Case |
|------|------|-------|----------|
| **Early** (`early: true`, default) | Before feature engineering | ~100x faster | Production (modalities only process sampled positions) |
| **Late** (`early: false`) | After all features computed | Slower | When you need features at all positions first |

Early sampling is the key to running the full-stack pipeline on a laptop.
Example: chr22 goes from 17.3M positions to 96K in <1s, then 7 modalities
process only the sampled positions.

### Three-tier Sampling

1. **Splice sites** (always kept): positions where `donor_prob > threshold`
   or `acceptor_prob > threshold`
2. **Proximity zone** (optional): positions within N bp of a splice site
3. **Background** (random sample): fraction of remaining positions

### Storage Estimates (86 columns, parquet, early sampling)

| Config | Retention | Full Genome Estimate |
|--------|-----------|---------------------|
| No sampling | 100% | ~500+ GB |
| Default (thresh=0.01, bg=0.5%) | ~0.6% | ~2-3 GB |
| Isoform discovery (thresh=0.005, window=50bp) | ~1.5% | ~14 GB |

### Design Note: No Label Leakage

Sampling uses **prediction scores only** (donor_prob, acceptor_prob),
not ground truth labels (TP/FP/FN/TN). This is intentional — feature
artifacts feed the meta-layer training pipeline, where labels must
remain hidden until evaluation.

## Running Long Processes

Genome-scale feature engineering can take 30+ minutes per chromosome
(conservation and epigenetic stream from remote bigWig servers). Here
are approaches for surviving laptop standby and terminal closures.

### Recommended: `caffeinate` + `nohup` (macOS)

```bash
caffeinate -s nohup mamba run -n agentic-spliceai python -u \
    06_multimodal_genome_workflow.py \
    --config configs/full_stack.yaml \
    --chromosomes all --resume \
    > /tmp/full_stack_features.log 2>&1 &

# Monitor progress
tail -f /tmp/full_stack_features.log
```

- `caffeinate -s` — prevent system sleep (even with lid closed, requires AC power)
- `nohup` — survive terminal closure
- `python -u` — unbuffered output (log stays current)
- `--resume` — skip completed chromosomes on restart

### tmux/screen (SSH / remote)

```bash
tmux new -s features
caffeinate -s mamba run -n agentic-spliceai python -u \
    06_multimodal_genome_workflow.py \
    --config configs/full_stack.yaml \
    --chromosomes all --resume
# Ctrl+B, D to detach; tmux attach -t features to reattach
```

### Crash Recovery with `--resume`

```bash
# First run — gets interrupted at chr3
caffeinate -s nohup mamba run -n agentic-spliceai python -u \
    06_multimodal_genome_workflow.py \
    --config configs/full_stack.yaml \
    --chromosomes all --resume \
    > /tmp/features.log 2>&1 &

# After interruption — resumes from where it left off
# (same command — --resume skips chromosomes with existing output)
```

### On a GPU pod (full genome, faster)

```bash
nohup python -u examples/features/06_multimodal_genome_workflow.py \
    --config examples/features/configs/full_stack.yaml \
    --chromosomes all --resume \
    > /workspace/output/features.log 2>&1 &
```

Expected runtime: ~2-3 hours locally (M1 MacBook, 4 chroms), ~6-8 hours
for full genome. Output with default sampling: ~2-3 GB.

## Quick Reference

```bash
# Single gene exploration (scripts 01-03, 05)
python 01_base_score_features.py --gene TP53
python 02_annotation_and_genomic.py --gene BRCA1 --model spliceai
python 05_multimodal_exploration.py --gene TP53

# YAML-driven production workflow (script 06)
python 06_multimodal_genome_workflow.py --config configs/default.yaml --chromosomes chr22
python 06_multimodal_genome_workflow.py --config configs/full_stack.yaml --chromosomes all --resume
python 06_multimodal_genome_workflow.py --config configs/full_stack.yaml --chromosomes chr19 chr22 --force

# Check what's been generated
ls -lah data/mane/GRCh38/openspliceai_eval/analysis_sequences/
```
