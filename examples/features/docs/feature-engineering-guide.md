# Feature Engineering Guide

Reference for the feature engineering workflow in `examples/features/`.
Covers the example scripts (01-05), modality catalog, position sampling,
and practical recipes for running genome-scale pipelines.

## Example Scripts — Progressive Storyline

The scripts build on each other, introducing concepts incrementally:

| Script | Focus | Modalities | Scope |
|--------|-------|-----------|-------|
| `01_base_score_features.py` | Base score features only | base_scores | Single gene |
| `02_annotation_and_genomic.py` | Standard 3-modality set | base_scores, annotation, genomic | Single gene |
| `03_configurable_modalities.py` | Per-modality config overrides | base_scores (configured) | Single gene |
| `04_genome_scale_workflow.py` | Production workflow + sampling | base_scores, annotation, genomic | Chromosomes |
| `05_multimodal_exploration.py` | Experimental signal exploration | conservation, epigenetic, expression | Single gene |

**Convention**: Scripts 02 and 04 share the same default modality set
(`base_scores, annotation, genomic`). When new modalities graduate from
experimental (05) to standard, both scripts update together.

## Modality Catalog

### Implemented (in `splice_engine/features/modalities/`)

| Modality | Columns | External Data | Notes |
|----------|---------|--------------|-------|
| **base_scores** | 43 | None | Derived from donor/acceptor/neither probs: entropy, ratios, rank stats |
| **annotation** | 3 | GTF (auto-resolved) | splice_type, transcript_id, exon context |
| **genomic** | 4 | None | relative_gene_position (strand-corrected), distances to gene boundaries, gc_content |
| **sequence** | varies | FASTA | k-mer frequencies, dinucleotide composition (heavier, opt-in) |

### Planned / Experimental (explored in script 05)

| Modality | Signal | Data Source | Status |
|----------|--------|------------|--------|
| **conservation** | PhyloP, PhastCons | UCSC bigWig (remote) | Explored in 05, not yet a registered modality |
| **epigenetic** | H3K36me3, H3K4me3 | ENCODE bigWig (remote S3) | Explored in 05, tissue-specific |
| **expression** | RNA-seq junction counts | STAR SJ.out.tab / GTEx | Explored in 05, requires junction files |
| **structural** | RNA secondary structure | ViennaRNA / predicted | Research phase |
| **variant** | ClinVar pathogenicity | ClinVar VCF | Planned (Phase 6) |

When a new modality is implemented as a registered class in `modalities/`:
1. Add to `modalities/__init__.py` auto-registration
2. Update default modality list in scripts 02 and 04 (if lightweight / no special data)
3. Update `FeaturePipelineConfig.modalities` default (if universally useful)

## Position Sampling

### Why sample?

The full genome has ~1.15 billion nucleotide positions, but only ~0.06%
have a splice probability above 0.01. Storing feature-enriched parquet
files for all positions requires ~271 GB — impractical for most machines.

### How it works

Three-tier sampling applied per chromosome after feature engineering:

1. **Splice sites** (always kept): positions where `donor_prob > threshold`
   or `acceptor_prob > threshold`
2. **Proximity zone** (optional): positions within N bp of a splice site
3. **Background** (random sample): fraction of remaining positions

### Storage estimates (47 columns, parquet)

| Config | Retention | Full Genome |
|--------|-----------|-------------|
| No sampling | 100% | ~271 GB |
| `--sample` (default: thresh=0.01, bg=0.5%) | ~0.6% | ~1.5 GB |
| `--sample --sample-bg-rate 0.001` | ~0.2% | ~0.4 GB |
| `--sample --sample-window 50` | ~5.4% | ~15 GB |
| `--sample --sample-window 200` | ~15% | ~41 GB |

### Design note: no label leakage

Sampling uses **prediction scores only** (donor_prob, acceptor_prob),
not ground truth labels (TP/FP/FN/TN). This is intentional — feature
artifacts feed the meta-layer training pipeline, where labels must
remain hidden until evaluation. The meta-spliceai `tn_sample_factor`
approach (which uses TP/FP/FN categories) is appropriate for evaluation
artifacts but not for pre-training feature generation.

## Running Long Processes

Genome-scale prediction + feature engineering can take 30+ minutes per
chromosome on Apple Silicon (MPS). Here are approaches for surviving
laptop standby, SSH disconnects, and terminal closures.

### Option 1: `caffeinate` + background (macOS — recommended for local)

`caffeinate` prevents macOS from sleeping while the process runs.
Combined with `nohup`, the process survives terminal closure:

```bash
# Prevent sleep + run in background, log to file
caffeinate -i nohup python 04_genome_scale_workflow.py \
    --chromosomes chr20 chr21 chr22 --sample \
    > feature_run.log 2>&1 &

# Check progress
tail -f feature_run.log

# Check if still running
jobs -l
# or
ps aux | grep 04_genome_scale
```

Flags:
- `caffeinate -i` — prevent idle sleep (process keeps running during lid-close
  if "Prevent sleeping when display is off" is enabled in Energy settings)
- `caffeinate -s` — prevent system sleep entirely (requires AC power)
- `nohup` — detach from terminal (survives terminal close)
- `&` — run in background

### Option 2: `tmux` or `screen` (recommended for SSH / remote)

Terminal multiplexers keep sessions alive across disconnects:

```bash
# Start a named session
tmux new -s features

# Run the command normally inside tmux
python 04_genome_scale_workflow.py \
    --chromosomes chr20 chr21 chr22 --sample

# Detach: Ctrl+B, then D
# Reattach later:
tmux attach -t features

# List sessions:
tmux ls
```

Combined with caffeinate for local use:

```bash
tmux new -s features
caffeinate -i python 04_genome_scale_workflow.py \
    --chromosomes chr20 chr21 chr22 --sample
# Ctrl+B, D to detach
```

### Option 3: `launchd` plist (macOS — survives reboot)

For truly unattended runs that must survive restarts:

```bash
# Create a one-shot launchd job
cat > ~/Library/LaunchAgents/com.spliceai.features.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.spliceai.features</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/zsh</string>
        <string>-c</string>
        <string>conda run -n agentic-spliceai python /path/to/04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 --sample</string>
    </array>
    <key>StandardOutPath</key>
    <string>/tmp/spliceai_features.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/spliceai_features.err</string>
    <key>RunAtLoad</key>
    <true/>
</dict>
</plist>
EOF

# Load and run
launchctl load ~/Library/LaunchAgents/com.spliceai.features.plist

# Check status
launchctl list | grep spliceai

# Remove when done
launchctl unload ~/Library/LaunchAgents/com.spliceai.features.plist
rm ~/Library/LaunchAgents/com.spliceai.features.plist
```

### Option 4: `--resume` for crash recovery

Regardless of which method you use, always design for interruption:

```bash
# First run — gets interrupted at chr3
caffeinate -i nohup python 04_genome_scale_workflow.py \
    --chromosomes chr1 chr2 chr3 chr4 chr5 --sample \
    > run.log 2>&1 &

# After interruption — resumes from where it left off
caffeinate -i nohup python 04_genome_scale_workflow.py \
    --chromosomes chr1 chr2 chr3 chr4 chr5 --sample --resume \
    >> run.log 2>&1 &
```

The `--resume` flag skips chromosomes that already have output files.
Predictions are also incrementally merged — only missing chromosomes
are predicted.

### Recommended recipe: full genome with sampling

```bash
# All canonical chromosomes, sampled, with crash recovery
ALL_CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 \
chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 \
chr21 chr22 chrX chrY"

caffeinate -i nohup python 04_genome_scale_workflow.py \
    --chromosomes $ALL_CHROMS \
    --sample --resume \
    > genome_features.log 2>&1 &

# Monitor
tail -f genome_features.log
```

Expected runtime: ~6-8 hours on M1 MacBook (MPS), ~1.5 GB output with
default sampling. Predictions (~110 GB TSV) are cached in the registry
for reuse.

## Quick Reference

```bash
# Single gene exploration (scripts 01-03, 05)
python 01_base_score_features.py --gene TP53
python 02_annotation_and_genomic.py --gene BRCA1 --model spliceai
python 03_configurable_modalities.py --gene UNC13A --context-window 5
python 05_multimodal_exploration.py --gene TP53

# Genome-scale (script 04)
python 04_genome_scale_workflow.py --chromosomes chr22              # full output
python 04_genome_scale_workflow.py --chromosomes chr22 --sample     # sampled
python 04_genome_scale_workflow.py --chromosomes chr22 --sample \
    --sample-window 50 --sample-bg-rate 0.001                       # custom sampling

# Incremental additions
python 04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 --resume --sample

# Fresh redo (remove old artifacts first)
rm -rf data/mane/GRCh38/openspliceai_eval/analysis_sequences/
python 04_genome_scale_workflow.py --chromosomes chr20 chr21 chr22 --sample
```
