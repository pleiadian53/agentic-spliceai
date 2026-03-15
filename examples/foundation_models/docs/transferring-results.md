# Transferring Results from GPU Pods

How to download experiment outputs (embeddings, metrics, plots, model checkpoints)
from a running SkyPilot/RunPod cluster back to your local machine.

## Quick Reference

```bash
# Download all outputs from a pod
mkdir -p ./output/<topic>/
rsync -Pavz <cluster>:/workspace/output/ ./output/<topic>/

# Example: exon classifier results
mkdir -p ./output/exon_classifier
rsync -Pavz sky-23ad-pleiadian53:/workspace/output/ ./output/exon_classifier/
```

---

## How It Works

### Source path convention

All experiment scripts write to `/workspace/output/<experiment>/` on the pod:

```
/workspace/output/
  sparse-evo2/           # 05_sparse_exon_classifier.py --foundation-model evo2
  sparse-splicebert/     # 05_sparse_exon_classifier.py --foundation-model splicebert
  sparse-hyenadna/       # 05_sparse_exon_classifier.py --foundation-model hyenadna
  sparse-dnabert/        # 05_sparse_exon_classifier.py --foundation-model dnabert
```

Each experiment directory contains:

```
sparse-<model>/
  sparse_embeddings.npz          # Frozen embeddings (0.9-1.5 MB)
  model/
    best_model.pt                # Classifier checkpoint (~4-30 MB)
    eval_metrics.json            # Test set metrics (AUROC, AUPRC, etc.)
  plots/
    loss_curves.png              # Train/val loss over epochs
    metric_curves.png            # AUROC/AUPRC over epochs
    training_overview.png        # Combined training summary
```

### Local destination convention

Group results by experiment topic under `<project>/output/`:

```
output/
  exon_classifier/       # Phase A: sparse exon classification
    sparse-evo2/
    sparse-splicebert/
    sparse-hyenadna/
    sparse-dnabert/
  splice_predictor/      # Phase B: dense splice site prediction (future)
```

---

## rsync Patterns

### Download everything

```bash
mkdir -p ./output/exon_classifier
rsync -Pavz <cluster>:/workspace/output/ ./output/exon_classifier/
```

Flags: `-P` (progress + partial), `-a` (archive/recursive), `-v` (verbose), `-z` (compress).

### Exclude large model checkpoints

Keep only metrics, plots, and embeddings (~2 MB per model vs ~30 MB with checkpoints):

```bash
rsync -Pavz --exclude='*.pt' \
  <cluster>:/workspace/output/ ./output/exon_classifier/
```

### Download only metrics and plots (lightest)

```bash
rsync -Pavz \
  --include='*/' --include='*.json' --include='*.png' --exclude='*' \
  <cluster>:/workspace/output/ ./output/exon_classifier/
```

### Download a single experiment

```bash
rsync -Pavz <cluster>:/workspace/output/sparse-evo2/ ./output/exon_classifier/sparse-evo2/
```

---

## Trailing Slash Semantics

This is the most common rsync pitfall:

| Source | Destination | Result |
|--------|------------|--------|
| `/workspace/output/` (trailing /) | `./output/exon_classifier/` | Contents of `output/` placed inside `exon_classifier/` |
| `/workspace/output` (no trailing /) | `./output/exon_classifier/` | Creates `exon_classifier/output/` (extra nesting) |

**Rule of thumb**: Always use trailing `/` on the source to copy *contents*, not the directory itself.

---

## Upload (Local to Pod)

For uploading code changes to a running pod:

```bash
# Upload specific files (preserving directory structure)
rsync -Pavz --relative \
  foundation_models/foundation_models/splicebert/model.py \
  examples/foundation_models/05_sparse_exon_classifier.py \
  <cluster>:~/sky_workdir/

# Upload all Python files
rsync -Pavz --include='*/' --include='*.py' --exclude='*' \
  ./ <cluster>:~/sky_workdir/
```

**Note**: Uploaded code goes to `~/sky_workdir/` (SkyPilot's working directory),
while experiment outputs go to `/workspace/output/` (persistent volume).

---

## Cluster Identification

Find your running cluster name:

```bash
sky status          # Lists all active clusters
```

Cluster names follow the pattern `sky-<4char>-<username>`, e.g. `sky-23ad-pleiadian53`.
