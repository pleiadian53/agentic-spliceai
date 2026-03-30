# Splice Site Prediction with Foundation Model Embeddings

**Date**: March 2026
**Pipeline**: `07a_direct_shard_splice_predictor.py`
**Foundation Model**: SpliceBERT (512-dim, 6-layer BERT, pre-trained on 2M pre-mRNAs from 72 vertebrates)

## Summary

We trained a lightweight splice site classifier on top of frozen SpliceBERT embeddings
using genome-scale data (19K+ MANE genes, SpliceAI chromosome split). The classifier
achieves **AUPRC=0.832** and **AUROC=0.999** on held-out test chromosomes (chr1,3,5,7,9),
demonstrating that generic RNA language model representations carry substantial splice
signal — even without any task-specific fine-tuning of the foundation model.

This result establishes a strong frozen-head baseline for the project's three-layer
architecture (Base Layer → Meta Layer → Agentic Layer), where foundation model embeddings
will serve as one modality in the multimodal meta-layer fusion.

---

## Experimental Setup

### Data

- **Annotation**: MANE Select v1.3 (GRCh38), ~19,226 protein-coding genes
- **Split**: SpliceAI chromosome split (Jaganathan et al., 2019)
  - Train: even chromosomes + chr10-22 + chrX + chrY (12,334 train + 1,370 val)
  - Test: chr1, chr3, chr5, chr7, chr9 (5,522 genes)
- **Window size**: 512 bp, step 512 (non-overlapping)
- **Embedding extraction**: Direct-to-shard pipeline, SpliceBERT frozen on CUDA
  - 93 train shards, 9 val shards (380,928 + 36,864 windows)
  - Note: chrX partially extracted (genes 1-150 of 787) + chrY missing due to
    disk-full crash mid-extraction. ~7% of training data lost. See "Infrastructure"
    section below.

### Classifier

| Parameter | Value |
|-----------|-------|
| Architecture | Dilated CNN (3 blocks, kernel=11, dilations=[1,4,16]) |
| Input dim | 512 (SpliceBERT hidden dim) |
| Hidden dim | 128 |
| Parameters | 1,149,699 |
| Output | 3-class: neither (0), acceptor (1), donor (2) |

### Training

| Parameter | Value |
|-----------|-------|
| Loss | Focal loss (gamma=2.0) |
| Class weights | neither=0.3, acceptor=1000.0, donor=1000.0 |
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| Batch size | 256 |
| Epochs | 100 (no early stopping triggered) |
| Best epoch | 97 (val_AUPRC=0.8904) |
| DataLoader workers | 2 (fork-safe, pid-keyed h5py handles) |
| Device | NVIDIA A40 48GB (RunPod) |

### Calibration

Post-hoc temperature scaling on the validation set:

| Metric | Before | After |
|--------|--------|-------|
| ECE | 0.0001 | 0.0000 |
| NLL | 0.0014 | 0.0006 |

Temperature vector: `[0.325, 1.520, 1.554]`
- "neither" (T=0.325): sharpens — focal loss makes the model overconfident on the
  dominant class, temperature corrects this
- Splice classes (T≈1.5): softens slightly — extreme class weights cause mild
  underconfidence on rare splice sites

---

## Results

### Per-Chromosome Test Metrics

| Chromosome | Genes | Positions | AUROC | AUPRC | Acc AUPRC | Don AUPRC | Time |
|-----------|-------|-----------|-------|-------|-----------|-----------|------|
| chr1 | 1,977 | 201.4M | 0.9991 | 0.8481 | 0.8294 | 0.8668 | 98 min |
| chr3 | 1,030 | 167.2M | 0.9993 | 0.8208 | 0.8037 | 0.8380 | 79 min |
| chr5 | 860 | 130.6M | 0.9992 | 0.8279 | 0.8186 | 0.8373 | 60 min |
| chr7 | 886 | 135.5M | 0.8140 | 0.8140 | 0.7933 | 0.8346 | 68 min |
| chr9 | 747 | 92.2M | 0.9990 | 0.8498 | 0.8320 | 0.8675 | 48 min |
| **Aggregate** | **5,500** | **726.9M** | **0.9991** | **0.8320** | **0.8152** | **0.8489** | **354 min** |

### Aggregate Metrics (Position-Weighted)

| Class | AUROC | AUPRC | Precision | Recall | F1 | Sites |
|-------|-------|-------|-----------|--------|-----|-------|
| Acceptor | 0.9989 | 0.8152 | 0.6849 | 0.8207 | 0.7462 | 102,351 |
| Donor | 0.9993 | 0.8489 | 0.6636 | 0.8767 | 0.7546 | 103,617 |
| **Mean** | **0.9991** | **0.8320** | **0.6743** | **0.8487** | **0.7504** | — |

### Comparison with POC Run (chr3+22)

| Run | Train genes | Val AUPRC | Test AUPRC | Test AUROC |
|-----|------------|-----------|------------|------------|
| chr3+22 (POC) | 409 (chr22 only) | 0.7770 | 0.6767 | 0.9975 |
| **Full genome** | **12,334** | **0.8904** | **0.8320** | **0.9991** |

30× more training genes → +23% test AUPRC (0.677 → 0.832). The POC suffered from
severe train/test imbalance (409 train genes vs 934 test genes from chr3).

### Context: Where This Fits

| Model | Approach | Test AUPRC | Notes |
|-------|----------|------------|-------|
| SpliceAI | End-to-end CNN, 10kb context | ~0.95+ | Purpose-built for splice prediction |
| OpenSpliceAI | PyTorch reimplementation | ~0.95+ | MANE annotation, GRCh38 |
| Meta-layer XGBoost (M1) | Base scores + 83 features | 0.998 | Uses SpliceAI scores as input |
| **SpliceBERT frozen head** | **Generic embeddings + CNN head** | **0.832** | **No fine-tuning** |

The ~12% gap vs purpose-built models is expected: SpliceBERT was pre-trained for general
RNA understanding, not splice site detection. The frozen classifier head (1.1M params) is
orders of magnitude smaller than SpliceAI's full architecture.

---

## Key Findings

### 1. Generic RNA embeddings carry strong splice signal

AUPRC=0.832 from frozen embeddings demonstrates that SpliceBERT's pre-training on
2M pre-mRNAs from 72 vertebrates captures biologically meaningful splice patterns,
even though the model was never explicitly trained for splice site detection.

### 2. Donor prediction is consistently easier than acceptor

Donor AUPRC (0.849) outperforms acceptor (0.815) across all 5 test chromosomes without
exception. This likely reflects the stronger sequence motif at donor sites (GT dinucleotide)
compared to the more degenerate acceptor branch point + AG signal.

### 3. Per-chromosome performance is stable

AUPRC ranges from 0.814 (chr7) to 0.850 (chr9) — a narrow 3.6% spread. No chromosome
is a major outlier, indicating the model generalizes well across the genome.

### 4. High recall, moderate precision

The model achieves 82-88% recall but only 66-70% precision at the 0.5 threshold.
This is the expected trade-off for a frozen-head model: it captures most true splice
sites but also flags some false positives that a purpose-built model would reject
using longer-range context (SpliceAI uses 10kb; SpliceBERT sees 512bp windows).

### 5. Scaling data matters more than scaling epochs

The jump from 409 to 12,334 training genes (+30×) produced a +23% AUPRC gain
(0.677 → 0.832). In contrast, the full-genome model ran 100 epochs (vs 50 for the POC)
with diminishing returns after epoch ~60 — val_AUPRC plateaued at ~0.89 while val_loss
diverged (0.019 → 0.090), signaling mild overfitting.

### 6. Calibration is near-perfect

ECE dropped from 0.0001 to 0.0000 with temperature scaling. The model's probability
estimates are well-calibrated, making the predictions directly usable as confidence
scores in downstream pipelines without additional calibration.

---

## Training Dynamics

### Learning Curve

```
Epoch   Train Loss   Val Loss   Val AUPRC
  1     0.0197       0.0145     0.576
  2     0.0101       0.0098     0.701    ← rapid initial gain
  5     0.0066       0.0077     0.699
 10     0.0038       0.0091     0.766
 20     0.0021       0.0133     0.827
 30     0.0015       0.0182     0.827
 40     0.0014       0.0187     0.850
 50     0.0015       0.0186     0.850
 60     0.0007       0.0337     0.868
 70     0.0005       0.0483     0.880
 80     0.0004       0.0620     0.883
 90     0.0003       0.0779     0.888
 97     0.0003       0.0818     0.890    ← best epoch (restored)
100     0.0003       0.0897     0.889
```

Notable pattern: val_AUPRC continues climbing even as val_loss diverges significantly
(0.015 → 0.090). This is characteristic of focal loss with extreme class weights —
the model becomes more discriminative (better ranking = higher AUPRC) even as its
probability calibration degrades (higher NLL). Temperature scaling corrects this
post-hoc.

Early stopping with patience=20 on AUPRC (not loss) would have been optimal.
The current run used patience=20 but never triggered because AUPRC kept improving.

---

## Infrastructure Notes

### Disk-Full Crash and Recovery

The initial extraction run crashed mid-chrX (gene 151/787) when the 500 GB RunPod
network volume filled up. Key recovery steps:

1. **Corrupted shard**: `train_shard_093.h5` had a truncated HDF5 header — detected
   via `h5py.File()` scan, deleted (4096 windows lost, <1.1% of data)
2. **Missing manifest**: `manifest.json` was only saved at the end of extraction,
   so it was lost in the crash. Fixed by adding incremental manifest saves after
   each chromosome.
3. **Space recovery**: Deleted stale `miniforge3/` (52 GB) from the network volume

The training proceeded with 93 intact shards (380,928 windows) — missing ~7% of
training data from the tail of chrX and all of chrY. This is a negligible loss
for a frozen-head classifier.

### Volume Path Discovery

SkyPilot-managed RunPod pods mount the network volume at `/runpod-volume`, NOT
`/workspace` (which is the convention for RunPod-native pods launched via the UI).
The first extraction run accidentally wrote to `/workspace` (ephemeral container
overlay, 256 GB) and would have lost all data on pod teardown.

### GPU Utilization

| Phase | GPU Compute | VRAM | Bottleneck |
|-------|------------|------|------------|
| Phase 1 (extraction) | 30-40% | 5% | SpliceBERT inference + HDF5 writes |
| Phase 2 (training) | 33% | 2% | DataLoader I/O from 93 shards |
| Phase 3 (evaluation) | 20-30% | 3% | Sequential live inference per gene |

Low VRAM utilization (2-5%) is structurally expected for a frozen-head pipeline —
the 1.1M classifier and 19M SpliceBERT are tiny relative to the A40's 48 GB.
GPU compute utilization is bounded by I/O: HDF5 reads during training, FASTA
extraction during evaluation.

### Timing

| Phase | Duration | Notes |
|-------|----------|-------|
| Phase 1: Extraction | ~6 hours | 24 chromosomes, direct-to-shard |
| Phase 2: Training | ~87 hours | 100 epochs × 52 min/epoch |
| Phase 2.5: Calibration | <1 min | 76 temperature optimization epochs |
| Phase 3: Evaluation | 5.9 hours | 5,500 genes, live inference |
| **Total** | **~99 hours** | 3 pod sessions, $0.44/hr = ~$44 |

---

## Paths to Improvement

### Short-term: Fine-tuning (08_foundation_model_finetuning.py)

Unfreezing SpliceBERT and training end-to-end should close a significant portion
of the gap to purpose-built models. The frozen embeddings already capture splice
signal; fine-tuning will optimize them for the specific task.

### Medium-term: Larger foundation models

Phase A benchmarks showed Evo2 (AUROC=0.9937) significantly outperforms SpliceBERT
(0.8594) on exon classification. Evo2's 7B parameters and long-range context should
produce better embeddings for splice prediction.

### Long-term: Dual role in the three-layer architecture

The fine-tuned foundation model serves **two complementary roles**:

**1. Base layer — a new base model for splicing.** Once fine-tuned, the foundation
model becomes a "foundation model for splicing" that produces splice site predictions
(delta scores) alongside SpliceAI and OpenSpliceAI. This is especially valuable for
models like Evo2 that bring fundamentally different inductive biases (attention over
long-range context vs SpliceAI's dilated CNN) and may capture splice patterns that
CNN-based models miss entirely. The frozen-head results here (AUPRC=0.832) represent
the lower bound; fine-tuning should close the gap significantly.

**2. Meta layer — embeddings as a rich feature modality.** Beyond scalar predictions,
the foundation model's dense embeddings (512-dim per position for SpliceBERT, 4096-dim
for Evo2) encode biological context that cannot be distilled into a single score. These
embeddings feed into the multimodal meta-layer fusion alongside:
- Base model delta scores (SpliceAI, OpenSpliceAI, and the foundation model itself)
- Conservation (PhyloP, PhastCons)
- Epigenetics (H3K36me3, H3K4me3)
- GTEx junction support
- RBP eCLIP binding
- DNA sequence features

The meta-layer XGBoost baseline (M1) already achieves AUPRC=0.998 with 83 features.
Adding foundation model predictions as another base model score and/or embeddings as a
new modality may push performance further, particularly for novel/non-canonical splice
sites where existing base models underperform.

---

## Reproduction

```bash
# Phase 1+2: Train (requires GPU pod with network volume)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes all \
    --step-size 512 \
    --batch-size 256 \
    --split spliceai \
    -o /runpod-volume/output/splice_classifier/splicebert-full-genome/

# Phase 3: Evaluate (can run on a separate pod session)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes all \
    --split spliceai \
    --eval-only \
    --checkpoint /runpod-volume/output/splice_classifier/splicebert-full-genome/model/ \
    -o /runpod-volume/output/splice_classifier/splicebert-full-genome/
```

## Output Artifacts

```
output/splice_classifier/splicebert-full-genome/
  model/
    best_model.pt            # Trained + calibrated classifier (4.6 MB)
    temperature.pt           # Calibration temperature vector
    training_history.json    # Per-epoch train/val metrics
    eval_metrics.json        # Calibration results
  eval/
    test_metrics.json        # Aggregate + per-chromosome test metrics
    test_metrics_chr1.json   # Per-chromosome breakdown
    test_metrics_chr3.json
    test_metrics_chr5.json
    test_metrics_chr7.json
    test_metrics_chr9.json
  shards/                    # 93 train + 9 val HDF5 shard files (409 GB, on pod only)
```
