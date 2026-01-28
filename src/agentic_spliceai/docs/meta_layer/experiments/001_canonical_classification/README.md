# Experiment 001: Canonical Classification

**Date**: December 2025  
**Status**: Completed  
**Outcome**: Partial Success (High accuracy, poor variant detection)

---

## Hypothesis

Training a multimodal meta-layer on canonical splice sites (from GTF annotations) will improve splice site classification and generalize to variant-induced splice changes.

---

## Setup

### Data
- **Training**: Base layer artifacts (`analysis_sequences_*.tsv`)
- **Labels**: GTF canonical splice sites (donor, acceptor, neither)
- **Evaluation**: SpliceVarDB variants

### Model
- **Architecture**: `MetaSpliceModel` (multimodal)
- **Sequence Encoder**: CNN (lightweight)
- **Score Encoder**: MLP
- **Fusion**: Concatenation

### Training
- **Epochs**: 30
- **Batch Size**: 64
- **Optimizer**: AdamW (lr=1e-4)
- **Loss**: CrossEntropyLoss with class weights

---

## Results

### Classification Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.11% |
| PR-AUC (macro) | 0.987 |
| Per-class AP | D: 0.98, A: 0.98, N: 0.99 |

### Variant Detection

| Metric | Base Model | Meta-Layer |
|--------|------------|------------|
| Detection Rate | 67% | **17%** |
| Mean |Δ| (SA) | 0.13 | 0.02 |

---

## Analysis

### Why Classification Succeeded
1. Canonical sites are well-defined (clear GT/AG patterns)
2. Base model scores are highly predictive
3. Task is relatively easy (≈random guessing baseline: 33%)

### Why Variant Detection Failed
1. **Training-evaluation mismatch**: Trained on canonical sites, evaluated on variants
2. **No variant-specific learning**: Model never sees variant effects during training
3. **Confidence suppression**: Model becomes overly confident in canonical labels

---

## Key Insight

> **High classification accuracy on canonical sites does NOT translate to variant effect detection.**

The meta-layer essentially learns to replicate base model behavior, which is already good at canonical sites but blind to variant-induced changes.

---

## Lessons Learned

1. **Train for evaluation task**: If goal is variant detection, must train on variant data
2. **SpliceVarDB is essential**: Only source of ground-truth variant effects
3. **Need different approach**: Classification → Delta prediction

---

## Next Steps

This experiment motivated the pivot to:
1. **Paired Delta Prediction** (Experiment 002)
2. **Validated Delta Prediction** (Experiment 004) - BEST

---

## Files

| File | Description |
|------|-------------|
| `models/meta_splice_model.py` | Model implementation |
| `workflows/canonical_training.py` | Training workflow |












