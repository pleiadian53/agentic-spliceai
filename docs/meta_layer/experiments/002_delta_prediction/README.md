# Experiment 002: Paired Delta Prediction

**Date**: December 2025  
**Status**: Completed  
**Outcome**: Moderate correlation (r=0.38), insufficient for practical use

---

## Hypothesis

A Siamese architecture that compares reference and alternate sequences can learn to predict splice site score changes (deltas) better than the base model.

---

## Setup

### Data
- **Input**: (ref_seq, alt_seq) pairs from SpliceVarDB variants
- **Target**: `base_model(alt) - base_model(ref)` (per-position deltas)
- **Context**: 501 nucleotides centered on variant

### Model Variations Tested

| Model | Architecture | Loss |
|-------|--------------|------|
| DeltaPredictorV2 | Simple CNN | MSE |
| SimpleCNNDeltaPredictor | Gated CNN | MSE |
| + Quantile | Gated CNN | Quantile (τ=0.9) |
| + Scaled | Gated CNN | Scaled MSE |
| + Temperature | Gated CNN | Temperature scaling |

---

## Results

### Correlation with True Deltas

| Variation | Pearson r | Notes |
|-----------|-----------|-------|
| V2 Original | -0.04 | No learning |
| V2 + 10x data | 0.002 | Data alone insufficient |
| Gated CNN | 0.36 | Architecture matters! |
| + Quantile loss | **0.38** | Best for this approach |
| + Scaling | 0.22 | Overfitting |
| + Temperature | -0.03 | No improvement |
| + Multi-task | -0.07 | Task interference |

### Best Model Performance

| Metric | Value |
|--------|-------|
| Pearson r | 0.38 |
| SA Detection (>0.1) | 22% |
| False Positive Rate | 8% |

---

## Analysis

### What Worked

1. **Gated CNN architecture**: Dilated convolutions capture long-range dependencies
2. **Quantile loss (τ=0.9)**: Focuses on large deltas, reduces noise learning
3. **LayerNorm + GELU**: Better than BatchNorm + ReLU

### What Limited Performance

1. **Target quality**: Base model deltas may be inaccurate for non-SA variants
2. **Training on wrong labels**: If base model predicts delta for normal variant, we learn noise
3. **Correlation ceiling**: r=0.38 suggests fundamental limitation

---

## Key Insight

> **The target (base model delta) is the limiting factor.**
>
> If base model prediction for a variant is wrong, we're training on wrong labels.
> This limits how much we can improve over the base model.

---

## Lessons Learned

1. **Architecture matters**: Gated CNN >> Simple CNN
2. **Loss matters**: Quantile loss >> MSE for sparse deltas
3. **Target quality limits learning**: Can't exceed base model accuracy with base model labels
4. **Need validated targets**: SpliceVarDB can help filter reliable training examples

---

## Next Steps

This experiment motivated:
1. **Binary Classification** (Experiment 003) - Decomposed approach
2. **Validated Delta Prediction** (Experiment 004) - Ground truth filtering

---

## Files

| File | Description |
|------|-------------|
| `models/delta_predictor.py` | Siamese architecture |
| `models/delta_predictor_v2.py` | Per-position output |
| `models/hyenadna_delta_predictor.py` | SimpleCNNDeltaPredictor |
| `models/delta_predictor_calibrated.py` | Calibration wrappers |












