# Experiment 004: Validated Delta Prediction (Single-Pass)

**Date**: December 2025  
**Status**: ✅ Completed  
**Outcome**: **r=0.41 (BEST SO FAR!)**

---

## Key Innovation

This experiment addresses a fundamental limitation of paired prediction:
**base model deltas may be inaccurate for non-splice-altering variants**.

### The Problem with Paired Prediction

```
Paired Prediction (Previous Approach):
  Target = base_model(alt) - base_model(ref)
  
  Issue: If variant is NOT splice-altering but base model predicts 
         a delta anyway, we're training on wrong labels!
```

### Our Solution: Validated Delta Targets

```
Validated Delta Prediction:
  If SpliceVarDB says "Splice-altering":
    Target = base_model(alt) - base_model(ref)  # Trust base model
  
  If SpliceVarDB says "Normal":
    Target = [0, 0, 0]  # Override base model - no effect!
  
  If SpliceVarDB says "Low-frequency" or "Conflicting":
    SKIP  # Uncertain, don't train on it
```

---

## Results

### Correlation (Splice-altering samples only)

| Model | Pearson r | p-value |
|-------|-----------|---------|
| Paired (Siamese) | 0.38 | - |
| **Validated (Single-Pass)** | **0.41** | 1.4e-07 |

**Improvement: +8% correlation**

### Binary Discrimination (SA vs Normal)

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.58 |
| PR-AUC | 0.62 |

### Detection at Threshold=0.1

| Metric | Value |
|--------|-------|
| SA detected | 18.7% |
| False positives | 6.0% |

---

## Why It Works Better

1. **Ground truth filtering**: SpliceVarDB provides validated labels
2. **No false learning**: Doesn't learn from incorrect base model predictions
3. **Cleaner signal**: Normal variants always have zero delta target
4. **Single-pass efficiency**: No reference sequence needed at inference

---

## Architecture

```
                    ValidatedDeltaPredictor
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  alt_seq [B, 4, 501] ──→ [Gated CNN (6 layers)] ──→ [B, 128]   │
│                                                      │          │
│  ref_base [B, 4] ──┬──→ [MLP Embed] ──→ [B, 128]    │          │
│  alt_base [B, 4] ──┘                       │         │          │
│                                            └────┬────┘          │
│                                                 │               │
│                                         concat [B, 256]         │
│                                                 │               │
│                                         [Delta Head]            │
│                                                 │               │
│                                          Δ [B, 3]               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Training Configuration

```python
# Data
samples = 2000 (balanced: 1000 SA, 1000 Normal)
context_size = 501 nt
test_chromosomes = ['21', '22']

# Model
hidden_dim = 128
n_layers = 6
dropout = 0.1
parameters = 3,011,843

# Training
epochs = 40
batch_size = 32
optimizer = AdamW(lr=5e-5, weight_decay=0.02)
scheduler = OneCycleLR(max_lr=5e-4)
```

---

## Comparison Summary

| Aspect | Paired Prediction | Validated Prediction |
|--------|-------------------|---------------------|
| Input | ref_seq + alt_seq | alt_seq + var_info |
| Target source | Base model (may be wrong) | SpliceVarDB-validated |
| Forward passes | 2 | **1** |
| Correlation | r=0.38 | **r=0.41** |
| Inference speed | Slower (2 passes) | **Faster (1 pass)** |

---

## Recommendations

1. **Use validated targets** for any delta prediction task
2. **Increase training data**: 2000 samples is limited
3. **Try longer context**: 501nt → 1001nt
4. **Add position attention** for interpretability
5. **Scale with HyenaDNA** on GPU

---

## Files

| File | Description |
|------|-------------|
| `models/validated_delta_predictor.py` | Model implementation |
| `methods/VALIDATED_DELTA_PREDICTION.md` | Design rationale |

---

*This approach is now the recommended method for delta prediction.*












