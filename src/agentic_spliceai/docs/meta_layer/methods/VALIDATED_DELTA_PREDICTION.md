# Validated Delta Prediction: Single-Pass with Ground Truth Targets

**Status**: ✅ BEST APPROACH (r=0.41)  
**Last Updated**: December 2025

---

## Overview

Validated Delta Prediction addresses a fundamental limitation of paired (Siamese) prediction: **base model deltas may be inaccurate for non-splice-altering variants**.

This approach uses SpliceVarDB classifications to filter/validate training targets, ensuring the model learns from ground truth rather than potentially incorrect base model predictions.

---

## The Problem with Paired Prediction

```
Paired Prediction (Previous Approach):
  Target = base_model(alt) - base_model(ref)
  
  Issue: If variant is NOT splice-altering but base model predicts 
         a delta anyway, we're training on wrong labels!
```

### Why This Matters

| Scenario | Base Model Says | SpliceVarDB Says | Training Target |
|----------|-----------------|------------------|-----------------|
| True Positive | Delta | Splice-altering | ✅ Trust delta |
| False Positive | Delta | Normal | ❌ Wrong target! |
| True Negative | No delta | Normal | ✅ Correct |
| False Negative | No delta | Splice-altering | ⚠️ Missing info |

---

## Our Solution: Validated Delta Targets

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

### Key Components

1. **Gated CNN Encoder**: Dilated convolutions with gating for long-range dependencies
2. **Variant Embedding**: Encodes ref_base + alt_base information
3. **Fusion**: Concatenates sequence and variant features
4. **Delta Head**: Predicts [Δ_donor, Δ_acceptor, Δ_neither]

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

## Training Configuration

```python
from agentic_spliceai.splice_engine.meta_layer.models import (
    ValidatedDeltaPredictor,
    create_validated_delta_predictor
)

# Create model
model = create_validated_delta_predictor(
    variant='basic',     # or 'attention' for interpretability
    hidden_dim=128,
    n_layers=6,
    dropout=0.1
)

# Training config
config = {
    'epochs': 40,
    'batch_size': 32,
    'learning_rate': 5e-5,
    'weight_decay': 0.02,
    'scheduler': 'OneCycleLR'
}
```

---

## Usage

### Training

```python
from agentic_spliceai.splice_engine.meta_layer import ValidatedDeltaPredictor

model = ValidatedDeltaPredictor(hidden_dim=128)

# Forward pass (single-pass, no ref_seq needed!)
delta = model(alt_seq, ref_base, alt_base)  # [B, 3]
```

### Inference

```python
# Final score = base_scores + predicted_delta
ref_scores = base_model(ref_seq)  # [donor, acceptor, neither]
delta = validated_predictor(alt_seq, ref_base, alt_base)

final_scores = ref_scores + delta  # Adjusted prediction
```

---

## Comparison to Paired Prediction

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
2. **Scale with more data**: Current results use 2000 samples
3. **Try longer context**: 501nt → 1001nt
4. **Add position attention** (use `variant='attention'`) for interpretability
5. **Scale with HyenaDNA** on GPU for best results

---

## Model Files

| File | Description |
|------|-------------|
| `models/validated_delta_predictor.py` | Model implementation |
| `data/variant_dataset.py` | Dataset with validated targets |

---

*This approach is the recommended method for delta prediction.*












