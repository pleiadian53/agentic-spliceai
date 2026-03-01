# Paired Delta Prediction: Siamese Architecture

**Status**: Tested (r=0.38)  
**Last Updated**: December 2025

---

## Overview

Paired Delta Prediction uses a Siamese architecture to predict splice site score changes by comparing reference and alternate sequences through shared encoder weights.

---

## Architecture

```
                      Paired Delta Predictor (Siamese)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ref_seq [B, 4, L]                     alt_seq [B, 4, L]              │
│         │                                      │                        │
│         ▼                                      ▼                        │
│   ┌─────────────┐                        ┌─────────────┐                │
│   │   Encoder   │ ◄── shared weights ──► │   Encoder   │                │
│   │  (Gated CNN)│                        │  (Gated CNN)│                │
│   └─────────────┘                        └─────────────┘                │
│         │                                      │                        │
│         ▼                                      ▼                        │
│   ref_emb [B, D]                         alt_emb [B, D]                 │
│         │                                      │                        │
│         └──────────────┬───────────────────────┘                        │
│                        │                                                │
│                        ▼                                                │
│              ┌──────────────────┐                                       │
│              │  diff = alt - ref │                                      │
│              └──────────────────┘                                       │
│                        │                                                │
│                        ▼                                                │
│              ┌──────────────────┐                                       │
│              │   Delta Head     │                                       │
│              │  (MLP + Output)  │                                       │
│              └──────────────────┘                                       │
│                        │                                                │
│                        ▼                                                │
│                   Δ [B, L, 2]                                           │
│               (Δ_donor, Δ_acceptor)                                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Characteristics

### Inputs
- **ref_seq**: Reference sequence [B, 4, L] (one-hot encoded)
- **alt_seq**: Alternate sequence [B, 4, L] (one-hot encoded)

### Target
- **base_delta**: `base_model(alt) - base_model(ref)`

### Output
- **Delta scores**: [B, L, 2] (per-position Δ_donor, Δ_acceptor)

---

## Results

### Training Variations

| Variation | Correlation | Notes |
|-----------|-------------|-------|
| V2 Original | r=-0.04 | No learning |
| V2 + 10x data | r=0.002 | Still no correlation |
| Gated CNN | r=0.36 | Architecture matters! |
| + Quantile loss (τ=0.9) | **r=0.38** | Best for paired |
| + Scaling | r=0.22 | Overfitting |
| + Temperature | r=-0.03 | No improvement |
| + Multi-task | r=-0.07 | Task interference |

### Best Configuration

```python
from agentic_spliceai.splice_engine.meta_layer.models import (
    SimpleCNNDeltaPredictor,
    create_calibrated_predictor
)

# Best architecture
base_model = SimpleCNNDeltaPredictor(
    hidden_dim=64,
    n_layers=6,
    kernel_size=15
)

# Best calibration: Quantile loss
model = create_calibrated_predictor(
    base_predictor=base_model,
    strategy='quantile',
    quantile=0.9  # Focus on large deltas
)
```

---

## Limitations

### 1. Target Quality Issue

The fundamental problem:

```
If variant is NOT splice-altering but base model predicts a delta:
  → We're training on WRONG targets
  → Model learns noise, not signal
```

### 2. Inference Overhead

Requires **two forward passes**:
1. Encode reference sequence
2. Encode alternate sequence
3. Compute difference

This is slower than single-pass approaches.

### 3. Correlation Ceiling

Even with best architecture (Gated CNN) and loss (Quantile), correlation is limited to r=0.38. This suggests the target (base model delta) is not reliable enough.

---

## When to Use

✅ **Use when**:
- You trust base model predictions
- Reference sequence is always available
- Inference speed is not critical

❌ **Don't use when**:
- Base model has known blind spots (most variants!)
- Single-pass efficiency is needed
- You have SpliceVarDB labels (use Validated Delta instead)

---

## Comparison to Validated Delta

| Aspect | Paired (this) | Validated |
|--------|---------------|-----------|
| Input | ref + alt | alt + var_info |
| Target | base_delta | SpliceVarDB-validated |
| Forward passes | 2 | 1 |
| Correlation | r=0.38 | **r=0.41** |
| Target quality | Variable | **Ground truth** |

---

## Model Files

| File | Description |
|------|-------------|
| `models/delta_predictor.py` | Original Siamese implementation |
| `models/delta_predictor_v2.py` | Per-position output version |
| `models/hyenadna_delta_predictor.py` | SimpleCNNDeltaPredictor |
| `models/delta_predictor_calibrated.py` | Calibration wrappers |

---

## Lessons Learned

1. **Architecture matters**: Gated CNN >> simple CNN
2. **Loss function matters**: Quantile loss >> MSE for sparse deltas
3. **Target quality is limiting factor**: Can't exceed base model accuracy
4. **Consider single-pass alternatives** when SpliceVarDB labels available

---

*For better results, consider using Validated Delta Prediction instead.*












