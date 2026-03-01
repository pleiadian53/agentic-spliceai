# Experiment 003: Binary Classification (Multi-Step Step 1)

**Date**: December 2025  
**Status**: Completed  
**Outcome**: AUC=0.61, F1=0.53 (needs improvement)

---

## Hypothesis

A binary classifier can learn to distinguish splice-altering from normal variants as the first step of a multi-step framework.

---

## Setup

### Data
- **Input**: alt_sequence + variant_info (ref_base, alt_base)
- **Labels**: SpliceVarDB classification (splice-altering vs normal)
- **Balance**: 50% splice-altering, 50% normal

### Model
- **Architecture**: `SpliceInducingClassifier`
- **Encoder**: Gated CNN (6 layers)
- **Variant Embedding**: MLP (ref_base, alt_base)
- **Output**: P(splice-altering)

---

## Results

### Binary Classification Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.61 |
| PR-AUC | 0.58 |
| F1 Score | 0.53 |
| Accuracy | 55% |

### Confusion Matrix

|  | Predicted Normal | Predicted SA |
|--|------------------|--------------|
| **Actual Normal** | 52% | 48% |
| **Actual SA** | 43% | 57% |

---

## Analysis

### What This Shows

1. **Better than random**: AUC=0.61 > 0.5
2. **Learnable signal exists**: Model finds some pattern
3. **Not yet useful**: F1=0.53 too low for practical use (need >0.7)

### Why Performance is Limited

1. **Context may be insufficient**: 501nt might not capture all relevant information
2. **Variant types vary**: Different mechanisms for donor/acceptor gain/loss
3. **Data quality**: Some SpliceVarDB labels may be noisy

---

## Key Insight

> **Binary classification is learnable, but current F1=0.53 is insufficient.**
>
> The signal exists (AUC > 0.5), but model needs improvement to be practically useful.

---

## Improvement Ideas

1. **Larger context**: 501nt → 1001nt or longer
2. **Position-aware features**: Add relative position to splice sites
3. **Data augmentation**: Reverse complement, small mutations
4. **Effect-type conditioning**: Different heads for different effect types
5. **HyenaDNA encoder**: Better sequence representation

---

## Relation to Multi-Step Framework

This experiment tests **Step 1** of the multi-step framework:

```
Step 1: Is splice-altering? → Binary [THIS EXPERIMENT]
Step 2: What type?          → Multi-class (TODO)
Step 3: Where?              → Localization (TODO)
Step 4: How strong?         → Regression (TODO)
```

Before proceeding to Steps 2-4, Step 1 should achieve F1 > 0.7.

---

## Files

| File | Description |
|------|-------------|
| `models/splice_classifier.py` | SpliceInducingClassifier |
| `models/splice_classifier.py` | EffectTypeClassifier (Step 2) |
| `models/splice_classifier.py` | UnifiedSpliceClassifier (Multi-task) |












