# Alternative Splice Site Prediction: Methodology Roadmap

**Created**: December 2025  
**Status**: Active Development  
**Last Updated**: December 2025

---

## Overview

This document tracks the progressive development of methods for detecting and predicting **alternative splice sites** induced by genetic variants.

### Goal
Predict whether and how a genetic variant affects splicing patterns, going beyond what current base models (SpliceAI, OpenSpliceAI) can detect.

### Challenge
Base models are trained on canonical splice sites and fail to capture many variant-induced alternative splice sites documented in SpliceVarDB.

---

## Method Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALTERNATIVE SPLICE SITE PREDICTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ PAIRED DELTA PREDICTION (Siamese)                                      │  │
│  │                                                                        │  │
│  │ • Input: ref_seq + alt_seq (BOTH needed)                              │  │
│  │ • Target: base_model(alt) - base_model(ref)                           │  │
│  │ • Output: [L, 2] per-position deltas                                  │  │
│  │                                                                        │  │
│  │ Status: Tested, r=0.38 correlation (not sufficient)                   │  │
│  │ Limitation: Learning from potentially inaccurate base model deltas    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ VALIDATED DELTA PREDICTION (Single-Pass with Ground Truth)             │  │
│  │                                                                        │  │
│  │ • Input: alt_seq + variant_info (ref_base, alt_base)                  │  │
│  │ • Target: SpliceVarDB-validated delta (ground truth filtering)        │  │
│  │ • Output: Δ directly (single forward pass)                            │  │
│  │                                                                        │  │
│  │ Status: ✅ IMPLEMENTED & TESTED - r=0.41 (BEST SO FAR!)               │  │
│  │ Advantage: Uses ground truth labels, efficient inference              │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ MULTI-STEP FRAMEWORK (Decomposed Problem)                              │  │
│  │                                                                        │  │
│  │ Step 1: Binary Classification                                         │  │
│  │   • "Is this variant splice-altering?"                                │  │
│  │   • Status: Tested, AUC=0.61, F1=0.53 (needs >0.7)                   │  │
│  │                                                                        │  │
│  │ Step 2: Effect Type Classification                                    │  │
│  │   • "What type of effect?" (gain/loss, donor/acceptor)               │  │
│  │   • Status: NOT YET IMPLEMENTED                                       │  │
│  │                                                                        │  │
│  │ Step 3: Position Localization                                         │  │
│  │   • "Where in the window is the effect?"                              │  │
│  │   • Status: NOT YET IMPLEMENTED                                       │  │
│  │                                                                        │  │
│  │ Step 4: Delta Magnitude                                               │  │
│  │   • "How strong is the effect at that position?"                      │  │
│  │   • Status: NOT YET IMPLEMENTED                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Development Timeline

### Canonical Classification (Completed ❌)

**Goal**: Improve splice site classification using meta-learning on base model artifacts

**Approach**:
- Multimodal model: sequence (CNN) + tabular features (base model scores)
- Labels: GTF canonical splice sites (donor, acceptor, neither)
- Sample weights from SpliceVarDB

**Results**:
- Classification accuracy: 99%
- Variant detection: 17% (FAILED)

**Conclusion**: High accuracy on canonical sites does NOT transfer to variant detection.

**Documentation**: `docs/experiments/001_canonical_classification/`

---

### Paired Delta Prediction (Completed ⚠️)

**Goal**: Predict delta scores directly using Siamese architecture

**Approach**:
- Input: ref_seq + alt_seq (paired)
- Target: base_model(alt) - base_model(ref)
- Architecture: Gated CNN with dilated convolutions

**Variations Tested**:

| Variation | Correlation | Notes |
|-----------|-------------|-------|
| V2 Original | r=-0.04 | No learning |
| V2 + 10x data | r=0.002 | Still no correlation |
| Gated CNN | r=0.36 | Architecture matters |
| + Quantile loss | r=0.38 | Best for this approach |
| + Scaling | r=0.22 | Overfitting |
| + Temperature | r=-0.03 | No improvement |
| + Multi-task | r=-0.07 | Task interference |

**Conclusion**: Moderate correlation achieved (r=0.38) but:
1. Targets (base model deltas) may be inaccurate
2. Not sufficient for practical use

**Documentation**: `docs/experiments/002_delta_prediction/`

---

### Validated Delta Prediction (COMPLETED ✅)

**Status**: Implemented and tested  
**Result**: r=0.41 correlation (best so far!)  
**Goal**: Use SpliceVarDB classifications to derive ground-truth delta targets

**Key Difference from Paired**:
- Paired: Target = base_model(alt) - base_model(ref) (possibly inaccurate)
- Validated: Target = SpliceVarDB-validated delta (ground truth filtering)

**Approach**:
```
Input: alt_sequence + variant_info (ref_base, alt_base)
Target: Δ derived from SpliceVarDB classification
Output: Δ directly (single forward pass)

Final score = base_scores + Δ
```

**Documentation**: `docs/experiments/004_validated_delta/`

---

### Multi-Step Framework (IN PROGRESS)

**Status**: Step 1 tested (needs improvement)  
**Goal**: Decompose the problem into manageable sub-tasks

**Step 1: Binary Classification**
- Question: "Is this variant splice-altering?"
- Results: AUC=0.61, F1=0.53 (needs F1 > 0.7)
- Status: Needs improvement

**Step 2-4**: Not yet implemented

**Documentation**: `docs/MULTI_STEP_FRAMEWORK.md`

---

## Current Implementation Status

### Models Implemented

| Model | File | Purpose | Status |
|-------|------|---------|--------|
| `DeltaPredictorV2` | `delta_predictor_v2.py` | Paired prediction | Tested |
| `SimpleCNNDeltaPredictor` | `hyenadna_delta_predictor.py` | Gated CNN encoder | Tested |
| `ValidatedDeltaPredictor` | `validated_delta_predictor.py` | Single-pass validated | **BEST** |
| `SpliceInducingClassifier` | `splice_classifier.py` | Binary classification | Tested |
| `EffectTypeClassifier` | `splice_classifier.py` | Multi-class effects | Implemented |
| `UnifiedSpliceClassifier` | `splice_classifier.py` | Multi-task | Implemented |

---

## Next Steps (Prioritized)

### Immediate (M1 Mac)

1. **Scale Validated Delta** with more training data
2. **Improve Multi-Step Step 1** (Binary Classification)
   - Try larger context (1001nt vs 501nt)
   - Add position-aware features
   - Data augmentation (reverse complement)
   - Target: F1 > 0.7

3. **Document methodology choices**

### With GPU (RunPods)

1. **HyenaDNA encoder** for all approaches
2. **Full SpliceVarDB** (~50K variants)
3. **Cross-validation** on larger scale

---

## Compute Resources

### Available

| Environment | Specs | Suitable For |
|-------------|-------|--------------|
| MacBook M1 | 16GB RAM, MPS | Quick iterations, small models |

### Needed (RunPods)

| Environment | Specs | Suitable For |
|-------------|-------|--------------|
| RTX 4090 | 24GB VRAM | HyenaDNA-small, larger batches |
| A40 | 48GB VRAM | HyenaDNA-medium |
| A100 | 80GB VRAM | HyenaDNA-large, fine-tuning |

---

## References

- **SpliceVarDB**: Source of ground-truth variant effect labels
- **LABELING_STRATEGY.md**: Detailed approach descriptions
- **ARCHITECTURE.md**: Model architectures

---

*This roadmap will be updated as methodology development progresses.*












