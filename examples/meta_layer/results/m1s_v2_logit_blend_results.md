# M1-S v2: Logit-Space Residual Blend Results

**Date**: 2026-04-09
**Model**: M1-S v2 (367K params, logit-space blend + learned per-class temperature)
**Training**: 50 epochs on A40 GPU pod, best at epoch 42
**Test set**: SpliceAI test split (chr1, 3, 5, 7, 9) — 5,522 MANE genes

---

## Motivation

The v1 M1-S model used a **probability-space residual blend** at inference:

```
output = alpha * softmax(meta_logits) + (1 - alpha) * softmax(base_scores)
```

Two bugs were discovered:

1. **Double-softmax**: `base_scores` are already probabilities, but `softmax()`
   was applied again, flattening a 99.8% prediction to 57.5%.
2. **Dead blend_alpha**: The blend was inference-only; `blend_alpha` never
   received gradients during training and stayed at its init value of 0.5.

These caused variant delta scores to be dampened 1.5-5x vs the base model.

## Architecture Change

v2 replaces the blend with a **logit-space product-of-experts** formulation,
active during both training and inference:

```
output = softmax((alpha * meta_logits + (1 - alpha) * log(base_probs)) / T)
```

- `alpha` is learned end-to-end (receives gradients during training)
- `T = [T_donor, T_acceptor, T_neither]` is a per-class learned temperature
  vector, subsuming post-hoc calibration

## Learned Parameters (epoch 42)

```
alpha = 0.535   (slightly meta-favoring; v1 was stuck at 0.500)
T     = [1.18, 0.94, 1.14]   (donor/neither softened, acceptor sharpened)
```

---

## M1 Results: v1 vs v2

### Canonical Splice Site Classification (MANE test set)

| Metric | v1 (prob blend) | **v2 (logit blend)** | Change |
|--------|----------------|---------------------|--------|
| **Val PR-AUC (best)** | 0.9899 (epoch 46) | **0.9954** (epoch 42) | **+0.0055** |
| **Test PR-AUC** | 0.9994 | **0.9996** | +0.0002 |
| Donor PR-AUC | 0.9899 | **0.9951** | +0.0052 |
| Acceptor PR-AUC | 0.9852 | **0.9911** | +0.0059 |
| **False Negatives** | 710 | **643** | **-9.4%** (67 more TPs) |
| **False Positives** | 15,883 | **13,427** | **-15.5%** (2,456 fewer) |
| True Positives | 101,500 | 101,567 | +67 |
| FN reduction vs base | 91.0% | 91.8% | +0.8pp |
| Accuracy | 99.99% | 99.99% | same |

**Key improvement**: v2 reduces FPs by 15.5% while *also* recovering more TPs.
The v1 model needed post-hoc temperature scaling (T=[0.81, 0.81, 1.17]) to
get FPs down to 14,195. The v2 model achieves **13,427 FPs without any post-hoc
calibration** — better than v1's calibrated result.

### Convergence Speed

v2 converged faster: PR-AUC 0.99+ by epoch 2 (v1 needed ~15 epochs).
Best metric at epoch 42 vs v1's epoch 46 — similar, but v2's per-epoch
wall time was ~50% less (270s vs ~540s) due to fixing the unnecessary
gradient accumulation on GPU (batch 16 x accum 1 vs batch 8 x accum 2).

### Training Dynamics

The learned parameters evolved during training:

| Epoch | alpha | T_donor | T_acceptor | T_neither | Val PR-AUC |
|-------|-------|---------|------------|-----------|------------|
| 1 | 0.490 | 1.14 | 0.98 | 1.03 | 0.9884 |
| 10 | 0.447 | 1.47 | 1.29 | 1.41 | 0.9900 |
| 25 | 0.472 | 1.32 | 1.05 | 1.29 | 0.9933 |
| 33 | 0.511 | 1.24 | 1.04 | 1.20 | 0.9945 |
| 42 | 0.528 | 1.20 | 0.97 | 1.16 | 0.9954 |
| 50 | 0.535 | 1.18 | 0.94 | 1.14 | 0.9937 |

- **alpha** drifted from 0.5 to 0.535 — the model learned to weight the
  meta-CNN slightly more than the base model
- **T_acceptor** converged below 1.0 (sharpening) — consistent with the
  largest per-class improvement being in acceptor PR-AUC (+0.006)
- **T_donor, T_neither** settled at ~1.2 (softening)

---

## Variant Delta Scores

The primary motivation for the logit-space blend was to fix dampened variant
deltas.  Results on validated test variants:

| Variant | Gene | Base DS_DL | v1 DS_DL | **v2 DS_DL** | Dampening |
|---------|------|-----------|---------|-------------|-----------|
| chr3:184300445 G>A | PSMD2 (+) | -0.96 | -0.19 (80% lost) | **-0.43** | 55% lost |
| chr11:47332565 C>A | MYBPC3 (-) | -1.00 | -0.68 (32% lost) | **-0.95** | **5% lost** |
| chr11:47333193 C>A | MYBPC3 (-) | -0.95 | -0.67 (29% lost) | **-0.89** | **7% lost** |

| Variant | Gene | Base DS_DG | v1 DS_DG | **v2 DS_DG** | |
|---------|------|-----------|---------|-------------|---|
| chr3:184300445 G>A | PSMD2 (+) | +0.06 | +0.46 | **+0.69** | meta amplifies |
| chr11:47332565 C>A | MYBPC3 (-) | +0.07 | +0.30 | **+0.38** | meta amplifies |
| chr11:47333193 C>A | MYBPC3 (-) | +0.27 | +0.48 | **+0.93** | meta amplifies |

**Key findings**:
- **Donor loss dampening largely eliminated**: MYBPC3 deltas recover 93-95% of
  base model signal (v1 recovered only 68-71%)
- **Cryptic donor gains amplified**: v2 meta model detects cryptic donors *more
  confidently* than the base model alone (e.g., +0.93 vs base +0.27 for
  MYBPC3 47333193). The multimodal context adds signal rather than suppressing it.
- **PSMD2 still partially dampened** (55% signal loss): this plus-strand variant
  at a weaker donor site (base=0.96) shows more dampening than the strong
  MYBPC3 donors (base~1.0). Suggests the blend weighting could be further
  optimized for variant analysis.

---

## Reproduction

```bash
# Training (A40 pod)
python -u examples/meta_layer/07_train_sequence_model.py \
    --mode m1 --device cuda --epochs 50 \
    --cache-dir /runpod-volume/output/meta_layer/gene_cache_mane \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --use-shards \
    --output-dir /runpod-volume/output/meta_layer/m1s_v2_logit_blend

# Evaluation (A40 pod)
python -u examples/meta_layer/08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --build-cache --device cuda

# Variant delta test (local CPU)
python examples/variant_analysis/01_single_variant_delta.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --config examples/variant_analysis/test_variants.yaml
```

---

## Files

| Artifact | Location |
|----------|----------|
| Best checkpoint | `output/meta_layer/m1s_v2_logit_blend/best.pt` |
| Config | `output/meta_layer/m1s_v2_logit_blend/config.pt` |
| Eval results | `output/meta_layer/m1s_v2_logit_blend/eval_results.json` |
| Training log | `output/meta_layer/m1s_v2_train.log` |
| v1 checkpoint (preserved) | `output/meta_layer/m1s_v1_prob_blend/` |

## Implications for OOD Generalization

The logit-space blend has a direct impact on **out-of-distribution (OOD)
generalization** — the model's ability to predict splice sites it was never
trained on.  The v1 probability-space blend actively suppressed the base
model's signal at alternative splice sites, causing the meta-layer to *hurt*
OOD performance (PR-AUC dropped from 0.749 to 0.704).  The v2 logit-space
blend reverses this: at sites where the meta-CNN is uncertain, the base
model's signal dominates the blend, enabling graceful degradation rather than
confident misclassification.

See [OOD Generalization in Splice Site Prediction](../docs/ood_generalization.md)
for a detailed analysis of this failure mode and the architectural principles
behind the fix.

## Related

- [M1-S v1 ablation study](m1s_ablation_study.md) — modality contributions (v1)
- [M1-P full-genome results](m1p_fullgenome_results.md) — position-level XGBoost baseline
- [M2 evaluation results](m2_evaluation_results.md) — alternative splice site evaluation
- [OOD generalization](../docs/ood_generalization.md) — why v1 hurt on unseen sites, why v2 fixes it
- [Variant analysis tutorial](../../../docs/variant_analysis/negative_strand_and_variant_effects.md)
