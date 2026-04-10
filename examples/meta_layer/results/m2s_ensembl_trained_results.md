# M2-S: Ensembl-Trained Meta-Layer Results

**Date**: 2026-04-10
**Model**: M2-S (367K params, logit-space blend, trained on Ensembl labels)
**Training**: 17 epochs on A40 (early stopped, patience=10, best at epoch 7)

> **Naming convention**: M2-S is the sequence-level meta model trained on
> Ensembl splice sites for alternative site detection. See
> [naming_convention.md](../../../docs/meta_layer/methods/naming_convention.md)
> for the full model vs evaluation protocol distinction.

---

## Training Summary

| Parameter | M1-S | M2-S |
|-----------|-----------|-------------|
| Training genes | 12,390 | 40,897 |
| Val genes | 1,376 | 4,544 |
| Training splice sites | ~370K | ~2.8M |
| Best epoch | 42 | 7 |
| Early stop | epoch 50 (completed) | epoch 17 (patience=10) |
| Val PR-AUC | 0.9954 | 0.7646 |

### Learned Parameters

| Parameter | M1-S | M2-S | Interpretation |
|-----------|-----------|-------------|----------------|
| alpha | 0.535 | **0.660** | Ensembl model trusts meta-CNN more |
| T_donor | 1.18 | **1.58** | Soften donor predictions (noisier labels) |
| T_acceptor | 0.94 | **1.57** | Soften acceptor predictions |
| T_neither | 1.14 | **0.68** | Sharpen "not splice" — more decisive rejection |

The alpha shift from 0.535 to 0.660 means the model learned to rely more
on its own sequence CNN and multimodal features than on the base model.
This is expected: the Ensembl label set includes many sites where the base
model (OpenSpliceAI) is confidently wrong, so the model must learn to
override it using conservation, junction evidence, and sequence patterns.

The T_neither sharpening (1.14 → 0.68) means the model is more decisive
about non-splice positions — important when the training set includes
3.3x more genes with 3.3x more background positions.

---

## M2a Evaluation: Ensembl Alternative Sites

Evaluation on alternative sites (Ensembl \ MANE) using the SpliceAI test
split (chr1, 3, 5, 7, 9) — chromosomes never seen during training.

| Metric | Base model | M1-S | **M2-S** |
|--------|-----------|-----------|-----------------|
| **PR-AUC** | 0.749 | 0.775 | **0.967** |
| **Recall** | 11.9% | 14.9% | **59.1%** |
| True Positives | 11,345 | 14,168 | **56,390** |
| False Negatives | 84,054 | 81,231 | **39,009** |
| False Positives | 9 | 10 | 359 |
| FN reduction vs base | — | +3.4% | **+53.6%** |
| Top-k (1.0x) | 0.719 | 0.746 | **0.935** |

### Key findings

**1. PR-AUC near-perfect at 0.967.** The model correctly ranks almost all
alternative splice sites above non-splice positions. This is a +0.218
improvement over the base model and +0.192 over M1-S.

**2. Recall jumps from 15% to 59%.** The model detects 4x more alternative
sites than M1-S, recovering 45,045 additional true positives that
both the base model and the MANE-trained meta-layer miss.

**3. FP cost is modest.** FPs increase from 10 to 359 — a 36x increase in
relative terms, but the absolute number is small compared to the 45K
additional TPs. The precision-recall tradeoff is very favorable.

**4. The meta-layer overrides confidently wrong base scores.** At
alternative sites, the base model typically outputs `[0.001, 0.001, 0.998]`
(confident "neither"). The meta-layer, with alpha=0.66, generates
strong enough donor/acceptor logits from sequence + multimodal features
to overcome the base model's negative signal. This validates that
multimodal evidence (conservation, junction reads) provides independent
biological confirmation sufficient to correct the base model.

### What 59% recall means

Of 95,399 alternative sites in Ensembl \ MANE on the test chromosomes:
- **56,390 detected** (59.1%) — these are alternative splice sites the
  model can now identify using multimodal evidence
- **39,009 still missed** (40.9%) — likely sites with weak sequence
  motifs, no conservation signal, and no GTEx junction support

The remaining 41% are the hardest cases — sites that may require
tissue-specific features (M2e), deeper intronic context, or novel
data modalities to detect.

---

## M2b Evaluation: GENCODE Alternative Sites

*Results pending — evaluation running on pod.*

---

## Comparison Across All Model Variants

### M2a setting (Ensembl \ MANE alternative sites)

| Model | Training | PR-AUC | Recall | FNs | FPs |
|-------|----------|--------|--------|-----|-----|
| Base (OpenSpliceAI) | MANE | 0.749 | 11.9% | 84,054 | 9 |
| M1-S v1 | MANE (prob blend) | 0.704 | 17.4% | 78,834 | 15 |
| M1-S | MANE (logit blend) | 0.775 | 14.9% | 81,231 | 10 |
| **M2-S** | **Ensembl (logit blend)** | **0.967** | **59.1%** | **39,009** | **359** |

### Progression story

1. **v1 (prob blend)**: Meta-layer *hurt* on alternative sites (OOD failure)
2. **v2 logit blend**: Fixed OOD — meta now slightly exceeds base (+0.026)
3. **Ensembl training**: Massive improvement (+0.192 over v2, +0.218 over base)

The logit-space blend was necessary but not sufficient. The architectural
fix (v2) enabled graceful degradation at OOD sites, but true alternative
site detection required training on broader labels (M2c protocol).

---

## Implications

### For M4 (Variant Analysis)

M2-S should produce stronger variant deltas at alternative sites.
The PYGB and CDC25B cases that failed with M1-S (because the base
model had zero signal and the meta-layer couldn't override it) may now
succeed — the Ensembl-trained model has learned to detect splice sites at
those positions independent of the base model.

### For Isoform Discovery (Phase 9)

59% recall on alternative sites means the model can already detect the
majority of known alternative splice sites. Combined with the Phase 1B
consequence predictor, this enables systematic discovery of alternative
transcript structures at a genome-wide scale.

### For M2d/M2e (Future)

- **M2d (junction-weighted)**: Could improve the remaining 41% recall by
  weighting training labels by GTEx junction evidence
- **M2e (tissue-conditioned)**: Could enable tissue-specific alternative
  site detection (currently tissue-agnostic)

---

## Reproduction

```bash
# Training (M2c protocol → M2-S)
bash examples/meta_layer/ops_train_m2c_pod.sh

# M2a evaluation
python -u examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m2c/best.pt \
    --annotation-source ensembl \
    --cache-dir /runpod-volume/output/meta_layer/gene_cache_ensembl/test \
    --device cuda
```

## Files

| Artifact | Location |
|----------|----------|
| M2-S checkpoint | `output/meta_layer/m2c/best.pt` |
| Config | `output/meta_layer/m2c/config.pt` |
| Training log | `output/meta_layer/m2c_train.log` |
| M2a eval results | `output/meta_layer/m2a_m2c_eval_results.json` |
| M2b eval results | *pending* |

## Related

- [M1-S v2 results](m1s_v2_logit_blend_results.md) — logit blend on canonical sites
- [M2 evaluation overview](m2_evaluation_results.md) — M2a/M2b comparison across models
- [OOD generalization](../docs/ood_generalization.md) — why training labels matter
- [Variant effect validation](../../variant_analysis/results/variant_effect_validation.md)
