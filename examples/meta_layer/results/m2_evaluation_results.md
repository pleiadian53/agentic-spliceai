# M2 Evaluation: Alternative Splice Site Generalization

**Date**: 2026-04-09 (updated)
**Model**: M1-S trained on MANE canonical sites only
**Question**: Does the meta-layer generalize to splice sites it was never trained on?

---

## Background

The M1-S model is trained exclusively on **MANE** splice sites (~370K sites,
~19K protein-coding genes).  MANE represents one canonical transcript per gene
— the most well-supported, clinically relevant isoform.

But human genes produce multiple transcript isoforms via alternative splicing.
Sites present in richer annotations (Ensembl, GENCODE) but absent from MANE
represent **alternative splice sites** — the delta set.

M2 evaluation tests whether M1-S, despite never seeing these sites as positive
labels during training, can still predict them correctly.

### Annotation hierarchy

| Source | Genes | Splice sites | Description |
|--------|-------|-------------|-------------|
| MANE | ~19K | ~370K | One canonical transcript per gene |
| Ensembl 112 | ~57K | ~10M | Comprehensive, includes pseudogenes |
| GENCODE v47 | ~62K | ~3.5M | GENCODE comprehensive annotation |

The alternative site sets:
- **M2a**: Ensembl \ MANE (~9.6M sites) — broad, includes many pseudogene sites
- **M2b**: GENCODE \ MANE (~3.1M sites) — curated, tiered confidence

These alternative sites are **out-of-distribution (OOD)** for the meta-layer
— they were never positive labels during training.  M2 evaluation directly
measures OOD generalization: does the meta-layer help or hurt on splice sites
it has never seen?  See [OOD Generalization](../docs/ood_generalization.md) for
the full analysis.

---

## M2a: Ensembl Alternative Sites

### v1 (probability-space blend) vs v2 (logit-space blend)

#### Alternative sites only (Ensembl \ MANE)

| Metric | Base model | v1 Meta | **v2 Meta** |
|--------|-----------|---------|-------------|
| **PR-AUC** | **0.749** | 0.704 | **0.775** |
| True Positives | 11,345 | 16,565 | 14,168 |
| False Negatives | 84,054 | 78,834 | 81,231 |
| False Positives | 9 | 15 | 10 |
| FN reduction | — | +6.2% | +3.4% |

#### Overall (all Ensembl sites on test chromosomes)

| Metric | Base model | v1 Meta | **v2 Meta** |
|--------|-----------|---------|-------------|
| **PR-AUC** | **0.827** | 0.779 | **0.812** |
| FN reduction | — | +8.9% | +3.4% |
| FP reduction | — | -50.4% | -11.1% |

### Analysis

**v1 finding — OOD failure (now superseded)**: The base model outperformed
the v1 meta model on alternative sites (PR-AUC 0.749 vs 0.704).  This is a
classic OOD failure: the meta-layer learned that "real" splice sites have
strong base scores, high conservation, and junction support.  Alternative
sites with moderate signals were actively suppressed.  The double-softmax bug
amplified this by flattening the base model's informative moderate-confidence
scores.  See [OOD Generalization](../docs/ood_generalization.md) for a
detailed analysis of this failure mode.

**v2 finding — OOD fixed**: The logit-space blend **reverses this**.  v2 meta
PR-AUC (0.775) now **exceeds** the base model (0.749) on alternative sites.
The product-of-experts formulation enables **graceful degradation**: when the
meta-CNN is uncertain about an OOD site, its logits are near zero and the base
model's signal dominates.  The meta-layer no longer overrides the base model's
informative predictions at sites outside its training distribution.

**TP count paradox**: v1 had more TPs (16,565) than v2 (14,168) despite
lower PR-AUC.  This is because v1 was more aggressive (lower threshold at
the argmax decision boundary due to double-softmax), calling more sites as
positive — but with worse precision.  v2 is more conservative and more
accurate.

### Top-K accuracy (v2)

| k | Base | v2 Meta | Delta |
|---|------|---------|-------|
| 0.5x | 0.510 | 0.520 | +0.010 |
| 1.0x | 0.719 | **0.746** | **+0.027** |
| 2.0x | 0.968 | **0.977** | +0.009 |
| 4.0x | 0.999 | 1.000 | +0.001 |

The meta model improves ranking at all k values, with the largest gain at
k=1.0x — the operating point where you predict exactly as many sites as
exist in the ground truth.

---

## M2b: GENCODE v47 Alternative Sites

**Annotation**: GENCODE v47 comprehensive (3.5M splice sites, 62K genes)
**Test set**: SpliceAI test split (chr1, 3, 5, 7, 9) — 22,349 genes, 11,046 evaluated

### Alternative sites only (GENCODE \ MANE)

| Metric | Base model | **v2 Meta** | Change |
|--------|-----------|-------------|--------|
| **PR-AUC** | 0.637 | **0.728** | **+0.091** |
| True Positives | 22,678 | **27,890** | **+5,212 recovered** |
| False Negatives | 125,754 | 120,542 | -5,212 (-4.1%) |
| **False Positives** | **10,178** | **1,441** | **-8,737 (-85.8%)** |

### M2a vs M2b comparison

| Metric | M2a (Ensembl) | M2b (GENCODE) |
|--------|--------------|---------------|
| N alternative sites | 95,399 | 148,432 |
| Base PR-AUC | 0.749 | 0.637 |
| Meta PR-AUC | 0.775 | 0.728 |
| **Meta - Base** | +0.026 | **+0.091** |
| TPs recovered | +2,823 | **+5,212** |
| **FP reduction** | +1 (negligible) | **-8,737 (-85.8%)** |
| FN reduction | -3.4% | -4.1% |

### Analysis

**The meta-layer's advantage is largest on the harder task.** GENCODE
alternative sites have lower base PR-AUC (0.637 vs 0.749) because they
include more rare isoforms and computationally predicted splice sites.
The meta-layer provides +0.091 PR-AUC improvement — 3.5x larger than
the M2a improvement (+0.026).

**FP elimination is the headline result.** The base model produces 10,178
false positives on GENCODE alternative sites. The meta-layer eliminates
85.8% of them (down to 1,441), while simultaneously recovering 5,212
additional true positives. The multimodal features (conservation, junction
evidence, chromatin) provide strong negative evidence at non-splice
positions, enabling the model to confidently reject false calls.

**GENCODE is a harder but more informative evaluation.** With 148K
alternative sites (55% more than Ensembl's 95K), GENCODE provides a
broader test of generalization. The meta-layer's consistent improvement
across both evaluation sets confirms the logit-space blend enables
genuine OOD generalization, not overfitting to one annotation's biases.

### Top-K accuracy (v2)

| k | Base | v2 Meta | Delta |
|---|------|---------|-------|
| 0.5x | 0.448 | 0.463 | +0.015 |
| 1.0x | 0.654 | **0.681** | **+0.027** |
| 2.0x | 0.957 | **0.973** | +0.016 |
| 4.0x | 0.999 | 1.000 | +0.001 |

---

## Implications for M2c (Ensembl-Trained M1-S)

### Updated assessment (post M2a + M2b)

The v2 meta-layer consistently improves over the base model on alternative
sites across both annotation sets — a clear win for the logit-space blend
and multimodal features.  However, significant room for improvement remains:

| Setting | Meta FNs | Total alt sites | Recall |
|---------|----------|----------------|--------|
| M2a (Ensembl) | 81,231 | 95,399 | 14.8% |
| M2b (GENCODE) | 120,542 | 148,432 | 18.8% |

The model recovers only 15-19% of alternative sites — the vast majority
are still missed. M2c training on Ensembl labels would expose the model to
these sites as positive examples during training, potentially pushing recall
much higher.

**The M2b FP result strongly motivates M2c**: the meta-layer already
eliminates 85.8% of base model FPs on GENCODE sites without any training
on those sites. If the model can maintain this FP suppression while also
learning to detect more alternative sites through broader training labels,
the precision-recall tradeoff would be very favorable.

M2c would retrain the meta-layer on Ensembl labels with confidence weighting.
This requires building an Ensembl train/val gene cache but **not retraining
the base model** (OpenSpliceAI remains frozen). Ops script is ready:
`examples/meta_layer/ops_train_m2c_pod.sh`.

---

## Reproduction

```bash
# M2a: Ensembl alternative sites
python -u examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --annotation-source ensembl \
    --build-cache \
    --base-scores-dir /runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --device cuda

# M2b: GENCODE alternative sites
python -u examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --annotation-source gencode \
    --gtf data/gencode/GRCh38/gencode.v47.annotation.gtf \
    --build-cache \
    --base-scores-dir /runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --device cuda
```

### Persistent gene caches (on RunPod volume)

```
/runpod-volume/output/meta_layer/
  gene_cache_mane/       58 GB  (train + val + test)
  gene_cache_ensembl/    9.2 GB (test only, reusable across models)
  gene_cache_gencode/    TBD    (test only, building)
```

---

## Files

| Artifact | Location |
|----------|----------|
| v1 M2a results | `output/meta_layer/m2a_eval/m2a_eval_results.json` |
| v2 M2a results | `output/meta_layer/m2a_v2_eval_results.json` |
| v2 M2b results | `output/meta_layer/m2b_v2_eval_results.json` |
| GENCODE ground truth | `data/gencode/GRCh38/splice_sites_enhanced.tsv` |

## Related

- [OOD generalization](../docs/ood_generalization.md) — why v1 failed OOD, why v2 fixes it, general principles
- [M1-S v2 results](m1s_v2_logit_blend_results.md) — logit blend canonical evaluation
- [M1-S ablation study](m1s_ablation_study.md) — modality contributions
- [M2 variant formulations](../../../docs/meta_layer/methods/05_m2_variant_formulations.md) — M2a-M2f design
- [Temperature scaling](../../../docs/ml_engineering/probability_calibration/temperature_scaling.md)
