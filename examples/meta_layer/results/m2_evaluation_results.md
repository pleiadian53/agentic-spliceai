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

*Results pending — evaluation running on A40 GPU pod.
GENCODE gene cache build in progress (22,349 test genes).*

### What M2b will show

GENCODE v47 comprehensive annotation provides:
- **3.5M splice sites** (vs Ensembl's 10M — more curated, fewer pseudogenes)
- Tiered confidence: sites shared with Ensembl (high confidence) vs
  GENCODE-only (computational predictions)
- Better ground truth for assessing generalization to well-supported
  alternative isoforms

### Expected results

| Site category | Description | Expected base PR-AUC |
|--------------|-------------|---------------------|
| GENCODE ∩ Ensembl \ MANE | Well-supported alternatives | Higher (validated) |
| GENCODE-only \ MANE | Rare isoforms, predictions | Lower (noisier labels) |

*This section will be updated when M2b results are available.*

---

## Implications for M2c (Ensembl-Trained M1-S)

The v2 M2a results change the calculus for M2c training:

**v1 conclusion** (now revised): Base model > meta model on alt sites →
meta-layer hurts on unseen sites → M2c training on Ensembl labels needed.

**v2 conclusion**: Meta model > base model on alt sites → logit-space blend
preserves base signal and adds multimodal context → M2c training is **less
urgent** but may still help for the ~81K false negatives remaining.

The decision to proceed to M2c should depend on M2b results:
- If v2 performs well on GENCODE high-confidence alternatives → the model
  generalizes adequately from MANE → defer M2c
- If v2 still misses many GENCODE Tier 1 sites → the model needs exposure
  to alternative sites during training → proceed to M2c

M2c would retrain the meta-layer on Ensembl labels with confidence weighting.
This requires building an Ensembl train/val gene cache but **not retraining
the base model** (OpenSpliceAI remains frozen).

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
| v2 M2b results | *pending* |
| GENCODE ground truth | `data/gencode/GRCh38/splice_sites_enhanced.tsv` |

## Related

- [OOD generalization](../docs/ood_generalization.md) — why v1 failed OOD, why v2 fixes it, general principles
- [M1-S v2 results](m1s_v2_logit_blend_results.md) — logit blend canonical evaluation
- [M1-S ablation study](m1s_ablation_study.md) — modality contributions
- [M2 variant formulations](../../../docs/meta_layer/methods/05_m2_variant_formulations.md) — M2a-M2f design
- [Temperature scaling](../../../docs/ml_engineering/probability_calibration/temperature_scaling.md)
