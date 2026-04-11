# Alternative Splice Site Evaluation Results

**Date**: 2026-04-10 (updated)
**Models evaluated**: M1-S (MANE-trained), M2-S (Ensembl-trained)
**Evaluation protocols**: Eval-Ensembl-Alt, Eval-GENCODE-Alt
**Question**: How well do meta-layer models detect alternative splice sites?

> See [naming_convention.md](../../../docs/meta_layer/methods/naming_convention.md)
> for model vs evaluation protocol definitions.

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

| Source | Genes | Unique sites | Rows (per-transcript) | Description |
|--------|-------|-------------|----------------------|-------------|
| MANE | 18,200 | 367K | 370K | One canonical transcript per gene |
| Ensembl 112 | 39,291 | 738K | 2.8M | Comprehensive, includes pseudogenes |
| GENCODE v47 | 54,117 | 928K | 3.5M | GENCODE comprehensive annotation |

The evaluation protocols use the set difference as the test set:
- **Eval-Ensembl-Alt**: Ensembl \ MANE (~371K unique sites)
- **Eval-GENCODE-Alt**: GENCODE \ MANE (~561K unique sites)

For M1-S, these alternative sites are **out-of-distribution (OOD)** — they
were never positive labels during M1-S training.  For M2-S (trained on
Ensembl), they are in-distribution.  Comparing M1-S vs M2-S on the same
protocol reveals the value of training on broader labels.
See [OOD Generalization](../docs/ood_generalization.md) for the full analysis.

> **Note on site counts**: Annotation files contain per-transcript rows
> (2.8M Ensembl, 3.5M GENCODE).  At the unique (chrom, position) level,
> Ensembl has ~738K and GENCODE has ~928K unique splice sites.

---

## Eval-Ensembl-Alt: M1-S on Ensembl Alternative Sites

### M1-S v1 (probability-space blend) vs v2 (logit-space blend)

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

## Eval-GENCODE-Alt: M1-S on GENCODE v47 Alternative Sites

**Annotation**: GENCODE v47 comprehensive (3.5M splice sites, 62K genes)
**Test set**: SpliceAI test split (chr1, 3, 5, 7, 9) — 22,349 genes, 11,046 evaluated

### Alternative sites only (GENCODE \ MANE)

| Metric | Base model | **v2 Meta** | Change |
|--------|-----------|-------------|--------|
| **PR-AUC** | 0.637 | **0.728** | **+0.091** |
| True Positives | 22,678 | **27,890** | **+5,212 recovered** |
| False Negatives | 125,754 | 120,542 | -5,212 (-4.1%) |
| **False Positives** | **10,178** | **1,441** | **-8,737 (-85.8%)** |

### Eval-Ensembl-Alt vs Eval-GENCODE-Alt comparison

| Metric | Eval-Ensembl-Alt | Eval-GENCODE-Alt |
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
the Eval-Ensembl-Alt improvement (+0.026).

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

## M2-S Results (Ensembl-Trained Model)

M2-S was trained on Ensembl labels (~2.8M sites, 63K genes) using the same
architecture as M1-S. Full training details in
[m2s_ensembl_trained_results.md](m2s_ensembl_trained_results.md).

### M2-S on Eval-Ensembl-Alt (all genes, no min_length skip)

| Metric | Base model | M1-S | **M2-S** |
|--------|-----------|------|---------|
| **PR-AUC** | 0.749 | 0.775 | **0.965** |
| **Recall** | 12.1% | 14.9% | **59.2%** |
| TPs | 12,474 | 14,168 | **61,181** |
| FNs | 90,838 | 81,231 | **42,131** |
| FPs | 11 | 10 | 450 |

### M2-S on Eval-GENCODE-Alt (all genes, no min_length skip)

| Metric | Base model | M1-S | **M2-S** |
|--------|-----------|------|---------|
| **PR-AUC** | 0.631 | 0.728 | **0.907** |
| **Recall** | 15.9% | — | **46.9%** |
| TPs | 25,796 | — | **76,040** |
| FNs | 136,360 | — | **86,116** |
| FPs | 11,737 | — | 1,962 |

### M2-S on Eval-MANE (canonical site tradeoff)

| Metric | Base model | M1-S | **M2-S** |
|--------|-----------|------|---------|
| PR-AUC | 0.983 | **0.9996** | 0.9990 |
| FNs | 8,434 | 643 | **144** |
| FPs | 8,617 | 13,427 | **1,910,382** |
| FN reduction | — | +91.8% | **+98.3%** |

**Tradeoff**: M2-S achieves near-perfect recall on canonical sites (only
144 FNs) but at the cost of massive FP overcalling (1.9M). The model
trained on 2.8M Ensembl sites calls splice sites much more aggressively.

**Use-case guidance**:
- **M1-S**: High precision for clinical variant interpretation (few FPs)
- **M2-S**: High recall for discovery workflows (few FNs, tolerant of FPs)

---

## Reproduction

```bash
# Eval-Ensembl-Alt: Ensembl alternative sites
python -u examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --annotation-source ensembl \
    --build-cache \
    --base-scores-dir /runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --device cuda

# Eval-GENCODE-Alt: GENCODE alternative sites
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
| M1-S v1 Eval-Ensembl-Alt | `output/meta_layer/m2a_eval/m2a_eval_results.json` |
| M1-S v2 Eval-Ensembl-Alt | `output/meta_layer/m2a_v2_eval_results.json` |
| M1-S v2 Eval-GENCODE-Alt | `output/meta_layer/m2b_v2_eval_results.json` |
| GENCODE ground truth | `data/gencode/GRCh38/splice_sites_enhanced.tsv` |

## Related

- [Naming convention](../../../docs/meta_layer/methods/naming_convention.md) — model vs evaluation protocol definitions
- [M2-S results](m2s_ensembl_trained_results.md) — Ensembl-trained model details
- [OOD generalization](../docs/ood_generalization.md) — why M1-S failed OOD, why logit blend fixes it
- [M1-S v2 results](m1s_v2_logit_blend_results.md) — logit blend canonical evaluation
- [M1-S ablation study](m1s_ablation_study.md) — modality contributions
- [Temperature scaling](../../../docs/ml_engineering/probability_calibration/temperature_scaling.md)
