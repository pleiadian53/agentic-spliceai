# M2 — Alternative Splice Sites

**Task.** Recognize the **alternative** splice sites that the curated MANE annotation leaves out —
defined as the set difference **Ensembl ∖ MANE**: competing donors/acceptors, minor isoforms, and
tissue-specific sites. This is the regime where multimodal refinement produces its largest, most
clear-cut win.

**Bottom line:** on alternative sites the base model is nearly blind — it recovers only ~17% of them.
The promoted **M2-S** recovers **~90%** at near-equal precision, lifting alternative-site PR-AUC from
**0.911 to 0.990** and cutting false negatives by **88%**.

---

## Why M2 exists: the OOD problem

M1-S is trained on MANE, so alternative (non-MANE) sites are **out-of-distribution** for it. Evaluated
there, the first M1-S blend actually scored *below the base model* — Ensembl-alt PR-AUC **0.704** vs
base **0.749** — a genuine OOD regression. The v2 logit-blend fixed the regression (**0.775**), but
0.775 is still far from usable: a MANE-only model, however well blended, cannot recognize sites its
training annotation never contained.

That motivated training a model directly on the broader Ensembl annotation — **M2-S**. The full
narrative is in
[`ood_generalization.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/ood_generalization.md).

---

## Current results (v4)

Model `m2s_v4_cleanannot` — same architecture as M1-S, trained on Ensembl labels
([Workflow Stage 5](../../workflows/meta_layer/05_training_m2s.md)) and evaluated with the dedicated
alternative-site protocol ([Stage 6](../../workflows/meta_layer/06_evaluation.md#m2-s-alternative-sites-the-metric-that-matters-for-m2)).

**Alternative sites** (Ensembl ∖ MANE; 70,923 alt sites / 14,724 genes, paralog-cleaned):

| | Base (OpenSpliceAI) | M2-S (v4) |
|-|---------------------|-----------|
| Alt-site macro PR-AUC | 0.911 | **0.990** |
| Donor / acceptor PR-AUC | 0.920 / 0.902 | 0.993 / 0.987 |
| Donor recall | 0.174 | **0.904** |
| FN reduction vs base | — | **88.2%** |

The base model finds **~17%** of alternative sites; M2-S finds **~90%** — a 5.3× recall gain — while
holding precision roughly level. Validation macro PR-AUC was 0.953 (epoch 6).

On the **overall** Ensembl test set, each model scored at its own F1-optimal threshold:

| | Base (OpenSpliceAI) | M2-S (v4) |
|-|---------------------|-----------|
| Its own F1-optimal threshold | 0.25 | 0.99 |
| Precision / recall | 0.832 / 0.654 | **0.866 / 0.797** |
| Macro F1 | 0.733 | **0.830** |
| FP / FN | 24,100 / 63,239 | **22,600 / 37,089** |

M2-S is ahead on precision as well as recall, so the alternative-site gain is not bought
by loosening the operating point. The thresholds differ by 4x, which is why quoting either
model at the other's cutoff is misleading.

!!! note "The Lab UI now uses the same yardstick"
    Both numbers above are scored on Ensembl, never on MANE — scoring a model built to find
    non-MANE sites against MANE alone counts every success as an error. Until 2026-08-10 the
    genome view did exactly that, because it took ground truth from the *base* model's annotation.
    It now resolves to the overlaid meta model's training annotation and labels every count with the
    truth set used, so the page and this table agree. See
    [Reading the numbers](../../bio_lab/05_reading_the_numbers.md#which-yardstick).

!!! warning "Alternative-site recovery is a discovery-mode tradeoff"
    Going from 17% to 90% recall necessarily admits more positives. At a fixed argmax threshold the
    overall false-positive *count* rises sharply (13,163 for base against 1,006,208 for M2-S), because
    argmax sits near 0.5 and M2-S is calibrated to operate near 0.99. Judge it by PR-AUC and by
    precision/recall at each model's own F1-optimal point (above), not by argmax FP counts.

!!! note "Corrected 2026-08"
    An earlier version quoted **precision 0.97 / recall 0.94 / F1 0.956 at threshold 0.65** here.
    Those came from a sweep whose negatives are 1%-subsampled, which inflates precision and pulls the
    apparent optimum down. See the [M1-S page](m1_canonical.md) for the mechanism. Recall-side numbers
    (the 17% → 90% headline, the FN reduction) were never affected.

**Cross-annotation check (GENCODE).** The same effect holds on GENCODE ∖ MANE: M2-S reaches PR-AUC
**0.907**, and even M1-S v2 improves GENCODE-alt to **0.728** vs base 0.637 (+0.091) with false
positives down 85.8% — evidence the gain generalizes beyond the Ensembl definition.

**Modality contribution.** As with the canonical models, **junction is the dominant channel** — removing
it drops overall PR-AUC by 0.029, the largest single-modality effect (removing all multimodal channels:
−0.053).

---

## Version history

| Version | Alt-site PR-AUC | Alt-site recall | Notes |
|---------|-----------------|-----------------|-------|
| v1 | 0.967 | ~59% | first Ensembl-trained model |
| v2 | 0.9665 | ~66% | learned blend (α = 0.665); val macro PR-AUC 0.833 |
| **v4 (`cleanannot`)** | **0.990** | **~90%** | **current promoted model**; FN −88% |

!!! note "Reconciling older numbers"
    Earlier docs report the v1/v2 alternative-site result two ways (recall ~59% vs ~66%) because one
    table predates the v2 retrain. Both are **superseded** by v4 — treat the v4 row above as
    authoritative and the earlier figures as historical context only.

---

## Takeaways

- Alternative-site recognition is the meta layer's **highest-leverage result**: recall from ~17% to
  ~90%, PR-AUC 0.911 → 0.990.
- The win comes from **training on the right annotation** (Ensembl), not from a fancier architecture —
  M2-S shares M1-S's design. The architecture-search post-mortem
  ([`M2/architecture_and_inductive_bias_lessons.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/M2/architecture_and_inductive_bias_lessons.md))
  reached the same conclusion: leverage is upstream in features and labels, not in fusion.
- Junction evidence again carries the multimodal signal — consistent with M1, and a useful contrast
  with [M3](m3_novel.md), where junction is deliberately withheld as the prediction target.
  **Now quantified**: see [M2-S Modality Attribution](m2_ablation.md). Multimodal evidence accounts for
  ~80% of the lift over base, junction support for ~70% of that, and the remaining channels turn out to
  be mutually redundant rather than inert.
