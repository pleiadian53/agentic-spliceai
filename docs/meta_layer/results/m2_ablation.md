# M2-S Modality Attribution — which signals carry the lift

**Question:** the M2-S alternative-site result is real, but *what produces it?* The multimodal evidence,
or simply training on a richer annotation than the base model saw? This page answers that by zeroing
each channel group at inference on the promoted checkpoint and measuring what the model loses.

!!! abstract "Protocol"
    **Model:** `m2s.concat_fusion.cleanannot` · **Set:** 14,724 held-out genes, 627.6 M positions ·
    **Method:** inference-time channel zeroing ([workflow](../../workflows/ablation/README.md)) ·
    **Metric:** per-class PR-AUC, which is threshold-free.

---

## Results

Per-class PR-AUC, and the drop each ablation causes relative to the unablated model:

| Zeroed | donor PR-AUC | Δ donor | acceptor PR-AUC | Δ acceptor |
|--------|-------------:|--------:|----------------:|-----------:|
| — (full model) | **0.9821** | — | **0.9805** | — |
| **ALL multimodal** | 0.9291 | **−0.0530** | 0.9188 | **−0.0617** |
| **junction support** | 0.9448 | **−0.0373** | 0.9324 | **−0.0481** |
| epigenetic | 0.9817 | −0.0004 | 0.9801 | −0.0004 |
| conservation | 0.9821 | −0.0000 | 0.9801 | −0.0004 |
| chromatin accessibility | 0.9821 | −0.0000 | 0.9805 | −0.0000 |
| RBP binding | 0.9822 | +0.0001 | 0.9805 | −0.0000 |
| *base model (OpenSpliceAI)* | *0.9148* | | *0.9069* | |

---

## Three findings

### 1. Multimodal evidence is ~80% of the meta lift

The base model scores donor PR-AUC 0.9148; M2-S reaches 0.9821, a lift of +0.0673. Zeroing every
multimodal channel gives back 0.0530 of that, so **79% of the improvement comes from the evidence
channels and 21% from sequence context and the architecture** (84% / 16% on acceptors).

This bears directly on the fair-comparison objection. M2-S is trained on Ensembl while the base model was
trained on MANE, so it is reasonable to ask whether the lift is just annotation exposure. Holding the
trained model and its annotation exposure fixed, four fifths of its advantage disappears when the
evidence channels are zeroed, so the evidence is doing most of the work rather than the annotation alone.

The caveat is that this does not settle the question completely. A model *retrained* on Ensembl without
multimodal inputs might recover some of that ground by leaning harder on sequence, and this protocol
cannot see that. What it establishes is that the deployed model depends on the evidence, not that no
evidence-free model could compete. Closing that gap requires the retraining route below.

### 2. One channel carries most of it

Junction support alone accounts for −0.0373 of the −0.0530 total, roughly **70% of the multimodal
contribution** (78% on acceptors). Epigenetic marks, conservation, chromatin accessibility and RBP
binding are each indistinguishable from zero when removed individually.

Junction support is RNA-seq split-read evidence. It is **not** the training label, which is the Ensembl
annotation, so this is not leakage. But it is the closest available signal to a direct observation of
the event being predicted, and it is exactly why the M3 novel-site model
[excludes these channels](m3_novel.md) — there, junction support *is* the target. The honest statement
of this result is narrower than "ten fused modalities": for alternative-site recovery, most of the win
is RNA-seq evidence that splicing occurs at a position.

### 3. The remaining channels are redundant, not useless

Individually the non-junction groups sum to −0.0376, yet removing everything costs −0.0530. The
**−0.0154 gap, 29% of the multimodal effect, is shared signal**: conservation, chromatin and epigenetic
marks carry overlapping information, so dropping any one is absorbed by the others while dropping all of
them removes what they have in common.

The methodological consequence is worth stating, because it is easy to get wrong: **leave-one-out
ablation systematically understates correlated features.** A channel measuring −0.0000 alone is not
evidence that the channel is worthless, only that it is not uniquely necessary. Ranking modalities by
single-channel ablation would have concluded that eight of the nine channels contribute nothing, which
the all-channel control refutes.

---

## Reading these numbers correctly

!!! warning "Do not quote the FN/FP counts from these result files"
    The stored JSONs report error counts at a fixed 0.5 threshold, where M2-S records 7,222 false
    negatives against the base model's 71,118 — an apparent 89.8% reduction. At the same threshold its
    false positives are **1,006,348 against the base model's 13,163**. Splice sites are under 0.5% of
    positions, so a fixed 0.5 cutoff is not a meaningful operating point and the "FN −89.8%" figure is
    an artifact of it. PR-AUC is threshold-free and is the metric this page uses throughout. See the
    [evaluation workflow](../../workflows/meta_layer/06_evaluation.md) for operating-point selection.

**This measures dependence, not necessity.** Channels were zeroed at inference on a model that was
trained with them present, which answers "does the trained model lean on this signal?" A model trained
without junction support might route around its absence and lose less. The retraining question needs
`03_modality_ablation.py`; the distinction is set out in the
[ablation workflow](../../workflows/ablation/README.md).

**Do not generalise to other variants.** Modality importance is task-specific. M3 finds multimodal
channels add almost nothing to genome-wide novel-site ranking, while the M3-R candidate frame finds them
strongly load-bearing. Ablate the variant you intend to make claims about.

---

## Related

- [M2 — Alternative Sites](m2_alternative.md) — the result this attributes
- [Ablation Studies](../../workflows/ablation/README.md) — how to reproduce it
- [M3 — Novel Sites](m3_novel.md) — where junction channels are excluded by design
