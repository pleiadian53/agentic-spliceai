# Meta-Layer Results & Findings

A single, reconciled view of how the meta-layer models perform — what each one is measured against,
its current headline numbers, and the finding it establishes. Each model has its own page; this index
is the cross-model summary and the guide to reading the numbers correctly.

!!! info "How this relates to the workflow series"
    The [Meta-Layer MLOps Workflow](../../workflows/meta_layer/README.md) tells you *how* to produce
    and evaluate these models. This series tells you *what came out* — the evaluated results and the
    conclusions drawn from them. Where a number depends on a protocol, the model page links back to the
    relevant workflow stage.

---

## Cross-model summary

Current, promoted models. All splice tasks are severely class-imbalanced, so the headline metric is
**PR-AUC** (and, for the discovery tasks, **precision@k / recall@k per gene**), never accuracy alone.
"vs base" is the improvement over the OpenSpliceAI base model on the same evaluation.

| Model | Task | Evaluation | Headline (current) | vs base |
|-------|------|-----------|--------------------|---------|
| [M1-P](m1_canonical.md#m1-p-position-level-baseline) | Canonical, position-level (XGBoost) | SpliceAI holdout | PR-AUC donor 0.9986 / acceptor 0.9972 | FN −62%, FP −68% |
| [M1-S](m1_canonical.md#m1-s-sequence-model-current) | Canonical, sequence CNN | SpliceAI holdout (MANE) | macro PR-AUC **0.9998** | 0.9986 → 0.9998 |
| [M2-S](m2_alternative.md#current-results-v4) | Alternative sites (Ensembl ∖ MANE) | Eval-Ensembl-Alt | alt-site PR-AUC **0.990**, recall ~**90%** | 0.911 → 0.990, FN −88% |
| [M3-S (v1)](m3_novel.md#m3-s-m3-v1-the-novel-site-ranker) | Novel sites (junction-supported) | anti-circular D1 / D2 | D1 P@5 **0.335**, D2 R@20 **0.79** | beats base (0.277 / 0.51) |
| [M3-anchor](m3_novel.md#m3-anchor-folding-disease-anchors-in-the-disease-cryptic-win) | Novel + disease anchors (budgeted fold) | anti-circular D1 / D2 | D2 R@20 **0.86** (SF3B1), D1 held | D2 0.79 → 0.86 over M3-v1 |
| [M3-R](m3_novel.md#m3-r-the-candidate-refiner-milestone) | Candidate refiner (rerank) | anti-circular D1 / D2 | **ties base** (honest negative) | within-gene flat |
| M4 | Perturbation-induced | — | *in progress* | — |

**The one-line story:** the canonical task (M1) is essentially solved; the alternative-site task (M2)
is where multimodal refinement pays off most dramatically; and the novel-site frontier (M3) is where a
learned recognizer (M3-v1) helps but genome-averaged multimodal evidence, by itself, does not
([M3-R](m3_novel.md#m3-r-the-candidate-refiner-milestone)).

**Which signals earn that lift** is answered separately by
[M2-S Modality Attribution](m2_ablation.md): multimodal evidence accounts for ~80% of M2-S's gain over
base, junction support alone for ~70% of that, and the remaining channels prove mutually redundant
rather than inert. It is also the clearest illustration of the operating-point convention below, since
the same run shows a 90% false-negative reduction and a 76x false-positive increase at a 0.5 threshold.

---

## How to read these numbers

A few conventions are applied consistently across every page, so the results can be compared honestly.

**Metric choice.** Splice sites are ~0.1–1% of positions. We report **PR-AUC** (ranking quality
independent of threshold) and, for discovery, **precision@k / recall@k per gene**. Accuracy is quoted
only where it adds context.

**Operating points are labeled.** Raw false-negative / false-positive *counts* depend entirely on the
decision threshold, and at the naïve argmax/0.5 threshold they are misleading under this imbalance. So
when counts appear, they are reported at the **F1-optimal** (or a stated matched-recall) operating
point, and the operating point is named. The same run at argmax will show very different FP counts —
that is a threshold artifact, not a ranking change.

**"Anti-circular" evaluation (M3).** For novel-site discovery, evaluating against the annotation the
model was trained on is circular. M3 results are reported on **independent truth sets** — ENCODE
long-read novel junctions (D1) and held-out disease anchors (D2) — with the training/annotated sites
subtracted out. See [M3 novel-site results](m3_novel.md).

**Model naming.** `M{task}-{S|P}`: task 1–4 (canonical / alternative / novel / perturbation), `-S` =
sequence model, `-P` = position-level. Full convention in
[naming_convention.md](../methods/naming_convention.md).

---

## Model status

| Model | Version | Status |
|-------|---------|--------|
| M1-P | full-genome | Reference baseline |
| M1-S | `m1s_v4_cleanannot` | **Promoted (canonical)** |
| M2-S | `m2s_v4_cleanannot` | **Promoted (alternative)** |
| M3-S | `m3_v1` | Best general novel-site ranker (research) |
| M3-S v1.1 | `m3s_v1_1_confirmed` | Tier 1 confirmed-only retrain — marginally worse than v1 (label noise was *not* the ceiling) |
| M3-anchor | `m3s_anchorpos` | Disease-anchor fold (5% budget) — improves held-out SF3B1 (D2 R@20 0.86) with no D1 cost; not yet promoted (single chromosome split) |
| M3-R | `m3r_candidate_refiner` | Honest negative — not promoted; global-AUC triage repurposable |
| M4 | — | In progress (perturbation-paired labels needed) |

Superseded versions (M1-S v1/v2 blends, M2-S v1/v2) are summarized as **version history** on each
model page — the design trail is preserved without competing with the current numbers.

---

## Pages

- **[M1 — Canonical splice sites](m1_canonical.md)** — M1-P baseline and the promoted M1-S sequence model.
- **[M2 — Alternative splice sites](m2_alternative.md)** — the OOD problem and the M2-S alternative-site result.
- **[M3 — Novel splice sites](m3_novel.md)** — the M3-v1 ranker and the M3-R candidate-refiner milestone.

**Sources.** Curated write-ups in
[`examples/meta_layer/results/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/results/)
and [`examples/meta_layer/docs/M3/`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/M3/);
machine-readable metrics under `output/meta_layer/<model>/`.
