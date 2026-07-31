# M3 — Novel Splice Site Discovery (topic)

Design notes and experimental results for **M3**, the *unconditional* novel
splice site discoverer: identify splice sites that are **absent from every
major annotation** (Ensembl, GENCODE, RefSeq) yet supported by junction
evidence. M3 is the candidate generator that feeds Phase 8 (isoform
discovery) and any later disease-conditional models.

> Topic subdirectory under `examples/meta_layer/docs/`, mirroring the
> [`M4/`](../M4/) layout. M3 material that used to live flat in `docs/`
> now lands here.

> **📂 Where the data-prep scripts live:** the M3 label workflow (positive
> pool, anchors, negatives) is under **[`examples/data_preparation/m3/`](../../../data_preparation/m3/)**
> (run-ordered `01`–`09` + its own README), NOT here — `examples/meta_layer/`
> holds the meta-model *training/eval*. Completing that workflow is a
> **precondition** to M3 Phase C training.

## Scope decision (settled 2026-05-13)

> **Naming.** Throughout this folder, **M3-v1** is the *training run* in
> `output/meta_layer/m3_v1/`, whose canonical ID is **`m3s.concat_fusion.cleanannot`**
> (variant M3-S · `concat_fusion` architecture · `cleanannot` corpus). The "v1" is a run
> index only — it is neither an architecture nor a corpus generation. `M3-v1.1` is the
> confirmed-only retrain of the same three axes. See
> [naming_convention.md](../../../../docs/meta_layer/methods/naming_convention.md).
> Note that `output/meta_layer/m3s/` is **not** this model — it is a `--smoke` plumbing run.

M3 v1 is **unconditional**: a per-position novelty score over the genome,
trained on a *pooled* positive set (GTEx-novel-filtered + Tier-1 published
disease catalogs), with junctions as the **label**, not an input.

Disease conditioning (the disease-Δ idea — "this site activates *in TDP-43
LOF / SF3B1-mutant context*") is **deferred** to M3-D / [M4](../M4/). That is
perturbation-effect prediction, a different thesis. Building unconditional
first gives us the catalog the conditional models select over. See
[`m3_design.md`](m3_design.md) §1 for the full rationale.

## Status

| Phase | What | Status |
|---|---|---|
| A1 | Cross-annotation audit | **Done** (corrected 2026-05-24). A minus-strand annotation bug inflated the original count; genuinely-novel GTEx sites = 748. See [`label_audit_A1.md`](label_audit_A1.md). |
| B1 | Positive-pool assembly | **Done** — pooled positives **154,113** (SpliceVault-dominated) + 6,351 held-out disease anchors. |
| B2 | Negatives + annotation mask | **Done** — 826K annotation mask (load-bearing: loss ignore-index + inference post-filter). The 308K sampled negatives are **unused by the recognizer**, which trains per-gene-window: every non-positive, non-masked position is already `neither`. See [`m3_design.md`](m3_design.md) §5. |
| B3 | ENCODE4 long-read held-out truth set | **Done** (2026-05-28) — 681,809 novel D1 sites; `longread_confirmed` flag added to the positive pool (77,879 / 154,113). |
| C | Architecture + training | **Done** (2026-05-30) — `output/meta_layer/m3_v1`; the M2-S backbone (`concat_fusion` architecture) with `mm_channels=7`, annotated sites masked, confirmed positives 2×. |
| D | Anti-circular evaluation (Tier 0) | **Done** (2026-07-26) — **M3-v1 is the best novel-site ranker**: D1 P@5 0.335 vs base 0.277; D2 R@20 0.791 vs 0.506. [`m3_eval_D_results.md`](m3_eval_D_results.md) |
| Tier 1 | Confirmed-only retrain — is label noise the ceiling? | **Done — no.** `m3s_v1_1_confirmed` marginally worse everywhere. |
| Tier 2 | M3-R candidate refiner — is the task framing the ceiling? | **Done — honest negative.** Trains to AUC 0.90 but ties base; its edge is between-gene, not within-gene. [`m3r_candidate_refiner_results.md`](m3r_candidate_refiner_results.md) |
| A2 | M1-S zero-shot probe on TDP-43 cryptic catalog | Superseded by Tier 0 (M1-S evaluated head-to-head on D1/D2; it underperforms base) |
| UI | Novel Site Explorer (`/novel/{gene}`) | **Done** — per-gene ranked candidates in the Bio Lab UI, held-out chromosomes only. |

## Contents
- [`m3_prerequisites.md`](m3_prerequisites.md) — the original go/no-go: is
  junction support a reliable novelty signal? What other data sources exist?
  Grounded in the `output/meta_layer/junction_coverage_audit*` outputs.
- [`m3_design.md`](m3_design.md) — the v1 design synthesis, **reconciled against
  what shipped**: unconditional-first scope, junctions-as-label, the architecture
  call (the M2-S attention ablation applies here too), and how the disease-Δ idea
  is folded in as pooled positives rather than conditioning. Sections carry
  ✅/⚠️ **Outcome** callouts marking which predictions held; §9 records the
  Tier 0/1/2 ladder and the locus-level-vs-position-level finding.
- [`label_audit_A1.md`](label_audit_A1.md) — experimental results from Phase
  A1: cross-checking the 67,490 GTEx-novel sides against GENCODE + RefSeq.
- [`m3_training_data.md`](m3_training_data.md) — what an M3 training example is,
  the 4 label classes, how junction evidence relates to labels, and the
  **validation status** of every pool (coordinate accuracy ✓ / functional ⚠️ pending B3).

## The one-line takeaway
**M3-v1 is the best novel-site ranker in the system** — the only meta model to
beat raw base on independent, anti-circular novel-site truth. The pool it trains
on is ~154K junction-supported, annotation-clean sites, carried almost entirely by
**SpliceVault** cryptic events (GTEx-only novel junctions are rare — 748 — once a
minus-strand annotation bug is fixed).

The open problem is **not** label quality, as originally believed: Tier 1 tested
that directly and cleaner labels made things slightly *worse*. Tier 2 then showed
reframing the task doesn't help either. The remaining lever is **position-level
features** — the external multimodal tracks are locus-level evidence, which is why
they lift M1/M2 but barely move novel-site discovery.

## Related
- [`../meta_model_variants_m1_m4.md`](../meta_model_variants_m1_m4.md) — the M1–M4 variant overview.
- [`../junction_coverage_findings.md`](../junction_coverage_findings.md) — strand asymmetry + biotype effects in the junction audit that seeds M3.
- [`../ood_generalization.md`](../ood_generalization.md) — why training labels matter for alt/novel-site detection.
- [`../M4/`](../M4/) — the disease-conditional sibling (perturbation as input).
- Audit script: `examples/meta_layer/11_junction_coverage_audit.py`.
