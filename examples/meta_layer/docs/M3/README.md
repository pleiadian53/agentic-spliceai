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
| B2 | Negative sampling | **Done** — 308,000 negatives (154K hard canonical-dinuc + 154K easy) + 826K annotation mask. Recognizer + post-filter framing (annotated sites masked, not decoy-negated). |
| A2 | M1-S zero-shot probe on TDP-43 cryptic catalog | Planned |
| B3 | ENCODE4 long-read held-out truth set | **Next** |
| C | Architecture + training | Planned |
| D | Evaluation (long-read, per-disease, calibration) | Planned |

## Contents
- [`m3_prerequisites.md`](m3_prerequisites.md) — the original go/no-go: is
  junction support a reliable novelty signal? What other data sources exist?
  Grounded in the `output/meta_layer/junction_coverage_audit*` outputs.
- [`m3_design.md`](m3_design.md) — the v1 design synthesis: unconditional-first
  scope, junctions-as-label, why the lever is **data/labels not architecture**
  (the M2-S attention ablation result applies here too), and how the
  disease-Δ idea is folded in as pooled positives rather than conditioning.
- [`label_audit_A1.md`](label_audit_A1.md) — experimental results from Phase
  A1: cross-checking the 67,490 GTEx-novel sides against GENCODE + RefSeq.
- [`m3_training_data.md`](m3_training_data.md) — what an M3 training example is,
  the 4 label classes, how junction evidence relates to labels, and the
  **validation status** of every pool (coordinate accuracy ✓ / functional ⚠️ pending B3).

## The one-line takeaway
The M3 novel pool is **~154K junction-supported, annotation-clean splice
sites**, carried almost entirely by **SpliceVault** cryptic events; GTEx-only
novel junctions are rare (748) once a minus-strand annotation bug is fixed.
The open problem is **label quality** (real cryptic biology vs alignment
artifact vs low-frequency noise), not finding candidates.

## Related
- [`../meta_model_variants_m1_m4.md`](../meta_model_variants_m1_m4.md) — the M1–M4 variant overview.
- [`../junction_coverage_findings.md`](../junction_coverage_findings.md) — strand asymmetry + biotype effects in the junction audit that seeds M3.
- [`../ood_generalization.md`](../ood_generalization.md) — why training labels matter for alt/novel-site detection.
- [`../M4/`](../M4/) — the disease-conditional sibling (perturbation as input).
- Audit script: `examples/meta_layer/11_junction_coverage_audit.py`.
