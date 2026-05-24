# M4 — Perturbation-Induced Splicing (topic)

Design notes and case studies for **M4**, the *conditional / counterfactual*
meta-model: predict the splicing change caused by perturbing a *trans*-acting
regulator (e.g. TDP-43 loss in ALS, SF3B1 mutation in cancer). This is the
disease-conditional arm of the meta layer, distinct from the unconditional
M1 (canonical) / M2 (alternative) / M3 (novel) variants.

> First topic subdirectory under `examples/meta_layer/docs/` — created as the
> docs grew enough to warrant organizing by topic. Flat docs above remain where
> they are; new M4 material lands here.

## Contents
- [`m4_conditional_design.md`](m4_conditional_design.md) — the core design:
  one conditional M4 (perturbation as input) instead of many per-gene variants;
  why it generalizes; what actually makes it work (perturbation-paired labels,
  not architecture); the current `rbp_n_bound` substrate and its limits; and the
  **TDP-43/ALS first-instance validation** (M2-S detects the UNC13A/STMN2 cryptic
  sites, but de-repression is not learned — and why).

## The one-line takeaway
Model the **regulator → splicing map once**, conditioned on *which regulator is
perturbed*; the disease is an **input**, not a new model. TDP-43/UNC13A/STMN2 is
the first instance used to validate the mechanism — deliberately not the thing
the architecture is built around.

## Related
- `../meta_model_variants_m1_m4.md` — the M1–M4 variant overview.
- `../m3_prerequisites.md`, `../ood_generalization.md`.
- Demo/experiment scripts: `examples/UI_integration/` (03 cryptic detection,
  04 TDP-43 ablation counterfactual, 05 Integrated Gradients, 06 demo synthesis).
