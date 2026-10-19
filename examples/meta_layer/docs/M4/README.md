# M4: Perturbation-Induced Splicing (topic)

Design notes and case studies for **M4**, the *conditional / counterfactual* meta-model: predict the splicing change caused by perturbing a *trans*-acting regulator (e.g. TDP-43 loss in ALS, SF3B1 mutation in cancer). This is the disease-conditional arm of the meta layer, distinct from the unconditional M1 (canonical) / M2 (alternative) / M3 (novel) variants.

> First topic subdirectory under `examples/meta_layer/docs/`, created as the docs grew enough to warrant organizing by topic. Flat docs above remain where they are; new M4 material lands here.

## Tutorial series (start here)

| Part | Topic |
|---|---|
| [1. Training data](01_training_data.md) | one example $(e, R)$, the label $y = \Delta\mathrm{PSI}$, the 229 features in $X$, columns that must never enter $X$ |
| [2. Model architecture](02_model_architecture.md) | the methods implemented today, the conditional target design ($h_e$, $z_R$, $c$, $p_0$), and why to predict perturbed PSI through a log-odds offset |
| [3. Training and evaluation](03_training_and_evaluation.md) | gene splits, control-batch-grouped regulator folds, settings S2/S3, few-shot protocol, metrics, results so far |
| [4. TabPFN and in-context learning](04_tabpfn_in_context_learning.md) | how a tabular foundation model works and how M4 uses it |

## Contents
- [`m4_conditional_design.md`](m4_conditional_design.md): the core design: one conditional M4 (perturbation as input) instead of many per-gene variants; why it generalizes; what actually makes it work (perturbation-paired labels, not architecture); the current `rbp_n_bound` substrate and its limits; and the **TDP-43/ALS first-instance validation** (M2-S detects the UNC13A/STMN2 cryptic sites, but de-repression is not learned, and why).

- [`m4_fewshot_benchmark.md`](m4_fewshot_benchmark.md): **benchmark v0 (2026-10-17)**: predict a held-out HepG2 knockdown regulator's signed ΔPSI on A3SS/A5SS events. Leakage-safe design (gene splits, coupled long/short forms, leave-one-out baseline PSI), results for S3 (new regulator + new genes: pooled zero-shot 58% direction accuracy) and S2 (event prior 67% once regulators that share control samples are kept in one fold; v0's 74% leaked shared control noise), and the cold-start framing for next steps.

## The one-line takeaway
Model the **regulator → splicing map once**, conditioned on *which regulator is perturbed*; the disease is an **input**, not a new model. TDP-43/UNC13A/STMN2 is the first instance used to validate the mechanism, deliberately not the thing the architecture is built around.

## Data and scripts
- Label + feature build: [`examples/data_preparation/m4/`](../../../data_preparation/m4/README.md) (`01` events, `03` site features, `04` regulator binding).
- Benchmark: `examples/meta_layer/17_m4_fewshot_benchmark.py` (shared helpers `_m4_common.py`; pod driver `ops_m4_pod.sh`).

## Related
- `../meta_model_variants_m1_m4.md`: the M1–M4 variant overview.
- `../M3/` (incl. `m3_prerequisites.md`), `../ood_generalization.md`.
- Demo/experiment scripts: `examples/UI_integration/` (03 cryptic detection, 04 TDP-43 ablation counterfactual, 05 Integrated Gradients, 06 demo synthesis).
