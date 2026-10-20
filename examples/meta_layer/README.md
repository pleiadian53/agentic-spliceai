# Meta Layer Examples

**Purpose:** Driver scripts for training, evaluating, and analyzing the meta-layer models (M1–M4) that refine the base model's splice predictions with multimodal evidence.

These are real R&D scripts. For the connected walkthrough of how they fit together, and for the evaluated results, see the published series:

- **Workflow (how to run it end to end):** [docs/workflows/meta_layer/](../../docs/workflows/meta_layer/README.md)
- **Results & findings (what came out):** [docs/meta_layer/results/](../../docs/meta_layer/results/README.md)

---

## Model variants

`M{task}-{S|P}`: task 1–4 (canonical / alternative / novel / perturbation), `-S` = sequence model, `-P` = position-level. See [naming_convention.md](../../docs/meta_layer/methods/naming_convention.md).

| Variant | Task | Status |
|---------|------|--------|
| M1-P | Canonical, position-level (XGBoost) | Reference baseline |
| M1-S | Canonical, sequence CNN | Promoted (`m1s_v4_cleanannot`) |
| M2-S | Alternative sites (Ensembl ∖ MANE) | Promoted (`m2s_v4_cleanannot`) |
| M3-S | Novel sites (junction-supported) | Best novel-site ranker (`m3_v1`) |
| M3-R | Candidate refiner (rerank) | Research (honest negative) |
| M4 | Perturbation response (RBP knockdown ΔPSI) | In progress |

---

## Layout

Scripts are grouped by the variant they serve. Each folder is numbered from `01` in the order you would run its scripts; `shared/` holds what several variants use. Label and feature preparation lives in [`examples/data_preparation/<variant>/`](../data_preparation/), which follows the same convention.

When a doc or note names a script, it names the folder too (`m3/01_evaluate_m3_novel.py`), never a bare number: `01` exists in every folder.

### `shared/`: used by several variants

| Script | Purpose |
|--------|---------|
| `01_train_sequence_model.py` | **Primary trainer** for M1-S / M2-S / M3-S. Variant via `--mode {m1,m2,m3}`, architecture via `--arch {concat_fusion,xattn_fusion}`. |
| `02_evaluate_sequence_model.py` | Held-out meta-vs-base evaluation (M1-S; M2-S canonical yardstick). Supports FASTA inference, ablation, calibration. |
| `03_junction_coverage_audit.py` | GTEx junction coverage of annotated sites (feeds M2 and the M3 label workflow). |
| `ops_bootstrap_pod.sh` | Links a pod's `data/` and `output/` to the network volume; sourced by every pod runner. |

### `m1/`: canonical sites

| Script | Purpose |
|--------|---------|
| `01_xgboost_baseline.py` | M1-P position-level XGBoost baseline. |
| `02_calibration_analysis.py` | Reliability / ECE of base vs meta probabilities. |
| `03_modality_ablation.py` | M1-P leave-one-modality-out contribution. |
| `ops_train_m1s_pod.sh`, `ops_eval_m1s_pod.sh`, `ops_ablation_m1s_pod.sh` | Pod runners: train M1-S (`shared/01_train_sequence_model.py`), evaluate it and run channel ablations (`shared/02_evaluate_sequence_model.py`). |

### `m2/`: alternative sites

| Script | Purpose |
|--------|---------|
| `01_evaluate_alternative_sites.py` | M2-S alternative-site evaluation (Ensembl ∖ MANE → `m2a`; GENCODE → `m2b`). |
| `02_evaluate_tissue_stratified.py` | Alternative-site recall stratified by the GTEx tissue supporting each site (consumes the per-site outcomes from `m2/01_evaluate_alternative_sites.py --dump-site-outcomes`). |
| `03_verify_evaluation_stats.py` | Recompute annotation stats and reprint metrics from result JSONs (cross-check of the results write-up). |
| `04_feature_redundancy_analysis.py` | Whether modalities carry signal complementary to base scores. |
| `ops_train_m2s_pod.sh`, `ops_eval_alt_sites_pod.sh`, `ops_eval_tissue_pod.sh`, `ops_ablation_m2s_pod.sh` | Pod runners: train M2-S, alternative-site and tissue evaluation, channel ablations. |

### `m3/`: novel sites

| Script | Purpose |
|--------|---------|
| `01_evaluate_m3_novel.py` | M3-S novel-site evaluation: anti-circular, per-gene precision@k / recall@k vs independent truth (D1/D2). M3-S itself is trained by `shared/01_train_sequence_model.py --mode m3`. |
| `02_m3_distance_baseline_probe.py` | How much of the evaluation a "GT/AG near a known site" ranker already recovers. |
| `03_train_candidate_refiner.py` | M3-R candidate refiner (XGBoost reranker, base-matched hard negatives). |
| `04_evaluate_candidate_classifier.py` | M3-R anti-circular head-to-head vs base. |
| `_m3_novel_eval.py` | Metric library shared by the M3 evaluators (self-test: `python examples/meta_layer/m3/_m3_novel_eval.py`). |
| `ops_train_m3_pod.sh`, `ops_eval_m3_novel_pod.sh` | Pod runners: train M3-S, build the evaluation feature cache. |

### `m4/`: perturbation response

| Script | Purpose |
|--------|---------|
| `01_m4_fewshot_benchmark.py` | Predict a held-out regulator's knockdown ΔPSI from k examples (settings S2/S3; zero, prior, XGBoost, TabPFN). |
| `_m4_common.py` | Event loading, features, gene/regulator splits, metrics. |
| `ops_m4_pod.sh` | Pod runner: site features (`examples/data_preparation/m4/03_build_site_features.py`), then the benchmark. |

M4 labels and features are built by [`examples/data_preparation/m4/`](../data_preparation/m4/README.md) (`01`–`06`).

### Old → new names (reorganised 2026-10-20)

Session notes and logs written before 2026-10-20 use the old flat names.

| Old | New |
|---|---|
| `01_xgboost_baseline.py` | `m1/01_xgboost_baseline.py` |
| `02_calibration_analysis.py` | `m1/02_calibration_analysis.py` |
| `03_modality_ablation.py` | `m1/03_modality_ablation.py` |
| `07_train_sequence_model.py` | `shared/01_train_sequence_model.py` |
| `08_evaluate_sequence_model.py` | `shared/02_evaluate_sequence_model.py` |
| `09_evaluate_alternative_sites.py` | `m2/01_evaluate_alternative_sites.py` |
| `10_verify_evaluation_stats.py` | `m2/03_verify_evaluation_stats.py` |
| `11_junction_coverage_audit.py` | `shared/03_junction_coverage_audit.py` |
| `12_feature_redundancy_analysis.py` | `m2/04_feature_redundancy_analysis.py` |
| `13_evaluate_m3_novel.py` | `m3/01_evaluate_m3_novel.py` |
| `14_train_candidate_refiner.py` | `m3/03_train_candidate_refiner.py` |
| `15_evaluate_candidate_classifier.py` | `m3/04_evaluate_candidate_classifier.py` |
| `16_evaluate_tissue_stratified.py` | `m2/02_evaluate_tissue_stratified.py` |
| `17_m4_fewshot_benchmark.py` | `m4/01_m4_fewshot_benchmark.py` |
| `18_m3_distance_baseline_probe.py` | `m3/02_m3_distance_baseline_probe.py` |
| `ops_*_pod.sh`, `_m3_novel_eval.py`, `_m4_common.py` | same name, in the folder of their variant (`ops_bootstrap_pod.sh` in `shared/`) |

---

## Directories

- `docs/`: methods and how-to notes: `evaluation_tutorial.md`, `evaluation_hierarchy.md`, `meta_model_variants_m1_m4.md`, plus `M2/`, `M3/`, `M4/` design notes.
- `results/`: curated result write-ups (also promoted to [docs/meta_layer/results/](../../docs/meta_layer/results/README.md)).
