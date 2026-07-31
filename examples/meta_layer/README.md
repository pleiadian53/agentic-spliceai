# Meta Layer Examples

**Purpose:** Driver scripts for training, evaluating, and analyzing the meta-layer models (M1–M4)
that refine the base model's splice predictions with multimodal evidence.

These are real R&D scripts, run in development order (the numbering reflects that order, so there are
gaps). For the connected, prose walkthrough of how they fit together, and for the evaluated results,
see the published series:

- **Workflow (how to run it end to end):** [docs/workflows/meta_layer/](../../docs/workflows/meta_layer/README.md)
- **Results & findings (what came out):** [docs/meta_layer/results/](../../docs/meta_layer/results/README.md)

---

## Model variants

`M{task}-{S|P}` — task 1–4 (canonical / alternative / novel / perturbation), `-S` = sequence model,
`-P` = position-level. See [naming_convention.md](../../docs/meta_layer/methods/naming_convention.md).

| Variant | Task | Status |
|---------|------|--------|
| M1-P | Canonical, position-level (XGBoost) | Reference baseline |
| M1-S | Canonical, sequence CNN | Promoted (`m1s_v4_cleanannot`) |
| M2-S | Alternative sites (Ensembl ∖ MANE) | Promoted (`m2s_v4_cleanannot`) |
| M3-S | Novel sites (junction-supported) | Best novel-site ranker (`m3_v1`) |
| M3-R | Candidate refiner (rerank) | Research (honest negative) |
| M4 | Perturbation-induced | In progress |

---

## Scripts

### Train

| Script | Purpose |
|--------|---------|
| `07_train_sequence_model.py` | **Primary trainer** for M1-S / M2-S / M3-S. Variant via `--mode {m1,m2,m3}`, architecture via `--arch {concat_fusion,xattn_fusion}`. |
| `01_xgboost_baseline.py` | M1-P position-level XGBoost baseline. |
| `14_train_candidate_refiner.py` | M3-R candidate-refiner (XGBoost reranker, base-matched hard negatives). |

### Evaluate

| Script | Purpose |
|--------|---------|
| `08_evaluate_sequence_model.py` | Held-out meta-vs-base evaluation (M1-S; M2-S canonical yardstick). Supports FASTA inference, ablation, calibration. |
| `09_evaluate_alternative_sites.py` | M2-S alternative-site evaluation (Ensembl ∖ MANE → `m2a`; GENCODE → `m2b`). |
| `13_evaluate_m3_novel.py` | M3 novel-site evaluation — anti-circular, per-gene precision@k / recall@k vs independent truth (D1/D2). |
| `15_evaluate_candidate_classifier.py` | M3-R Phase-2 anti-circular head-to-head vs base. |
| `10_verify_evaluation_stats.py` | Recompute annotation stats + reprint metrics from result JSONs (cross-check). |

### Analyze

| Script | Purpose |
|--------|---------|
| `02_calibration_analysis.py` | Reliability / ECE of base vs meta probabilities. |
| `03_modality_ablation.py` | M1-P leave-one-modality-out contribution. |
| `11_junction_coverage_audit.py` | GTEx junction coverage of annotated sites (feeds M2/M3). |
| `12_feature_redundancy_analysis.py` | Whether modalities carry signal complementary to base scores. |

### GPU-pod runners

`ops_*.sh` wrap the train/eval scripts for RunPod (see the workflow series'
[GPU Pods runbook](../../docs/workflows/meta_layer/08_gpu_pods.md)): `ops_train_{m1s,m2s}_pod.sh`,
`ops_eval_{m1s,alt_sites}_pod.sh`, `ops_ablation_{m1s,m2s}_pod.sh`, `ops_bootstrap_pod.sh`.

---

## Directories

- `docs/` — methods and how-to notes: `evaluation_tutorial.md`, `evaluation_hierarchy.md`,
  `meta_model_variants_m1_m4.md`, plus `M2/`, `M3/`, `M4/` design notes.
- `results/` — curated result write-ups (also promoted to
  [docs/meta_layer/results/](../../docs/meta_layer/results/README.md)).
- `_m3_novel_eval.py` — shared metric library for the M3 evaluators.
