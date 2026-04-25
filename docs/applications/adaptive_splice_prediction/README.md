# Adaptive Splice Prediction (M1/M2)

**Goals served**: adaptive splice prediction

**Tier**: Active

**Last updated**: 2026-04

---

## Problem

Base-layer predictions are canonical — they capture the splice sites seen
during training but miss context-dependent variation (tissue specificity,
alternative sites, variant-induced events). The meta layer refines base
predictions with multimodal evidence (conservation, junction support,
RBP binding, chromatin accessibility, etc.) and specializes for two task
regimes: **M1 — canonical** (sharpen what the base model already sees)
and **M2 — alternative** (recover sites the base model misses on
annotation sources richer than MANE).

## User-facing functionality

- Train an M1-S sequence-level adaptive model on MANE canonical labels
- Train an M2-S sequence-level adaptive model on Ensembl/GENCODE alternative labels
- Produce context-aware per-nucleotide predictions that combine base model
  scores with multimodal features via logit-space blend
- Evaluate calibration, modality importance (SHAP / gain), and OOD
  generalization to alternative sites
- Ablate individual modalities to quantify contribution

## Driving examples

- [`examples/meta_layer/01_xgboost_baseline.py`](../../../examples/meta_layer/01_xgboost_baseline.py) — M1-P position-level XGBoost baseline
- [`examples/meta_layer/02_calibration_analysis.py`](../../../examples/meta_layer/02_calibration_analysis.py) — post-hoc calibration analysis
- [`examples/meta_layer/03_modality_ablation.py`](../../../examples/meta_layer/03_modality_ablation.py) — modality-by-modality ablation
- [`examples/meta_layer/07_train_sequence_model.py`](../../../examples/meta_layer/07_train_sequence_model.py) — train M1-S / M2-S sequence CNN
- [`examples/meta_layer/08_evaluate_sequence_model.py`](../../../examples/meta_layer/08_evaluate_sequence_model.py) — evaluate M1-S on held-out chromosomes
- [`examples/meta_layer/09_evaluate_alternative_sites.py`](../../../examples/meta_layer/09_evaluate_alternative_sites.py) — evaluate on Ensembl/GENCODE alternative sites
- [`examples/meta_layer/10_verify_evaluation_stats.py`](../../../examples/meta_layer/10_verify_evaluation_stats.py) — confidence intervals and statistical verification
- [`examples/meta_layer/11_junction_coverage_audit.py`](../../../examples/meta_layer/11_junction_coverage_audit.py) — junction coverage diagnostic
- Pod ops: [`ops_train_m1s_pod.sh`](../../../examples/meta_layer/ops_train_m1s_pod.sh), [`ops_train_m2s_pod.sh`](../../../examples/meta_layer/ops_train_m2s_pod.sh), [`ops_ablation_m1s_pod.sh`](../../../examples/meta_layer/ops_ablation_m1s_pod.sh), [`ops_ablation_m2s_pod.sh`](../../../examples/meta_layer/ops_ablation_m2s_pod.sh)

## `src/` surface

- `agentic_spliceai.splice_engine.meta_layer.core.feature_schema` — canonical 116-column schema
- `agentic_spliceai.splice_engine.meta_layer.models.sequence_model` — 2-stream dilated CNN with logit-space blend
- `agentic_spliceai.splice_engine.meta_layer.training.*` — training loops, checkpointing, loss functions
- `agentic_spliceai.splice_engine.meta_layer.inference.*` — inference utilities
- `agentic_spliceai.splice_engine.eval.splitting` — balanced chromosome split for SpliceAI convention

## Evaluation

- **Datasets**: MANE (canonical), Ensembl canonical + alternative (M2), GENCODE alternative
- **Baselines**: OpenSpliceAI base model, XGBoost M1-P (Tree SHAP), v1 probability-space residual blend
- **Key metrics**: PR-AUC, ROC-AUC, accuracy, FP/FN per splice type; alternative-site recall
- **Results**:
  - [M1-P full-genome results](../../../examples/meta_layer/results/m1p_fullgenome_results.md) — XGBoost baseline
  - [M1-S ablation study](../../../examples/meta_layer/results/m1s_ablation_study.md)
  - [M1-S v2 logit-blend results](../../../examples/meta_layer/results/m1s_v2_logit_blend_results.md) — PR-AUC 0.9954, FPs -15.5% vs base
  - [M2-S Ensembl-trained results](../../../examples/meta_layer/results/m2s_ensembl_trained_results.md) — 59% recall on alternative sites
  - [Alternative site evaluation](../../../examples/meta_layer/results/alternative_site_evaluation_results.md) — M2-S PR-AUC 0.965 vs base 0.749

**Key finding**: logit-space residual blend (v2) exceeds base model on
both canonical (PR-AUC 0.9954 > 0.99) and alternative sites (0.775 > 0.749)
— v1 probability-space blend hurt on alternative sites due to
overcommitment to the meta-CNN when uncertain.

## Maturity tier and signals

**Current tier**: Active

**Signals supporting the tier**:

- 11 example scripts spanning baseline → training → evaluation → diagnostics
- M1-S v2 and M2-S trained with reproducible results
- Ablation and calibration analyses committed
- Stable args for training scripts since logit-blend refactor (April 2026)
- Depended on by Variant Effect Analysis and (planned) Novel Isoform Discovery
- Pod-based training ops scripts codify reproducibility

## Graduation signals

**To advance to Mature, the application needs**:

- Canonical driver script for production inference (currently evaluation scripts serve this role)
- Inference-path test coverage in `tests/`
- M3-S (novel site discovery) trained and evaluated, demonstrating the full M1-M4 framework
- Documented stable CLI wrapping `07_train_sequence_model.py` or a dedicated inference entry point

## Known limitations

- M2-S OOD generalization degrades on unseen genes (see [ood_generalization.md](../../../examples/meta_layer/docs/ood_generalization.md))
- Junction modality coverage uneven across tissues (see [junction_coverage_findings.md](../../../examples/meta_layer/docs/junction_coverage_findings.md))
- `shap` package broken with numpy 2.4 — use XGBoost `pred_contribs=True` instead
- Inference requires pod for large-scale work; local inference only feasible per-chromosome

## Related

- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — base layer input
- [Multimodal Feature Engineering](../multimodal_features/README.md) — upstream features
- [Variant Effect Analysis](../variant_analysis/README.md) — uses M1-S/M2-S for delta scoring
- [Novel Isoform Discovery](../novel_isoform_discovery/README.md) — future M3 application
- [Meta Layer Methods](../../meta_layer/methods/) — M1-M4 framework, naming convention
- [Roadmap: Phase 6](../../ROADMAP.md)
