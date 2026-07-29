# Analysis Workflows

End-to-end, multi-stage workflows that string the individual layers of Agentic-SpliceAI into a single
runnable pipeline.

## Meta-Layer MLOps (M1-S / M2-S / M3)

The **[Meta-Layer MLOps Workflow](meta_layer/README.md)** is the golden path for the sequence-level
meta-models — from raw GTF/FASTA to a promoted, evaluated checkpoint:

1. [Data Preparation](meta_layer/01_data_preparation.md) — GTF/FASTA → splice-site labels
2. [Base Scoring](meta_layer/02_base_scoring.md) — OpenSpliceAI per-nucleotide predictions
3. [Feature Engineering](meta_layer/03_feature_engineering.md) — the 9-modality feature stack
4. [Training M1-S](meta_layer/04_training_m1s.md) and [M2-S](meta_layer/05_training_m2s.md)
5. [Evaluation](meta_layer/06_evaluation.md) — meta-vs-base, canonical and alternative sites
6. [Reporting & Promotion](meta_layer/07_reporting.md) — results, provenance, canonical registry
7. [GPU Pods](meta_layer/08_gpu_pods.md) — running it all at genome scale on RunPod

**M3 (novel sites)** reuses Stages 1–3, then branches into its own sub-series — novel sites have no
annotation to label against, so they need separate label curation and an anti-circular evaluation:

9. [M3 Label Curation](meta_layer/09_m3_label_curation.md) — junctions, long-read, disease catalogs
10. [M3 Training](meta_layer/10_training_m3.md) — recognizer (M3-S) and candidate refiner (M3-R)
11. [M3 Evaluation](meta_layer/11_m3_evaluation.md) — per-gene precision@k against independent truth

## Related workflows

- [Splice Prediction Guide](../tutorials/SPLICE_PREDICTION_GUIDE.md) — base-layer prediction
- [Data Preparation CLI](../base_layer/DATA_PREPARATION_CLI.md) — data-processing commands
- [Agency Patterns](../agency/MEMORY_PATTERNS.md) — agentic workflow patterns

## Planned

- Variant impact assessment
- Agentic validation of M3 novel-site candidates (literature, expression, conservation)
