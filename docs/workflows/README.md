# Analysis Workflows

End-to-end, multi-stage workflows that string the individual layers of Agentic-SpliceAI into a single
runnable pipeline.

## Meta-Layer MLOps (M1-S / M2-S)

The **[Meta-Layer MLOps Workflow](meta_layer/README.md)** is the golden path for the sequence-level
meta-models — from raw GTF/FASTA to a promoted, evaluated checkpoint:

1. [Data Preparation](meta_layer/01_data_preparation.md) — GTF/FASTA → splice-site labels
2. [Base Scoring](meta_layer/02_base_scoring.md) — OpenSpliceAI per-nucleotide predictions
3. [Feature Engineering](meta_layer/03_feature_engineering.md) — the 9-modality feature stack
4. [Training M1-S](meta_layer/04_training_m1s.md) and [M2-S](meta_layer/05_training_m2s.md)
5. [Evaluation](meta_layer/06_evaluation.md) — meta-vs-base, canonical and alternative sites
6. [Reporting & Promotion](meta_layer/07_reporting.md) — results, provenance, canonical registry
7. [GPU Pods](meta_layer/08_gpu_pods.md) — running it all at genome scale on RunPod

## Related workflows

- [Splice Prediction Guide](../tutorials/SPLICE_PREDICTION_GUIDE.md) — base-layer prediction
- [Data Preparation CLI](../base_layer/DATA_PREPARATION_CLI.md) — data-processing commands
- [Agency Patterns](../agency/MEMORY_PATTERNS.md) — agentic workflow patterns

## Planned

- Variant impact assessment
- Novel isoform discovery pipeline (M3 → agentic validation)
