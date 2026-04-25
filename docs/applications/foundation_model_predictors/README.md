# Foundation Model Predictors

**Goals served**: adaptive splice prediction (alternative base-layer predictors)

**Tier**: Experimental (sub-project)

**Last updated**: 2026-04

---

## Problem

SpliceAI and OpenSpliceAI are trained on curated annotations (GENCODE,
MANE) and may under-represent context-dependent and non-canonical splice
sites. DNA foundation models (Evo2, SpliceBERT, HyenaDNA, others) trained
on broad genomic corpora provide alternative per-nucleotide
representations that may capture complementary signal. This application
explores the feasibility of using foundation-model embeddings as inputs
to splice-site classifiers, either frozen-head or end-to-end fine-tuned.

This is a **sub-project** at `foundation_models/` with its own
`pyproject.toml`, environments, and conventions. It has a separate
lifecycle from the main pipeline.

## User-facing functionality

- Extract per-nucleotide embeddings from foundation models (Evo2 7B, 40B,
  SpliceBERT, HyenaDNA, others) on cloud GPU infrastructure
- Train sparse or dense exon classifiers on the extracted embeddings
- Compare classifier performance across base models and architectures
- End-to-end fine-tune foundation models for splice prediction

## Driving examples

- [`examples/foundation_models/01_synthetic_pipeline.py`](../../../examples/foundation_models/01_synthetic_pipeline.py) — synthetic data pipeline (no GPU, <30s)
- [`examples/foundation_models/02_embedding_extraction.py`](../../../examples/foundation_models/02_embedding_extraction.py) — extract embeddings from foundation models
- [`examples/foundation_models/03_train_and_evaluate.py`](../../../examples/foundation_models/03_train_and_evaluate.py) — train classifier on extracted embeddings
- [`examples/foundation_models/04_extract_and_train.py`](../../../examples/foundation_models/04_extract_and_train.py) — combined extraction + training
- [`examples/foundation_models/05_sparse_exon_classifier.py`](../../../examples/foundation_models/05_sparse_exon_classifier.py) — Evo2 paper reproduction
- [`examples/foundation_models/06_dense_splice_predictor.py`](../../../examples/foundation_models/06_dense_splice_predictor.py) — dense per-nucleotide predictor
- [`examples/foundation_models/07_genome_scale_splice_predictor.py`](../../../examples/foundation_models/07_genome_scale_splice_predictor.py) — genome-scale inference
- [`examples/foundation_models/07a_direct_shard_splice_predictor.py`](../../../examples/foundation_models/07a_direct_shard_splice_predictor.py) — direct shard prediction
- [`examples/foundation_models/08_foundation_model_finetuning.py`](../../../examples/foundation_models/08_foundation_model_finetuning.py) — end-to-end fine-tuning

Ops scripts:

- [`ops_provision_cluster.py`](../../../examples/foundation_models/ops_provision_cluster.py) — SkyPilot cluster provisioning
- [`ops_stage_data.py`](../../../examples/foundation_models/ops_stage_data.py) — direct rsync data staging
- [`ops_run_pipeline.py`](../../../examples/foundation_models/ops_run_pipeline.py) — pipeline execution
- [`ops_compute_check.py`](../../../examples/foundation_models/ops_compute_check.py) — GPU environment verification

## `src/` surface

This sub-project has its own source tree at `foundation_models/foundation_models/`:

- `foundation_models.gpu_runner` — SkyPilot config builder + launcher
- `foundation_models.chunking` — sequence chunking (bfloat16 handling)
- `foundation_models.fp8_patch` — FP8 monkey-patch for GPUs < compute 8.9
- Model-specific adapters (Evo2, SpliceBERT, HyenaDNA, Pangolin, etc.)

Integration with main pipeline (pending):

- `agentic_spliceai.splice_engine.features.modalities.fm_embeddings` —
  modality shim for foundation-model features (currently commented out,
  awaiting full Evo2 extraction)

## Evaluation

- **Models evaluated**: SpliceAI (baseline), OpenSpliceAI, Evo2 7B, Evo2 40B (planned), SpliceBERT, HyenaDNA
- **GPU profiles**: 8 SkyPilot-configured (rtx4000ada, rtxa5000, rtx5090, rtx4090, l4, a40, a100, h100)
- **Throughput**: Evo2 7B embeddings ~100 bp/s (INT8) on M1 vs ~10K bp/s on A40
- **Current status**: Phase A complete (4 models), HyenaDNA bug fixed,
  embedding diagnostics added, Evo2 7B dual-strand extraction in progress

## Maturity tier and signals

**Current tier**: Experimental (sub-project)

**Signals supporting the tier**:

- Separate `pyproject.toml` and environment
- 8 example scripts + 4 ops scripts
- Multi-model support implemented (SkyPilot + RunPod)
- Error catalog at `dev/errors/foundation_models/` (6 documented issues)

**Why not Active (in main pipeline sense)**:

- Not yet integrated as a modality in the main feature pipeline
- No head-to-head benchmark against SpliceAI/OpenSpliceAI on splice tasks
- Cost-to-benefit unclear — Evo2 extraction is expensive

## Graduation signals

**To advance the `fm_embeddings` modality from commented-out to Active**
(integration into the main pipeline):

- Complete Evo2 7B full-genome extraction
- Add modality shim in `agentic_spliceai.splice_engine.features.modalities.fm_embeddings`
- Benchmark M1-S with and without FM embeddings modality on a standard test chromosome
- Document the extraction pipeline as a reproducible recipe

The sub-project itself may or may not mature into a main-pipeline
application — that decision waits for benchmark evidence.

## Known limitations

- Cloud GPU dependency (SkyPilot + RunPod) — no local path for full Evo2 7B
- FP8 requires compute 8.9+ (A40, A100 need monkey-patch)
- Evo2 40B needs A100 80GB minimum
- `rsync` symlink handling requires `.gitignore` care (use `/data` not `data/*`)
- Sub-project's own conventions differ from main pipeline — don't assume interchangeability

## Related

- [Multimodal Feature Engineering](../multimodal_features/README.md) — primary integration point if graduated
- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — incumbent base-layer for comparison
- [Foundation Models README](../../../foundation_models/README.md) — sub-project documentation
- [Roadmap: Phase 5](../../ROADMAP.md)
