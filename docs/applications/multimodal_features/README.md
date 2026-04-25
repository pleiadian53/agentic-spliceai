# Multimodal Feature Engineering

**Goals served**: cross-cutting (feeds all downstream prediction applications)

**Tier**: Active

**Last updated**: 2026-04

---

## Problem

Splice prediction benefits from evidence beyond the DNA sequence:
cross-species conservation, chromatin accessibility, histone marks,
RNA-seq junction reads, RBP binding, foundation-model embeddings. Each
modality contributes orthogonal signal, but combining them requires a
consistent schema, shared coordinate alignment, and memory-bounded
streaming for genome-scale operation.

This application provides a YAML-configurable, 10-modality feature
pipeline that produces a single aligned parquet per chromosome —
consumable by any downstream predictor.

## User-facing functionality

- Generate per-position features from 10 modalities (116 columns total):
  base scores, annotation, sequence, genomic, conservation, epigenetic,
  junction, RBP eCLIP, chromatin accessibility, foundation-model
  embeddings (optional)
- Swap modality sets via YAML config profiles (default, full_stack,
  isoform_discovery, meta_m3_novel)
- Run at genome scale with memory monitoring, per-chromosome parquet
  outputs, and `--augment` for incremental modality addition
- Verify feature alignment and modality completeness post hoc

## Driving examples

- [`examples/features/01_base_score_features.py`](../../../examples/features/01_base_score_features.py) — base scores (43 cols)
- [`examples/features/02_annotation_and_genomic.py`](../../../examples/features/02_annotation_and_genomic.py) — annotation + genomic features
- [`examples/features/03_configurable_modalities.py`](../../../examples/features/03_configurable_modalities.py) — YAML-driven modality selection
- [`examples/features/04_genome_scale_workflow.py`](../../../examples/features/04_genome_scale_workflow.py) — genome-scale feature generation
- [`examples/features/05_multimodal_exploration.py`](../../../examples/features/05_multimodal_exploration.py) — exploratory multimodal analysis
- [`examples/features/06_multimodal_genome_workflow.py`](../../../examples/features/06_multimodal_genome_workflow.py) — full-stack genome workflow (canonical driver)
- [`examples/features/06a_ephemeral_genome_workflow.py`](../../../examples/features/06a_ephemeral_genome_workflow.py) — ephemeral (predict → feature → delete) variant for bounded disk
- [`examples/features/07_streaming_fm_scalars.py`](../../../examples/features/07_streaming_fm_scalars.py) — foundation model scalar streaming
- [`examples/features/config_loader.py`](../../../examples/features/config_loader.py) — YAML config loader
- [`examples/features/verify_feature_alignment.py`](../../../examples/features/verify_feature_alignment.py) — alignment verification
- [`examples/features/check_modality_completeness.py`](../../../examples/features/check_modality_completeness.py) — completeness audit

Configuration profiles:

- [`examples/features/configs/`](../../../examples/features/configs/) — 4 YAML profiles

Per-modality tutorials:

- [`examples/features/docs/`](../../../examples/features/docs/) — modality-specific guides (junction reads, RBP eCLIP, conservation, etc.)

## `src/` surface

**Library (stable):**

- `agentic_spliceai.splice_engine.features.FeaturePipeline` — modality auto-registration, protocol-based dispatch
- `agentic_spliceai.splice_engine.features.FeatureWorkflow` — genome-scale orchestration
- `agentic_spliceai.splice_engine.features.modalities.*` — 10 modality implementations
- `agentic_spliceai.splice_engine.features.verification` — position alignment checks
- `agentic_spliceai.splice_engine.features.dense_feature_extractor` — memory-bounded extraction
- `agentic_spliceai.splice_engine.utils.memory_monitor` — RSS monitoring, graceful abort

**Application package** (`src/agentic_spliceai/applications/multimodal_features/`):

- `profiles.py` — profile catalog (reads `examples/features/configs/*.yaml`)
- `tracks.py` — external-track catalog (conservation, ENCODE epigenetic) + `fetch_conservation_tracks()`
- `manifest.py` — versioned `FeatureManifest` (profile + inputs + per-chrom artifacts + hashes + tracks used)
- `pipeline.py` — `prepare_features()` orchestrator + `resolve_canonical_features_dir()` helper
- `status.py` — `FeaturePrepStatus` readiness query (per-chromosome)
- `steps.py` — thin wrappers over `FeatureWorkflow` + conservation fetch + validate
- `cli.py` — unified subcommand CLI

**CLI entry point:** `agentic-spliceai-features`

```bash
agentic-spliceai-features list-profiles
agentic-spliceai-features list-tracks --build GRCh38
agentic-spliceai-features fetch-tracks --build GRCh38                         # downloads PhyloP, PhastCons
agentic-spliceai-features status --canonical --build GRCh38                   # read-only production check
agentic-spliceai-features status --output-dir output/features/my_run          # throwaway dir check
agentic-spliceai-features prepare --profile full_stack --build GRCh38 \
    --chromosomes 22 \
    --input-dir data/mane/GRCh38/openspliceai_eval/precomputed \
    --output-dir output/features/chr22_full_stack
agentic-spliceai-features validate --output-dir data/mane/GRCh38/openspliceai_eval/analysis_sequences
```

**Production-safety guards** (same pattern as `data_preparation`):

- `prepare` requires either `--output-dir` (throwaway) or `--inplace` (canonical)
- Library-level `resume=True` is the default → per-chromosome parquets are preserved; `--no-resume` to override
- `status`, `list-profiles`, `list-tracks`, and `validate` are read-only

**External-track fetching:**

- `fetch-tracks --modality conservation --build GRCh38` — downloads PhyloP + PhastCons bigWigs from UCSC to the configured cache
- `fetch-tracks --modality epigenetic --build GRCh38 [--cell-lines K562 HepG2] [--marks h3k4me3]` — downloads ENCODE ChIP-seq fold-change bigWigs by accession. URL pattern: `https://www.encodeproject.org/files/<ENCFF...>/@@download/<ENCFF...>.bigWig`. Filters by cell line / mark when supplied; otherwise fetches the full configured panel.

**Optional experiment tracking** (silent fallback when wandb is not
installed or no API key is set):

```bash
agentic-spliceai-features prepare --profile full_stack --build GRCh38 \
    --input-dir  data/mane/GRCh38/openspliceai_eval/precomputed \
    --output-dir output/features/chr22_full_stack \
    --chromosomes 22 \
    --track --tracking-project agentic-spliceai-multimodal-features
```

Logs per-step durations, rows, success flags, and the resulting
`feature_manifest.json` as a W&B artifact. Shared with base_layer and
data_preparation via
[`applications._common.tracking`](../../../src/agentic_spliceai/applications/_common/tracking.py).

## Evaluation

- **Scale verified**: 24/24 chromosomes complete (2.88 GB total, 116 cols each)
- **Location**: `data/mane/GRCh38/openspliceai_eval/analysis_sequences/`
- **Alignment**: position-level verification via `verification.py`
- **Downstream impact**: junction_has_support = #2 feature by SHAP (31.3%), FN reduction -60/-70% (donor/acceptor) on M1-P XGBoost
- **Feature catalog**: [docs/multimodal_feature_engineering/feature_catalog.md](../../multimodal_feature_engineering/feature_catalog.md)

## Maturity tier and signals

**Current tier**: Active (moving toward Mature)

**Signals supporting the tier**:

- 10 modalities (9 active + 1 fm_embeddings commented out) with 116 columns
- Full genome completed (24/24 chromosomes)
- 4 YAML profiles covering distinct modeling objectives
- Depended on by Adaptive Splice Prediction and Variant Effect Analysis
- Memory-bounded streaming verified on 16GB MacBook (peak 2.51 GB chr1 full-stack)
- Per-modality tutorials in `examples/features/docs/`
- **Packaged application** at `src/agentic_spliceai/applications/multimodal_features/` with versioned manifest, readiness API (per-chromosome), external-track catalog, and dedicated CLI (`agentic-spliceai-features`)
- **Production-path completeness check** verified: `status --canonical --build GRCh38` correctly enumerates all 24 existing per-chromosome parquets read-only (chr22 validated: 96,467 rows, alignment columns present)

## Graduation signals

**To advance to Mature, the application needs**:

- FM embeddings modality wired back in with Evo2 full-genome extraction
- Versioned feature schema (116 columns is a frozen surface)
- Test coverage for modality registration and alignment invariants
- ENCODE-accession-driven downloads for epigenetic tracks (currently `remote_fallback` at run-time only; the `fetch-tracks` CLI currently covers conservation only)
- Full meta-layer pre-flight integration (parallel to the base-layer pre-flight)

## Known limitations

- Conservation bigWig streaming dominates runtime (~2 hrs for 562K positions)
- pyBigWig connection timeouts on laptop standby (kill and `--resume`)
- Foundation-model embedding modality disabled pending Evo2 extraction completion
- Junction modality coverage uneven across tissues (see junction coverage audit)
- RBP eCLIP data limited to ENCODE set — new RBPs require peak re-aggregation

## Related

- [Adaptive Splice Prediction](../adaptive_splice_prediction/README.md) — primary consumer
- [Variant Effect Analysis](../variant_analysis/README.md) — locus-level features used at ref and alt
- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — base-score modality source
- [Foundation Model Predictors](../foundation_model_predictors/README.md) — FM embedding modality source (experimental)
- [Feature Catalog](../../multimodal_feature_engineering/feature_catalog.md)
- [Roadmap: Phase 4](../../ROADMAP.md)
