# Feature Engineering Examples

**Package**: `src/agentic_spliceai/splice_engine/features/`
**Modalities**: 9 registered modalities producing 100 feature columns per genomic position
**Config**: YAML-driven workflow with 4 profiles (default, full_stack, isoform_discovery, meta_m3_novel)

---

## Learning Path

**New to multimodal features?**
1. Start with `01_base_score_features.py` — 43 engineered features from base model predictions
2. Run `02_annotation_and_genomic.py` — add ground truth labels and positional features

**Adding external data modalities:**
3. Use `03_configurable_modalities.py` — conservation + epigenetic marks via bigWig streaming
4. Explore `05_multimodal_exploration.py` — interactive feature analysis

**Genome-scale feature generation (main pipeline):**
5. Use `06_multimodal_genome_workflow.py` — YAML-driven, all 9 modalities, per-chromosome parquet output
6. Use `06a_ephemeral_genome_workflow.py` — memory-bounded variant (predict → featurize → delete per chromosome)

**Verification:**
7. Run `verify_feature_alignment.py` — validate position alignment across all modalities

---

## Available Examples

### 01_base_score_features.py
**Base Score Feature Engineering (43 columns)**

Derives engineered features from raw splice site probabilities: context scores, gradients, peak detection, cross-type comparisons, entropy.

```bash
python examples/features/01_base_score_features.py --chromosomes chr22
```

---

### 02_annotation_and_genomic.py
**Annotation + Genomic Context (7 additional columns)**

Adds ground truth splice labels from GTF and positional features (relative gene position, GC content).

---

### 03_configurable_modalities.py
**Configurable Modality Stack**

Demonstrates the `FeaturePipeline` registry — enable/disable modalities, customize configs per modality.

---

### 05_multimodal_exploration.py
**Interactive Multimodal Feature Analysis**

Explore feature distributions, correlations, and modality contributions across splice sites.

---

### 06_multimodal_genome_workflow.py
**YAML-Driven Genome-Scale Feature Engineering (Main Pipeline)**

The primary workflow script. Loads a YAML config, runs predictions, samples positions, and applies all enabled modalities. Supports `--augment` for incremental modality addition to existing artifacts.

```bash
# Full-stack features for chr22
python examples/features/06_multimodal_genome_workflow.py \
    --config examples/features/configs/full_stack.yaml \
    --chromosomes chr22

# Augment existing artifacts with new modalities
python examples/features/06_multimodal_genome_workflow.py \
    --config examples/features/configs/full_stack.yaml \
    --chromosomes chr22 --augment
```

**Key flags**: `--config`, `--chromosomes`, `--augment`, `--resume`, `--memory-limit`

---

### 06a_ephemeral_genome_workflow.py
**Ephemeral Genome-Scale Workflow (Memory-Bounded)**

Interleaved predict → featurize → delete cycle per chromosome. Bounded disk usage — only one chromosome's predictions exist at a time.

```bash
python examples/features/06a_ephemeral_genome_workflow.py \
    --config examples/features/configs/full_stack.yaml \
    --chromosomes chr22 --ephemeral
```

---

### verify_feature_alignment.py
**Position Alignment Verification**

Validates that all modality features are correctly aligned to the same genomic positions. Checks schema completeness, label-score consistency, null patterns, and value ranges.

```bash
python examples/features/verify_feature_alignment.py --chromosomes chr22 --verbose
```

---

## YAML Configs

Located in `examples/features/configs/`:

| Config | Modalities | Columns | Use Case |
|--------|-----------|---------|----------|
| `default.yaml` | 3 (base_scores, annotation, genomic) | ~50 | Quick exploration |
| `full_stack.yaml` | 9 (all modalities) | 100 | M1/M2 meta-layer training |
| `isoform_discovery.yaml` | 9 (lower thresholds) | 100 | M2 alternative site detection |
| `meta_m3_novel.yaml` | 7 (excludes junction) | ~82 | M3 novel site prediction (junction as target) |

---

## Modality Reference

| # | Modality | Columns | Data Source | Tutorial |
|---|----------|---------|-------------|----------|
| 1 | `base_scores` | 43 | Base model predictions | — |
| 2 | `annotation` | 3 | GTF annotations | — |
| 3 | `sequence` | 3 | Reference FASTA | — |
| 4 | `genomic` | 4 | Gene coordinates + sequence | — |
| 5 | `conservation` | 9 | UCSC PhyloP/PhastCons bigWig | — |
| 6 | `epigenetic` | 12 | ENCODE H3K36me3/H3K4me3 ChIP-seq | [Tutorial](docs/epigenetic-marks-tutorial.md) |
| 7 | `junction` | 12 | GTEx v8 RNA-seq (353K junctions, 54 tissues) | — |
| 8 | `rbp_eclip` | 8 | ENCODE eCLIP (K562, HepG2) | [Tutorial](docs/rbp-eclip-tutorial.md) |
| 9 | `chrom_access` | 6 | ENCODE ATAC-seq (5 cell lines) | [Tutorial](docs/chromatin-accessibility-tutorial.md) |

**Full feature catalog**: [`docs/multimodal_feature_engineering/feature_catalog.md`](../../docs/multimodal_feature_engineering/feature_catalog.md)

---

## Related Documentation

- [Feature Catalog](../../docs/multimodal_feature_engineering/feature_catalog.md) — complete 100-column reference
- [M1-M4 Model Variants](../meta_layer/docs/meta_model_variants_m1_m4.md) — how features feed into meta-layer training
- [Roadmap](../../docs/ROADMAP.md) — Phase 4 (feature engineering) and Phase 6 (meta-layer training)
