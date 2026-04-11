# Meta-Layer Documentation

**Last Updated**: April 2026
**Status**: Active Development (Phase 6 — M1-S training, M2 series design)

---

## Overview

The meta-layer is a multimodal refinement system that takes frozen base
model predictions (OpenSpliceAI, SpliceAI) and improves them using
conservation, epigenetic, chromatin, junction, and RBP evidence. It
follows the universal `[L, 3]` protocol: given an L-nucleotide sequence,
output per-position (donor, acceptor, neither) scores.

Four model variants (M1-M4) address progressively harder prediction
tasks, from canonical site recalibration to perturbation-induced splice
site discovery.

---

## Methods (read in order)

| # | Document | Topic |
|---|----------|-------|
| 00 | [Model Variants M1-M4](methods/00_model_variants_m1_m4.md) | Variant definitions, architecture, results, status |
| 01 | [Label Hierarchy & Weak Supervision](methods/01_label_hierarchy_and_weak_supervision.md) | Label levels (L0-L4), weak-supervision framing, SpliceVarDB |
| 02 | [Annotation-Driven Splice Prediction](methods/02_annotation_driven_splice_prediction.md) | Annotation as latent variable, GENCODE \ MANE, tier-based confidence |
| 03 | [Virtual Transcripts & Junction Pairing](methods/03_virtual_transcripts_and_junction_pairing.md) | Representation gap, donor-acceptor pairing, Level 3.5 |
| 04 | [Data Sources & Landscape](methods/04_data_sources_and_landscape.md) | GTEx, SpliceVault, ENCODE data + ML landscape analysis |
| 05 | [M2 Variant Formulations](methods/05_m2_variant_formulations.md) | M2-S model + evaluation protocols |
| — | [Naming Convention](methods/naming_convention.md) | Model vs eval protocol definitions |

### Reading guide

- **Start here**: 00 (model overview) — defines M1-M4 and current results
- **Naming**: [naming_convention.md](methods/naming_convention.md) — models (M1-S, M2-S) vs eval protocols (Eval-MANE, etc.)
- **Understand the problem**: 01 (label hierarchy) — why this isn't standard supervised learning
- **Understand M2**: 02 (annotation strategy) + 05 (M2 protocols) — the current research frontier
- **Deeper context**: 03 (junction pairing) + 04 (data landscape) — for M3/M4 planning

---

## Architecture

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design principles |

---

## Meta-SpliceAI Archive

The predecessor project (Meta-SpliceAI) ran four experiments on
variant-level splice prediction using SpliceVarDB:

| ID | Experiment | Result |
|----|------------|--------|
| 001 | Canonical Classification | 99.1% acc, 17% variant detection |
| 002 | Paired Delta Prediction | r=0.38 |
| 003 | Binary Classification | AUC=0.61, F1=0.53 |
| 004 | Validated Delta Prediction | **r=0.41 (best)** |

Key finding: the bottleneck is label quality (binary variant-level labels),
not model capacity. This motivated the shift to annotation-tier and
junction-weighted labeling in the M2 series.

Full archive: [meta-spliceai-archive/](meta-spliceai-archive/)

---

## Related Code

| Component | Path |
|-----------|------|
| MetaSpliceModel | `src/.../meta_layer/models/meta_splice_model_v3.py` |
| DenseFeatureExtractor | `src/.../features/dense_feature_extractor.py` |
| SequenceLevelDataset | `src/.../meta_layer/data/sequence_level_dataset.py` |
| Shard packing | `src/.../meta_layer/data/shard_packing.py` |
| Training script | `examples/meta_layer/07_train_sequence_model.py` |
| XGBoost baseline (M1-P) | `examples/meta_layer/01_xgboost_baseline.py` |
| M1-P full-genome results | `examples/meta_layer/docs/m1_fullgenome_results.md` |
| Feature configs | `examples/features/configs/` |
