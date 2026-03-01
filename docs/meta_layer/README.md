# Meta-Layer Documentation

**Last Updated**: December 2025  
**Status**: Active Development

---

## Overview

This directory contains documentation for the **meta-learning layer** that recalibrates base model splice site predictions to improve variant effect detection.

---

## Document Index

### Core Architecture

| Document | Description | Status |
|----------|-------------|--------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design | ✅ |
| [LABELING_STRATEGY.md](LABELING_STRATEGY.md) | Label derivation from SpliceVarDB | ✅ |
| [DATA_FORMAT_AND_LEAKAGE.md](DATA_FORMAT_AND_LEAKAGE.md) | Data format and avoiding leakage | ✅ |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | Step-by-step training instructions | ✅ |

### Methodology

| Document | Description | Status |
|----------|-------------|--------|
| [methods/README.md](methods/README.md) | Method taxonomy overview | ✅ |
| [methods/ROADMAP.md](methods/ROADMAP.md) | Development roadmap | ✅ |
| [methods/PAIRED_DELTA_PREDICTION.md](methods/PAIRED_DELTA_PREDICTION.md) | Siamese/paired prediction | ✅ |
| [methods/VALIDATED_DELTA_PREDICTION.md](methods/VALIDATED_DELTA_PREDICTION.md) | Single-pass with ground truth (BEST) | ✅ |
| [MULTI_STEP_FRAMEWORK.md](MULTI_STEP_FRAMEWORK.md) | Decomposed classification | ✅ |

### Experiments

| ID | Experiment | Outcome | Details |
|----|------------|---------|---------|
| 001 | Canonical Classification | Partial Success | [docs](experiments/001_canonical_classification/) |
| 002 | Paired Delta Prediction | r=0.38 | [docs](experiments/002_delta_prediction/) |
| 003 | Binary Classification | AUC=0.61 | [docs](experiments/003_binary_classification/) |
| 004 | **Validated Delta (BEST)** | **r=0.41** | [docs](experiments/004_validated_delta/) |

---

## Quick Links

### Getting Started
1. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system overview
2. Review [methods/ROADMAP.md](methods/ROADMAP.md) for methodology context
3. Check [experiments/004_validated_delta/](experiments/004_validated_delta/) for best approach

### For Training
1. Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
2. Use [methods/VALIDATED_DELTA_PREDICTION.md](methods/VALIDATED_DELTA_PREDICTION.md) approach

### For Understanding Results
1. Check [experiments/README.md](experiments/README.md) for experiment index
2. Review individual experiment docs for details

---

## Package Documentation vs Project Documentation

This documentation is **package-level** (inside `src/agentic_spliceai/docs/`), focusing on:
- Implementation details and R&D insights
- Experiment logs and methodology development
- Technical deep-dives

For **project-level** documentation (user-facing guides), see:
- `/docs/` - High-level project documentation
- `/README.md` - Project overview

---

## Related Code

| Package | Path | Description |
|---------|------|-------------|
| `meta_layer.core` | `splice_engine/meta_layer/core/` | Configuration, artifact loading |
| `meta_layer.models` | `splice_engine/meta_layer/models/` | Model implementations |
| `meta_layer.training` | `splice_engine/meta_layer/training/` | Training pipelines |
| `meta_layer.workflows` | `splice_engine/meta_layer/workflows/` | High-level workflows |

---

## Naming Conventions

We avoid cryptic names like "Phase 1", "Approach A", "Approach B" in favor of descriptive names:

| Old Name | New Name | Rationale |
|----------|----------|-----------|
| Phase 1 | Canonical Classification | Describes training data |
| Approach A | Paired Delta Prediction | Describes input format |
| Approach B | Validated Delta Prediction | Describes target source |
| Phase 1 Workflow | CanonicalTrainingWorkflow | Self-explanatory |

---

*Ported from meta_spliceai with improved naming conventions.*












