# Stage 5: Essential Components for Porting

**Purpose**: Identify the minimal set of modules needed to port base layer functionality  
**Date**: December 10, 2025

---

## Overview

This document traces the dependency tree starting from `run_base_model.py` to identify:
1. **Essential modules** that must be ported or imported
2. **Optional modules** that can be deferred
3. **Legacy/deprecated modules** that should be skipped

Since data is symlinked between `meta-spliceai` and `agentic-spliceai`, the focus is on **code dependencies**, not data files.

---

## 1. Entry Point Trace

### Primary Entry Point

```
meta_spliceai/run_base_model.py
    ├── run_base_model_predictions()  # Main API
    └── predict_splice_sites()        # Simplified API
```

### Direct Imports from run_base_model.py

```python
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
```

---

## 2. Essential Module Categories

### Category A: System Infrastructure (MUST PORT)

These modules provide foundational services used throughout the system.

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/system/genomic_resources/` | Path resolution, config, schema | ~16 files |
| `meta_spliceai/system/output_resources/` | Output path management | 5 files |
| `meta_spliceai/system/artifact_manager.py` | Artifact tracking, mode routing | 12KB |
| `meta_spliceai/system/config.py` | Project root, system config | 2KB |
| `meta_spliceai/system/global_constants.py` | Shared constants | 1KB |

### Category B: Core Workflow (MUST PORT)

These modules implement the main prediction workflow.

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/run_base_model.py` | User-facing API | 12KB |
| `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py` | Main workflow | 61KB |
| `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py` | Data loading | 41KB |
| `meta_spliceai/splice_engine/meta_models/workflows/sequence_data_utils.py` | Sequence extraction | 29KB |

### Category C: Core Data Types & Config (MUST PORT)

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/splice_engine/meta_models/core/data_types.py` | SpliceAIConfig, GeneManifest | 23KB |
| `meta_spliceai/splice_engine/meta_models/core/model_config.py` | OpenSpliceAIConfig, factory | 16KB |
| `meta_spliceai/splice_engine/meta_models/core/analyzer.py` | Analyzer class | 22KB |

### Category D: Evaluation & Feature Generation (MUST PORT)

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/splice_engine/meta_models/core/enhanced_workflow.py` | Feature generation | 30KB |
| `meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py` | Error analysis | 89KB |
| `meta_spliceai/splice_engine/meta_models/io/handlers.py` | Data I/O | 47KB |

### Category E: Utility Modules (MUST PORT)

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/splice_engine/meta_models/utils/sequence_utils.py` | Sequence loading | 21KB |
| `meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py` | Position adjustments | 28KB |
| `meta_spliceai/splice_engine/utils_df.py` | DataFrame utilities | 22KB |
| `meta_spliceai/splice_engine/utils_fs.py` | File system utilities | 14KB |
| `meta_spliceai/splice_engine/utils_bio.py` | Biological utilities | 82KB |
| `meta_spliceai/splice_engine/utils_doc.py` | Printing utilities | 17KB |
| `meta_spliceai/splice_engine/workflow_utils.py` | Workflow helpers | 14KB |
| `meta_spliceai/splice_engine/analysis_utils.py` | Analysis helpers | 106KB |

### Category F: Model Loading (MUST PORT)

| Module | Purpose | Size |
|--------|---------|------|
| `meta_spliceai/splice_engine/run_spliceai_workflow.py` | SpliceAI model loading, prediction | 329KB |
| `meta_spliceai/splice_engine/evaluate_models.py` | Evaluation functions | 196KB |
| `meta_spliceai/splice_engine/model_evaluator.py` | File handler | 27KB |
| `meta_spliceai/foundation_models/model_loader.py` | Model loading abstraction | ~5KB |

---

## 3. Complete Dependency Tree

```text
run_base_model.py
│
├── splice_engine/meta_models/workflows/splice_prediction_workflow.py
│   ├── splice_engine/run_spliceai_workflow.py
│   │   ├── foundation_models/model_loader.py
│   │   ├── splice_engine/utils.py
│   │   ├── splice_engine/visual_analyzer.py (OPTIONAL - visualization)
│   │   ├── splice_engine/model_evaluator.py
│   │   ├── splice_engine/utils_fs.py
│   │   ├── splice_engine/utils_df.py
│   │   ├── splice_engine/extract_genomic_features.py
│   │   ├── splice_engine/utils_bio.py
│   │   ├── splice_engine/evaluate_models.py
│   │   ├── splice_engine/performance_analyzer.py (OPTIONAL - analysis)
│   │   └── splice_engine/utils_doc.py
│   │
│   ├── splice_engine/meta_models/workflows/data_preparation.py
│   │   └── system/genomic_resources/* (entire package)
│   │
│   ├── splice_engine/meta_models/core/data_types.py
│   │   └── splice_engine/meta_models/core/analyzer.py
│   │       └── system/genomic_resources/registry.py
│   │
│   ├── splice_engine/meta_models/core/enhanced_workflow.py
│   │   └── splice_engine/meta_models/core/enhanced_evaluation.py
│   │
│   ├── splice_engine/meta_models/io/handlers.py
│   │
│   ├── splice_engine/meta_models/utils/sequence_utils.py
│   │
│   └── splice_engine/meta_models/utils/infer_splice_site_adjustments.py
│
└── system/artifact_manager.py
    └── system/config.py
```

---

## 4. Essential Files List (Detailed)

### 4.1 System Package (`meta_spliceai/system/`)

| File | Purpose | Dependencies |
|------|---------|--------------|
| `__init__.py` | Package exports | - |
| `config.py` | Project root detection | - |
| `global_constants.py` | Shared constants | - |
| `artifact_manager.py` | Artifact tracking | config.py |

### 4.2 Genomic Resources (`meta_spliceai/system/genomic_resources/`)

**Port entire package** - all files are interconnected.

| File | Purpose |
|------|---------|
| `__init__.py` | Public API (40+ exports) |
| `config.py` | Config dataclass, load_config() |
| `registry.py` | Registry class (path resolution) |
| `schema.py` | Column standardization |
| `derive.py` | GenomicDataDeriver |
| `splice_sites.py` | Splice site extraction |
| `validators.py` | Data validation |
| `gene_mapper.py` | Gene ID mapping |
| `gene_mapper_enhanced.py` | Enhanced mapping |
| `external_id_mapper.py` | MANE ↔ Ensembl mapping |
| `gene_selection.py` | Gene sampling |
| `build_naming.py` | Build name standardization |
| `download.py` | Ensembl file download |
| `cli.py` | Command-line interface |

### 4.3 Output Resources (`meta_spliceai/system/output_resources/`)

**Port entire package** - manages output paths.

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports |
| `config.py` | Output configuration |
| `manager.py` | Output path manager |
| `registry.py` | Output registry |

### 4.4 Meta Models Core (`meta_spliceai/splice_engine/meta_models/core/`)

| File | Purpose | Port? |
|------|---------|-------|
| `__init__.py` | Package exports | YES |
| `data_types.py` | SpliceAIConfig, GeneManifest | YES |
| `model_config.py` | OpenSpliceAIConfig, factory | YES |
| `analyzer.py` | Analyzer class | YES |
| `enhanced_workflow.py` | Feature generation | YES |
| `enhanced_evaluation.py` | Error analysis | YES |
| `schema_adapters.py` | Schema conversion | YES |
| `schema_utils.py` | Schema utilities | YES |
| `position_analysis.py` | Position analysis | OPTIONAL |
| `test_schema_mismatch.py` | Tests | NO |

### 4.5 Meta Models Workflows (`meta_spliceai/splice_engine/meta_models/workflows/`)

| File | Purpose | Port? |
|------|---------|-------|
| `__init__.py` | Package exports | YES |
| `splice_prediction_workflow.py` | Main workflow | YES |
| `data_preparation.py` | Data loading | YES |
| `sequence_data_utils.py` | Sequence extraction | YES |
| `splice_prediction_workflow_v1.py` | Legacy | NO |
| `chunked_meta_processor.py` | Chunked processing | OPTIONAL |
| `data_generation.py` | Data generation | OPTIONAL |
| `inference_workflow_utils.py` | Inference utils | OPTIONAL |
| `selective_*.py` | Selective processing | OPTIONAL |
| `splice_inference_workflow.py` | Inference workflow | OPTIONAL |
| `demo_*.py` | Demos | NO |

### 4.6 Meta Models I/O (`meta_spliceai/splice_engine/meta_models/io/`)

| File | Purpose | Port? |
|------|---------|-------|
| `__init__.py` | Package exports | YES |
| `handlers.py` | MetaModelDataHandler | YES |
| `datasets.py` | Dataset utilities | OPTIONAL |

### 4.7 Meta Models Utils (`meta_spliceai/splice_engine/meta_models/utils/`)

| File | Purpose | Port? |
|------|---------|-------|
| `__init__.py` | Package exports | YES |
| `sequence_utils.py` | Sequence loading | YES |
| `infer_splice_site_adjustments.py` | Position adjustments | YES |
| `chrom_utils.py` | Chromosome utilities | YES |
| `dataframe_utils.py` | DataFrame helpers | YES |
| `annotation_utils.py` | Annotation helpers | YES |
| `model_utils.py` | Model utilities | OPTIONAL |
| `score_adjustment.py` | Score adjustment | OPTIONAL |
| `feature_enrichment.py` | Feature enrichment | OPTIONAL |
| `comprehensive_evaluation.py` | Evaluation | OPTIONAL |
| `analyze_splice_adjustment.py` | Analysis | NO |
| `verify_splice_adjustment.py` | Verification | NO |

### 4.8 Splice Engine Root (`meta_spliceai/splice_engine/`)

| File | Purpose | Port? |
|------|---------|-------|
| `__init__.py` | Package exports | YES |
| `run_spliceai_workflow.py` | SpliceAI prediction | YES (partial) |
| `evaluate_models.py` | Evaluation functions | YES (partial) |
| `model_evaluator.py` | File handler | YES |
| `utils_df.py` | DataFrame utilities | YES |
| `utils_fs.py` | File system utilities | YES |
| `utils_bio.py` | Biological utilities | YES |
| `utils_doc.py` | Printing utilities | YES |
| `workflow_utils.py` | Workflow helpers | YES |
| `analysis_utils.py` | Analysis helpers | YES (partial) |
| `extract_genomic_features.py` | Feature extraction | YES (partial) |
| `visual_analyzer.py` | Visualization | OPTIONAL |
| `performance_analyzer.py` | Performance analysis | OPTIONAL |
| `demo_*.py` | Demos | NO |
| `train_*.py` | Training scripts | NO |
| `error_sequence_*.py` | Error models | NO |

### 4.9 Foundation Models (`meta_spliceai/foundation_models/`)

| File | Purpose | Port? |
|------|---------|-------|
| `model_loader.py` | Model loading abstraction | YES |

---

## 5. Optional/Deferrable Modules

### Can Skip for Minimal Port

| Module | Why Optional | When Needed |
|--------|--------------|-------------|
| `visual_analyzer.py` | Visualization only | When plotting results |
| `performance_analyzer.py` | Analysis only | When analyzing performance |
| `splice_engine/meta_layer/` | Meta-model training | When training meta-models |
| `splice_engine/error_model/` | Error modeling | When training error models |
| `splice_engine/feature_importance/` | Feature analysis | When analyzing features |
| `splice_engine/model_training/` | Model training | When training models |
| `splice_engine/openspliceai_recalibration/` | Recalibration | When recalibrating |

### Legacy/Deprecated (DO NOT PORT)

| Module | Why Skip |
|--------|----------|
| `run_spliceai_workflow_v0.py` | Old version |
| `run_spliceai_workflow-2.py` | Backup/old version |
| `splice_prediction_workflow_v1.py` | Old version |
| `demo_*.py` files | Demo scripts |
| `train_*.py` files | Training scripts (separate concern) |
| `*_dist.py` files | Distributed versions |
| `case_studies/` | Case study notebooks |

---

## 6. Porting Strategy

### Option A: Import from meta-spliceai (Recommended for Now)

Since both projects share the same data via symlink, you can import directly:

```python
# In agentic_spliceai/splice_engine/api.py
from meta_spliceai.run_base_model import run_base_model_predictions, BaseModelConfig
from meta_spliceai.system.genomic_resources import Registry, load_config
```

**Pros**: No code duplication, automatic updates  
**Cons**: Tight coupling, requires meta-spliceai installation

### Option B: Copy Essential Modules

Copy the essential modules to `agentic_spliceai/splice_engine/`:

```text
agentic_spliceai/splice_engine/
├── system/
│   ├── genomic_resources/    # Copy entire package
│   ├── output_resources/     # Copy entire package
│   ├── artifact_manager.py
│   └── config.py
├── meta_models/
│   ├── core/
│   ├── workflows/
│   ├── io/
│   └── utils/
├── utils/
│   ├── utils_df.py
│   ├── utils_fs.py
│   ├── utils_bio.py
│   └── utils_doc.py
└── run_base_model.py
```

**Pros**: Independent, can modify freely  
**Cons**: Code duplication, manual sync needed

### Option C: Hybrid Approach (Recommended Long-term)

1. **Import system infrastructure** from meta-spliceai (genomic_resources, etc.)
2. **Port and simplify** the workflow modules
3. **Create thin wrappers** in agentic-spliceai

```python
# agentic_spliceai/splice_engine/api.py
from meta_spliceai.system.genomic_resources import Registry, load_config
from meta_spliceai.run_base_model import run_base_model_predictions

def predict_splice_sites(genes, **kwargs):
    """Agentic-SpliceAI wrapper for base model predictions."""
    return run_base_model_predictions(target_genes=genes, **kwargs)
```

---

## 7. Summary: Minimal Essential Set

### Must Have (Core Functionality)

```text
meta_spliceai/
├── run_base_model.py                                    # Entry point
├── system/
│   ├── genomic_resources/                               # Entire package (~16 files)
│   ├── output_resources/                                # Entire package (~5 files)
│   ├── artifact_manager.py
│   └── config.py
├── foundation_models/
│   └── model_loader.py
└── splice_engine/
    ├── meta_models/
    │   ├── core/
    │   │   ├── data_types.py
    │   │   ├── model_config.py
    │   │   ├── analyzer.py
    │   │   ├── enhanced_workflow.py
    │   │   ├── enhanced_evaluation.py
    │   │   └── schema_*.py
    │   ├── workflows/
    │   │   ├── splice_prediction_workflow.py
    │   │   ├── data_preparation.py
    │   │   └── sequence_data_utils.py
    │   ├── io/
    │   │   └── handlers.py
    │   └── utils/
    │       ├── sequence_utils.py
    │       ├── infer_splice_site_adjustments.py
    │       └── chrom_utils.py
    ├── run_spliceai_workflow.py (partial - prediction functions)
    ├── evaluate_models.py (partial - evaluation functions)
    ├── model_evaluator.py
    ├── extract_genomic_features.py (partial)
    ├── utils_df.py
    ├── utils_fs.py
    ├── utils_bio.py
    ├── utils_doc.py
    └── workflow_utils.py
```

### File Count Summary

| Category | Files | Approx. Size |
|----------|-------|--------------|
| System infrastructure | ~25 files | ~100KB |
| Core workflow | ~10 files | ~250KB |
| Utilities | ~10 files | ~300KB |
| **Total Essential** | **~45 files** | **~650KB** |

---

## 8. Next Steps

1. **Decide on porting strategy** (Import vs Copy vs Hybrid)
2. **Create agentic-spliceai wrapper API** in `splice_engine/api.py`
3. **Test basic functionality** with a simple gene prediction
4. **Iterate** to add optional features as needed

---

**Next**: Stage 6 - Create Minimal Port
