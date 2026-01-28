# Stage 6: Clean Port Design (Revised)

**Purpose**: Design a clean, well-organized module structure for porting essential functionality  
**Date**: December 11, 2025 (Revised)

---

## Design Goals

1. **Clean organization** - Logical module grouping, no legacy cruft
2. **Minimal dependencies** - Only port what's needed
3. **Better naming** - Clear, descriptive module names
4. **Maintainable** - Easy to understand and extend
5. **Independent** - No dependency on meta-spliceai after porting
6. **Clear layer separation** - Distinct `base_layer` and `meta_layer` packages

---

## Architecture Overview

The system has two distinct layers:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC-SPLICEAI                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         META LAYER                                   │  │
│   │   (Multimodal deep learning for alternative splice site prediction) │  │
│   │   - Captures variant-induced splice changes                         │  │
│   │   - Disease, stress, epigenetic effects                             │  │
│   │   - Uses base layer predictions as input features                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ Predictions + Features                 │
│                                    │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         BASE LAYER                                   │  │
│   │   (Foundation splice site prediction using pre-trained models)      │  │
│   │   - SpliceAI (GRCh37/Ensembl)                                       │  │
│   │   - OpenSpliceAI (GRCh38/MANE)                                      │  │
│   │   - Extensible for new base models                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                    ▲                                        │
│                                    │ Genomic Data                           │
│                                    │                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      SHARED INFRASTRUCTURE                           │  │
│   │   config/, resources/, utils/                                        │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Directory Structure

```text
agentic_spliceai/splice_engine/
├── __init__.py                    # Public API exports
├── api.py                         # User-facing API (exists)
├── cli.py                         # CLI interface (exists)
├── README.md                      # Documentation (exists)
│
├── config/                        # Configuration management
│   ├── __init__.py
│   ├── base_config.py             # BaseModelConfig, SpliceAIConfig, OpenSpliceAIConfig
│   ├── genomic_config.py          # Genomic resources config
│   └── settings.yaml              # YAML configuration file
│
├── resources/                     # Resource management
│   ├── __init__.py
│   ├── registry.py                # Path resolution
│   ├── schema.py                  # Schema standardization
│   ├── validators.py              # Data validation
│   ├── gene_mapper.py             # Gene ID mapping
│   └── artifact_manager.py        # Artifact tracking
│
├── utils/                         # Shared utility functions
│   ├── __init__.py
│   ├── dataframe.py               # DataFrame utilities
│   ├── filesystem.py              # File system utilities
│   ├── bio.py                     # Biological utilities
│   └── display.py                 # Printing/display utilities
│
├── base_layer/                    # BASE LAYER: Foundation predictions
│   ├── __init__.py
│   ├── data/                      # Data loading and preparation
│   │   ├── __init__.py
│   │   ├── preparation.py         # Data preparation functions
│   │   ├── sequences.py           # Sequence loading/extraction
│   │   └── annotations.py         # Annotation loading
│   │
│   ├── models/                    # Base model implementations
│   │   ├── __init__.py
│   │   ├── loader.py              # Model loading abstraction
│   │   ├── spliceai.py            # SpliceAI-specific code
│   │   └── openspliceai.py        # OpenSpliceAI-specific code
│   │
│   ├── prediction/                # Prediction workflow
│   │   ├── __init__.py
│   │   ├── workflow.py            # Main prediction workflow
│   │   ├── evaluation.py          # Evaluation functions
│   │   └── features.py            # Feature generation
│   │
│   └── io/                        # Input/Output handling
│       ├── __init__.py
│       └── handlers.py            # File handlers
│
├── meta_layer/                    # META LAYER: Multimodal deep learning
│   ├── __init__.py
│   ├── core/                      # Core components
│   │   ├── __init__.py
│   │   ├── config.py              # Meta layer config
│   │   ├── artifact_loader.py     # Load base layer artifacts
│   │   └── feature_schema.py      # Feature definitions
│   │
│   ├── models/                    # Neural network models
│   │   ├── __init__.py
│   │   ├── meta_splice_model.py   # Main multimodal model
│   │   ├── score_encoder.py       # Encode base layer scores
│   │   └── sequence_encoder.py    # Encode sequences
│   │
│   ├── inference/                 # Inference pipeline
│   │   ├── __init__.py
│   │   ├── predictor.py           # Prediction interface
│   │   └── full_coverage.py       # Full genome inference
│   │
│   ├── training/                  # Training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py             # Training logic
│   │
│   └── configs/                   # Model configurations
│       └── default.yaml
│
└── docs/                          # Documentation (exists)
    └── ...
```

---

## Module Mapping: meta-spliceai → agentic-spliceai

### Shared Infrastructure

#### Configuration (`config/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/meta_models/core/data_types.py` | `config/base_config.py` | SpliceAIConfig, GeneManifest |
| `splice_engine/meta_models/core/model_config.py` | `config/base_config.py` | OpenSpliceAIConfig, factory |
| `system/genomic_resources/config.py` | `config/genomic_config.py` | Config dataclass, load_config |
| `configs/genomic_resources.yaml` | `config/settings.yaml` | YAML config |

#### Resources (`resources/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `system/genomic_resources/registry.py` | `resources/registry.py` | Registry class |
| `system/genomic_resources/schema.py` | `resources/schema.py` | Schema standardization |
| `system/genomic_resources/validators.py` | `resources/validators.py` | Validation |
| `system/genomic_resources/gene_mapper.py` | `resources/gene_mapper.py` | Gene mapping |
| `system/genomic_resources/gene_mapper_enhanced.py` | `resources/gene_mapper.py` | Merge into one |
| `system/artifact_manager.py` | `resources/artifact_manager.py` | Artifact tracking |

#### Utils (`utils/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/utils_df.py` | `utils/dataframe.py` | DataFrame utils |
| `splice_engine/utils_fs.py` | `utils/filesystem.py` | File system utils |
| `splice_engine/utils_bio.py` | `utils/bio.py` | Bio utils |
| `splice_engine/utils_doc.py` | `utils/display.py` | Display utils |
| `splice_engine/workflow_utils.py` | `utils/dataframe.py` | Merge into df utils |

### Base Layer (`base_layer/`)

#### Data (`base_layer/data/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/meta_models/workflows/data_preparation.py` | `base_layer/data/preparation.py` | Data prep functions |
| `splice_engine/meta_models/workflows/sequence_data_utils.py` | `base_layer/data/sequences.py` | Sequence extraction |
| `splice_engine/meta_models/utils/sequence_utils.py` | `base_layer/data/sequences.py` | Merge sequence utils |
| `splice_engine/extract_genomic_features.py` | `base_layer/data/annotations.py` | Annotation extraction |

#### Models (`base_layer/models/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `foundation_models/model_loader.py` | `base_layer/models/loader.py` | Model loading abstraction |
| `splice_engine/run_spliceai_workflow.py` (partial) | `base_layer/models/spliceai.py` | SpliceAI prediction |
| OpenSpliceAI adapter code | `base_layer/models/openspliceai.py` | OpenSpliceAI prediction |

#### Prediction (`base_layer/prediction/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/meta_models/workflows/splice_prediction_workflow.py` | `base_layer/prediction/workflow.py` | Main workflow |
| `splice_engine/meta_models/core/enhanced_evaluation.py` | `base_layer/prediction/evaluation.py` | Evaluation |
| `splice_engine/meta_models/core/enhanced_workflow.py` | `base_layer/prediction/features.py` | Feature generation |
| `splice_engine/evaluate_models.py` | `base_layer/prediction/evaluation.py` | Merge evaluation |

#### I/O (`base_layer/io/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/meta_models/io/handlers.py` | `base_layer/io/handlers.py` | Data handlers |
| `splice_engine/model_evaluator.py` | `base_layer/io/handlers.py` | Merge handlers |

### Meta Layer (`meta_layer/`)

| Source (meta-spliceai) | Target (agentic-spliceai) | Notes |
|------------------------|---------------------------|-------|
| `splice_engine/meta_layer/core/config.py` | `meta_layer/core/config.py` | Meta layer config |
| `splice_engine/meta_layer/core/artifact_loader.py` | `meta_layer/core/artifact_loader.py` | Load base artifacts |
| `splice_engine/meta_layer/core/feature_schema.py` | `meta_layer/core/feature_schema.py` | Feature definitions |
| `splice_engine/meta_layer/models/meta_splice_model.py` | `meta_layer/models/meta_splice_model.py` | Main model |
| `splice_engine/meta_layer/models/score_encoder.py` | `meta_layer/models/score_encoder.py` | Score encoder |
| `splice_engine/meta_layer/models/sequence_encoder.py` | `meta_layer/models/sequence_encoder.py` | Sequence encoder |
| `splice_engine/meta_layer/inference/predictor.py` | `meta_layer/inference/predictor.py` | Prediction |
| `splice_engine/meta_layer/inference/full_coverage_inference.py` | `meta_layer/inference/full_coverage.py` | Full genome |
| `splice_engine/meta_layer/training/` | `meta_layer/training/` | Training pipeline |

---

## Porting Priority

### Phase 1: Shared Infrastructure

1. **`config/`** - Configuration classes
2. **`resources/`** - Registry, schema, validators
3. **`utils/`** - Utility functions

### Phase 2: Base Layer

4. **`base_layer/data/`** - Data loading and preparation
5. **`base_layer/models/`** - SpliceAI and OpenSpliceAI model loading
6. **`base_layer/prediction/`** - Workflow, evaluation, features
7. **`base_layer/io/`** - File handlers

### Phase 3: Meta Layer

8. **`meta_layer/core/`** - Config, artifact loader, feature schema
9. **`meta_layer/models/`** - Neural network models
10. **`meta_layer/inference/`** - Prediction pipeline
11. **`meta_layer/training/`** - Training pipeline

### Phase 4: Integration

12. **Update `api.py`** - Wire up base_layer and meta_layer
13. **Update `cli.py`** - Wire up CLI
14. **Testing** - Verify functionality

---

## Key Design Decisions

### 1. Flatten the Hierarchy

**Before (meta-spliceai)**:
```
system/genomic_resources/registry.py
splice_engine/meta_models/core/data_types.py
splice_engine/meta_models/workflows/splice_prediction_workflow.py
```

**After (agentic-spliceai)**:
```
resources/registry.py
config/base_config.py
prediction/workflow.py
```

### 2. Consolidate Related Modules

**Before**: Multiple scattered utility files
```
utils_df.py, utils_fs.py, utils_bio.py, utils_doc.py, workflow_utils.py
```

**After**: Organized utils package
```
utils/dataframe.py, utils/filesystem.py, utils/bio.py, utils/display.py
```

### 3. Clear Separation of Concerns

| Package | Responsibility |
|---------|----------------|
| `config/` | Configuration only |
| `resources/` | Resource discovery and validation |
| `data/` | Data loading and preparation |
| `prediction/` | Prediction workflow |
| `models/` | Model loading and inference |
| `io/` | File I/O |
| `utils/` | Shared utilities |

### 4. Remove Legacy Code

**Skip entirely**:
- `*_v0.py`, `*_v1.py` versions
- `demo_*.py` scripts
- `train_*.py` scripts
- Visualization code (can add later)
- Performance analysis (can add later)

---

## Public API Design

### Top-Level Exports (`__init__.py`)

```python
# Configuration
from .config import BaseModelConfig, SpliceAIConfig, OpenSpliceAIConfig

# Main API
from .api import (
    predict_splice_sites,
    run_base_model_predictions,
    get_splice_predictions
)

# Resources
from .resources import Registry, load_config

# Version
__version__ = "0.1.0"
```

### User-Facing API (`api.py`)

```python
def predict_splice_sites(
    genes: Union[str, List[str]],
    base_model: str = 'spliceai',
    **kwargs
) -> pl.DataFrame:
    """Quick prediction for genes."""
    ...

def run_base_model_predictions(
    base_model: str = 'spliceai',
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    config: Optional[BaseModelConfig] = None,
    **kwargs
) -> Dict[str, Any]:
    """Full prediction workflow with all options."""
    ...

def get_splice_predictions(
    gene_id: str,
    position: int,
    window: int = 5000
) -> Dict[str, Any]:
    """Get predictions for a specific genomic region."""
    ...
```

---

## File Size Estimates

| Package | Estimated Size | Files |
|---------|----------------|-------|
| `config/` | ~40KB | 3 |
| `resources/` | ~80KB | 5 |
| `data/` | ~100KB | 4 |
| `prediction/` | ~150KB | 3 |
| `models/` | ~50KB | 3 |
| `io/` | ~50KB | 2 |
| `utils/` | ~100KB | 4 |
| **Total** | **~570KB** | **24** |

This is significantly smaller than the ~650KB from meta-spliceai because we're:
1. Removing legacy code
2. Consolidating related modules
3. Skipping optional features

---

## Next Steps

1. **Create directory structure** - Set up empty packages
2. **Port Phase 1** - config/, resources/, utils/
3. **Port Phase 2** - data/, io/
4. **Port Phase 3** - models/, prediction/
5. **Wire up API** - Update api.py and cli.py
6. **Test** - Verify with simple predictions

---

## Configuration Decisions (Confirmed)

### 1. Config Location
- `settings.yaml` lives in `splice_engine/config/`
- This is specific to splice engine data configuration

### 2. Data Root
- Default: `data/` at project root (symlinked)
- Configurable via `settings.yaml` or environment variable
- Pattern: `<data_root>/<annotation_source>/<build>/`

### 3. Model Weights Location
- Pattern: `<data_root>/models/<base_model>/`
- SpliceAI: `data/models/spliceai/`
- OpenSpliceAI: `data/models/openspliceai/`
- New models: `data/models/<model_name>/`

### 4. Scope
- Port both SpliceAI and OpenSpliceAI
- System must support multiple base models
- Focus is on meta layer (multimodal deep learning)

---

## Import Path Updates

When porting, imports must be updated from meta-spliceai paths to agentic-spliceai paths:

### Before (meta-spliceai)

```python
from meta_spliceai.system.genomic_resources import Registry, load_config
from meta_spliceai.splice_engine.meta_models.core.data_types import SpliceAIConfig
from meta_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow import (
    run_enhanced_splice_prediction_workflow
)
from meta_spliceai.splice_engine.meta_layer.inference import MetaSplicePredictor
```

### After (agentic-spliceai)

```python
from agentic_spliceai.splice_engine.resources import Registry, load_config
from agentic_spliceai.splice_engine.config import SpliceAIConfig
from agentic_spliceai.splice_engine.base_layer.prediction import (
    run_enhanced_splice_prediction_workflow
)
from agentic_spliceai.splice_engine.meta_layer.inference import MetaSplicePredictor
```

### Import Mapping Summary

| Old Import Path | New Import Path |
|-----------------|-----------------|
| `meta_spliceai.system.genomic_resources` | `agentic_spliceai.splice_engine.resources` |
| `meta_spliceai.system.artifact_manager` | `agentic_spliceai.splice_engine.resources` |
| `meta_spliceai.splice_engine.meta_models.core.data_types` | `agentic_spliceai.splice_engine.config` |
| `meta_spliceai.splice_engine.meta_models.workflows` | `agentic_spliceai.splice_engine.base_layer.prediction` |
| `meta_spliceai.splice_engine.meta_models.io` | `agentic_spliceai.splice_engine.base_layer.io` |
| `meta_spliceai.splice_engine.utils_*` | `agentic_spliceai.splice_engine.utils.*` |
| `meta_spliceai.splice_engine.meta_layer` | `agentic_spliceai.splice_engine.meta_layer` |

---

## File Count Summary

| Package | Files | Approx. Size |
|---------|-------|--------------|
| `config/` | 4 | ~40KB |
| `resources/` | 6 | ~80KB |
| `utils/` | 5 | ~100KB |
| `base_layer/` | 12 | ~300KB |
| `meta_layer/` | 12 | ~80KB |
| **Total** | **~39 files** | **~600KB** |

---

## Next Steps

1. **Create directory structure** - Set up packages with `__init__.py`
2. **Port Phase 1** - config/, resources/, utils/
3. **Port Phase 2** - base_layer/
4. **Port Phase 3** - meta_layer/
5. **Wire up API** - Update api.py and cli.py
6. **Test** - Verify with simple predictions

---

**Ready to proceed with creating the directory structure.**
