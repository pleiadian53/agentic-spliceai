# Base Model Prediction Gap Analysis

**Objective**: Verify if agentic-spliceai can fully replicate the base model prediction functionality from meta-spliceai.

**Date**: December 2025

---

## Call Chain Analysis

### Entry Points in meta-spliceai

```
1. CLI: run_base_model (pyproject.toml entry point)
   └── meta_spliceai/cli/run_base_model_cli.py::main()
       └── meta_spliceai/run_base_model.py::run_base_model_predictions()
           └── splice_engine/meta_models/workflows/splice_prediction_workflow.py::run_enhanced_splice_prediction_workflow()

2. Shell Script: process_chromosomes_sequential_smart.sh
   └── Calls `run_base_model` CLI for each chromosome
   └── Handles checkpointing and resumption
```

### Core Workflow Dependencies

The `run_enhanced_splice_prediction_workflow()` function requires:

| Module | Source File | Purpose | Status in agentic-spliceai |
|--------|-------------|---------|---------------------------|
| `SpliceAIConfig` | `meta_models/core/data_types.py` | Configuration dataclass | ✅ Ported as `BaseModelConfig` |
| `GeneManifest` | `meta_models/core/data_types.py` | Track gene processing | ✅ Ported |
| `predict_splice_sites_for_genes` | `run_spliceai_workflow.py` | Core prediction function | ❌ **NOT PORTED** |
| `prepare_gene_annotations` | `workflows/data_preparation.py` | Load GTF annotations | ❌ **NOT PORTED** |
| `prepare_splice_site_annotations` | `workflows/data_preparation.py` | Extract splice sites | ❌ **NOT PORTED** |
| `prepare_genomic_sequences` | `workflows/data_preparation.py` | Load FASTA sequences | ❌ **NOT PORTED** |
| `load_spliceai_models` | `workflows/data_preparation.py` | Load model weights | ❌ **NOT PORTED** |
| `enhanced_process_predictions_with_all_scores` | `core/enhanced_workflow.py` | Process predictions | ❌ **NOT PORTED** |
| `extract_analysis_sequences` | `workflows/sequence_data_utils.py` | Extract sequences | ❌ **NOT PORTED** |
| `MetaModelDataHandler` | `io/handlers.py` | I/O operations | ❌ **NOT PORTED** |
| `ModelEvaluationFileHandler` | `model_evaluator.py` | File handling | ❌ **NOT PORTED** |
| `standardize_splice_sites_schema` | `system/genomic_resources/schema.py` | Schema normalization | ✅ Ported |
| `Registry` | `system/genomic_resources/registry.py` | Path resolution | ✅ Ported |

---

## What's Ported vs. Missing

### ✅ PORTED (Phase 1 Complete)

**Configuration Layer** (`splice_engine/config/`):
- `Config` dataclass for genomic resources
- `load_config()` function
- `settings.yaml` with defaults

**Resources Layer** (`splice_engine/resources/`):
- `Registry` class for path resolution
- Schema standardization functions

**Utilities** (`splice_engine/utils/`):
- DataFrame utilities
- Display/printing utilities
- Filesystem utilities

**Base Layer Types** (`splice_engine/base_layer/`):
- `BaseModelConfig` (ported from `SpliceAIConfig`)
- `BaseModelRunner` (placeholder implementation)
- `GeneManifest`, `GeneManifestEntry`
- `PredictionResult`

**Meta Layer Types** (`splice_engine/meta_layer/`):
- `MetaLayerConfig`
- `FeatureSchema`, `DEFAULT_SCHEMA`
- Label encoding/decoding

### ❌ NOT PORTED (Required for Full Functionality)

#### Priority 1: Core Prediction Pipeline

| Component | Source | Lines | Complexity |
|-----------|--------|-------|------------|
| `run_enhanced_splice_prediction_workflow` | `workflows/splice_prediction_workflow.py` | ~1200 | High |
| `predict_splice_sites_for_genes` | `run_spliceai_workflow.py` | ~500 | High |
| `enhanced_process_predictions_with_all_scores` | `core/enhanced_workflow.py` | ~400 | Medium |

#### Priority 2: Data Preparation

| Component | Source | Lines | Complexity |
|-----------|--------|-------|------------|
| `prepare_gene_annotations` | `workflows/data_preparation.py` | ~150 | Medium |
| `prepare_splice_site_annotations` | `workflows/data_preparation.py` | ~200 | Medium |
| `prepare_genomic_sequences` | `workflows/data_preparation.py` | ~150 | Medium |
| `handle_overlapping_genes` | `workflows/data_preparation.py` | ~100 | Low |

#### Priority 3: Model Loading & Inference

| Component | Source | Lines | Complexity |
|-----------|--------|-------|------------|
| `load_spliceai_models` | `workflows/data_preparation.py` | ~100 | Medium |
| OpenSpliceAI adapter | `meta_models/openspliceai_adapter/` | ~500 | High |
| SpliceAI model wrapper | `base_models/` | ~300 | Medium |

#### Priority 4: I/O & Artifact Management

| Component | Source | Lines | Complexity |
|-----------|--------|-------|------------|
| `MetaModelDataHandler` | `meta_models/io/handlers.py` | ~300 | Medium |
| `ModelEvaluationFileHandler` | `model_evaluator.py` | ~200 | Medium |
| Artifact manager | `core/data_types.py` | ~150 | Medium |

#### Priority 5: Sequence Utilities

| Component | Source | Lines | Complexity |
|-----------|--------|-------|------------|
| `load_chromosome_sequence_streaming` | `utils/sequence_utils.py` | ~100 | Medium |
| `extract_analysis_sequences` | `workflows/sequence_data_utils.py` | ~200 | Medium |
| FASTA handling | `utils_bio.py` | ~500 | Medium |

---

## Full Coverage Mode Requirements

For **full coverage mode** (every nucleotide gets 3 scores), the following is required:

### Prediction Output Format

```python
# For each gene, produce:
{
    'gene_id': str,
    'gene_length': int,
    'predictions': {
        'donor_scores': np.ndarray,      # shape: (gene_length,)
        'acceptor_scores': np.ndarray,   # shape: (gene_length,)
        'neither_scores': np.ndarray,    # shape: (gene_length,)
    },
    'positions': np.ndarray,  # 1-indexed genomic positions
}
```

### Key Functions for Full Coverage

1. **`predict_splice_sites_for_genes()`** in `run_spliceai_workflow.py`:
   - Loads model ensemble
   - Iterates through genes
   - Calls model on padded sequences
   - Returns per-nucleotide scores

2. **Model Inference**:
   - SpliceAI: TensorFlow model, outputs `(batch, seq_len, 3)`
   - OpenSpliceAI: PyTorch model, same output shape

3. **Sequence Handling**:
   - Context window: 5000nt on each side (10000nt total context)
   - Padding for genes near chromosome boundaries
   - Strand-aware reverse complement

---

## Estimated Porting Effort

| Phase | Components | Est. Lines | Est. Time |
|-------|------------|------------|-----------|
| Phase 2A | Data Preparation | ~600 | 2-3 hours |
| Phase 2B | Core Prediction | ~2000 | 4-6 hours |
| Phase 2C | I/O Handlers | ~500 | 1-2 hours |
| Phase 2D | Model Loading | ~400 | 2-3 hours |
| Phase 2E | Sequence Utils | ~800 | 2-3 hours |
| **Total** | | ~4300 | **12-17 hours** |

---

## Recommendation

### Option A: Full Port (Recommended for Production)

Port all missing components to achieve complete independence from meta-spliceai.

**Pros**:
- Clean, self-contained codebase
- Can add agentic enhancements without meta-spliceai dependency
- Better for long-term maintenance

**Cons**:
- Significant effort (~15 hours)
- Need to maintain two codebases

### Option B: Thin Wrapper (Quick Start)

Keep meta-spliceai as a dependency, wrap its functions.

**Pros**:
- Minimal effort
- Immediate functionality

**Cons**:
- Dependency on meta-spliceai
- Harder to add agentic enhancements
- Version coupling issues

### Option C: Incremental Port (Balanced)

1. Start with Option B (wrapper)
2. Port components incrementally as needed
3. Eventually achieve full independence

**Recommended**: Start with **Option C**, prioritizing:
1. Data preparation (most reusable)
2. Core prediction workflow
3. Model loading

---

## Next Steps

1. **Immediate**: Create wrapper functions that delegate to meta-spliceai
2. **Short-term**: Port data preparation module
3. **Medium-term**: Port core prediction workflow
4. **Long-term**: Port model loading and achieve full independence

---

## Files to Port (Priority Order)

```
# Phase 2A: Data Preparation
meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py

# Phase 2B: Core Prediction  
meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py
meta_spliceai/splice_engine/meta_models/core/enhanced_workflow.py
meta_spliceai/splice_engine/run_spliceai_workflow.py (partial)

# Phase 2C: I/O Handlers
meta_spliceai/splice_engine/meta_models/io/handlers.py
meta_spliceai/splice_engine/model_evaluator.py (partial)

# Phase 2D: Model Loading
meta_spliceai/splice_engine/meta_models/utils/model_utils.py
meta_spliceai/splice_engine/base_models/

# Phase 2E: Sequence Utilities
meta_spliceai/splice_engine/meta_models/utils/sequence_utils.py
meta_spliceai/splice_engine/meta_models/workflows/sequence_data_utils.py
meta_spliceai/splice_engine/utils_bio.py (partial)
```
