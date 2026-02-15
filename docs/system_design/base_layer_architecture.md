# Base Layer Architecture

**Purpose**: Foundation for splice site prediction with extensible base model support  
**Component**: `splice_engine/base_layer/`  
**Status**: 🚧 Design complete, refactoring in progress

---

## Overview

The base layer provides an **abstract interface** for splice site prediction models, allowing:
- Multiple base models (SpliceAI, OpenSpliceAI, future models)
- Standardized prediction workflow
- Consistent data formats
- Coordinate system handling

---

## Current Status

This is an active refactoring area. For complete details, see:

📋 **Primary Documentation**:
- [dev/refactoring/BASE_LAYER_REFACTORING.md](../../dev/refactoring/BASE_LAYER_REFACTORING.md)
- [dev/refactoring/CRITICAL_BUG_FIX_COORDINATE_ADJUSTMENT.md](../../dev/refactoring/CRITICAL_BUG_FIX_COORDINATE_ADJUSTMENT.md)

---

## Key Concepts

### 1. Abstract Base Model Interface

All base models implement a common interface:

```python
class BaseModel(ABC):
    """Abstract interface for base prediction models."""
    
    @abstractmethod
    def predict(
        self,
        sequence: str,
        gene_annotations: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """Predict splice sites for a sequence."""
        pass
```

### 2. Coordinate System Conventions

**CRITICAL**: Base models have different coordinate conventions:
- **SpliceAI**: 0-based, predictions at position
- **OpenSpliceAI**: 0-based, predictions offset by +1

See [CRITICAL_BUG_FIX_COORDINATE_ADJUSTMENT.md](../../dev/refactoring/CRITICAL_BUG_FIX_COORDINATE_ADJUSTMENT.md) for details.

### 3. Prediction Pipeline

Standardized workflow:
1. Load genome and annotations
2. Prepare analysis sequences
3. Run base model predictions
4. Adjust coordinates (model-specific)
5. Detect peaks
6. Save artifacts

---

## Architecture Components

### 1. Base Model Abstraction

```python
# Concrete implementations
class SpliceAIModel(BaseModel):
    """SpliceAI base model wrapper."""
    coordinate_offset = 0
    
class OpenSpliceAIModel(BaseModel):
    """OpenSpliceAI base model wrapper."""
    coordinate_offset = 1
```

### 2. Data Preparation

```python
# Prepare analysis sequences
from splice_engine.base_layer.data_prep import prepare_analysis_sequences

sequences = prepare_analysis_sequences(
    genes=genes,
    genome=genome,
    context_length=10000
)
```

### 3. Prediction Workflow

```python
# Run predictions
from splice_engine.base_layer.workflow import run_base_predictions

results = run_base_predictions(
    base_model="spliceai",
    chromosome="chr1",
    genes=genes,
    genome=genome
)
```

---

## Refactoring Progress

| Component | Status | Notes |
|-----------|--------|-------|
| Base model interface | ✅ Designed | Abstract class defined |
| SpliceAI wrapper | ⚠️ In progress | Coordinate adjustment needed |
| OpenSpliceAI wrapper | ⚠️ In progress | Coordinate adjustment needed |
| Data preparation | ⚠️ Partial | Core functions ported |
| Prediction workflow | 📝 Planned | Not yet implemented |
| Coordinate adjustment | ✅ Core done | Integration pending |

---

## Next Steps

1. ✅ Complete coordinate adjustment integration
2. 🚧 Refactor prediction workflow
3. 📝 Implement chunking and checkpointing
4. 📝 Add comprehensive testing

---

## Related Documentation

- [dev/refactoring/BASE_LAYER_REFACTORING.md](../../dev/refactoring/BASE_LAYER_REFACTORING.md) - Detailed refactoring plan
- [dev/refactoring/COORDINATE_CONVENTIONS.md](../../dev/base_layer/coordinate_systems/COORDINATE_CONVENTIONS.md) - Coordinate system details

---

**Status**: 🚧 Active development area  
**Last Updated**: February 15, 2026  
**Next Review**: After Phase 1.5 completion
