# Package Organization Guide

**Date**: 2026-01-30  
**Audience**: Developers working on experimental/research features

## 📂 Current Project Structure

```
agentic-spliceai/
├── src/                          # Core production code (pip installable)
│   └── agentic_spliceai/
│       ├── __init__.py
│       └── splice_engine/
│           ├── base_layer/       # SpliceAI/OpenSpliceAI
│           ├── meta_layer/       # Multimodal DL
│           ├── prediction/       # Core prediction logic
│           └── resources/        # Resource management
│
├── examples/                     # Driver scripts (fast iteration)
│   ├── base_layer/
│   ├── data_preparation/
│   └── ...
│
├── notebooks/                    # Educational Jupyter notebooks
│
├── scripts/                      # Utilities, validation, tools
│
├── tests/                        # Unit & integration tests
│
├── dev/                          # Private development docs
│   ├── sessions/
│   ├── planning/
│   └── research/
│
├── docs/                         # Public documentation
│
└── (experimental packages)?      # ← WHERE TO PUT EXPERIMENTAL CODE?
```

---

## 🧪 Where to Put Experimental Features?

### Decision Framework

For experimental features like **Evo2 foundation model**, consider these factors:

| Factor | Put in `src/` | Put in parallel package | Put in `examples/` |
|--------|---------------|-------------------------|-------------------|
| **Stability** | Stable, tested | Experimental, evolving | Demo/prototype |
| **Integration** | Core system | Optional add-on | Quick test |
| **Dependencies** | Standard deps | Heavy deps (GPU models) | Minimal deps |
| **Audience** | All users | Research users | Developers |
| **Installation** | Always installed | Optional install | Not installed |
| **Maintenance** | Long-term | Research cycle | Temporary |

---

## 📋 Recommendation for Evo2

### ✅ Option 1: Parallel Package (RECOMMENDED)

**Structure**:
```
agentic-spliceai/
├── src/                          # Core production code
│   └── agentic_spliceai/
│
├── foundation_models/            # ← Parallel experimental package
│   ├── __init__.py
│   ├── README.md
│   ├── pyproject.toml           # Separate dependencies
│   ├── environment-evo2.yml     # GPU-specific environment
│   └── evo2/
│       ├── __init__.py
│       ├── model.py             # Evo2 integration
│       ├── inference.py         # Inference logic
│       └── adapters/
│           └── spliceai_adapter.py  # Bridge to splice_engine
│
├── examples/
│   └── foundation_models/       # Evo2 examples
│       └── 01_evo2_prediction.py
│
└── notebooks/
    └── foundation_models/       # Evo2 tutorials
        └── 01_evo2_basics.ipynb
```

**Benefits**:
- ✅ **Optional installation**: Users can install core without heavy deps
- ✅ **Separate dependencies**: Evo2 needs specific GPU libs, separate `environment-evo2.yml`
- ✅ **Clear boundaries**: Experimental code doesn't pollute production
- ✅ **Easy testing**: Can run core tests without Evo2 dependencies
- ✅ **Flexible deployment**: Deploy to pod without installing on local machine
- ✅ **Independent evolution**: Update Evo2 without touching core

**Installation**:
```bash
# Core only (local development)
pip install -e .

# Core + Evo2 (pod with A40)
pip install -e .
pip install -e ./foundation_models

# Or with conda
mamba env create -f foundation_models/environment-evo2.yml
```

**Imports** (from examples):
```python
# Add both packages to path
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'foundation_models'))

# Import core
from agentic_spliceai.splice_engine.base_layer import ...

# Import experimental
from evo2.model import Evo2SplicePredictor
from evo2.adapters import adapt_to_spliceai_format
```

---

### Option 2: Subdirectory in `src/` (If stable)

**Structure**:
```
src/
└── agentic_spliceai/
    ├── splice_engine/
    ├── experimental/             # ← Experimental features
    │   ├── __init__.py
    │   ├── foundation_models/
    │   │   ├── __init__.py
    │   │   ├── evo2/
    │   │   └── ...
    │   └── README.md
    └── ...
```

**Use when**:
- Features are **relatively stable**
- Dependencies are **manageable** (not too heavy)
- You want **unified installation**
- Features **will eventually become core**

**Drawbacks**:
- ❌ All users install experimental dependencies
- ❌ Harder to separate GPU-specific code
- ❌ More coupling with core system

---

### Option 3: Separate Git Submodule (If very independent)

**Structure**:
```
agentic-spliceai/
├── src/
├── submodules/
│   └── agentic-evo2/            # ← Separate git repo as submodule
│       ├── .git
│       ├── README.md
│       ├── pyproject.toml
│       └── evo2/
└── ...
```

**Use when**:
- Feature is **very independent** (could be its own project)
- Multiple projects might use it
- Different development team/cycle
- Want separate version control

---

## 💡 Specific Recommendation for Evo2

### Recommended Structure

**Use Option 1 (Parallel Package)** because:
1. **Heavy GPU dependencies** (need A40, separate environment)
2. **Pod-specific testing** (won't run on local Mac)
3. **Research-oriented** (may change rapidly)
4. **Optional feature** (not all users need it)

**Implementation**:

```
agentic-spliceai/
├── src/agentic_spliceai/         # Core (always installed)
│
├── foundation_models/            # Experimental (optional)
│   ├── __init__.py
│   ├── README.md                 # Installation & usage
│   ├── pyproject.toml
│   │   [project]
│   │   name = "agentic-spliceai-foundation-models"
│   │   dependencies = [
│   │       "agentic-spliceai",   # Depends on core
│   │       "evo-model>=2.0",     # Evo2 specific
│   │       "transformers>=4.30",
│   │       "torch>=2.0+cu118"
│   │   ]
│   │
│   ├── environment-evo2.yml      # Pod environment
│   │   name: agentic-spliceai-evo2
│   │   channels: [nvidia, pytorch, conda-forge]
│   │   dependencies:
│   │     - python=3.11
│   │     - pytorch::pytorch>=2.0
│   │     - pytorch::pytorch-cuda=11.8
│   │     - evo-model
│   │
│   ├── evo2/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── model.py             # Evo2ModelWrapper
│   │   ├── inference.py         # Inference pipeline
│   │   └── adapters/
│   │       └── spliceai_adapter.py
│   │
│   └── tests/
│       └── test_evo2_integration.py
│
├── examples/foundation_models/
│   ├── README.md
│   ├── 01_evo2_setup.md         # Setup on pod
│   └── 02_evo2_prediction.py    # Usage example
│
└── notebooks/foundation_models/
    └── 01_evo2_splice_prediction.ipynb
```

---

## 🔧 Integration Pattern

### Core → Experimental Bridge

**In `src/agentic_spliceai/splice_engine/base_layer/models/`**:
```python
# registry.py - Model registry with dynamic loading

_MODEL_REGISTRY = {
    'spliceai': 'agentic_spliceai.splice_engine.base_layer.prediction.core.load_spliceai',
    'openspliceai': 'agentic_spliceai.splice_engine.base_layer.prediction.core.load_openspliceai',
}

def register_foundation_model(name: str, loader_path: str):
    """Register experimental foundation model (optional)."""
    _MODEL_REGISTRY[name] = loader_path

def load_model(model_name: str, **kwargs):
    """Load model by name (core or experimental)."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    
    module_path, func_name = _MODEL_REGISTRY[model_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    loader = getattr(module, func_name)
    return loader(**kwargs)
```

**In `foundation_models/evo2/__init__.py`**:
```python
# Auto-register when foundation_models is imported
from agentic_spliceai.splice_engine.base_layer.models.registry import register_foundation_model

register_foundation_model(
    'evo2',
    'evo2.model.load_evo2_splice_model'
)
```

**Usage**:
```python
# Works with or without foundation_models installed

# Core models (always available)
runner = BaseModelRunner()
result = runner.run_single_model(model_name='openspliceai', ...)

# Experimental models (if foundation_models installed)
try:
    import evo2  # Triggers registration
    result = runner.run_single_model(model_name='evo2', ...)
except ImportError:
    print("Evo2 not available - install foundation_models package")
```

---

## 📦 Installation Workflows

### Local Development (Core Only)
```bash
cd agentic-spliceai
pip install -e .                  # Core only
python examples/base_layer/01_phase1_prediction.py
```

### Pod Development (Core + Evo2)
```bash
# On pod with A40
cd agentic-spliceai

# Create Evo2 environment
mamba env create -f foundation_models/environment-evo2.yml
conda activate agentic-spliceai-evo2

# Install both packages
pip install -e .                  # Core
pip install -e ./foundation_models  # Evo2

# Test
python examples/foundation_models/02_evo2_prediction.py
```

### CI/CD Testing
```yaml
# .github/workflows/test.yml
jobs:
  test-core:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -e .
      - run: pytest tests/
  
  test-evo2:
    runs-on: [self-hosted, gpu-a40]  # Pod runner
    steps:
      - uses: actions/checkout@v3
      - run: conda env create -f foundation_models/environment-evo2.yml
      - run: pip install -e . && pip install -e ./foundation_models
      - run: pytest foundation_models/tests/
```

---

## 📝 Summary & Decision

### For Evo2 Foundation Model: Use **Parallel Package** ✅

**Rationale**:
1. **Deployment flexibility**: Run core locally, Evo2 on pod
2. **Dependency isolation**: Heavy GPU libs separate
3. **Development speed**: Rapid iteration without core changes
4. **Optional feature**: Users can choose to install
5. **Clear boundaries**: Experimental vs production

**File to Create**:
```bash
# Create parallel package
mkdir -p foundation_models/evo2/adapters
touch foundation_models/{__init__.py,README.md,pyproject.toml}
touch foundation_models/evo2/{__init__.py,model.py,inference.py}
touch foundation_models/evo2/adapters/spliceai_adapter.py
touch foundation_models/environment-evo2.yml
```

**Import Pattern** (in examples):
```python
# examples/foundation_models/02_evo2_prediction.py

import sys
from pathlib import Path

# Add both packages
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'foundation_models'))

# Import core
from agentic_spliceai.splice_engine.base_layer import ...

# Import experimental
from evo2.model import Evo2SplicePredictor
```

---

## ✅ Implementation Status (March 2026)

**Option 1 (Parallel Package) was adopted.** The `foundation_models/` sub-project is fully operational:

```
foundation_models/
├── pyproject.toml                    # Separate dependencies
├── README.md                         # Setup + hardware requirements
├── foundation_models/
│   ├── evo2/                         # Evo2-based exon classifier
│   │   ├── config.py                 # Evo2Config (device auto-detect)
│   │   ├── model.py                  # HuggingFace wrapper
│   │   ├── embedder.py               # Chunked extraction + HDF5 cache
│   │   └── classifier.py             # ExonClassifier (linear/MLP/CNN/LSTM)
│   ├── classifiers/                  # Splice classifiers
│   │   └── splice_classifier.py      # Direct shard predictor
│   └── utils/                        # Quantization, chunking
├── configs/
│   ├── gpu_config.yaml               # Infrastructure defaults (GPU, volume, deps)
│   └── skypilot/                     # SkyPilot cloud deployment (RunPod)
└── docs/                             # Sub-project documentation
```

**Key achievements**:
- 4 classifier architectures (linear, MLP, CNN, LSTM)
- Device-aware quantization routing (INT8 for MPS/CPU, bitsandbytes for CUDA)
- SkyPilot + RunPod cloud workflows (A40/A100 GPU)
- GPU task runner with generic SkyPilot config builder + launcher
- Ops scripts for cluster provisioning, data staging, and pipeline execution
- Direct shard splice predictor for foundation model fine-tuning

**See**: [`foundation_models/README.md`](../../foundation_models/README.md) for current setup and hardware requirements

---

*This guide follows the pattern established by `genai-lab` and other research-oriented projects in your workspace.*
