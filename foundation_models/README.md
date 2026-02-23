# Foundation Models for Agentic-SpliceAI

**Status**: Experimental  
**Purpose**: Integration of genomic foundation models (Evo2, Evo1, Nucleotide Transformer) as feature extractors for splice prediction and isoform discovery.

---

## Overview

This package provides optional foundation model integration for agentic-spliceai. Foundation models are large pre-trained genomic language models that can be used as:

1. **Feature extractors**: Generate embeddings for downstream tasks
2. **Zero-shot predictors**: Direct sequence likelihood scoring
3. **Fine-tunable backbones**: Adapter-based task-specific tuning

**Current models**:
- ✅ **Evo2** (7B, 40B): Arc Institute's genomic foundation model
- 🔄 Evo1 (planned)
- 🔄 Nucleotide Transformer (planned)

---

## Why Separate Package?

Foundation models are kept separate from the core `agentic_spliceai` package because:

1. **Heavy dependencies**: Require PyTorch with CUDA, transformers, 10s of GB model weights
2. **Compute requirements**: Need GPU for reasonable inference speed
3. **Experimental**: Rapidly evolving research code
4. **Optional**: Core functionality works without foundation models

---

## Installation

### Option 1: Local Development (MacBook Pro M1)

**Requirements**:
- MacBook Pro with M1/M2/M3 chip
- 16GB+ RAM recommended
- ~20GB disk space for model weights

**Setup**:
```bash
cd agentic-spliceai

# Create environment (CPU/MPS only)
mamba env create -f foundation_models/environment-evo2-local.yml
conda activate aspliceai-evo2-local

# Install packages
pip install -e .  # Core package
pip install -e ./foundation_models  # Foundation models

# Test installation
python foundation_models/examples/01_load_evo2_local.py
```

**What you can do**:
- Load Evo2 7B (quantized to INT8)
- Extract embeddings for genes (8-32kb windows)
- Train lightweight classifiers
- Prototype experiments

**Limitations**:
- 40B model too large
- Slower inference (~10x vs GPU)
- Limited batch size (1-2)

---

### Option 2: GPU Pod (A40/A100)

**Requirements**:
- NVIDIA GPU with 40GB+ VRAM (A40, A100)
- CUDA 11.8+
- ~100GB disk space for 40B model

**Setup**:
```bash
cd agentic-spliceai

# Create environment (GPU)
mamba env create -f foundation_models/environment-evo2-pod.yml
conda activate aspliceai-evo2-pod

# Install packages
pip install -e .
pip install -e ./foundation_models

# Test installation
python foundation_models/examples/01_load_evo2_pod.py
```

**What you can do**:
- Load Evo2 40B (full precision or FP16)
- Process entire genomes
- Large batch sizes (8-16)
- Fast inference
- Fine-tuning with adapters

---

## Quick Start

### Extract Embeddings for a Gene

```python
from foundation_models.evo2 import Evo2Embedder

# Load model (auto-detects device)
embedder = Evo2Embedder(
    model_size="7b",  # or "40b" on GPU
    quantize=True,    # INT8 for M1 Mac
)

# Extract embeddings
sequence = "ATCGATCGATCG..."  # Your DNA sequence
embeddings = embedder.encode(sequence)  # Shape: [seq_len, hidden_dim]

print(f"Embeddings shape: {embeddings.shape}")
# Output: Embeddings shape: torch.Size([12, 2560])
```

### Train Exon Classifier

```python
from foundation_models.evo2 import ExonClassifier

# Create classifier
classifier = ExonClassifier(
    input_dim=2560,  # Evo2 7B hidden dim
    hidden_dim=256,
    num_layers=2,
    architecture="mlp"
)

# Train on embeddings
classifier.fit(
    train_embeddings=train_embeddings,
    train_labels=train_labels,
    val_embeddings=val_embeddings,
    val_labels=val_labels,
    epochs=50
)

# Predict
exon_probs = classifier.predict(test_embeddings)
```

### Integration with Core Package

```python
# Core agentic-spliceai functionality
from agentic_spliceai.splice_engine.base_layer import BaseModelRunner

# Foundation model embeddings
from foundation_models.evo2 import Evo2Embedder

# Compare base models with Evo2
runner = BaseModelRunner()
openspliceai_pred = runner.run_single_model('openspliceai', gene='BRCA1')

embedder = Evo2Embedder()
evo2_embeddings = embedder.encode(sequence)

# Use both for meta-layer fusion (future)
```

---

## Current Experiments

### 1. Exon Classifier (In Progress)

**Goal**: Reproduce Evo2 paper's single-nucleotide exon classification

**Status**: Week 1 - Package setup  
**Target**: AUROC 0.82-0.99 across 8 held-out species

**See**: `dev/foundation_and_adaptor/logs/evo2_experiment_plan.md`

---

## Package Structure

```
foundation_models/
├── README.md                          # This file
├── pyproject.toml                     # Package metadata
├── setup.py                           # Alternative setup
│
├── environment-evo2-local.yml         # M1 Mac (7B, INT8)
├── environment-evo2-pod.yml           # GPU (7B/40B, full)
│
├── foundation_models/                 # Python package
│   ├── __init__.py
│   ├── evo2/
│   │   ├── __init__.py
│   │   ├── config.py                  # Evo2Config
│   │   ├── model.py                   # Evo2Model wrapper
│   │   ├── embedder.py                # Embedding extraction
│   │   ├── classifier.py              # Exon classifier
│   │   └── adapters/
│   │       ├── lora.py                # LoRA fine-tuning
│   │       └── spliceai_adapter.py    # Bridge to core
│   │
│   └── utils/
│       ├── quantization.py            # INT8/INT4 helpers
│       └── chunking.py                # Sequence chunking
│
├── examples/
│   ├── 01_load_evo2_local.py          # M1 Mac demo
│   ├── 02_extract_embeddings.py       # Embedding extraction
│   ├── 03_train_classifier.py         # Classifier training
│   └── 04_evaluate_species.py         # Cross-species eval
│
├── notebooks/
│   └── evo2_exon_classifier.ipynb     # Tutorial
│
└── tests/
    ├── test_evo2_model.py
    └── test_embeddings.py
```

---

## Model Comparison

| Model | Parameters | Context | Quantization | M1 Mac | GPU |
|-------|------------|---------|--------------|--------|-----|
| **Evo2 7B** | 7B | 1M bp | INT8 | ✅ Slow | ✅ Fast |
| **Evo2 40B** | 40B | 1M bp | FP16/INT8 | ❌ Too large | ✅ Fast |
| **Evo1** | ~7B | 131K bp | INT8 | ✅ Slow | ✅ Fast |
| **Nucleotide Transformer** | 500M-2.5B | 6kb-12kb | - | ✅ Fast | ✅ Fast |

**Recommendation**:
- **Local prototyping**: Evo2 7B (INT8)
- **Production**: Evo2 40B (FP16) on GPU
- **Comparison baseline**: Nucleotide Transformer, Evo1

---

## Performance Expectations

### M1 MacBook Pro (16GB RAM)

**Evo2 7B (INT8 quantized)**:
- **Loading time**: ~2 minutes
- **Inference speed**: ~100 bp/second (single sequence)
- **Memory usage**: ~8GB
- **Max sequence length**: 32,768 bp (practical limit)

### NVIDIA A40 (48GB VRAM)

**Evo2 7B (FP16)**:
- **Loading time**: ~30 seconds
- **Inference speed**: ~10,000 bp/second (batch size 8)
- **Memory usage**: ~14GB
- **Max sequence length**: 1,000,000 bp

**Evo2 40B (FP16)**:
- **Loading time**: ~2 minutes
- **Inference speed**: ~5,000 bp/second (batch size 4)
- **Memory usage**: ~80GB (requires A100 80GB)
- **Max sequence length**: 1,000,000 bp

---

## Integration with Agentic-SpliceAI

### Base Layer Integration

Foundation models can be registered in the base layer model registry:

```python
# In core package: src/agentic_spliceai/splice_engine/base_layer/models/registry.py

def load_model(model_name: str, **kwargs):
    """Load model by name. Auto-discovers foundation models."""
    
    # Try to import foundation_models (if installed)
    try:
        import foundation_models.evo2  # Triggers auto-registration
    except ImportError:
        pass  # Not installed, that's okay
    
    # Load model
    if model_name == 'evo2':
        from foundation_models.evo2 import Evo2Model
        return Evo2Model(**kwargs)
    elif model_name in ['spliceai', 'openspliceai']:
        # Core models (always available)
        ...
```

### Meta Layer Integration

Evo2 embeddings can be used as input modality:

```python
# Multi-modal fusion with Evo2 embeddings

meta_model = MetaPredictor(
    modalities={
        'base_scores': openspliceai_predictions,  # Base layer
        'evo2_embeddings': evo2_embeddings,       # Foundation model (new!)
        'conservation': phylop_scores,            # Conservation
        'context': patient_variants,              # Patient context
    }
)

# Predict with fusion
meta_predictions = meta_model.predict()

# Delta scores for novel site detection
delta = meta_predictions - openspliceai_predictions
novel_sites = delta > 0.3  # High confidence novel
```

---

## Development Workflow

### Local Development (M1 Mac)

1. **Prototype on small genes** (BRCA1, TP53)
2. **Test classifier architectures** (linear, MLP, CNN)
3. **Debug data pipeline**
4. **Rapid iteration** (no pod queue)

### GPU Pod Scaling

1. **Scale to full genomes** (chr21, entire training set)
2. **Use 40B model** for best performance
3. **Cross-species evaluation** (8 held-out species)
4. **Production runs** (final experiments)

---

## Troubleshooting

### Issue: Model too large for M1 Mac

**Solution**: Use 7B model with INT8 quantization
```python
embedder = Evo2Embedder(model_size="7b", quantize=True)
```

### Issue: CUDA out of memory on GPU

**Solutions**:
1. Reduce batch size: `batch_size=1`
2. Use gradient checkpointing: `model.gradient_checkpointing_enable()`
3. Reduce sequence length: `max_length=32768`
4. Use INT8 quantization: `load_in_8bit=True`

### Issue: Slow inference on M1 Mac

**Expected**: M1 is ~10x slower than GPU  
**Solution**: Pre-compute and cache embeddings
```bash
python examples/02_extract_embeddings.py --cache --output embeddings.h5
```

### Issue: ImportError when using core package

**Solution**: Foundation models are optional. Install with:
```bash
pip install -e ./foundation_models
```

Or skip foundation model features if not needed.

---

## Citing

If you use Evo2 in your research, please cite:

```bibtex
@article{hie2025evo2,
  title={Genome modeling and design across all domains of life with Evo 2},
  author={Hie, Brian L and Hsu, Patrick D and Goodarzi, Hani and others},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.02.18.638918}
}
```

---

## Contributing

This is experimental research code. Contributions welcome!

**Areas for contribution**:
1. Additional foundation models (Evo1, Nucleotide Transformer)
2. New downstream tasks (splice site prediction, variant effect)
3. Optimization (quantization, pruning)
4. Documentation and tutorials

---

## Support

For questions about this package:
- Open an issue on GitHub
- Check `dev/foundation_and_adaptor/logs/` for experiment notes
- See examples in `examples/` directory

For Evo2 model questions:
- Arc Institute: https://arcinstitute.org/tools/evo
- GitHub: https://github.com/ArcInstitute/evo2
- Paper: https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1

---

**Last Updated**: February 15, 2026  
**Version**: 0.1.0-alpha  
**Status**: Active Development
