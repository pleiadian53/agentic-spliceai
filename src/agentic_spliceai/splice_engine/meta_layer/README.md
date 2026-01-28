# Meta-Layer: Base-Model-Agnostic Multimodal Meta-Learning

**Status**: 🚧 In Development  
**Version**: 0.1.0  
**Last Updated**: December 2025  
**Ported from**: meta_spliceai/splice_engine/meta_layer/

---

## Overview

The Meta-Layer is a **multimodal deep learning system** that recalibrates base model splice site predictions to:

1. **Correct FPs/FNs** - Reduce false positives and false negatives from base models
2. **Predict context-dependent splicing** - Account for variant-induced alternative splicing
3. **Maintain consistency** - Output same format as base layer (per-nucleotide probabilities)

### Key Design Principle: Base-Model-Agnostic

Just like the base layer supports any splice prediction model, the meta-layer works with **any base model** via a single parameter:

```python
from agentic_spliceai.splice_engine.meta_layer import MetaSpliceModel, run_canonical_training

# Works with SpliceAI (GRCh37)
result = run_canonical_training(base_model='spliceai', epochs=30)

# Works with OpenSpliceAI (GRCh38/MANE)
result = run_canonical_training(base_model='openspliceai', epochs=30)
```

---

## Model Categories

### 1. Classification Models (Canonical Splice Site Classification)

For recalibrating base model predictions on canonical splice sites:

```python
from agentic_spliceai.splice_engine.meta_layer import (
    MetaSpliceModel,      # Per-window classification (501nt → [1, 3])
    MetaSpliceModelV2,    # Sequence-to-sequence (L nt → [L, 3])
)
```

### 2. Splice Effect Classifiers (Variant Classification)

For predicting whether variants affect splicing:

```python
from agentic_spliceai.splice_engine.meta_layer import (
    SpliceInducingClassifier,   # Binary: Is this variant splice-altering?
    EffectTypeClassifier,       # Multi-class: What type of effect?
    UnifiedSpliceClassifier,    # Multi-task with position attention
)
```

### 3. Delta Prediction Models (Variant Effect Magnitude)

For predicting how much variants change splice site scores:

```python
from agentic_spliceai.splice_engine.meta_layer import (
    DeltaPredictor,             # Siamese network (paired prediction)
    SimpleCNNDeltaPredictor,    # Gated CNN (BEST calibrated)
    ValidatedDeltaPredictor,    # Single-pass with SpliceVarDB targets (BEST SO FAR, r=0.41)
)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         META-LAYER ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  INPUT: Base Layer Artifacts (analysis_sequences_*.tsv)        │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • 501nt contextual sequences                                   │   │
│  │  • Base model scores (donor, acceptor, neither)                 │   │
│  │  • 50+ derived features                                         │   │
│  │  • Labels (splice_type from GTF annotations)                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SEQUENCE ENCODER (Modality 1)                                  │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Options: HyenaDNA, Gated CNN (lightweight)                     │   │
│  │  Output: [B, D] sequence embeddings                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  SCORE ENCODER (Modality 2)                                     │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  MLP: [50+ features] → [D] score embeddings                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  FUSION LAYER                                                    │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Cross-attention or concatenation                                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Recalibrated probabilities / Delta scores              │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  Classification: P(donor), P(acceptor), P(neither)              │   │
│  │  Delta: Δ_donor, Δ_acceptor, Δ_neither                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Package Structure

```
meta_layer/
├── __init__.py                 # Package entry point
├── README.md                   # This file
│
├── core/
│   ├── __init__.py
│   ├── config.py               # MetaLayerConfig
│   ├── artifact_loader.py      # Load base layer artifacts
│   ├── feature_schema.py       # Standardized feature definitions
│   └── path_manager.py         # Safe read/write path resolution
│
├── models/
│   ├── __init__.py
│   ├── sequence_encoder.py     # DNA LM wrapper (HyenaDNA, CNN)
│   ├── score_encoder.py        # MLP for score features
│   ├── meta_splice_model.py    # Classification model (V1)
│   ├── meta_splice_model_v2.py # Seq2seq model (V2)
│   ├── splice_classifier.py    # Variant effect classifiers
│   ├── delta_predictor.py      # Siamese delta predictor
│   ├── delta_predictor_v2.py   # Per-position delta
│   ├── validated_delta_predictor.py  # SpliceVarDB-validated (BEST)
│   ├── hyenadna_delta_predictor.py   # SimpleCNNDeltaPredictor
│   ├── hyenadna_encoder.py     # HyenaDNA integration
│   └── delta_predictor_calibrated.py # Calibration strategies
│
├── data/
│   ├── __init__.py
│   ├── dataset.py              # MetaLayerDataset
│   ├── splicevardb_loader.py   # SpliceVarDB integration
│   └── variant_dataset.py      # VariantDeltaDataset
│
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop
│   ├── evaluator.py            # Metrics (PR-AUC, top-k)
│   └── variant_evaluator.py    # Variant effect evaluation
│
├── inference/
│   ├── __init__.py
│   ├── predictor.py            # Inference engine
│   ├── base_model_predictor.py # Base model wrapper
│   ├── full_coverage_inference.py   # From scratch
│   └── full_coverage_predictor.py   # From artifacts
│
├── workflows/
│   ├── __init__.py
│   └── canonical_training.py   # Canonical classification training
│
├── configs/
│   ├── default.yaml            # Default configuration
│   ├── lightweight.yaml        # M1 Mac optimized
│   └── hyenadna.yaml           # GPU training with HyenaDNA
│
└── docs/                       # Package documentation
```

---

## Quick Start

### 1. Train on Canonical Splice Sites

```python
from agentic_spliceai.splice_engine.meta_layer import run_canonical_training

# Train with lightweight CNN (CPU-friendly for M1 Mac)
result = run_canonical_training(
    base_model='openspliceai',
    epochs=30,
    sequence_encoder='cnn',
    output_dir='./experiments/canonical_v1'
)

print(f"Test PR-AUC: {result.canonical_test_metrics['pr_auc_macro']:.4f}")
if result.variant_evaluation:
    print(result.variant_evaluation.summary())
```

### 2. Create Delta Predictor

```python
from agentic_spliceai.splice_engine.meta_layer import (
    ValidatedDeltaPredictor,
    create_validated_delta_predictor
)

# Create model
model = create_validated_delta_predictor(
    variant='attention',  # With position attention
    hidden_dim=128,
    n_layers=6
)

# Forward pass
delta = model(alt_seq, ref_base, alt_base)  # [B, 3]
```

### 3. Use Calibrated Predictor

```python
from agentic_spliceai.splice_engine.meta_layer import (
    SimpleCNNDeltaPredictor,
    create_calibrated_predictor
)

# Create base predictor
base = SimpleCNNDeltaPredictor(hidden_dim=64)

# Wrap with quantile calibration (BEST for large deltas)
model = create_calibrated_predictor(
    base_predictor=base,
    strategy='quantile',
    quantile=0.9
)
```

---

## Training Workflows

| Workflow | Description | Status |
|----------|-------------|--------|
| `canonical_training.py` | Train on canonical sites, evaluate on variants | ✅ Implemented |
| `validated_delta_training.py` | Delta prediction with SpliceVarDB targets | 📋 TODO |
| `hyenadna_training.py` | GPU training with HyenaDNA | 📋 TODO |

---

## Best Approaches (from R&D)

Based on experiments in meta-spliceai:

1. **ValidatedDeltaPredictor** with SpliceVarDB targets: r=0.41 correlation
2. **SimpleCNNDeltaPredictor** with quantile loss (τ=0.9): Best calibration
3. **Gated CNN with dilated convolutions**: Captures long-range patterns
4. **LayerNorm + GELU**: Better than BatchNorm + ReLU for this task

Things that **didn't work**:
- Simple scaling/temperature calibration
- MSE loss alone (need quantile loss)
- More data without architecture improvements

---

## Configurations

| Config | Environment | Sequence Encoder | Notes |
|--------|-------------|------------------|-------|
| `default.yaml` | General | CNN | Balanced defaults |
| `lightweight.yaml` | M1 Mac | CNN | 32-dim, reduced epochs |
| `hyenadna.yaml` | GPU | HyenaDNA | Full capacity |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- polars, numpy
- scikit-learn
- tqdm

Optional:
- transformers (for HyenaDNA)
- CUDA GPU (for full training)

---

## Status

| Component | Status |
|-----------|--------|
| Core config | ✅ Complete |
| Artifact loader | ✅ Complete |
| Models - Classification | ✅ Complete |
| Models - Delta prediction | ✅ Complete |
| Dataset preparation | ✅ Complete |
| Training pipeline | ✅ Complete |
| Evaluation | ✅ Complete |
| Inference | ✅ Complete |
| Workflows | 🚧 In progress |
| CLI | 📋 Planned |

---

*Ported from meta_spliceai with updated import paths and improved naming conventions.*  
*Last Updated: December 2025*












