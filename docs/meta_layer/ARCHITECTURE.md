# Meta-Layer Architecture

**Last Updated**: December 2025  
**Status**: Design Document

---

## Overview

The Meta-Layer is a **multimodal deep learning system** that recalibrates base model splice site predictions by combining:

1. **Sequence embeddings** from DNA language models (HyenaDNA) or CNN encoders
2. **Score embeddings** from base model features
3. **Cross-modal fusion** to leverage both modalities

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           META-LAYER SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        DATA LAYER                                      │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  Base Layer Artifacts          SpliceVarDB                             │ │
│  │  ─────────────────────         ───────────                             │ │
│  │  • analysis_sequences_*.tsv    • 50K validated variants                │ │
│  │  • 501nt context windows       • Splice-altering classifications      │ │
│  │  • 50+ derived features        • hg19/hg38 coordinates                 │ │
│  │  • GTF-based labels            • Sample weights                        │ │
│  │                                                                        │ │
│  │                    ↓                           ↓                       │ │
│  │                    └───────────┬───────────────┘                       │ │
│  │                                ↓                                       │ │
│  │                    ┌───────────────────────┐                           │ │
│  │                    │  ArtifactLoader       │                           │ │
│  │                    │  (base-model-agnostic)│                           │ │
│  │                    └───────────────────────┘                           │ │
│  │                                ↓                                       │ │
│  │                    ┌───────────────────────┐                           │ │
│  │                    │  MetaLayerDataset     │                           │ │
│  │                    │  (PyTorch Dataset)    │                           │ │
│  │                    └───────────────────────┘                           │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                   ↓                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                        MODEL LAYER                                     │ │
│  ├────────────────────────────────────────────────────────────────────────┤ │
│  │                                                                        │ │
│  │  ┌─────────────────────┐     ┌─────────────────────┐                  │ │
│  │  │  Sequence Encoder   │     │  Score Encoder      │                  │ │
│  │  ├─────────────────────┤     ├─────────────────────┤                  │ │
│  │  │  Options:           │     │  MLP Network:       │                  │ │
│  │  │  • HyenaDNA (SSM)   │     │  [50+ feat] → [D]   │                  │ │
│  │  │  • Gated CNN        │     │  LayerNorm + GELU   │                  │ │
│  │  │  • Lightweight CNN  │     │                     │                  │ │
│  │  │                     │     │                     │                  │ │
│  │  │  Output: [B, D]     │     │  Output: [B, D]     │                  │ │
│  │  └─────────────────────┘     └─────────────────────┘                  │ │
│  │           ↓                           ↓                               │ │
│  │           └───────────┬───────────────┘                               │ │
│  │                       ↓                                               │ │
│  │           ┌───────────────────────┐                                   │ │
│  │           │    Fusion Layer       │                                   │ │
│  │           ├───────────────────────┤                                   │ │
│  │           │  Options:             │                                   │ │
│  │           │  • Cross-attention    │                                   │ │
│  │           │  • Concatenation      │                                   │ │
│  │           │  • Gated fusion       │                                   │ │
│  │           │                       │                                   │ │
│  │           │  Output: [B, D*2]     │                                   │ │
│  │           └───────────────────────┘                                   │ │
│  │                       ↓                                               │ │
│  │           ┌───────────────────────┐                                   │ │
│  │           │   Output Head         │                                   │ │
│  │           ├───────────────────────┤                                   │ │
│  │           │  Classification:      │                                   │ │
│  │           │    P(donor), P(acc),  │                                   │ │
│  │           │    P(neither)         │                                   │ │
│  │           │                       │                                   │ │
│  │           │  OR Delta Prediction: │                                   │ │
│  │           │    Δ_donor, Δ_acc,    │                                   │ │
│  │           │    Δ_neither          │                                   │ │
│  │           └───────────────────────┘                                   │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Base-Model-Agnostic

The meta-layer works with **any base model** through automatic artifact routing:

```python
from agentic_spliceai.splice_engine.meta_layer import MetaLayerConfig

# Works with any base model - just change the parameter
config = MetaLayerConfig(base_model='openspliceai')  # or 'spliceai', 'newmodel'

# Automatic path resolution
config.artifacts_dir  # → data/mane/GRCh38/openspliceai_eval/meta_models
config.genome_build   # → GRCh38
```

### 2. Multimodal Fusion

Combines two complementary modalities:

| Modality | Information | Encoder |
|----------|-------------|---------|
| **Sequence** | Long-range dependencies, motifs | HyenaDNA (SSM) or Gated CNN |
| **Scores** | Base model knowledge, local patterns | MLP |

### 3. Multiple Output Modes

| Mode | Output | Use Case |
|------|--------|----------|
| Classification | P(donor), P(acceptor), P(neither) | Canonical site recalibration |
| Delta Prediction | Δ_donor, Δ_acceptor, Δ_neither | Variant effect prediction |
| Binary | P(splice-altering) | Variant screening |

### 4. Scalable Compute

Supports both local development (M1 Mac) and cloud training (RunPods):

| Environment | Encoder | Batch Size | Training Time |
|-------------|---------|------------|---------------|
| M1 Mac (16GB) | Gated CNN | 32 | ~2 hours (subset) |
| RunPods GPU | HyenaDNA | 128 | ~6 hours (full) |

---

## Model Components

### Sequence Encoders

```python
from agentic_spliceai.splice_engine.meta_layer.models import (
    SequenceEncoderFactory,
    CNNEncoder,
    HyenaDNAEncoder,
    GatedCNNEncoder
)

# Factory pattern for encoder selection
encoder = SequenceEncoderFactory.create('cnn', output_dim=256)
```

| Encoder | Description | Best For |
|---------|-------------|----------|
| `HyenaDNAEncoder` | State-space model | GPU training, best quality |
| `GatedCNNEncoder` | Dilated convolutions + gating | CPU/GPU, good balance |
| `CNNEncoder` | Simple multi-scale CNN | CPU, lightweight |

### Score Encoder

```python
from agentic_spliceai.splice_engine.meta_layer.models import ScoreEncoder

encoder = ScoreEncoder(num_features=50, hidden_dim=256)
# MLP: [50 features] → LayerNorm → GELU → [256 dim]
```

### Delta Predictors

| Model | Architecture | Best Result | Notes |
|-------|--------------|-------------|-------|
| `DeltaPredictor` | Siamese (paired) | r=0.38 | Needs both ref + alt |
| `SimpleCNNDeltaPredictor` | Gated CNN | r=0.38 | With quantile loss |
| `ValidatedDeltaPredictor` | Single-pass | **r=0.41** | BEST SO FAR |

---

## Training Pipeline

```
1. Data Loading
   └── ArtifactLoader.load_analysis_sequences()
       └── Returns: sequences, features, labels

2. Batch Creation
   └── MetaLayerDataset + DataLoader
       └── Tokenizes sequences
       └── Normalizes features
       └── Applies sample weights

3. Forward Pass
   └── sequence → SequenceEncoder → seq_emb
   └── features → ScoreEncoder → score_emb
   └── (seq_emb, score_emb) → Fusion → combined_emb
   └── combined_emb → OutputHead → logits/deltas

4. Loss Computation
   └── Classification: CrossEntropyLoss(logits, labels, weight=sample_weights)
   └── Delta: QuantileLoss(deltas, targets, tau=0.9)

5. Optimization
   └── AdamW with learning rate scheduling
   └── Gradient accumulation for large batches
```

---

## Evaluation Metrics

### Classification Tasks

| Metric | Description | Target |
|--------|-------------|--------|
| **PR-AUC** | Area under Precision-Recall curve | > 0.95 |
| **Top-k Accuracy** | % of top-k predictions correct | > 0.90 (k=100) |

### Delta Prediction Tasks

| Metric | Description | Target |
|--------|-------------|--------|
| **Pearson r** | Correlation with true deltas | > 0.5 |
| **Detection Rate** | % of splice-altering detected | > 50% |

---

## Related Documentation

- [LABELING_STRATEGY.md](LABELING_STRATEGY.md) - Label creation
- [methods/ROADMAP.md](methods/ROADMAP.md) - Methodology development
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Training instructions

---

*Ported from meta_spliceai with updated import paths.*












