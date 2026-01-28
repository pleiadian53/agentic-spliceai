"""
Meta-Layer: Base-Model-Agnostic Multimodal Meta-Learning for Splice Site Prediction
====================================================================================

This package implements a multimodal deep learning meta-layer that recalibrates
base model splice site predictions using:
1. Contextual DNA sequences (via HyenaDNA or other DNA language models)
2. Base model score features (donor, acceptor, neither probabilities)
3. Derived features (entropy, peak patterns, signal strength)

Key Features:
- Base-model-agnostic: Works with any base model (SpliceAI, OpenSpliceAI, etc.)
- Multimodal: Combines sequence and score information
- Scalable: Supports CPU (M1) and GPU (RunPods) training
- Variant-aware: Integrates SpliceVarDB for context-dependent splice sites
- Integrated: Uses genomic_resources for consistent path resolution

Model Categories:
----------------
1. Classification Models (canonical splice site classification):
   - MetaSpliceModel: Per-window classification (501nt → [1, 3])
   - MetaSpliceModelV2: Sequence-to-sequence (L nt → [L, 3])
   - SpliceInducingClassifier: Binary "Is this variant splice-altering?"
   - EffectTypeClassifier: Multi-class effect type classification

2. Delta Prediction Models (variant effect magnitude):
   - DeltaPredictor: Siamese network (paired prediction)
   - SimpleCNNDeltaPredictor: Gated CNN with dilated convolutions (BEST calibrated)
   - ValidatedDeltaPredictor: Single-pass with SpliceVarDB targets (BEST SO FAR, r=0.41)

Quick Start:
-----------
>>> from agentic_spliceai.splice_engine.meta_layer import MetaLayerConfig, MetaSpliceModel
>>> from agentic_spliceai.splice_engine.meta_layer.data import MetaLayerDataset
>>> 
>>> # Configure for OpenSpliceAI
>>> config = MetaLayerConfig(base_model='openspliceai')
>>> print(config.artifacts_dir)  # Uses genomic_resources
>>> 
>>> # Create model
>>> model = MetaSpliceModel(
...     sequence_encoder='cnn',
...     num_score_features=50,
...     hidden_dim=256
... )

Training:
--------
>>> from agentic_spliceai.splice_engine.meta_layer.workflows import run_canonical_training
>>> result = run_canonical_training(base_model='openspliceai', epochs=30)
>>> print(f"Test PR-AUC: {result.canonical_test_metrics['pr_auc_macro']:.4f}")

Documentation:
-------------
See `docs/` for detailed documentation:
- ARCHITECTURE.md: System architecture and design principles
- LABELING_STRATEGY.md: How labels are created from base layer + variants
- ALTERNATIVE_SPLICING_PIPELINE.md: From scores to exon-intron predictions

Ported from: meta_spliceai/splice_engine/meta_layer/
"""

__version__ = "0.1.0"
__author__ = "Agentic-SpliceAI Team"

# Core configuration
from .core.config import MetaLayerConfig
from .core.artifact_loader import ArtifactLoader
from .core.feature_schema import FeatureSchema, LABEL_ENCODING, LABEL_DECODING, DEFAULT_SCHEMA
from .core.path_manager import MetaLayerPathManager, get_path_manager

# Models - Classification
from .models import (
    MetaSpliceModel,
    MetaSpliceModelV2,
    ScoreOnlyModel,
    SequenceEncoderFactory,
    CNNEncoder,
    ScoreEncoder,
    SpliceInducingClassifier,
    EffectTypeClassifier,
    UnifiedSpliceClassifier,
    create_meta_model_v2,
    create_splice_classifier,
)

# Models - Delta Prediction
from .models import (
    DeltaPredictor,
    DeltaPredictorV2,
    SimpleCNNDeltaPredictor,
    ValidatedDeltaPredictor,
    ValidatedDeltaPredictorWithAttention,
    create_delta_predictor,
    create_delta_predictor_v2,
    create_validated_delta_predictor,
    create_calibrated_predictor,
)

# Data
from .data import (
    MetaLayerDataset,
    create_dataloaders,
    prepare_training_data,
    SpliceVarDBLoader,
    load_splicevardb,
    VariantDeltaDataset,
    prepare_variant_data,
)

# Workflows
from .workflows import (
    CanonicalTrainingWorkflow,
    CanonicalTrainingConfig,
    CanonicalTrainingResult,
    run_canonical_training,
)

__all__ = [
    # Version
    "__version__",
    # Core
    "MetaLayerConfig",
    "ArtifactLoader",
    "FeatureSchema",
    "LABEL_ENCODING",
    "LABEL_DECODING",
    "DEFAULT_SCHEMA",
    "MetaLayerPathManager",
    "get_path_manager",
    # Models - Classification
    "MetaSpliceModel",
    "MetaSpliceModelV2",
    "ScoreOnlyModel",
    "SequenceEncoderFactory",
    "CNNEncoder",
    "ScoreEncoder",
    "SpliceInducingClassifier",
    "EffectTypeClassifier",
    "UnifiedSpliceClassifier",
    "create_meta_model_v2",
    "create_splice_classifier",
    # Models - Delta Prediction
    "DeltaPredictor",
    "DeltaPredictorV2",
    "SimpleCNNDeltaPredictor",
    "ValidatedDeltaPredictor",
    "ValidatedDeltaPredictorWithAttention",
    "create_delta_predictor",
    "create_delta_predictor_v2",
    "create_validated_delta_predictor",
    "create_calibrated_predictor",
    # Data
    "MetaLayerDataset",
    "create_dataloaders",
    "prepare_training_data",
    "SpliceVarDBLoader",
    "load_splicevardb",
    "VariantDeltaDataset",
    "prepare_variant_data",
    # Workflows
    "CanonicalTrainingWorkflow",
    "CanonicalTrainingConfig",
    "CanonicalTrainingResult",
    "run_canonical_training",
]
