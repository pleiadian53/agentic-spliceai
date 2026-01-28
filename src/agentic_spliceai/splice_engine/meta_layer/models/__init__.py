"""
Model components for the meta-layer.

Classification Models (for canonical splice site classification):
- MetaSpliceModel: Per-window classification (501nt → [1, 3])
- MetaSpliceModelV2: Sequence-to-sequence (L nt → [L, 3]) - MATCHES BASE MODEL FORMAT
- SpliceInducingClassifier: Binary "Is this variant splice-altering?"
- EffectTypeClassifier: Multi-class effect type (gain/loss, donor/acceptor)
- UnifiedSpliceClassifier: Multi-task with position attention

Delta Prediction Models (for variant effect prediction):
- DeltaPredictor: Siamese network (paired prediction)
- DeltaPredictorV2: Per-position delta output [L, 2]
- SimpleCNNDeltaPredictor: Gated CNN with dilated convolutions (BEST)
- ValidatedDeltaPredictor: Single-pass with SpliceVarDB-validated targets (BEST SO FAR)
- DeltaPredictorWithEncoder: Configurable encoder (HyenaDNA or lightweight CNN)

Encoders:
- sequence_encoder.py: DNA language model wrappers (HyenaDNA, CNN, etc.)
- score_encoder.py: MLP for base model score features
- hyenadna_encoder.py: HyenaDNA integration with lightweight fallback

Calibration:
- delta_predictor_calibrated.py: Scaling, temperature, quantile, hybrid options

Ported from: meta_spliceai/splice_engine/meta_layer/models/
"""

from .sequence_encoder import (
    SequenceEncoderFactory,
    CNNEncoder,
    HyenaDNAEncoder,
    IdentityEncoder
)
from .score_encoder import ScoreEncoder, AttentiveScoreEncoder
from .meta_splice_model import MetaSpliceModel, ScoreOnlyModel, CrossAttentionFusion
from .meta_splice_model_v2 import MetaSpliceModelV2, create_meta_model_v2
from .delta_predictor import (
    DeltaPredictor,
    DeltaPredictorWithClassifier,
    WeightedMSELoss,
    create_delta_predictor
)
from .delta_predictor_v2 import DeltaPredictorV2, create_delta_predictor_v2
from .hyenadna_delta_predictor import SimpleCNNDeltaPredictor
from .hyenadna_encoder import (
    LightweightEncoder,
    DeltaPredictorWithEncoder,
    create_encoder,
    create_delta_predictor as create_delta_predictor_with_encoder
)
from .delta_predictor_calibrated import (
    ScaledDeltaPredictor,
    TemperatureScaledPredictor,
    QuantileDeltaPredictor,
    HybridDeltaPredictor,
    create_calibrated_predictor
)
from .splice_classifier import (
    SpliceInducingClassifier,
    EffectTypeClassifier,
    UnifiedSpliceClassifier,
    GatedCNNEncoder,
    create_splice_classifier
)
from .validated_delta_predictor import (
    ValidatedDeltaPredictor,
    ValidatedDeltaPredictorWithAttention,
    create_validated_delta_predictor
)

__all__ = [
    # Sequence encoders
    "SequenceEncoderFactory",
    "CNNEncoder",
    "HyenaDNAEncoder",
    "IdentityEncoder",
    "LightweightEncoder",
    "GatedCNNEncoder",
    # Score encoders
    "ScoreEncoder",
    "AttentiveScoreEncoder",
    # Classification models (canonical splice site classification)
    "MetaSpliceModel",      # V1: Per-window [1, 3] output
    "MetaSpliceModelV2",    # V2: Sequence-to-sequence [L, 3] output
    "ScoreOnlyModel",
    "CrossAttentionFusion",
    "create_meta_model_v2",
    # Splice effect classifiers (variant classification)
    "SpliceInducingClassifier",   # Binary: Is splice-altering?
    "EffectTypeClassifier",       # Multi-class: What type of effect?
    "UnifiedSpliceClassifier",    # Multi-task with attention
    "create_splice_classifier",
    # Delta prediction models (variant effect magnitude)
    "DeltaPredictor",             # Siamese network (paired)
    "DeltaPredictorV2",           # Per-position delta [L, 2]
    "DeltaPredictorWithClassifier",
    "DeltaPredictorWithEncoder",
    "SimpleCNNDeltaPredictor",    # Gated CNN (BEST calibrated)
    "ValidatedDeltaPredictor",    # Single-pass validated (BEST SO FAR)
    "ValidatedDeltaPredictorWithAttention",
    "WeightedMSELoss",
    "create_delta_predictor",
    "create_delta_predictor_v2",
    "create_delta_predictor_with_encoder",
    "create_validated_delta_predictor",
    "create_encoder",
    # Calibrated predictors
    "ScaledDeltaPredictor",
    "TemperatureScaledPredictor",
    "QuantileDeltaPredictor",
    "HybridDeltaPredictor",
    "create_calibrated_predictor",
]
