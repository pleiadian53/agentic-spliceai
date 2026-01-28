"""
Inference components for the meta-layer.

Modules:
- predictor.py: Main inference engine and delta scoring
- full_coverage_inference.py: Full coverage by running base model on new genes
- full_coverage_predictor.py: Full coverage from pre-computed artifacts (faster)
- base_model_predictor.py: Wrapper for base model (OpenSpliceAI/SpliceAI) inference

Two approaches for full coverage:
1. FullCoveragePredictor (from artifacts):
   - Uses pre-computed artifacts with all features
   - Fast: just runs meta-layer on existing data
   - For genes already in artifacts
   
2. FullCoverageInference (from scratch):
   - Runs base model with save_nucleotide_scores=True
   - Generates features for ALL positions
   - For new genes not in artifacts
"""

from .predictor import MetaLayerPredictor, DeltaScorer, PredictionResult
from .full_coverage_inference import (
    FullCoveragePredictor as FullCoverageFromScratch,
    FullCoverageResult,
    predict_full_coverage
)
from .full_coverage_predictor import (
    FullCoveragePredictor,  # Default: from pre-computed artifacts
    create_full_coverage_predictor
)
from .base_model_predictor import (
    BaseModelPredictor,
    DeltaScoreResult,
    get_base_model_predictor
)

__all__ = [
    # Standard prediction (subsampled positions)
    "MetaLayerPredictor",
    "DeltaScorer", 
    "PredictionResult",
    # Full coverage from pre-computed artifacts (PREFERRED)
    "FullCoveragePredictor",
    "create_full_coverage_predictor",
    # Full coverage from scratch (for new genes)
    "FullCoverageFromScratch",
    "FullCoverageResult",
    "predict_full_coverage",
    # Base model prediction
    "BaseModelPredictor",
    "DeltaScoreResult",
    "get_base_model_predictor",
]
