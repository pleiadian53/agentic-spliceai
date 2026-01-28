"""
Training components for the meta-layer.

- trainer.py: Training loop
- evaluator.py: Metrics (PR-AUC, top-k, AP, calibration)
- variant_evaluator.py: Variant effect evaluation (Phase 1 Approach A)
"""

from .trainer import Trainer, TrainingConfig, TrainingResult, train_meta_model
from .evaluator import Evaluator, EvaluationResult
from .variant_evaluator import (
    VariantEffectEvaluator,
    VariantEvaluationResult,
    DeltaResult,
    evaluate_variant_effects
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "TrainingResult",
    "train_meta_model",
    "Evaluator",
    "EvaluationResult",
    "VariantEffectEvaluator",
    "VariantEvaluationResult",
    "DeltaResult",
    "evaluate_variant_effects",
]
