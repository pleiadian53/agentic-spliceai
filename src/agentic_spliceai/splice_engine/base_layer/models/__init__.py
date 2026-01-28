"""Base model implementations for splice site prediction.

This package provides:
- Model configuration classes
- Model runner for predictions
- Model loading utilities

Exports:
    BaseModelConfig: Configuration for base models
    SpliceAIConfig: Alias for BaseModelConfig
    OpenSpliceAIConfig: Configuration for OpenSpliceAI
    BaseModelRunner: Runner for model predictions
    BaseModelResult: Result container
    ComparisonResult: Model comparison result
"""

from .config import (
    BaseModelConfig,
    SpliceAIConfig,
    OpenSpliceAIConfig,
    create_config,
)

from .runner import (
    BaseModelRunner,
    BaseModelResult,
    ComparisonResult,
)

__all__ = [
    'BaseModelConfig',
    'SpliceAIConfig',
    'OpenSpliceAIConfig',
    'create_config',
    'BaseModelRunner',
    'BaseModelResult',
    'ComparisonResult',
]
