"""Base Layer: Foundation splice site prediction.

This package provides the base layer functionality for splice site prediction
using pre-trained models like SpliceAI and OpenSpliceAI.

The base layer:
- Loads and runs pre-trained splice site prediction models
- Generates per-nucleotide splice site scores (donor, acceptor, neither)
- Provides evaluation and feature extraction
- Outputs predictions that feed into the meta layer

Subpackages:
    data: Data types and loading utilities
    models: Base model implementations (SpliceAI, OpenSpliceAI)
    prediction: Prediction workflow and evaluation
    io: Input/output handling

Example:
    >>> from agentic_spliceai.splice_engine.base_layer import BaseModelConfig
    >>> config = BaseModelConfig(base_model='spliceai')
"""

from .models import (
    BaseModelConfig,
    SpliceAIConfig,
    OpenSpliceAIConfig,
    create_config,
    BaseModelRunner,
    BaseModelResult,
    ComparisonResult,
)

from .data import (
    GeneManifestEntry,
    GeneManifest,
    PredictionResult,
)

__all__ = [
    # Configuration
    'BaseModelConfig',
    'SpliceAIConfig',
    'OpenSpliceAIConfig',
    'create_config',
    # Runner
    'BaseModelRunner',
    'BaseModelResult',
    'ComparisonResult',
    # Data types
    'GeneManifestEntry',
    'GeneManifest',
    'PredictionResult',
]
