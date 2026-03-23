"""Multimodal feature engineering framework.

Cross-layer package that transforms base layer predictions into
enriched feature sets for meta layer training. Each evidence type
(base scores, DNA sequence, annotations, genomic context) is
encapsulated as a configurable Modality.

Key exports:
- ``Modality``, ``ModalityConfig``, ``ModalityMeta`` — modality protocol
- ``FeaturePipeline``, ``FeaturePipelineConfig`` — pipeline orchestration
- ``FeatureWorkflow``, ``FeatureWorkflowResult`` — genome-scale processing

Example
-------
>>> from agentic_spliceai.splice_engine.features import (
...     FeaturePipeline, FeaturePipelineConfig
... )
>>> config = FeaturePipelineConfig(modalities=['base_scores'])
>>> pipeline = FeaturePipeline(config)
>>> enriched_df = pipeline.transform(predictions_df)
"""

from .modality import Modality, ModalityConfig, ModalityMeta
from .pipeline import FeaturePipeline, FeaturePipelineConfig
from .sampling import PositionSamplingConfig, sample_positions
from .workflow import FeatureWorkflow, FeatureWorkflowResult, detect_existing_modalities

# Trigger modality auto-registration
from . import modalities as _modalities  # noqa: F401

__all__ = [
    "Modality",
    "ModalityConfig",
    "ModalityMeta",
    "FeaturePipeline",
    "FeaturePipelineConfig",
    "PositionSamplingConfig",
    "sample_positions",
    "FeatureWorkflow",
    "FeatureWorkflowResult",
    "detect_existing_modalities",
]
