"""Built-in modalities for the feature engineering pipeline.

Auto-registers all built-in modalities with the FeaturePipeline registry
on import. External modalities can be registered via
``FeaturePipeline.register()``.
"""

from ..pipeline import FeaturePipeline
from .base_scores import BaseScoreConfig, BaseScoreModality

# Register built-in modalities
FeaturePipeline.register("base_scores", BaseScoreModality, BaseScoreConfig)


def _register_annotation() -> None:
    from .annotation import AnnotationConfig, AnnotationModality

    FeaturePipeline.register("annotation", AnnotationModality, AnnotationConfig)


def _register_sequence() -> None:
    from .sequence import SequenceConfig, SequenceModality

    FeaturePipeline.register("sequence", SequenceModality, SequenceConfig)


def _register_genomic() -> None:
    from .genomic import GenomicContextConfig, GenomicContextModality

    FeaturePipeline.register("genomic", GenomicContextModality, GenomicContextConfig)


def _register_conservation() -> None:
    from .conservation import ConservationConfig, ConservationModality

    FeaturePipeline.register("conservation", ConservationModality, ConservationConfig)


# Register modalities that may have heavier dependencies lazily
_register_annotation()
_register_sequence()
_register_genomic()
_register_conservation()
