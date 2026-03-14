"""
Foundation Models for Agentic-SpliceAI

Integration of genomic foundation models (Evo2, Evo1, Nucleotide Transformer)
for splice prediction and isoform discovery.

This is an experimental package with heavy GPU dependencies, kept separate
from the core agentic_spliceai package for flexibility.
"""

__version__ = "0.1.0"

# Auto-register foundation models with core package (if installed)
def _register_foundation_models():
    """
    Auto-register foundation models with agentic-spliceai base layer.
    
    This allows foundation models to be used seamlessly with the core
    package's BaseModelRunner, if both packages are installed.
    """
    try:
        from agentic_spliceai.splice_engine.base_layer.models.registry import (
            register_foundation_model
        )

        # Register Evo2
        register_foundation_model(
            name='evo2',
            loader_path='foundation_models.evo2.model:load_evo2_model'
        )

    except Exception:
        # Core package not installed, or registry not yet implemented.
        # A failed import can leave partially-initialized entries (None values)
        # in sys.modules, which breaks subsequent imports of the same packages.
        # Clean these up to prevent cascading failures.
        import sys
        broken = [k for k in sys.modules
                  if k.startswith('agentic_spliceai.') and sys.modules[k] is None]
        for k in broken:
            del sys.modules[k]


# Attempt registration on import
_register_foundation_models()


# Base classes and registry (always available — no heavy deps)
from foundation_models.base import (  # noqa: F401
    BaseEmbeddingModel,
    ModelMetadata,
    get_model_metadata,
    list_available_models,
    load_embedding_model,
)

__all__ = [
    # Base
    "BaseEmbeddingModel",
    "ModelMetadata",
    "get_model_metadata",
    "list_available_models",
    "load_embedding_model",
]

# Expose Evo2 APIs (optional — requires CUDA + evo2 package)
try:
    from foundation_models.evo2 import (  # noqa: F401
        Evo2Model,
        Evo2Embedder,
        Evo2Config,
        ExonClassifier,
    )

    __all__ += [
        "Evo2Model",
        "Evo2Embedder",
        "Evo2Config",
        "ExonClassifier",
    ]
except ImportError:
    pass

# Expose SpliceBERT APIs (optional — requires transformers + sentencepiece)
try:
    from foundation_models.splicebert import (  # noqa: F401
        SpliceBERTConfig,
        SpliceBERTModel,
    )

    __all__ += [
        "SpliceBERTConfig",
        "SpliceBERTModel",
    ]
except ImportError:
    pass

# Expose DNABERT-2 APIs (optional — requires transformers + einops)
try:
    from foundation_models.dnabert import (  # noqa: F401
        DNABERT2Config,
        DNABERT2Model,
    )

    __all__ += [
        "DNABERT2Config",
        "DNABERT2Model",
    ]
except ImportError:
    pass
