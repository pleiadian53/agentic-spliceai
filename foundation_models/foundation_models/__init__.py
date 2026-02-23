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
        
    except ImportError:
        # Core package not installed, that's okay
        pass


# Attempt registration on import
_register_foundation_models()


# Expose key APIs
try:
    from foundation_models.evo2 import (
        Evo2Model,
        Evo2Embedder,
        Evo2Config,
        ExonClassifier,
    )
    
    __all__ = [
        "Evo2Model",
        "Evo2Embedder",
        "Evo2Config",
        "ExonClassifier",
    ]
    
except ImportError as e:
    # Dependencies not installed yet
    import warnings
    warnings.warn(
        f"Could not import Evo2 components: {e}\n"
        "Please install dependencies: pip install -e ."
    )
    __all__ = []
