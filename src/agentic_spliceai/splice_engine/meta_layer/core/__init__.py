"""
Core components for the meta-layer.

- config.py: MetaLayerConfig for configuring the meta-layer
- artifact_loader.py: Load base layer artifacts for any base model
- feature_schema.py: Standardized feature definitions
- path_manager.py: Safe read/write path management
"""

from .config import MetaLayerConfig
from .artifact_loader import ArtifactLoader
from .feature_schema import FeatureSchema, LABEL_ENCODING, LABEL_DECODING, DEFAULT_SCHEMA
from .path_manager import MetaLayerPathManager, get_path_manager

__all__ = [
    "MetaLayerConfig",
    "ArtifactLoader",
    "FeatureSchema",
    "LABEL_ENCODING",
    "LABEL_DECODING",
    "DEFAULT_SCHEMA",
    "MetaLayerPathManager",
    "get_path_manager",
]
