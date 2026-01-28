"""Splice Engine - Splice site prediction with base and meta layers.

This module provides a clean, well-organized interface for splice site prediction
using pre-trained models like SpliceAI and OpenSpliceAI, with support for
meta-layer enhancements.

Architecture:
    - config/: Configuration management (genomic resources, model settings)
    - resources/: Resource management (registry, schema standardization)
    - utils/: Shared utility functions
    - base_layer/: Foundation splice site prediction (SpliceAI, OpenSpliceAI)
    - meta_layer/: Multimodal deep learning for alternative splice sites

Example:
    >>> from agentic_spliceai.splice_engine import BaseModelConfig, Registry
    >>> config = BaseModelConfig(base_model='openspliceai')
    >>> registry = Registry(build='GRCh38_MANE')
"""

__version__ = "0.1.0"

# Configuration
from .config import (
    Config,
    load_config,
    get_project_root,
    find_project_root,
)

# Resources
from .resources import (
    Registry,
    get_genomic_registry,
    standardize_splice_sites_schema,
    standardize_all_schemas,
)

# Base Layer
from .base_layer import (
    BaseModelConfig,
    SpliceAIConfig,
    OpenSpliceAIConfig,
    create_config,
    BaseModelRunner,
    BaseModelResult,
    GeneManifest,
    PredictionResult,
)

# Meta Layer
from .meta_layer import (
    MetaLayerConfig,
    FeatureSchema,
    DEFAULT_SCHEMA,
    LABEL_ENCODING,
    LABEL_DECODING,
)

# Meta Models (Prediction Workflows)
from .meta_models import (
    run_enhanced_splice_prediction_workflow,
    run_base_model_predictions,
    validate_workflow_config,
    prepare_gene_annotations,
    prepare_splice_site_annotations,
)

# Utils (commonly used)
from .utils import (
    is_dataframe_empty,
    smart_read_csv,
    ensure_directory,
    print_emphasized,
)

__all__ = [
    # Version
    '__version__',
    # Configuration
    'Config',
    'load_config',
    'get_project_root',
    'find_project_root',
    # Resources
    'Registry',
    'get_genomic_registry',
    'standardize_splice_sites_schema',
    'standardize_all_schemas',
    # Base Layer
    'BaseModelConfig',
    'SpliceAIConfig',
    'OpenSpliceAIConfig',
    'create_config',
    'BaseModelRunner',
    'BaseModelResult',
    'GeneManifest',
    'PredictionResult',
    # Meta Layer
    'MetaLayerConfig',
    'FeatureSchema',
    'DEFAULT_SCHEMA',
    'LABEL_ENCODING',
    'LABEL_DECODING',
    # Meta Models (Prediction Workflows)
    'run_enhanced_splice_prediction_workflow',
    'run_base_model_predictions',
    'validate_workflow_config',
    'prepare_gene_annotations',
    'prepare_splice_site_annotations',
    # Utils
    'is_dataframe_empty',
    'smart_read_csv',
    'ensure_directory',
    'print_emphasized',
]
