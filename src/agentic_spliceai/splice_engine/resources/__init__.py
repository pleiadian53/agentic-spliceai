"""Resource management for splice engine.

This package provides utilities for:
- Model-specific resource resolution (ModelResources)
- Path resolution (Registry)
- Schema standardization
- Data validation
- Gene ID mapping
- Artifact tracking

Exports:
    ModelResources: Model-specific resource manager
    get_model_resources: Get resources for a base model
    list_available_models: List configured models
    Registry: Path resolution for genomic resources
    get_genomic_registry: Get cached registry instance
    standardize_splice_sites_schema: Standardize column names
    standardize_all_schemas: Standardize multiple datasets
"""

from .registry import Registry, get_genomic_registry
from .model_resources import (
    ModelResources,
    get_model_resources,
    list_available_models,
    get_model_info,
)
from .schema import (
    standardize_splice_sites_schema,
    standardize_gene_features_schema,
    standardize_transcript_features_schema,
    standardize_exon_features_schema,
    standardize_all_schemas,
    get_standard_column_mapping,
    SPLICE_SITE_COLUMN_MAPPING,
    GENE_FEATURE_COLUMN_MAPPING,
)

__all__ = [
    # Model resources (NEW!)
    'ModelResources',
    'get_model_resources',
    'list_available_models',
    'get_model_info',
    # Registry
    'Registry',
    'get_genomic_registry',
    # Schema
    'standardize_splice_sites_schema',
    'standardize_gene_features_schema',
    'standardize_transcript_features_schema',
    'standardize_exon_features_schema',
    'standardize_all_schemas',
    'get_standard_column_mapping',
    'SPLICE_SITE_COLUMN_MAPPING',
    'GENE_FEATURE_COLUMN_MAPPING',
]
