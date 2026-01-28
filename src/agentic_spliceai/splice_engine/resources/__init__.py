"""Resource management for splice engine.

This package provides utilities for:
- Path resolution (Registry)
- Schema standardization
- Data validation
- Gene ID mapping
- Artifact tracking

Exports:
    Registry: Path resolution for genomic resources
    get_genomic_registry: Get cached registry instance
    standardize_splice_sites_schema: Standardize column names
    standardize_all_schemas: Standardize multiple datasets
"""

from .registry import Registry, get_genomic_registry
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
    'Registry',
    'get_genomic_registry',
    'standardize_splice_sites_schema',
    'standardize_gene_features_schema',
    'standardize_transcript_features_schema',
    'standardize_exon_features_schema',
    'standardize_all_schemas',
    'get_standard_column_mapping',
    'SPLICE_SITE_COLUMN_MAPPING',
    'GENE_FEATURE_COLUMN_MAPPING',
]
