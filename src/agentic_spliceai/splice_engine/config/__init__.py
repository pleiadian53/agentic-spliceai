"""Configuration management for splice engine.

This package provides configuration classes and utilities for:
- Base model configuration (SpliceAI, OpenSpliceAI)
- Genomic resources configuration
- Data path resolution

Exports:
    Config: Genomic resources configuration dataclass
    load_config: Load configuration from YAML file
    get_project_root: Get the project root directory
    find_project_root: Find project root by markers
"""

from .genomic_config import (
    Config,
    load_config,
    get_project_root,
    find_project_root,
    filename,
)

__all__ = [
    'Config',
    'load_config',
    'get_project_root',
    'find_project_root',
    'filename',
]
