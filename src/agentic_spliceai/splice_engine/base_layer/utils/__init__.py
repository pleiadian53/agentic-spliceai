"""Utility functions for the base layer.

This package provides:
- Coordinate adjustment detection and application
- Caching for adjustment parameters
- Helper functions for splice site analysis
"""

from .coordinate_adjustment import (
    # Core adjustment functions
    normalize_strand,
    adjust_scores_hardcoded,
    apply_custom_adjustments,
    
    # Empirical detection
    empirical_infer_adjustments,
    auto_detect_adjustments,
    
    # Caching
    get_adjustment_cache_path,
    load_cached_adjustments,
    save_adjustments_to_cache,
    
    # Convenience
    get_or_detect_adjustments,
)

__all__ = [
    # Core adjustments
    'normalize_strand',
    'adjust_scores_hardcoded',
    'apply_custom_adjustments',
    
    # Detection
    'empirical_infer_adjustments',
    'auto_detect_adjustments',
    
    # Caching
    'get_adjustment_cache_path',
    'load_cached_adjustments',
    'save_adjustments_to_cache',
    
    # Convenience
    'get_or_detect_adjustments',
]
