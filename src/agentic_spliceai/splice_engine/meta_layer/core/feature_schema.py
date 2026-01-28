"""
Feature schema definitions for the meta-layer.

Defines the standardized feature set used across all base models,
ensuring consistent input format for the meta-layer.

Ported from: meta_spliceai/splice_engine/meta_layer/core/feature_schema.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FeatureSchema:
    """
    Standardized feature schema for meta-layer training.
    
    All base models should produce artifacts with these features
    (or a subset thereof) to be compatible with the meta-layer.
    
    Attributes
    ----------
    SEQUENCE_COL : str
        Column containing the contextual DNA sequence.
    
    BASE_SCORE_COLS : list of str
        Core splice site probability columns from base model.
    
    CONTEXT_SCORE_COLS : list of str
        Neighboring position scores for context.
    
    DERIVED_FEATURE_COLS : list of str
        Engineered features derived from base scores.
    
    LABEL_COLS : list of str
        Label columns for training.
    
    METADATA_COLS : list of str
        Metadata columns (not used in training).
    """
    
    # Sequence column (501nt window)
    SEQUENCE_COL: str = 'sequence'
    
    # Core base model scores
    BASE_SCORE_COLS: List[str] = field(default_factory=lambda: [
        'donor_score',
        'acceptor_score', 
        'neither_score'
    ])
    
    # Context scores (±2 positions)
    CONTEXT_SCORE_COLS: List[str] = field(default_factory=lambda: [
        'context_score_m2',  # -2 positions
        'context_score_m1',  # -1 position
        'context_score_p1',  # +1 position
        'context_score_p2'   # +2 positions
    ])
    
    # Derived probability features
    PROBABILITY_FEATURE_COLS: List[str] = field(default_factory=lambda: [
        'relative_donor_probability',
        'splice_probability',
        'donor_acceptor_diff',
        'splice_neither_diff',
        'donor_acceptor_logodds',
        'splice_neither_logodds',
        'probability_entropy'
    ])
    
    # Local context features
    CONTEXT_PATTERN_COLS: List[str] = field(default_factory=lambda: [
        'context_neighbor_mean',
        'context_asymmetry',
        'context_max'
    ])
    
    # Donor-specific pattern features
    DONOR_PATTERN_COLS: List[str] = field(default_factory=lambda: [
        'donor_diff_m1',
        'donor_diff_m2',
        'donor_diff_p1',
        'donor_diff_p2',
        'donor_surge_ratio',
        'donor_is_local_peak',
        'donor_weighted_context',
        'donor_peak_height_ratio',
        'donor_second_derivative',
        'donor_signal_strength',
        'donor_context_diff_ratio'
    ])
    
    # Acceptor-specific pattern features
    ACCEPTOR_PATTERN_COLS: List[str] = field(default_factory=lambda: [
        'acceptor_diff_m1',
        'acceptor_diff_m2',
        'acceptor_diff_p1',
        'acceptor_diff_p2',
        'acceptor_surge_ratio',
        'acceptor_is_local_peak',
        'acceptor_weighted_context',
        'acceptor_peak_height_ratio',
        'acceptor_second_derivative',
        'acceptor_signal_strength',
        'acceptor_context_diff_ratio'
    ])
    
    # Comparative features
    COMPARATIVE_COLS: List[str] = field(default_factory=lambda: [
        'donor_acceptor_peak_ratio',
        'type_signal_difference',
        'score_difference_ratio',
        'signal_strength_ratio'
    ])
    
    # Label columns (ground truth)
    LABEL_COLS: List[str] = field(default_factory=lambda: [
        'splice_type',  # Ground truth: 'donor', 'acceptor', ''
    ])
    
    # LEAKAGE COLUMNS - NEVER use these as features!
    # These columns directly encode or correlate with the label.
    LEAKAGE_COLS: List[str] = field(default_factory=lambda: [
        'splice_type',         # The target label itself
        'pred_type',           # Base model prediction (derived from splice_type comparison)
        'true_position',       # Exact coordinate of real splice site
        'predicted_position',  # Tightly correlated with label
        'is_correct',          # Whether base model was correct
        'error_type',          # FP/FN/TP/TN classification
    ])
    
    # Metadata columns (not for training - high cardinality, poor generalization)
    METADATA_COLS: List[str] = field(default_factory=lambda: [
        'gene_id',
        'transcript_id',
        'gene_name',
        'gene_type',
        'chrom',
        'strand',
        'position',
        'absolute_position',
        'window_start',
        'window_end',
        'transcript_count',
    ])
    
    def get_all_feature_cols(self) -> List[str]:
        """Get all feature columns (excluding sequence, labels, metadata, leakage)."""
        return (
            self.BASE_SCORE_COLS +
            self.CONTEXT_SCORE_COLS +
            self.PROBABILITY_FEATURE_COLS +
            self.CONTEXT_PATTERN_COLS +
            self.DONOR_PATTERN_COLS +
            self.ACCEPTOR_PATTERN_COLS +
            self.COMPARATIVE_COLS
        )
    
    def get_minimal_feature_cols(self) -> List[str]:
        """Get minimal feature set for lightweight models."""
        return (
            self.BASE_SCORE_COLS +
            self.CONTEXT_SCORE_COLS +
            self.PROBABILITY_FEATURE_COLS[:4]  # Most important probability features
        )
    
    def get_excluded_cols(self) -> List[str]:
        """Get all columns that should NEVER be used as features.
        
        This includes:
        - LEAKAGE_COLS: Columns that leak label information
        - METADATA_COLS: High-cardinality columns that don't generalize
        - LABEL_COLS: The target labels themselves
        """
        return list(set(
            self.LEAKAGE_COLS +
            self.METADATA_COLS +
            self.LABEL_COLS
        ))
    
    def get_training_cols(self) -> List[str]:
        """Get all columns needed for training (features + sequence + label)."""
        return [self.SEQUENCE_COL] + self.get_all_feature_cols() + self.LABEL_COLS
    
    def is_leaky_column(self, col_name: str) -> bool:
        """Check if a column is a known leakage source."""
        return col_name in self.LEAKAGE_COLS
    
    def is_metadata_column(self, col_name: str) -> bool:
        """Check if a column is metadata (should not be used as feature)."""
        return col_name in self.METADATA_COLS


# Label encoding
LABEL_ENCODING = {
    'donor': 0,
    'acceptor': 1,
    'neither': 2,
    '': 2  # Empty string also maps to 'neither'
}

LABEL_DECODING = {
    0: 'donor',
    1: 'acceptor',
    2: 'neither'
}

# Default schema instance
DEFAULT_SCHEMA = FeatureSchema()
