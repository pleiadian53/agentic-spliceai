"""
Position coordinate types and conversion utilities.

This module resolves the ambiguity of the 'position' column which can represent
either absolute genomic coordinates or strand-dependent relative positions
depending on the context.

Ported from: meta_spliceai/splice_engine/meta_models/core/position_types.py

Coordinate Systems:
-------------------
1. ABSOLUTE: Genomic coordinates from reference assembly (e.g., chr17:41,196,312)
   - Used in: GTF files, splice_sites_enhanced.tsv, genomic annotation files
   - Independent of strand
   - Always increasing from 5' to 3' on the reference

2. RELATIVE: Strand-dependent positions within a gene (1-indexed)
   - Used in: Prediction artifacts, nucleotide_scores.tsv, meta-model training data
   - Position 1 = 5' end in transcription space
   - Positive strand: position 1 = gene_start (lowest coordinate)
   - Negative strand: position 1 = gene_end (highest coordinate)

Usage:
------
    from agentic_spliceai.splice_engine.base_layer.data.position_types import (
        PositionType, 
        absolute_to_relative, 
        relative_to_absolute,
        validate_position_range
    )
    
    # Convert absolute to relative
    rel_pos = absolute_to_relative(41196312, gene_start=41196312, gene_end=41277500, strand='+')
    # Returns: 1
    
    # Convert relative to absolute  
    abs_pos = relative_to_absolute(1, gene_start=41196312, gene_end=41277500, strand='+')
    # Returns: 41196312
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Optional, Tuple
import numpy as np


class PositionType(Enum):
    """
    Enumeration of position coordinate types.
    
    ABSOLUTE: Genomic coordinates from reference assembly
    RELATIVE: Strand-dependent positions within a gene (1-indexed, 5' to 3')
    """
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


@dataclass
class GeneCoordinates:
    """
    Gene coordinate information for position conversions.
    
    Attributes
    ----------
    gene_start : int
        Start position in absolute genomic coordinates (lower value)
    gene_end : int
        End position in absolute genomic coordinates (higher value)
    strand : str
        Strand orientation ('+' or '-')
    gene_id : str, optional
        Gene identifier for debugging/logging
    """
    gene_start: int
    gene_end: int
    strand: str
    gene_id: Optional[str] = None
    
    def __post_init__(self):
        if self.strand not in ('+', '-'):
            raise ValueError(f"Invalid strand: {self.strand}. Must be '+' or '-'")
        if self.gene_start > self.gene_end:
            raise ValueError(
                f"gene_start ({self.gene_start}) must be <= gene_end ({self.gene_end})"
            )
    
    @property
    def length(self) -> int:
        """Gene length in nucleotides (1-indexed, inclusive)."""
        return self.gene_end - self.gene_start + 1


def absolute_to_relative(
    absolute_pos: Union[int, List[int], np.ndarray],
    gene_start: int,
    gene_end: int,
    strand: str
) -> Union[int, List[int], np.ndarray]:
    """
    Convert absolute genomic coordinate(s) to relative position(s).
    
    Relative positions are 1-indexed and run 5' to 3' in transcription space:
    - Positive strand: position 1 = gene_start
    - Negative strand: position 1 = gene_end
    
    Parameters
    ----------
    absolute_pos : int, List[int], or np.ndarray
        Absolute genomic coordinate(s)
    gene_start : int
        Gene start in absolute coordinates (lower value)
    gene_end : int
        Gene end in absolute coordinates (higher value)
    strand : str
        Strand orientation ('+' or '-')
        
    Returns
    -------
    int, List[int], or np.ndarray
        Relative position(s) (1-indexed, 5' to 3' in transcription space)
        
    Examples
    --------
    >>> # Positive strand gene
    >>> absolute_to_relative(41196312, gene_start=41196312, gene_end=41277500, strand='+')
    1
    >>> absolute_to_relative(41196313, gene_start=41196312, gene_end=41277500, strand='+')
    2
    
    >>> # Negative strand gene (BRCA1)
    >>> absolute_to_relative(41277500, gene_start=41196312, gene_end=41277500, strand='-')
    1
    >>> absolute_to_relative(41277499, gene_start=41196312, gene_end=41277500, strand='-')
    2
    """
    if strand == '+':
        if isinstance(absolute_pos, (list, np.ndarray)):
            return [p - gene_start + 1 for p in absolute_pos] if isinstance(absolute_pos, list) else absolute_pos - gene_start + 1
        return absolute_pos - gene_start + 1
    elif strand == '-':
        if isinstance(absolute_pos, (list, np.ndarray)):
            return [gene_end - p + 1 for p in absolute_pos] if isinstance(absolute_pos, list) else gene_end - absolute_pos + 1
        return gene_end - absolute_pos + 1
    else:
        raise ValueError(f"Invalid strand: {strand}. Must be '+' or '-'")


def relative_to_absolute(
    relative_pos: Union[int, List[int], np.ndarray],
    gene_start: int,
    gene_end: int,
    strand: str
) -> Union[int, List[int], np.ndarray]:
    """
    Convert relative position(s) to absolute genomic coordinate(s).
    
    Relative positions are 1-indexed and run 5' to 3' in transcription space:
    - Positive strand: position 1 = gene_start
    - Negative strand: position 1 = gene_end
    
    Parameters
    ----------
    relative_pos : int, List[int], or np.ndarray
        Relative position(s) (1-indexed)
    gene_start : int
        Gene start in absolute coordinates (lower value)
    gene_end : int
        Gene end in absolute coordinates (higher value)
    strand : str
        Strand orientation ('+' or '-')
        
    Returns
    -------
    int, List[int], or np.ndarray
        Absolute genomic coordinate(s)
        
    Examples
    --------
    >>> # Positive strand gene
    >>> relative_to_absolute(1, gene_start=41196312, gene_end=41277500, strand='+')
    41196312
    >>> relative_to_absolute(2, gene_start=41196312, gene_end=41277500, strand='+')
    41196313
    
    >>> # Negative strand gene (BRCA1)
    >>> relative_to_absolute(1, gene_start=41196312, gene_end=41277500, strand='-')
    41277500
    >>> relative_to_absolute(2, gene_start=41196312, gene_end=41277500, strand='-')
    41277499
    """
    if strand == '+':
        if isinstance(relative_pos, (list, np.ndarray)):
            return [gene_start + p - 1 for p in relative_pos] if isinstance(relative_pos, list) else gene_start + relative_pos - 1
        return gene_start + relative_pos - 1
    elif strand == '-':
        if isinstance(relative_pos, (list, np.ndarray)):
            return [gene_end - p + 1 for p in relative_pos] if isinstance(relative_pos, list) else gene_end - relative_pos + 1
        return gene_end - relative_pos + 1
    else:
        raise ValueError(f"Invalid strand: {strand}. Must be '+' or '-'")


def validate_position_range(
    position: int,
    position_type: PositionType,
    gene_start: int,
    gene_end: int,
    strict: bool = False
) -> bool:
    """
    Validate that a position is within expected range for its type.
    
    Parameters
    ----------
    position : int
        Position value to validate
    position_type : PositionType
        Type of position (ABSOLUTE or RELATIVE)
    gene_start : int
        Gene start in absolute coordinates
    gene_end : int
        Gene end in absolute coordinates
    strict : bool, optional
        If True, raise ValueError for invalid positions. If False, return False.
        
    Returns
    -------
    bool
        True if position is valid
        
    Raises
    ------
    ValueError
        If strict=True and position is invalid
    """
    gene_length = gene_end - gene_start + 1
    
    if position_type == PositionType.ABSOLUTE:
        valid = gene_start <= position <= gene_end
        error_msg = f"Absolute position {position} outside gene range [{gene_start}, {gene_end}]"
    else:  # RELATIVE
        valid = 1 <= position <= gene_length
        error_msg = f"Relative position {position} outside range [1, {gene_length}]"
    
    if not valid and strict:
        raise ValueError(error_msg)
    
    return valid


def infer_position_type(
    positions: Union[List[int], np.ndarray],
    gene_start: int,
    gene_end: int,
    threshold: float = 0.9
) -> Tuple[PositionType, float]:
    """
    Infer whether positions are absolute or relative based on their values.
    
    This is a heuristic helper for debugging/migration. In production code,
    position types should be explicitly known and tracked.
    
    Parameters
    ----------
    positions : List[int] or np.ndarray
        Position values to analyze
    gene_start : int
        Gene start in absolute coordinates
    gene_end : int
        Gene end in absolute coordinates
    threshold : float, optional
        Confidence threshold (fraction of positions that must match), by default 0.9
        
    Returns
    -------
    Tuple[PositionType, float]
        Inferred position type and confidence score
        
    Examples
    --------
    >>> # Absolute positions (values around 41 million for chr17 gene)
    >>> infer_position_type([41196312, 41196313, 41196314], 41196312, 41277500)
    (PositionType.ABSOLUTE, 1.0)
    
    >>> # Relative positions (values 1, 2, 3...)
    >>> infer_position_type([1, 2, 3, 4, 5], 41196312, 41277500)
    (PositionType.RELATIVE, 1.0)
    """
    if len(positions) == 0:
        return PositionType.RELATIVE, 0.0
    
    gene_length = gene_end - gene_start + 1
    positions_array = np.array(positions)
    
    # Count positions that fall in absolute range
    in_absolute_range = np.sum((positions_array >= gene_start) & (positions_array <= gene_end))
    absolute_score = in_absolute_range / len(positions)
    
    # Count positions that fall in relative range
    in_relative_range = np.sum((positions_array >= 1) & (positions_array <= gene_length))
    relative_score = in_relative_range / len(positions)
    
    # Additional heuristic: if positions are much larger than gene_length, likely absolute
    max_pos = np.max(positions_array)
    if max_pos > gene_length * 100:  # Way larger than gene length
        return PositionType.ABSOLUTE, absolute_score
    
    # If both could match (gene spans 1 to gene_length and happens to overlap),
    # use the magnitude as a tiebreaker
    if absolute_score >= threshold and relative_score >= threshold:
        # Check if median position is closer to gene coordinates or small numbers
        median_pos = np.median(positions_array)
        if median_pos > gene_length:
            return PositionType.ABSOLUTE, absolute_score
        else:
            return PositionType.RELATIVE, relative_score
    
    if absolute_score >= relative_score:
        return PositionType.ABSOLUTE, absolute_score
    else:
        return PositionType.RELATIVE, relative_score


# Convenience functions for batch operations
def convert_positions_batch(
    positions: Union[List[int], np.ndarray],
    gene_start: int,
    gene_end: int,
    strand: str,
    from_type: PositionType,
    to_type: PositionType
) -> Union[List[int], np.ndarray]:
    """
    Convert a batch of positions between coordinate systems.
    
    Parameters
    ----------
    positions : List[int] or np.ndarray
        Positions to convert
    gene_start : int
        Gene start in absolute coordinates
    gene_end : int
        Gene end in absolute coordinates
    strand : str
        Strand orientation ('+' or '-')
    from_type : PositionType
        Source coordinate type
    to_type : PositionType
        Target coordinate type
        
    Returns
    -------
    List[int] or np.ndarray
        Converted positions (same type as input)
    """
    if from_type == to_type:
        return positions
    
    if from_type == PositionType.ABSOLUTE and to_type == PositionType.RELATIVE:
        return absolute_to_relative(positions, gene_start, gene_end, strand)
    elif from_type == PositionType.RELATIVE and to_type == PositionType.ABSOLUTE:
        return relative_to_absolute(positions, gene_start, gene_end, strand)
    else:
        raise ValueError(f"Unknown conversion: {from_type} -> {to_type}")

