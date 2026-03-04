"""Modality abstraction for multimodal feature engineering.

Defines the protocol that all evidence modalities must implement:
- ModalityConfig: typed, serializable configuration
- ModalityMeta: schema metadata (name, version, columns)
- Modality ABC: validate → transform contract

Each modality adds columns to a DataFrame without modifying existing ones,
enabling composable, order-independent feature pipelines.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Base configuration for a modality. Subclass per modality.

    Attributes
    ----------
    enabled : bool
        Whether this modality is active in the pipeline.
    """

    enabled: bool = True


@dataclass(frozen=True)
class ModalityMeta:
    """Schema metadata describing a modality's inputs and outputs.

    Attributes
    ----------
    name : str
        Registry key (e.g., 'base_scores', 'sequence').
    version : str
        Semantic version for reproducibility tracking.
    output_columns : tuple of str
        Columns this modality adds to the DataFrame.
    required_inputs : frozenset of str
        Columns that must exist in the input DataFrame.
    optional_inputs : frozenset of str
        Columns used if present but not required.
    description : str
        Human-readable description of the modality.
    """

    name: str
    version: str
    output_columns: tuple[str, ...]
    required_inputs: frozenset[str]
    optional_inputs: frozenset[str] = frozenset()
    description: str = ""


class Modality(ABC):
    """Abstract base class for feature engineering modalities.

    Each modality encapsulates one type of evidence (base scores,
    DNA sequence, annotations, etc.) and transforms it into features.

    Subclasses must implement:
    - ``meta`` property: declare inputs/outputs
    - ``transform()``: add columns to the DataFrame
    - ``default_config()``: return a sensible default configuration

    Rules:
    - Modalities are stateless across chunks
    - ``transform()`` must only ADD columns, never modify or drop
    - All operations should be vectorized Polars (no row iteration)
    - Context features must use ``.over('gene_id')`` to avoid cross-gene leakage
    """

    def __init__(self, config: ModalityConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def meta(self) -> ModalityMeta:
        """Return schema metadata for this modality."""

    def validate(self, available_columns: Set[str]) -> List[str]:
        """Check that prerequisites are met.

        Parameters
        ----------
        available_columns : set of str
            Columns present in the input DataFrame (including outputs
            from prior modalities in the pipeline).

        Returns
        -------
        list of str
            Error messages. Empty list means validation passed.
        """
        errors: List[str] = []
        missing = self.meta.required_inputs - available_columns
        if missing:
            errors.append(
                f"Modality '{self.meta.name}' requires columns {sorted(missing)} "
                f"but they are not available."
            )
        return errors

    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add this modality's feature columns to the DataFrame.

        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame. Must contain all ``meta.required_inputs``.

        Returns
        -------
        pl.DataFrame
            DataFrame with new columns appended. Existing columns
            must not be modified or dropped.
        """

    @classmethod
    @abstractmethod
    def default_config(cls) -> ModalityConfig:
        """Return the default configuration for this modality."""
