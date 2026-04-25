"""Base predictor protocol.

Every predictor registered in the base-layer application must satisfy
``BasePredictor``: per-nucleotide 3-class scores (neither, acceptor, donor)
over a target set of genes or genomic regions.

This is the single contract that makes the base layer pluggable. SpliceAI,
OpenSpliceAI, and foundation-model-derived classifiers (e.g., SpliceBERT +
SpliceClassifier) all satisfy it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Set, runtime_checkable

import polars as pl


@dataclass
class PredictionResult:
    """Result from a base predictor.

    The contract is **per-nucleotide 3-class probabilities** expressed as
    two columns (``donor_prob``, ``acceptor_prob``). The implicit third
    class is ``1 - donor_prob - acceptor_prob`` (neither).

    Attributes
    ----------
    predictor_name : str
        Identifier from the registry (e.g., ``"openspliceai"``).
    positions : pl.DataFrame
        One row per predicted position. Must contain columns:
        ``chrom``, ``position``, ``donor_prob``, ``acceptor_prob``.
        May contain additional predictor-specific columns
        (``pred_type``, strand, gene, etc.).
    processed_genes : Set[str]
        Genes successfully processed.
    missing_genes : Set[str]
        Target genes that could not be processed.
    runtime_seconds : float
        Wall-clock runtime for this prediction call.
    metadata : Dict[str, object]
        Free-form metadata (model version, context window, chunk size, etc.).
    error : Optional[str]
        Error message if the call failed; ``None`` on success.
    """

    predictor_name: str
    positions: pl.DataFrame
    processed_genes: Set[str] = field(default_factory=set)
    missing_genes: Set[str] = field(default_factory=set)
    runtime_seconds: float = 0.0
    metadata: Dict[str, object] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        n = self.positions.height if self.positions is not None else 0
        status = "ok" if self.success else f"error: {self.error}"
        return (
            f"PredictionResult(predictor={self.predictor_name}, "
            f"positions={n}, genes={len(self.processed_genes)}, "
            f"runtime={self.runtime_seconds:.2f}s, {status})"
        )


@runtime_checkable
class BasePredictor(Protocol):
    """Protocol every base-layer predictor must satisfy.

    Implementations may be thin wrappers over existing runners (SpliceAI,
    OpenSpliceAI) or full inference pipelines (foundation-model-derived
    classifiers). The protocol defines the minimum surface the CLI and
    evaluator need to talk to any predictor uniformly.
    """

    #: Short identifier used in the registry and on the CLI.
    name: str

    #: Genomic build this predictor was trained on (``"GRCh37"``, ``"GRCh38"``).
    training_build: str

    #: Annotation source used for training (``"mane"``, ``"ensembl"``, etc.).
    annotation_source: str

    def predict_genes(
        self,
        genes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        """Run prediction over a list of gene symbols.

        Parameters
        ----------
        genes
            List of gene symbols (e.g. ``["BRCA1", "TP53"]``).
        threshold
            Splice-site probability threshold used for classification.
            Downstream consumers (evaluator) may override.
        verbosity
            0 = silent, 1 = progress, 2 = debug.
        """
        ...

    def predict_chromosomes(
        self,
        chromosomes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        """Run prediction over whole chromosomes.

        Parameters
        ----------
        chromosomes
            Chromosome identifiers (with or without ``chr`` prefix).
        threshold
            See :meth:`predict_genes`.
        verbosity
            See :meth:`predict_genes`.
        """
        ...

    def describe(self) -> Dict[str, object]:
        """Return a dict describing this predictor for logging and UI.

        Should include at minimum: ``name``, ``training_build``,
        ``annotation_source``, ``training_annotation``, and any
        artifact paths.
        """
        ...


def validate_prediction_result(result: PredictionResult) -> None:
    """Validate that a ``PredictionResult`` satisfies the per-nt 3-class
    contract.

    Called by the CLI and evaluator before consuming a predictor's output.
    Raises ``ValueError`` on violations so integration bugs surface early.
    """
    if not result.success:
        return  # errored results are returned as-is; validation skipped

    if result.positions is None:
        raise ValueError(
            f"Predictor {result.predictor_name!r} returned no positions DataFrame."
        )

    required = {"chrom", "position", "donor_prob", "acceptor_prob"}
    missing = required - set(result.positions.columns)
    if missing:
        raise ValueError(
            f"Predictor {result.predictor_name!r} PredictionResult is missing "
            f"required columns: {sorted(missing)}. Found: "
            f"{sorted(result.positions.columns)}"
        )
