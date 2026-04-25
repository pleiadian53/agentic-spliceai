"""SpliceAI predictor adapter.

Thin wrapper over the existing ``BaseModelRunner`` that exposes SpliceAI
through the ``BasePredictor`` protocol.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import polars as pl

from ..protocol import PredictionResult
from ..registry import register_predictor

logger = logging.getLogger(__name__)

_PREDICTOR_NAME = "spliceai"


class SpliceAIPredictor:
    """BasePredictor adapter for SpliceAI (GRCh37, Ensembl-trained)."""

    name = _PREDICTOR_NAME
    training_build = "GRCh37"
    annotation_source = "ensembl"

    def __init__(self) -> None:
        # Lazy-import the runner so that registry population doesn't force
        # pulling in the full splice_engine dependency tree.
        from agentic_spliceai.splice_engine.base_layer.models.runner import (
            BaseModelRunner,
        )

        self._runner = BaseModelRunner()

    def predict_genes(
        self,
        genes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        return _run(
            self._runner,
            model_name=self.name,
            genes=genes,
            threshold=threshold,
            verbosity=verbosity,
        )

    def predict_chromosomes(
        self,
        chromosomes: List[str],
        *,
        threshold: float = 0.5,
        verbosity: int = 1,
    ) -> PredictionResult:
        genes = _resolve_chromosome_genes(
            chromosomes=chromosomes,
            build=self.training_build,
            annotation_source=self.annotation_source,
        )
        return _run(
            self._runner,
            model_name=self.name,
            genes=genes,
            threshold=threshold,
            verbosity=verbosity,
        )

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "training_build": self.training_build,
            "annotation_source": self.annotation_source,
            "training_annotation": "GENCODE V24lift37",
            "paper": "Jaganathan et al., Cell 2019",
            "notes": "Ensemble of 5 models, TensorFlow backend.",
        }


@register_predictor(
    _PREDICTOR_NAME,
    description={
        "training_build": "GRCh37",
        "annotation_source": "ensembl",
        "paper": "Jaganathan et al., Cell 2019",
        "notes": "Ensemble of 5 models, TensorFlow backend.",
    },
)
def _factory() -> SpliceAIPredictor:
    return SpliceAIPredictor()


# ---------------------------------------------------------------------------
# Shared helpers (used by both spliceai.py and openspliceai.py).
# ---------------------------------------------------------------------------


def _run(
    runner,
    *,
    model_name: str,
    genes: List[str],
    threshold: float,
    verbosity: int,
) -> PredictionResult:
    """Call BaseModelRunner and map its output to PredictionResult."""
    from agentic_spliceai.splice_engine.resources.schema import ensure_chrom_column

    test_name = f"app_base_layer_{model_name}"
    bm_result = runner.run_single_model(
        model_name=model_name,
        target_genes=list(genes),
        test_name=test_name,
        threshold=threshold,
        verbosity=verbosity,
    )

    positions = bm_result.positions
    if positions is not None and positions.height > 0:
        positions = ensure_chrom_column(positions)

    return PredictionResult(
        predictor_name=model_name,
        positions=positions if positions is not None else pl.DataFrame(),
        processed_genes=set(bm_result.processed_genes),
        missing_genes=set(bm_result.missing_genes),
        runtime_seconds=float(bm_result.runtime_seconds),
        metadata={
            "threshold": threshold,
            "paths": dict(bm_result.paths),
        },
        error=bm_result.error,
    )


def _resolve_chromosome_genes(
    *,
    chromosomes: List[str],
    build: str,
    annotation_source: str,
) -> List[str]:
    """Resolve a list of chromosomes to the gene symbols they contain.

    Uses the resource manager to pick the right GTF, then loads protein-
    coding gene symbols from it.
    """
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        filter_by_chromosomes,
        load_gene_annotations,
    )

    gene_df = load_gene_annotations(
        build=build,
        annotation_source=annotation_source,
    )

    filtered = filter_by_chromosomes(gene_df, chromosomes, build=build)

    # Prefer gene_name; fall back to gene_id when symbol missing.
    gene_col = "gene_name" if "gene_name" in filtered.columns else "gene_id"
    genes = sorted({str(g) for g in filtered[gene_col].to_list() if g})
    return genes
