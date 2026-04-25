"""OpenSpliceAI predictor adapter.

Thin wrapper over the existing ``BaseModelRunner`` that exposes
OpenSpliceAI through the ``BasePredictor`` protocol.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from ..protocol import PredictionResult
from ..registry import register_predictor
from .spliceai import _resolve_chromosome_genes, _run

logger = logging.getLogger(__name__)

_PREDICTOR_NAME = "openspliceai"


class OpenSpliceAIPredictor:
    """BasePredictor adapter for OpenSpliceAI (GRCh38, MANE-trained)."""

    name = _PREDICTOR_NAME
    training_build = "GRCh38"
    annotation_source = "mane"

    def __init__(self) -> None:
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
            "training_annotation": "MANE v1.3 RefSeq",
            "paper": "OpenSpliceAI GitHub",
            "notes": "Ensemble of 5 models, PyTorch backend (MPS-capable).",
        }


@register_predictor(
    _PREDICTOR_NAME,
    description={
        "training_build": "GRCh38",
        "annotation_source": "mane",
        "training_annotation": "MANE v1.3 RefSeq",
        "paper": "OpenSpliceAI GitHub",
        "notes": "Ensemble of 5 models, PyTorch backend (MPS-capable).",
    },
)
def _factory() -> OpenSpliceAIPredictor:
    return OpenSpliceAIPredictor()
