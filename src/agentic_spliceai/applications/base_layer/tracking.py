"""Weights & Biases tracker for base-layer evaluations.

Thin shim over :mod:`agentic_spliceai.applications._common.tracking` that
adds the predictor-specific context (``predictor.name``,
``training_build``, ``annotation_source``) to the run config. Keeps the
``evaluator.py`` call site unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .._common.tracking import WandbTracker, start_run as _common_start_run
from .protocol import BasePredictor

logger = logging.getLogger(__name__)


def start_run(
    *,
    predictor: BasePredictor,
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    threshold: float = 0.5,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    run_name: Optional[str] = None,
) -> Optional[WandbTracker]:
    """Start a W&B run for a base-layer evaluation.

    Silent fallback: returns ``None`` when wandb is unavailable.
    """
    config: Dict[str, Any] = {
        "predictor": predictor.name,
        "training_build": predictor.training_build,
        "annotation_source": predictor.annotation_source,
        "threshold": threshold,
    }
    if genes is not None:
        config["target_mode"] = "genes"
        config["n_genes"] = len(genes)
        if len(genes) <= 50:
            config["genes"] = list(genes)
    elif chromosomes is not None:
        config["target_mode"] = "chromosomes"
        config["chromosomes"] = list(chromosomes)

    merged_tags = [predictor.name, *(tags or [])]

    return _common_start_run(
        app="base_layer",
        run_kind="evaluate",
        config=config,
        project=project,
        tags=merged_tags,
        run_name=run_name,
    )


__all__ = ["WandbTracker", "start_run"]
