"""Application-level orchestration for base-layer prediction.

Packaged version of the prediction flow demonstrated in
``examples/base_layer/01,02,04,05``. Thin wrapper: resolves the predictor
from the registry, dispatches to either gene-list or chromosome-list
entry, validates the output, and persists results to disk.

This module does not re-implement prediction logic. It composes the
existing library (``splice_engine.base_layer``) through the
``BasePredictor`` protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import polars as pl

from .protocol import BasePredictor, PredictionResult, validate_prediction_result
from .registry import get_predictor

logger = logging.getLogger(__name__)


def run_prediction(
    *,
    predictor_name: str,
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
    verbosity: int = 1,
    predictor: Optional[BasePredictor] = None,
) -> PredictionResult:
    """Run a registered predictor over genes or chromosomes.

    Exactly one of ``genes`` or ``chromosomes`` must be provided.

    Parameters
    ----------
    predictor_name
        Registered predictor name (e.g. ``"spliceai"``, ``"openspliceai"``).
        Ignored when ``predictor`` is passed directly.
    genes
        Gene symbols to predict.
    chromosomes
        Chromosome identifiers (with or without ``chr`` prefix).
    threshold
        Splice-site probability threshold for classification.
    output_dir
        Optional directory to save the positions parquet and a run manifest
        JSON. If ``None``, results are not persisted.
    verbosity
        0 = silent, 1 = progress, 2 = debug.
    predictor
        Pre-instantiated predictor. When provided, ``predictor_name`` is
        taken from ``predictor.name``. Used for testing and custom wrappers.
    """
    if (genes is None) == (chromosomes is None):
        raise ValueError(
            "Exactly one of 'genes' or 'chromosomes' must be provided."
        )

    if predictor is None:
        predictor = get_predictor(predictor_name)
    predictor_name = predictor.name

    if verbosity >= 1:
        logger.info(
            "Running predictor %r (build=%s, annotation=%s)",
            predictor_name, predictor.training_build, predictor.annotation_source,
        )

    if genes is not None:
        result = predictor.predict_genes(
            genes=genes, threshold=threshold, verbosity=verbosity,
        )
    else:
        result = predictor.predict_chromosomes(
            chromosomes=chromosomes, threshold=threshold, verbosity=verbosity,
        )

    validate_prediction_result(result)

    if output_dir is not None:
        _persist(result, output_dir=Path(output_dir), verbosity=verbosity)

    return result


def _persist(
    result: PredictionResult,
    *,
    output_dir: Path,
    verbosity: int,
) -> None:
    """Persist a PredictionResult to disk.

    Layout::

        <output_dir>/
        ├── predictions.parquet  # positions
        └── run_manifest.json    # predictor info + counts + timestamp
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "predictions.parquet"
    manifest_path = output_dir / "run_manifest.json"

    if result.positions is not None and result.positions.height > 0:
        result.positions.write_parquet(parquet_path)
        if verbosity >= 1:
            logger.info(
                "Wrote %d positions to %s",
                result.positions.height, parquet_path,
            )

    manifest = {
        "predictor": result.predictor_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "n_positions": int(result.positions.height) if result.positions is not None else 0,
        "processed_genes": sorted(result.processed_genes),
        "missing_genes": sorted(result.missing_genes),
        "runtime_seconds": float(result.runtime_seconds),
        "metadata": _jsonable(result.metadata),
        "error": result.error,
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    if verbosity >= 1:
        logger.info("Wrote run manifest to %s", manifest_path)


def _jsonable(obj):
    """Best-effort conversion of predictor-specific metadata to JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
