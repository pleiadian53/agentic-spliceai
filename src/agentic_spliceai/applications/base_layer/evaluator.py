"""Application-level evaluation for base-layer predictors.

Packaged version of ``examples/base_layer/03_prediction_with_evaluation.py``.
Runs a registered predictor and compares the per-nucleotide scores against
MANE / Ensembl splice-site annotations.

Reuses existing evaluation primitives from
``splice_engine.base_layer.prediction.evaluation`` — this module only
orchestrates and records results (with optional wandb tracking).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .protocol import BasePredictor, PredictionResult
from .registry import get_predictor
from .runner import run_prediction

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from a base-predictor evaluation run."""

    predictor_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    prediction_result: Optional[PredictionResult] = None
    runtime_seconds: float = 0.0
    output_dir: Optional[Path] = None
    tracking_run_id: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


def evaluate_predictor(
    *,
    predictor_name: str,
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    threshold: float = 0.5,
    output_dir: Optional[Path] = None,
    track: bool = False,
    tracking_project: Optional[str] = None,
    tracking_tags: Optional[List[str]] = None,
    verbosity: int = 1,
    predictor: Optional[BasePredictor] = None,
) -> EvaluationResult:
    """Run prediction + evaluation against annotated splice sites.

    Parameters
    ----------
    predictor_name
        Registered predictor name. Ignored when ``predictor`` is passed.
    genes, chromosomes
        Exactly one must be provided — see :func:`run_prediction`.
    threshold
        Splice-site probability threshold.
    output_dir
        Where to write metrics JSON, predictions parquet, and run manifest.
    track
        When ``True``, log metrics + artifacts to Weights & Biases.
        Falls back silently when ``wandb`` is unavailable.
    tracking_project
        W&B project name. Defaults to ``$WANDB_PROJECT`` or
        ``"agentic-spliceai-base-layer"``.
    tracking_tags
        Optional list of tags (e.g., ``["chr21", "baseline"]``).
    verbosity
        0 = silent, 1 = progress, 2 = debug.
    predictor
        Pre-instantiated predictor (skip registry lookup).
    """
    import time

    if predictor is None:
        predictor = get_predictor(predictor_name)
    predictor_name = predictor.name

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    tracker = None
    if track:
        from . import tracking

        tracker = tracking.start_run(
            predictor=predictor,
            genes=genes,
            chromosomes=chromosomes,
            threshold=threshold,
            project=tracking_project,
            tags=tracking_tags,
        )

    try:
        pred_result = run_prediction(
            predictor_name=predictor_name,
            genes=genes,
            chromosomes=chromosomes,
            threshold=threshold,
            output_dir=output_dir,
            verbosity=verbosity,
            predictor=predictor,
        )

        if not pred_result.success:
            raise RuntimeError(
                f"Prediction failed for {predictor_name}: {pred_result.error}"
            )

        metrics = _compute_metrics(
            pred_result=pred_result,
            predictor=predictor,
            genes=genes,
            chromosomes=chromosomes,
            threshold=threshold,
            verbosity=verbosity,
        )

        if output_dir is not None:
            _persist_metrics(metrics, output_dir=output_dir)

        if tracker is not None:
            tracker.log_metrics(metrics)
            if output_dir is not None:
                tracker.log_artifact(
                    output_dir / "predictions.parquet",
                    artifact_type="predictions",
                )

        runtime = time.time() - t0
        if verbosity >= 1:
            _print_summary(metrics, runtime=runtime, predictor_name=predictor_name)

        return EvaluationResult(
            predictor_name=predictor_name,
            metrics=metrics,
            prediction_result=pred_result,
            runtime_seconds=runtime,
            output_dir=output_dir,
            tracking_run_id=(tracker.run_id if tracker is not None else None),
        )

    except Exception as exc:
        runtime = time.time() - t0
        logger.exception("Evaluation failed: %s", exc)
        return EvaluationResult(
            predictor_name=predictor_name,
            metrics={},
            prediction_result=None,
            runtime_seconds=runtime,
            output_dir=output_dir,
            error=str(exc),
        )
    finally:
        if tracker is not None:
            tracker.finish()


# ---------------------------------------------------------------------------
# Internals — reuse existing library utilities.
# ---------------------------------------------------------------------------


def _compute_metrics(
    *,
    pred_result: PredictionResult,
    predictor: BasePredictor,
    genes: Optional[List[str]],
    chromosomes: Optional[List[str]],
    threshold: float,
    verbosity: int,
) -> Dict[str, Any]:
    """Compute evaluation metrics by joining predictions against annotations.

    Uses existing library primitives from ``splice_engine.base_layer``.
    """
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        prepare_splice_site_annotations,
    )
    from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
        evaluate_splice_site_predictions,
    )

    if verbosity >= 1:
        logger.info(
            "Preparing splice-site annotations (build=%s, source=%s)...",
            predictor.training_build, predictor.annotation_source,
        )

    # target_genes is a hint for which annotations to load. When running
    # chromosomes, we rely on the predictor's processed_genes.
    target_genes = list(genes) if genes else sorted(pred_result.processed_genes)

    annotations = prepare_splice_site_annotations(
        target_genes=target_genes,
        build=predictor.training_build,
        annotation_source=predictor.annotation_source,
    )

    if verbosity >= 1:
        logger.info(
            "Evaluating %d predicted positions against %d annotated sites...",
            pred_result.positions.height,
            getattr(annotations, "height", len(annotations))
            if annotations is not None else 0,
        )

    eval_metrics = evaluate_splice_site_predictions(
        predictions_df=pred_result.positions,
        annotations_df=annotations,
        threshold=threshold,
    )

    # Normalize to a flat dict. evaluate_splice_site_predictions may return
    # a DataFrame, dict, or custom object depending on library vintage.
    metrics = _flatten_eval(eval_metrics)
    metrics["predictor"] = predictor.name
    metrics["training_build"] = predictor.training_build
    metrics["annotation_source"] = predictor.annotation_source
    metrics["threshold"] = threshold
    metrics["n_positions"] = int(pred_result.positions.height)
    metrics["n_processed_genes"] = len(pred_result.processed_genes)
    metrics["n_missing_genes"] = len(pred_result.missing_genes)
    if chromosomes is not None:
        metrics["chromosomes"] = sorted({str(c) for c in chromosomes})
    return metrics


def _flatten_eval(obj: Any) -> Dict[str, Any]:
    """Flatten library-produced eval output into a dict of scalars."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items()}
    if isinstance(obj, pl.DataFrame):
        if obj.height == 0:
            return {}
        # Single-row frame -> dict; multi-row -> record keyed by first col.
        if obj.height == 1:
            return dict(zip(obj.columns, obj.row(0)))
        return {"rows": obj.to_dicts()}
    # Duck-type for custom result objects.
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"result": obj}


def _persist_metrics(metrics: Dict[str, Any], *, output_dir: Path) -> None:
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as fh:
        json.dump(metrics, fh, indent=2, default=str)
    logger.info("Wrote evaluation metrics to %s", metrics_path)


def _print_summary(metrics: Dict[str, Any], *, runtime: float, predictor_name: str) -> None:
    print()
    print("=" * 70)
    print(f"Evaluation summary — predictor={predictor_name}")
    print("=" * 70)
    for key in (
        "n_positions", "n_processed_genes", "n_missing_genes",
        "precision", "recall", "f1", "accuracy",
        "donor_pr_auc", "acceptor_pr_auc",
    ):
        if key in metrics:
            val = metrics[key]
            if isinstance(val, float):
                print(f"  {key:>22}: {val:.4f}")
            else:
                print(f"  {key:>22}: {val}")
    print(f"  {'runtime_seconds':>22}: {runtime:.2f}")
    print("=" * 70)
    print()
