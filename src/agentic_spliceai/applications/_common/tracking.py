"""Generalised Weights & Biases experiment tracker.

Shared by the ``base_layer``, ``data_preparation``, and
``multimodal_features`` applications. Each caller supplies

- ``app``       — which application is running (e.g., ``"base_layer"``)
- ``run_kind``  — what the run does (e.g., ``"evaluate"``, ``"prepare"``)
- ``config``    — a flat dict of run metadata (predictor name, profile, build, ...)

and receives a :class:`WandbTracker` (or ``None`` if wandb is unavailable).

Silent fallback
---------------

- If the ``wandb`` package is not installed, :func:`start_run` returns ``None``.
- If ``WANDB_API_KEY`` is unset and ``WANDB_MODE`` is not ``offline``/``disabled``,
  :func:`start_run` returns ``None``.
- All public methods on :class:`WandbTracker` are safe to call even when
  the underlying run is ``None``.

This keeps every application usable without forcing wandb into every
environment — tracking is strictly opt-in via ``track=True`` on the CLI
or Python API.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_PREFIX = "agentic-spliceai"


class WandbTracker:
    """Opaque wrapper over a ``wandb.Run``.

    Callers should treat the returned object as opaque and use the
    methods below. If ``run`` is ``None`` the tracker is a no-op.
    """

    def __init__(self, run: Any) -> None:
        self._run = run

    @property
    def run_id(self) -> Optional[str]:
        return getattr(self._run, "id", None)

    @property
    def active(self) -> bool:
        return self._run is not None

    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log a dict of metrics.

        Scalar entries are logged as step metrics. Non-scalar entries are
        written to the run summary (when wandb allows) so they still show
        up in the UI.
        """
        if self._run is None:
            return
        scalars: Dict[str, Any] = {}
        non_scalars: Dict[str, Any] = {}
        for k, v in metrics.items():
            (scalars if _is_scalar(v) else non_scalars)[k] = v
        if scalars:
            try:
                self._run.log(scalars)
            except Exception as exc:  # noqa: BLE001
                logger.warning("wandb.log failed: %s", exc)
        if non_scalars:
            try:
                for k, v in non_scalars.items():
                    self._run.summary[k] = v
            except Exception:  # noqa: BLE001
                pass

    def log_artifact(
        self,
        path: Path,
        *,
        artifact_type: str = "data",
        name: Optional[str] = None,
    ) -> None:
        """Register a file or directory as a W&B artifact."""
        if self._run is None:
            return
        p = Path(path)
        if not p.exists():
            logger.debug("Artifact path does not exist, skipping: %s", p)
            return
        try:
            import wandb  # lazy import

            artifact_name = name or p.stem
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            if p.is_dir():
                artifact.add_dir(str(p))
            else:
                artifact.add_file(str(p))
            self._run.log_artifact(artifact)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log artifact %s: %s", p, exc)

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            pass


def start_run(
    *,
    app: str,
    run_kind: str,
    config: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    tags: Optional[List[str]] = None,
    run_name: Optional[str] = None,
) -> Optional[WandbTracker]:
    """Open a W&B run for the given ``app`` and ``run_kind``.

    Parameters
    ----------
    app
        Application identifier (``"base_layer"``, ``"data_preparation"``,
        ``"multimodal_features"``). Added to ``config`` and tags for
        filtering in the W&B UI.
    run_kind
        Run type (``"evaluate"``, ``"prepare"``, ``"fetch"``, ...).
    config
        Free-form metadata dict; merged with ``{"app", "run_kind"}``.
    project
        Explicit W&B project. When omitted, falls back to
        ``$WANDB_PROJECT`` and then ``agentic-spliceai-<app>``.
    tags
        Extra tags; the returned run is also tagged with ``app`` and
        ``run_kind`` automatically.
    run_name
        Optional explicit run name.

    Returns
    -------
    WandbTracker or None
        ``None`` when wandb is not installed or no API key is available.
    """
    try:
        import wandb  # lazy import
    except ImportError:
        logger.info("wandb not installed; tracking disabled.")
        return None

    if (
        not os.environ.get("WANDB_API_KEY")
        and os.environ.get("WANDB_MODE") not in ("offline", "disabled")
    ):
        logger.info(
            "WANDB_API_KEY not set and WANDB_MODE is not offline/disabled; "
            "tracking disabled. Run `wandb login` or set the env var."
        )
        return None

    resolved_project = (
        project
        or os.environ.get("WANDB_PROJECT")
        or f"{_DEFAULT_PROJECT_PREFIX}-{app}"
    )

    merged_config: Dict[str, Any] = {"app": app, "run_kind": run_kind}
    if config:
        merged_config.update(config)

    merged_tags: List[str] = [app, run_kind]
    if tags:
        merged_tags.extend(t for t in tags if t not in merged_tags)

    try:
        run = wandb.init(
            project=resolved_project,
            config=merged_config,
            name=run_name,
            tags=merged_tags,
            reinit=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("wandb.init failed: %s", exc)
        return None

    return WandbTracker(run)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_scalar(value: Any) -> bool:
    """True for int/float/str/bool; False for lists, dicts, paths, etc."""
    return isinstance(value, (int, float, str, bool))
