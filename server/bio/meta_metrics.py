"""Surface meta-layer evaluation results (base-vs-meta) for the Metrics Dashboard.

The meta-layer eval JSONs under ``output/meta_layer/<run>/`` carry BOTH the base
model and the meta model in a single file, so they naturally drive a base-vs-meta
comparison. This module discovers those files — driven by the promoted-model
registry in ``settings.yaml`` (``meta_models``), so only current/canonical models
appear — and normalizes their two on-disk shapes into one comparison payload:

- **M1-style** (``eval_results.json``): top-level ``meta_model`` / ``base_model``
  dicts plus ``fn_reduction_pct`` and ``meta_topk`` / ``base_topk`` — one scope.
- **M2-style** (``m2a_eval_results.json`` / ``m2b_eval_results.json``): nested
  ``overall`` and ``alternative_sites`` scopes, each with the same base/meta shape.

Kept deliberately separate from the base-model metrics endpoints (which read a
different schema under ``examples/base_layer/output/``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agentic_spliceai.splice_engine.resources import (
    get_meta_model_config,
    list_available_meta_models,
)

from . import config

logger = logging.getLogger(__name__)

# M2-style result filename -> the eval annotation it compares MANE against.
_M2_FILES: dict[str, str] = {
    "m2a_eval_results.json": "Ensembl",
    "m2b_eval_results.json": "GENCODE",
}


def _overall_recall(block: dict[str, Any]) -> float | None:
    """Overall site recall = TP / (TP + FN) across donor+acceptor for a model block."""
    tp = block.get("tp_count")
    fn = block.get("fn_count")
    if tp is None or fn is None or (tp + fn) == 0:
        return None
    return tp / (tp + fn)


def _per_class(block: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    """Extract donor/acceptor recall, precision, F1, and PR-AUC for one model block."""
    praucs = block.get("pr_aucs", {}) or {}
    out: dict[str, dict[str, float | None]] = {}
    for cls in ("donor", "acceptor"):
        out[cls] = {
            "recall": block.get(f"{cls}_recall"),
            "precision": block.get(f"{cls}_precision"),
            "f1": block.get(f"{cls}_f1"),
            "pr_auc": praucs.get(cls),
        }
    return out


def _topk(scope_block: dict[str, Any]) -> dict[str, Any]:
    """Normalize overall top-k accuracy for base and meta into aligned arrays."""
    meta = (scope_block.get("meta_topk") or {}).get("overall") or {}
    base = (scope_block.get("base_topk") or {}).get("overall") or {}
    if not meta or not base:
        return {}
    levels = sorted(meta.keys(), key=float)
    return {
        "levels": levels,
        "base": [base.get(k) for k in levels],
        "meta": [meta.get(k) for k in levels],
    }


def _normalize_scope(
    key: str, label: str, scope_block: dict[str, Any], caveat: str | None = None
) -> dict[str, Any]:
    """Turn one {meta_model, base_model, fn_reduction_pct, *_topk} block into a
    comparison scope: headline deltas + per-class + confusion counts + top-k."""
    meta = scope_block["meta_model"]
    base = scope_block["base_model"]
    headline = [
        {
            "label": "Recall",
            "base": _overall_recall(base),
            "meta": _overall_recall(meta),
            "fmt": "pct",
        },
        {
            "label": "Macro PR-AUC",
            "base": base.get("macro_pr_auc"),
            "meta": meta.get("macro_pr_auc"),
            "fmt": "prob",
        },
        {
            "label": "False negatives",
            "base": base.get("fn_count"),
            "meta": meta.get("fn_count"),
            "delta_pct": scope_block.get("fn_reduction_pct"),
            "fmt": "count_reduction",
        },
    ]
    return {
        "key": key,
        "label": label,
        "headline": headline,
        "per_class": {"base": _per_class(base), "meta": _per_class(meta)},
        "counts": {
            "base": {
                "tp": base.get("tp_count"),
                "fp": base.get("fp_count"),
                "fn": base.get("fn_count"),
            },
            "meta": {
                "tp": meta.get("tp_count"),
                "fp": meta.get("fp_count"),
                "fn": meta.get("fn_count"),
            },
        },
        "topk": _topk(scope_block),
        "caveat": caveat,
    }


def _discover() -> list[tuple[str, Path, str, str | None, str, str]]:
    """Discover base-vs-meta runs for the PROMOTED meta models (settings.yaml).

    Returns ``(run_id, path, kind, eval_annotation, variant, display_name)``. A model
    with an alternative-site eval (``<dir>_alt_eval/``) is surfaced through that file
    (its ``overall`` scope subsumes the standalone ``eval_results.json``); otherwise
    its own ``eval_results.json`` is used. ``run_id`` is slash-free (safe URL param).
    """
    found: list[tuple[str, Path, str, str | None, str, str]] = []
    try:
        names = list_available_meta_models()
    except Exception as e:  # registry missing/misconfigured — fall back to no meta runs
        logger.warning("meta-metrics: could not list meta models (%s)", e)
        return found

    for name in names:
        try:
            spec = get_meta_model_config(name)
        except Exception as e:
            logger.warning("meta-metrics: no config for %s (%s)", name, e)
            continue
        model_dir = config.PROJECT_ROOT / spec["dir"]
        variant = spec.get("variant", "")
        display = spec.get("name", variant or name)

        alt_dir = model_dir.parent / f"{model_dir.name}_alt_eval"
        alt_hits = [
            (alt_dir / fname, ann) for fname, ann in _M2_FILES.items() if (alt_dir / fname).exists()
        ]
        if alt_hits:
            for path, annotation in alt_hits:
                run_id = f"{name}__{path.name.split('_')[0]}"  # e.g. m2s_..__m2a
                found.append((run_id, path, "m2", annotation, variant, display))
        else:
            m1 = model_dir / "eval_results.json"
            if m1.exists():
                found.append((name, m1, "m1", None, variant, display))
    return found


def _build_run(
    run_id: str, path: Path, kind: str, annotation: str | None, variant: str, display: str
) -> dict[str, Any]:
    data = json.loads(path.read_text())
    model = variant or data.get("model", "meta")

    if kind == "m1":
        caveat = None
        fp_red = data.get("fp_reduction_pct")
        if fp_red is not None and fp_red < 0:
            caveat = (
                "At the fixed 0.5 operating point the meta model trades a small rise in "
                "false positives for far higher recall; ranking quality (PR-AUC, top-k) still "
                "improves. Reported on leakage-clean held-out chromosomes."
            )
        scope_block = {
            "meta_model": data["meta_model"],
            "base_model": data["base_model"],
            "fn_reduction_pct": data.get("fn_reduction_pct"),
            "meta_topk": data.get("meta_topk"),
            "base_topk": data.get("base_topk"),
        }
        return {
            "run_id": run_id,
            "model": model,
            "label": display,
            "kind": "m1",
            "metadata": {
                "n_genes": data.get("n_genes"),
                "n_positions": data.get("n_positions"),
                "test_chromosomes": data.get("test_chromosomes"),
                "annotation_source": data.get("annotation_source"),
            },
            "scopes": [_normalize_scope("test", "Held-out test set", scope_block, caveat)],
        }

    # kind == "m2"
    tissue = None
    tissue_path = path.parent / "tissue_stratified.json"
    if tissue_path.exists():
        try:
            tissue = json.loads(tissue_path.read_text())
        except (json.JSONDecodeError, OSError):
            tissue = None

    scopes = []
    if "alternative_sites" in data:
        scopes.append(
            _normalize_scope(
                "alternative_sites",
                f"Alternative sites ({annotation} ∖ MANE)",
                data["alternative_sites"],
            )
        )
    if "overall" in data:
        scopes.append(
            _normalize_scope("overall", "Overall (all annotated sites)", data["overall"])
        )
    return {
        "run_id": run_id,
        "model": model,
        "label": f"{model} · alternative sites ({annotation})",
        "kind": "m2",
        "metadata": {
            "n_genes": data.get("n_genes"),
            "eval_annotation": data.get("eval_annotation", (annotation or "").lower()),
            "n_alternative_sites": data.get("n_alternative_sites"),
            "n_shared_sites": data.get("n_shared_sites"),
            "test_chromosomes": data.get("test_chromosomes"),
        },
        "scopes": scopes,
        "tissue": tissue,
    }


def _headline_teaser(run: dict[str, Any]) -> str:
    """Short one-line delta for the run selector chip (uses the first scope)."""
    scopes = run.get("scopes") or []
    if not scopes:
        return ""
    fn = next((h for h in scopes[0]["headline"] if h["label"] == "False negatives"), None)
    if fn and fn.get("delta_pct") is not None:
        return f"FN −{fn['delta_pct']:.0f}%"
    return ""


def list_meta_runs() -> list[dict[str, Any]]:
    """Return lightweight summaries of every available meta-layer comparison run."""
    runs = []
    for run_id, path, kind, annotation, variant, display in _discover():
        try:
            run = _build_run(run_id, path, kind, annotation, variant, display)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning("meta-metrics: could not summarize %s (%s)", path, e)
            continue
        runs.append(
            {
                "run_id": run_id,
                "model": run["model"],
                "label": run["label"],
                "kind": run["kind"],
                "n_genes": run["metadata"].get("n_genes"),
                "headline": _headline_teaser(run),
            }
        )
    return runs


def get_meta_run(run_id: str) -> dict[str, Any] | None:
    """Return the full normalized comparison payload for one run, or None if unknown."""
    for rid, path, kind, annotation, variant, display in _discover():
        if rid == run_id:
            return _build_run(rid, path, kind, annotation, variant, display)
    return None
