"""Read per-model operating points out of a stored held-out threshold sweep.

A base model and a meta model refined from it have very different score
distributions, so a single shared cutoff flatters whichever one it happens to
suit. Comparing them at one threshold is not a fair comparison, and 0.5 in
particular is not a meaningful operating point under this class imbalance. The
held-out sweep written by :meth:`StreamingEvaluator.sweep_thresholds` already
holds the per-model answer; this module turns it into the one number a caller
needs.

**Why the correction here exists.** The sweep accumulators retain every splice
site but only ``neither_subsample_rate`` of the "neither" positions, so a raw
false-positive count over the retained rows understates the truth by
``1 / rate``. Precision comes out inflated at every threshold, and inflated most
where the model fires most, which drags the apparent F1 optimum far below its
real value. On the M2-S held-out sweep the uncorrected curve peaks at 0.65
while the corrected one peaks at 0.99, and the uncorrected base curve peaks at
0.01. Those are operating points somebody would act on, so they have to be
right.

Sweeps written after the fix carry ``prevalence_corrected: True`` and are used
as-is. Older sweeps are corrected at read time; that path is approximate,
because a stored row records only a total false-positive count and cannot say
which of those came from the fully-retained other splice class. In practice
"neither" positions outnumber splice sites by three orders of magnitude, so the
approximation holds: :func:`validate` checks it against the argmax precision the
same eval computed independently, and on both promoted models it agrees to
within 5%.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Rate used by every sweep written before ``prevalence_corrected`` existed.
#: Matches the ``_ModelAccumulator`` default; only applied to legacy sweeps.
LEGACY_NEITHER_SUBSAMPLE_RATE = 0.01

_CLASSES = ("donor", "acceptor")


def _corrected_rows(rows: list[dict], weight: float) -> list[dict]:
    """Re-derive precision and F1 with false positives weighted back to prevalence.

    ``tp``/``fn``/recall are untouched: every positive is retained, so they are
    already exact.
    """
    out = []
    for r in rows:
        tp, fn = r["tp"], r["fn"]
        fp = r["fp"] * weight
        prec = tp / max(tp + fp, 1.0)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        out.append({
            "threshold": r["threshold"],
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": int(round(fp)), "fp_observed": r["fp"], "fn": fn,
        })
    return out


def _macro_optimum(per_class: dict[str, list[dict]]) -> dict[str, Any] | None:
    """Threshold maximizing the mean of donor and acceptor F1.

    One number per model rather than two, because the UI offers one slider.
    Donor and acceptor optima are near-identical in practice; both are still
    returned so a caller can show them if they diverge.
    """
    usable = {c: rows for c, rows in per_class.items() if rows}
    if not usable:
        return None
    by_threshold: dict[float, list[float]] = {}
    for rows in usable.values():
        for r in rows:
            by_threshold.setdefault(r["threshold"], []).append(r["f1"])
    n = len(usable)
    scored = [(t, sum(f) / len(f)) for t, f in by_threshold.items() if len(f) == n]
    if not scored:
        return None
    threshold, macro_f1 = max(scored, key=lambda x: x[1])
    at = {c: next(r for r in rows if r["threshold"] == threshold) for c, rows in usable.items()}
    return {
        "threshold": threshold,
        "macro_f1": round(macro_f1, 4),
        "per_class": at,
        "at_grid_edge": threshold == max(by_threshold),
    }


def validate(sweep_rows: list[dict], argmax_precision: float | None) -> dict[str, Any] | None:
    """Check a corrected curve against the argmax precision from the same eval.

    The eval's ``class_counts`` precision is computed over *all* positions, with
    no subsampling anywhere, so it is an independent measurement of the same
    quantity the corrected curve estimates near 0.5. Agreement is evidence the
    correction is sound; a large gap means the stored rate is wrong for this file.
    """
    if argmax_precision is None or not sweep_rows:
        return None
    row = min(sweep_rows, key=lambda r: abs(r["threshold"] - 0.5))
    est = row["precision"]
    err = abs(est - argmax_precision) / max(argmax_precision, 1e-9)
    return {
        "corrected_precision_at_0.5": round(est, 4),
        "argmax_precision": round(argmax_precision, 4),
        "relative_error": round(err, 4),
        "agrees": err < 0.10,
    }


def operating_points(eval_json: dict[str, Any]) -> dict[str, Any] | None:
    """Per-model F1-optimal thresholds from an eval JSON's ``threshold_sweep``.

    Returns ``None`` when the eval carries no sweep. Otherwise returns the
    optimum for ``base`` and ``meta``, how the correction was applied, and a
    validation record per model.

    Examples
    --------
    >>> import json
    >>> from pathlib import Path
    >>> from agentic_spliceai.splice_engine.eval.operating_points import operating_points
    >>> d = json.loads(Path("output/meta_layer/m2s_v4_cleanannot/eval_results.json").read_text())
    >>> op = operating_points(d)
    >>> op["models"]["base"]["threshold"], op["models"]["meta"]["threshold"]
    (0.25, 0.99)
    """
    sweep = eval_json.get("threshold_sweep")
    if not sweep:
        return None

    already_corrected = bool(sweep.get("prevalence_corrected"))
    rate = sweep.get("neither_subsample_rate", LEGACY_NEITHER_SUBSAMPLE_RATE)
    weight = 1.0 if already_corrected else (1.0 / rate if rate > 0 else 1.0)

    models: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    for who, block_key in (("base", "base_model"), ("meta", "meta_model")):
        per_class_raw = sweep.get(who) or {}
        per_class = {
            c: (per_class_raw.get(c) or []) if already_corrected
            else _corrected_rows(per_class_raw.get(c) or [], weight)
            for c in _CLASSES
        }
        opt = _macro_optimum(per_class)
        if opt is None:
            continue
        models[who] = opt
        v = validate(per_class["donor"], (eval_json.get(block_key) or {}).get("donor_precision"))
        if v is not None:
            validation[who] = v
            if not v["agrees"]:
                logger.warning(
                    "operating_points: corrected precision for %s disagrees with argmax "
                    "(%.4f vs %.4f); the stored subsample rate may be wrong",
                    who, v["corrected_precision_at_0.5"], v["argmax_precision"],
                )

    if not models:
        return None
    return {
        "models": models,
        "correction": "at_write_time" if already_corrected else "at_read_time",
        "approximate": not already_corrected,
        "neither_subsample_rate": rate,
        "validation": validation,
        "note": (
            "F1-optimal on the held-out evaluation, per model. Precision is corrected "
            "back to genome-wide prevalence; without that correction these thresholds "
            "read far too low."
        ),
    }
