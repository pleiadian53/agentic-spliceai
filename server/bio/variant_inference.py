"""Variant-effect (ref vs alt) scoring for the Bio Lab UI.

Scores a single nucleotide change with the base model and a meta model, and
returns the per-position deltas plus the SpliceAI-convention summary scores.

Two things make this cheaper to serve than the genome view:

**No feature cache.** ``use_multimodal`` is off, so **any gene works**, not only
the handful with a warmed Phase-A ``.npz``. The justification is that for an SNV
the dense features are identical between ref and alt
(``mm_alt = mm_ref.copy()`` in ``variant_runner``), carrying no
variant-specific information; the arm's status doc calls the multimodal stack
"architecturally dead weight" for max-|Δ| ranking.

.. warning::
   This is **not** the setting the published benchmarks ran under, and the
   difference is not cosmetic. ``run_benchmark(use_multimodal=False)`` is only
   the *function* default; ``main()`` passes ``not args.no_multimodal``, whose
   CLI default is **True**, and the pod orchestrator passes ``--bigwig-cache``
   without ``--no-multimodal``. So the published ClinVar and MutSpliceDB numbers
   used multimodal features **on**.

   Measured on MutSpliceDB (434 rows common to both): turning them off leaves
   ``base_max_delta`` alone (median change 1.4e-5, which is CUDA-vs-CPU noise,
   since the base model never reads these channels) but moves
   ``meta_max_delta`` by a median of 0.04-0.06, with 200+ rows moving more than
   0.05. "Cancels in the subtraction" is true of the *features*, not of the
   model's nonlinear response to them.

   Any number quoted from those benchmarks therefore describes a different
   configuration than this page serves. Treat the page as a qualitative
   explorer, not as a reproduction of the published run.

**One window, not one gene.** The model sees ``window_size + context_padding``
bases centred on the variant, so a call is two base-model passes and two meta
passes rather than a whole-gene sweep.

What this deliberately does not claim: the delta is not a pathogenicity score.
On ClinVar the meta layer ties the base model (ROC-AUC 0.751 vs 0.754). Where it
earns its keep is *what kind* of splicing change, so that is what the page leads
with.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from agentic_spliceai.splice_engine.resources import get_meta_model_config, get_model_resources

from . import config

logger = logging.getLogger(__name__)

_DONOR, _ACCEPTOR = 0, 1

# One runner per meta model, loaded once. Loading is ~4 s (checkpoint + FASTA
# index + base model); scoring afterwards is well under a second.
_runners: dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=1)

#: LRU of scored variants, keyed by (chrom, pos, ref, alt, strand, meta model).
_cache: OrderedDict[tuple, dict] = OrderedDict()
MAX_CACHED_VARIANTS = 64

#: Plotted window around the variant. The model always sees the full
#: window_size; this only bounds what is sent to the browser, and 300 bp
#: comfortably contains the ±50 bp SpliceAI scoring radius plus context.
PLOT_RADIUS = 300

#: The track is sent **dense**, with no score floor. The genome view thresholds
#: because a whole gene is 10^4-10^5 positions; here the window is 2*PLOT_RADIUS+1,
#: so the whole thing is a few hundred numbers. Thresholding it would be actively
#: misleading: a variant's delta is extremely peaky (often 3 positions above 0.01),
#: and a line through 3 surviving points interpolates across hundreds of bases that
#: are actually flat zero.


def _fasta_path() -> Path:
    return Path(get_model_resources("openspliceai").get_fasta_path())


def _get_runner_sync(meta_model_name: str):
    """Load (and cache) a VariantRunner for one meta model."""
    if meta_model_name in _runners:
        return _runners[meta_model_name]

    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )

    spec = get_meta_model_config(meta_model_name)  # raises ValueError if unknown
    model_dir = config.PROJECT_ROOT / spec["dir"]
    ckpt = model_dir / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Meta model '{meta_model_name}' checkpoint not found at {ckpt}"
        )
    logger.info("Loading VariantRunner for %s (first load, will cache)", meta_model_name)
    runner = VariantRunner(
        meta_checkpoint=ckpt,
        fasta_path=_fasta_path(),
        base_model=spec.get("base_model", "openspliceai"),
        device="cpu",
    )
    _runners[meta_model_name] = runner
    return runner


def _track(delta: np.ndarray, window_start: int, variant_pos: int) -> dict[str, list]:
    """Deltas near the variant as parallel arrays for plotting.

    Only donor and acceptor are returned. The ``neither`` channel carries no
    splice-altering meaning and would dominate the y-range.
    """
    lo = max(0, variant_pos - window_start - PLOT_RADIUS)
    hi = min(delta.shape[0], variant_pos - window_start + PLOT_RADIUS + 1)
    sl = delta[lo:hi, :2]
    return {
        "positions": list(range(window_start + lo, window_start + hi)),
        "donor": np.round(sl[:, _DONOR], 4).tolist(),
        "acceptor": np.round(sl[:, _ACCEPTOR], 4).tolist(),
    }


def _summary(delta: np.ndarray, window_start: int, variant_pos: int,
             radius: int = 50) -> dict[str, Any]:
    """SpliceAI-convention DS scores, plus where each one sits.

    Restricted to ±``radius`` of the variant, matching OpenSpliceAI's default
    ``dist_var=50``. Positions are returned alongside the magnitudes because
    "how far away" is the part a reader needs to judge a call.
    """
    centre = delta.shape[0] // 2
    lo, hi = max(0, centre - radius), min(delta.shape[0], centre + radius + 1)
    sl = delta[lo:hi, :2]
    out: dict[str, Any] = {}
    for name, chan, sign in (
        ("DS_DG", _DONOR, +1), ("DS_DL", _DONOR, -1),
        ("DS_AG", _ACCEPTOR, +1), ("DS_AL", _ACCEPTOR, -1),
    ):
        col = sl[:, chan] * sign
        i = int(col.argmax())
        out[name] = {
            "score": round(float(col[i]), 4),
            "position": window_start + lo + i,
            "offset": window_start + lo + i - variant_pos,
        }
    top = max(out, key=lambda k: out[k]["score"])
    out["max"] = {"metric": top, **out[top]}
    return out


def _score_sync(chrom: str, position: int, ref: str, alt: str, strand: str,
                gene: str | None, meta_model_name: str) -> dict[str, Any]:
    """Run one variant through base + meta and package the result."""
    runner = _get_runner_sync(meta_model_name)
    result = runner.run(
        chrom, position, ref, alt,
        gene=gene or "", strand=strand, use_multimodal=False,
    )

    payload: dict[str, Any] = {
        "chrom": result.chrom,
        "position": result.position,
        "ref": ref,
        "alt": alt,
        "strand": strand,
        "gene": gene or "",
        "meta_model": meta_model_name,
        "window_start": result.window_start,
        "window_length": result.window_length,
        "plot_radius": PLOT_RADIUS,
        "scoring_radius": 50,
        "base": {
            "summary": _summary(result.base_delta, result.window_start, result.position),
            "track": _track(result.base_delta, result.window_start, result.position),
        },
        "meta": {
            "summary": _summary(result.delta, result.window_start, result.position),
            "track": _track(result.delta, result.window_start, result.position),
        },
        "events": [
            {
                "event_type": e.event_type,
                "position": e.position,
                "delta": round(float(e.delta), 4),
                "offset": e.distance_from_variant,
            }
            for e in result.events[:10]
        ],
        "n_events": len(result.events),
    }
    return payload


async def score_variant(chrom: str, position: int, ref: str, alt: str,
                        strand: str, gene: str | None,
                        meta_model_name: str) -> dict[str, Any]:
    """Score one variant, reusing a cached result when the same call repeats."""
    key = (chrom, position, ref.upper(), alt.upper(), strand, meta_model_name)
    if key in _cache:
        _cache.move_to_end(key)
        logger.info("Variant cache hit: %s:%d %s>%s", chrom, position, ref, alt)
        return _cache[key]

    loop = asyncio.get_event_loop()
    payload = await loop.run_in_executor(
        _executor, _score_sync, chrom, position, ref.upper(), alt.upper(),
        strand, gene, meta_model_name,
    )
    _cache[key] = payload
    while len(_cache) > MAX_CACHED_VARIANTS:
        _cache.popitem(last=False)
    return payload


def cache_stats() -> dict[str, Any]:
    """LRU contents, for the dev cache-introspection endpoint."""
    return {
        "size": len(_cache),
        "max": MAX_CACHED_VARIANTS,
        "runners_loaded": sorted(_runners),
        "keys": [
            {"chrom": k[0], "position": k[1], "ref": k[2], "alt": k[3],
             "strand": k[4], "meta_model": k[5]}
            for k in _cache
        ],
    }
