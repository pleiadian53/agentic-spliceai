"""Per-gene annotation tracks for the genome view.

The genome view draws one gene's predictions against ground truth. Ground truth
has always been **MANE alone**, which is the wrong yardstick for M2-S: that model
exists to find *alternative* sites, and the established protocol scores it on the
**delta set** — sites in Ensembl but not in MANE (see
``examples/meta_layer/09_evaluate_alternative_sites.py``, which builds the MANE
site set and counts a site alternative when it is absent from it).

This module serves several annotations at once so the page can stack them the way
a genome browser does, with the delta as its own track.

Reads the prebuilt ``splice_sites_track.parquet`` files
(``examples/data_preparation/05_build_annotation_track_parquets.py``): per-gene
lookup is ~7 ms there versus ~1 s against the source TSVs.

.. important::
   Chromosomes are **bare** ("1", not "chr1") throughout. MANE and GENCODE ship
   ``chr1`` while Ensembl ships ``1``; comparing them unnormalized yields zero
   overlap and therefore a delta set containing *every* Ensembl site. The builder
   normalizes at write time and :func:`_bare` guards the query side.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

from . import config

logger = logging.getLogger(__name__)

# (chrom, position, splice_type) — the identity used to compare annotations.
# Strand is excluded deliberately: it is implied by the gene, and including it
# would split a site that two overlapping genes report on opposite strands.
_KEY = ["chrom", "position", "splice_type"]

_cache: Dict[str, Optional[pl.DataFrame]] = {}


def _bare(chrom: str) -> str:
    return chrom[3:] if str(chrom).startswith("chr") else str(chrom)


def _sites_path(annotation_key: str) -> Optional[Path]:
    spec = config.ANNOTATIONS.get(annotation_key) or {}
    p = spec.get("sites")
    return Path(p) if p else None


def _load(annotation_key: str) -> Optional[pl.DataFrame]:
    """Lazily load one annotation's track parquet (cached per process)."""
    if annotation_key in _cache:
        return _cache[annotation_key]
    path = _sites_path(annotation_key)
    df = None
    if path and path.exists():
        try:
            df = pl.read_parquet(path)
            logger.info("Annotation track %s: %d sites", annotation_key, df.height)
        except Exception as e:
            logger.warning("Could not read %s: %s", path, e)
    elif path:
        logger.info(
            "Annotation track %s not built (%s). Run "
            "examples/data_preparation/05_build_annotation_track_parquets.py",
            annotation_key, path,
        )
    _cache[annotation_key] = df
    return df


def available_track_sources() -> List[str]:
    """Annotation keys whose track parquet exists on disk."""
    return [k for k in config.ANNOTATIONS if (p := _sites_path(k)) and p.exists()]


def _gene_sites(annotation_key: str, gene_name: str, chrom: str) -> Optional[pl.DataFrame]:
    df = _load(annotation_key)
    if df is None:
        return None
    sub = df.filter(
        (pl.col("gene_name") == gene_name) & (pl.col("chrom") == _bare(chrom))
    ).select(_KEY)
    return sub.unique()


def _as_track(key: str, label: str, df: Optional[pl.DataFrame], note: str = "") -> dict:
    if df is None:
        return {"key": key, "label": label, "n": 0, "donor": [], "acceptor": [],
                "available": False, "note": note}
    donor = df.filter(pl.col("splice_type") == "donor")["position"].sort().to_list()
    acceptor = df.filter(pl.col("splice_type") == "acceptor")["position"].sort().to_list()
    return {"key": key, "label": label, "n": len(donor) + len(acceptor),
            "donor": donor, "acceptor": acceptor, "available": True, "note": note}


#: Truth sets the genome view can score against. ``delta`` is intentionally
#: absent: TP/FP/FN is undefined against a *subset* of truth, because a call
#: away from the subset may be a perfectly correct canonical call. Measured on
#: TARDBP at 0.9, scoring the base model on the delta reports 9 "false
#: positives" that are all correct MANE calls. The delta is reported as
#: **recall** instead (the alt-sites badge).
TRUTH_SETS = ("mane", "ensembl", "gencode")


def truth_sets_for_build(build: str) -> tuple:
    """Truth sets with a built track parquet on ``build``.

    Scoring across builds is silently catastrophic rather than merely wrong:
    SpliceAI (GRCh37) on TP53 against the GRCh38 MANE track reports 0/21/0,
    because the GRCh37 gene span shares no coordinates with the GRCh38 gene, so
    every call falls outside every truth site. Callers offer only what this
    returns.
    """
    have = set(available_track_sources())
    return tuple(t for t in TRUTH_SETS if f"{t}.{build}" in have)


def truth_sites(gene_name: str, chrom: str, truth: str,
                build: str = "GRCh38") -> Optional[dict]:
    """Donor/acceptor positions for one gene under one annotation.

    ``truth`` is a bare source name (``mane`` / ``ensembl`` / ``gencode``),
    resolved against ``build`` to a track parquet. Returns ``None`` when that
    source/build combination has no track — never a different build's track.
    """
    key = f"{truth}.{build}"
    df = _gene_sites(key, gene_name, chrom)
    if df is None:
        return None
    return {
        "donor": set(df.filter(pl.col("splice_type") == "donor")["position"].to_list()),
        "acceptor": set(df.filter(pl.col("splice_type") == "acceptor")["position"].to_list()),
    }


def score_against_truth(
    positions: list, donor_prob: list, acceptor_prob: list,
    truth: dict, gene_start: int, gene_end: int, threshold: float,
) -> dict:
    """TP/FP/FN for one model against an explicit truth set.

    Re-derived here rather than routed through ``evaluate_splice_site_predictions``
    because that function takes its annotations from the *base model's* resources,
    and ``prepare_splice_site_annotations(annotation_source="ensembl")`` resolves
    genes through MANE-shaped ids — it returns 10 sites for TARDBP where the
    Ensembl track has 40. Scoring against a silently truncated truth set is worse
    than not offering the option.

    Conventions, chosen to match the offline evaluator:

    - **Exact position** match, no tolerance window.
    - **Type-specific**: a donor call on an acceptor truth site is a false
      positive, not a hit.
    - Restricted to ``[gene_start, gene_end]``. Truth sites outside the scored
      window are excluded from the denominator entirely, because no prediction
      exists there: they are unmeasured, not missed. Another annotation's gene is
      frequently longer than the base model's, so a wider truth set routinely has
      sites beyond the window. The exclusion is not a judgement about a model's
      scope, and it is not a property of the offline evaluation, which runs on
      the eval annotation's own spans. See ``dev/planning/UI_layer/BACKLOG.md``.
    """
    idx = {p: i for i, p in enumerate(positions)}
    in_win = lambda p: gene_start <= p <= gene_end          # noqa: E731
    tp = fp = fn = 0
    markers = []
    for stype, probs in (("donor", donor_prob), ("acceptor", acceptor_prob)):
        want = {p for p in truth[stype] if in_win(p)}
        for p in want:
            i = idx.get(p)
            if i is not None and probs[i] > threshold:
                tp += 1
                markers.append({"position": p, "site_type": stype, "pred_type": "TP"})
            else:
                fn += 1
                markers.append({"position": p, "site_type": stype, "pred_type": "FN"})
        for p, i in idx.items():
            if probs[i] > threshold and p not in want:
                fp += 1
                markers.append({"position": p, "site_type": stype, "pred_type": "FP"})
    return {"n_tp": tp, "n_fp": fp, "n_fn": fn, "n_truth": tp + fn, "markers": markers}


def gene_annotation_tracks(gene_name: str, chrom: str) -> dict:
    """Annotation tracks for one gene, including the derived delta sets.

    Returns MANE, ``Ensembl \\ MANE`` (the alternative-site delta), full Ensembl,
    and ``GENCODE \\ Ensembl`` when each source is available. Absent sources come
    back with ``available: False`` rather than raising, so a partially-built
    install still renders what it has.
    """
    mane = _gene_sites("mane.GRCh38", gene_name, chrom)
    ens = _gene_sites("ensembl.GRCh38", gene_name, chrom)
    gc = _gene_sites("gencode.GRCh38", gene_name, chrom)

    def minus(a: Optional[pl.DataFrame], b: Optional[pl.DataFrame]):
        if a is None or b is None:
            return None
        return a.join(b, on=_KEY, how="anti")

    tracks = [
        _as_track("mane", "MANE (canonical)", mane,
                  "The annotation M1-S trains on and the genome view has always scored against."),
        _as_track("ensembl_minus_mane", "Ensembl \\ MANE (alternative)", minus(ens, mane),
                  "The delta set. This is what M2-S is built to find; scoring it against MANE "
                  "alone measures the wrong thing."),
        _as_track("ensembl", "Ensembl 112 (all)", ens,
                  "Every Ensembl transcript's sites, canonical included."),
        _as_track("gencode_minus_ensembl", "GENCODE \\ Ensembl", minus(gc, ens),
                  "GENCODE adds ~136,858 sites genome-wide over Ensembl, mostly outside "
                  "protein-coding genes — often empty for a single coding gene."),
    ]

    # A MANE site absent from Ensembl is not an error: genome-wide 1,340 exist,
    # so canonical is NOT a subset of alternative. Surface the count rather than
    # letting it look like a join bug.
    mane_only = minus(mane, ens)
    return {
        "gene_name": gene_name,
        "chrom": _bare(chrom),
        "tracks": tracks,
        "n_mane_not_in_ensembl": 0 if mane_only is None else mane_only.height,
    }
