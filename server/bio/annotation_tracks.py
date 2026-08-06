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
