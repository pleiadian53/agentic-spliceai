#!/usr/bin/env python
"""Build regulator-specific binding features: where does the knocked-down RBP bind?

The meta-layer RBP channel (``rbp_eclip`` / ``rbp_n_bound``) is identity-blind: it
counts peaks of *any* RBP. M4 conditions on *which* regulator was perturbed, so for
every (event site, regulator R) pair present in ``kd_events.parquet`` this script
measures R's own eCLIP binding near the site. These features exist before the
knockdown outcome is observed (eCLIP is a separate experiment), so they are valid
inputs.

Per (site, R), with peaks on the site's strand:

* ``bind_w50`` / ``bind_w300`` / ``bind_w1000`` — R peaks overlapping site ± w nt.
* ``bind_intron300`` / ``bind_exon300`` — the ±300 window split by side. RBP effects
  are position-dependent (binding in the intron vs the exon can have opposite
  effects), so the side is oriented by strand and splice-site type: an acceptor's
  intron is upstream in transcript order, a donor's is downstream.
* ``dist`` — nt to the nearest R peak (0 if overlapping), capped at ``--max-dist``.

Cell context: the knockdowns are HepG2, so R's **HepG2** peaks are used when they
exist; otherwise K562 (``eclip_cell = "K562"``, cross-cell, flagged); otherwise all
features are null (``eclip_cell = "none"``). 104 of the 185 regulators have eCLIP.

Output: ``data/mane/GRCh38/m4_labels/regulator_site_features.parquet`` keyed by
``(chrom, position, strand, rbp)``.

Usage:
    python examples/data_preparation/m4/04_build_regulator_features.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[3]
M4_LABELS = REPO / "data/mane/GRCh38/m4_labels"
EVENTS = M4_LABELS / "kd_events.parquet"
ECLIP = REPO / "data/mane/GRCh38/rbp_data/eclip_peaks.parquet"  # ENCODE K562 + HepG2
OUT = M4_LABELS / "regulator_site_features.parquet"

KEY = ["chrom", "position", "strand"]
CELL_PREFERENCE = ("HepG2", "K562")


def event_sites() -> pl.DataFrame:
    """Unique (site, splice_type, rbp) triples over the long, short, and partner sites."""
    ev = pl.scan_parquet(EVENTS)
    alt = pl.when(pl.col("event_type") == "A3SS").then(pl.lit("acceptor")).otherwise(
        pl.lit("donor"))
    partner = pl.when(pl.col("event_type") == "A3SS").then(pl.lit("donor")).otherwise(
        pl.lit("acceptor"))
    return pl.concat([
        ev.select("chrom", pl.col(f"{r}_site").alias("position"), "strand", "rbp",
                  (partner if r == "partner" else alt).alias("splice_type"))
        for r in ("long", "short", "partner")
    ]).unique().collect()


def load_peaks() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-RBP peaks from the preferred cell line (1-based closed), and the cell chosen."""
    peaks = pl.read_parquet(ECLIP).with_columns(
        pl.col("chrom").str.replace(r"^chr", ""),
        (pl.col("start") + 1).alias("start"),  # BED 0-based start → 1-based closed
    )
    rank = {c: i for i, c in enumerate(CELL_PREFERENCE)}
    chosen = (
        peaks.select("rbp", "cell_line").unique()
        .with_columns(pl.col("cell_line").replace_strict(rank, default=99).alias("_r"))
        .sort("_r").unique("rbp", keep="first").select("rbp", "cell_line")
    )
    return peaks.join(chosen, on=["rbp", "cell_line"], how="semi"), chosen


def _count_overlaps(starts: np.ndarray, ends: np.ndarray, lo: np.ndarray,
                    hi: np.ndarray) -> np.ndarray:
    """# intervals [s, e] overlapping [lo, hi]: #(s <= hi) − #(e < lo). Inputs sorted."""
    return np.searchsorted(starts, hi, side="right") - np.searchsorted(ends, lo, side="left")


def site_features(pos: np.ndarray, intron_upstream: np.ndarray, starts: np.ndarray,
                  ends: np.ndarray, max_dist: int) -> dict[str, np.ndarray]:
    """Binding features of one regulator around many sites on one (chrom, strand)."""
    out = {f"bind_w{w}": _count_overlaps(starts, ends, pos - w, pos + w)
           for w in (50, 300, 1000)}
    up = _count_overlaps(starts, ends, pos - 300, pos - 1)
    down = _count_overlaps(starts, ends, pos + 1, pos + 300)
    out["bind_intron300"] = np.where(intron_upstream, up, down)
    out["bind_exon300"] = np.where(intron_upstream, down, up)

    left_i = np.searchsorted(ends, pos, side="left") - 1  # last peak ending before pos
    right_i = np.searchsorted(starts, pos, side="right")  # first peak starting after pos
    d_left = np.where(left_i >= 0, pos - ends[np.clip(left_i, 0, None)], max_dist)
    d_right = np.where(right_i < len(starts),
                       starts[np.clip(right_i, None, len(starts) - 1)] - pos, max_dist)
    overlap = _count_overlaps(starts, ends, pos, pos) > 0
    out["dist"] = np.minimum(np.where(overlap, 0, np.minimum(d_left, d_right)), max_dist)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-dist", type=int, default=10_000)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()

    sites = event_sites()
    peaks, chosen = load_peaks()
    print(f"{sites.height:,} (site, regulator) pairs over {sites['rbp'].n_unique()} regulators; "
          f"eCLIP for {chosen.height} RBPs ({dict(chosen.group_by('cell_line').len().iter_rows())})")

    # A site's intron is upstream (transcript order) for an acceptor, downstream for a donor;
    # in genomic coordinates "upstream" flips on the minus strand.
    sites = sites.with_columns(
        ((pl.col("splice_type") == "acceptor") == (pl.col("strand") == "+")).alias("_intron_up"))

    with_eclip = sites.join(chosen, on="rbp", how="inner")
    parts = []
    for (chrom, strand, rbp), grp in with_eclip.group_by(["chrom", "strand", "rbp"]):
        pk = peaks.filter((pl.col("chrom") == chrom) & (pl.col("strand") == strand)
                          & (pl.col("rbp") == rbp))
        starts = np.sort(pk["start"].to_numpy())
        ends = np.sort(pk["end"].to_numpy())
        if len(starts) == 0:
            feats = {c: np.zeros(grp.height, dtype=np.int64) for c in (
                "bind_w50", "bind_w300", "bind_w1000", "bind_intron300", "bind_exon300")}
            feats["dist"] = np.full(grp.height, args.max_dist)
        else:
            feats = site_features(grp["position"].to_numpy(), grp["_intron_up"].to_numpy(),
                                  starts, ends, args.max_dist)
        parts.append(grp.select(*KEY, "rbp", "splice_type", "cell_line").with_columns(
            [pl.Series(k, v) for k, v in feats.items()]))
    feat = pl.concat(parts).rename({"cell_line": "eclip_cell"})

    no_eclip = sites.join(chosen, on="rbp", how="anti").select(
        *KEY, "rbp", "splice_type").with_columns(pl.lit("none").alias("eclip_cell"))
    table = pl.concat([feat, no_eclip], how="diagonal_relaxed").sort([*KEY, "rbp"])
    assert table.height == sites.height, "every (site, regulator) pair must be emitted"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out)
    bound = table.filter(pl.col("eclip_cell") != "none")
    print(f"Wrote {args.out}: {table.height:,} rows ({time.time() - t0:.0f}s)")
    print(f"  eclip_cell: {dict(table.group_by('eclip_cell').len().iter_rows())}")
    print(f"  with eCLIP: frac bind_w300>0 = {(bound['bind_w300'] > 0).mean():.3f}, "
          f"median dist = {bound['dist'].median():.0f} nt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
