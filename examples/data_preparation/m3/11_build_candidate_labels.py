#!/usr/bin/env python
"""Build the M3-R candidate-refiner labeled table (Tier 2, Phase 0 substrate).

Reframes M3 as candidate refinement: the base model proposes candidate splice
sites; a classifier reranks each **real cryptic vs artifact** from multimodal
evidence. This builds the local, base-score-matched labeled table it trains on.

Why base-matched negatives matter: real novel sites are intrinsically LOW base
score (median ~0.04 — that is *why* they are cryptic). The existing
``negatives.parquet`` is dinucleotide decoys with ~0 base score, so a classifier
trained on them would just relearn the base score. Here, negatives are sampled
from the **same candidate pool** as the positives and **stratified to match the
positives' base-score histogram per splice type**, so the base score alone cannot
separate the classes — the classifier is forced to use conservation / RBP /
epigenetic / chromatin.

Labels:
  - positive (real) = feature-parquet position in ``positives_pooled`` (optionally
    the ``longread_confirmed`` subset — the Tier 1 label cleanup).
  - negative (artifact) = base-proposed position (max(donor_prob, acceptor_prob)
    > --min-base) NOT in (annotation ∪ positives_pooled ∪ long-read truth),
    base-matched to the positives.

Everything is LOCAL — the 116-col peak-preserving feature parquets already hold
every candidate's multimodal vector. No pod, no bigWig.

Output: ``data/mane/GRCh38/m3_labels/candidate_labels.parquet`` — all feature
columns + ``cand_label`` (1=real, 0=artifact), ``cand_splice_type``,
``cand_base_score`` (the proposing donor/acceptor probability), and the
``chrom/position/strand`` keys.

Usage:
    python examples/data_preparation/m3/11_build_candidate_labels.py \
        --confirmed-only --neg-ratio 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _example_utils import setup_example_environment  # noqa: E402
setup_example_environment()

REPO = Path(__file__).resolve().parents[3]  # examples/data_preparation/m3/<file> -> repo root
FEATURE_DIR = REPO / "data/mane/GRCh38/openspliceai_eval/analysis_sequences"
M3_LABELS = REPO / "data/mane/GRCh38/m3_labels"
D1_TRUTH = REPO / "data/encode_longread/GRCh38/longread_truth_novel.parquet"
OUT = M3_LABELS / "candidate_labels.parquet"

KEY = ["chrom", "position", "strand"]  # NOT splice_type — empty for non-annotated rows


def _bare(col: str = "chrom") -> pl.Expr:
    return pl.col(col).cast(pl.String).str.replace_all(r"^chr", "").alias("chrom")


def _norm_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Bare chrom + int position + string strand, for a clean join."""
    return df.with_columns(
        _bare("chrom"),
        pl.col("position").cast(pl.Int64),
        pl.col("strand").cast(pl.String),
    )


def load_features() -> pl.DataFrame:
    """All 24 feature parquets, chrom normalised to bare, with the proposing score."""
    frames = []
    for p in sorted(FEATURE_DIR.glob("analysis_sequences_chr*.parquet")):
        frames.append(pl.read_parquet(p))
    # Column ORDER differs across per-chrom parquets → align by name (diagonal).
    df = pl.concat(frames, how="diagonal")
    df = _norm_keys(df)
    # The proposing candidate: whichever base probability is higher.
    df = df.with_columns(
        cand_base_score=pl.max_horizontal("donor_prob", "acceptor_prob"),
        cand_argmax_type=pl.when(pl.col("donor_prob") >= pl.col("acceptor_prob"))
        .then(pl.lit("donor")).otherwise(pl.lit("acceptor")),
    )
    return df


def base_matched_sample(
    pos: pl.DataFrame, negpool: pl.DataFrame, ratio: float, n_bins: int, seed: int,
) -> pl.DataFrame:
    """Sample negatives whose per-type base-score histogram matches the positives.

    Bins ``cand_base_score`` in [0,1]; per (splice_type, bin) draws
    ``ratio × n_pos`` negatives (capped at what's available). This removes base
    score as a discriminator between the classes.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    def binned(df: pl.DataFrame) -> pl.DataFrame:
        b = np.clip(np.digitize(df["cand_base_score"].to_numpy(), edges[1:-1]), 0, n_bins - 1)
        return df.with_columns(pl.Series("bin", b))

    posb = binned(pos)
    negb = binned(negpool)
    want = (
        posb.group_by(["cand_splice_type", "bin"]).len()
        .with_columns((pl.col("len") * ratio).ceil().cast(pl.Int64).alias("n_want"))
    )
    out = []
    for row in want.iter_rows(named=True):
        cell = negb.filter(
            (pl.col("cand_splice_type") == row["cand_splice_type"]) & (pl.col("bin") == row["bin"])
        )
        if cell.height == 0:
            continue
        take = min(row["n_want"], cell.height)
        out.append(cell.sample(n=take, seed=seed))
    return pl.concat(out).drop("bin") if out else negpool.head(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirmed-only", action="store_true",
                    help="positives = long-read-confirmed subset (Tier 1 cleanup)")
    ap.add_argument("--min-base", type=float, default=0.01,
                    help="negative candidate pool: max(donor,acceptor) prob > this")
    ap.add_argument("--neg-ratio", type=float, default=3.0, help="negatives per positive")
    ap.add_argument("--n-bins", type=int, default=25, help="base-score histogram bins")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    print("Loading feature parquets (24 chroms)...")
    feat = load_features()
    print(f"  {feat.height:,} feature positions")

    pos_pool = _norm_keys(pl.read_parquet(M3_LABELS / "positives_pooled.parquet"))
    ann = _norm_keys(pl.read_parquet(M3_LABELS / "annotation_mask.parquet")).select(KEY).unique()
    d1 = _norm_keys(pl.read_parquet(D1_TRUTH)).select(KEY).unique()
    # D2 disease anchors are a HELD-OUT eval set — they must never be sampled as
    # "artifact" negatives, or Phase 2's D2 recall is poisoned (an eval-truth site
    # trained as fake). Exclude ALL anchors (novel + annotated) from the neg pool.
    d2 = _norm_keys(pl.read_parquet(M3_LABELS / "disease_anchors.parquet")).select(KEY).unique()

    # ── Positives: feature rows that ARE real novel sites ────────────────
    p = pos_pool
    if args.confirmed_only:
        p = p.filter(pl.col("longread_confirmed"))
    p_keys = p.select([*KEY, pl.col("splice_type").alias("cand_splice_type")])
    positives = (
        feat.join(p_keys, on=KEY, how="inner")
        .with_columns(
            cand_label=pl.lit(1, dtype=pl.Int8),
            # proposing score for the KNOWN type
            cand_base_score=pl.when(pl.col("cand_splice_type") == "donor")
            .then(pl.col("donor_prob")).otherwise(pl.col("acceptor_prob")),
        )
    )
    print(f"  positives (real, {'confirmed' if args.confirmed_only else 'all'}): {positives.height:,}")

    # ── Negative candidate pool: base-proposed, not real, not eval-truth ──
    # Exclude positives, annotation, AND both eval sets (D1 long-read + D2 anchors)
    # so no held-out truth site can ever be labeled an artifact.
    real_keys = pl.concat([pos_pool.select(KEY), ann, d1, d2]).unique()
    negpool = (
        feat.filter(pl.col("cand_base_score") > args.min_base)
        .join(real_keys, on=KEY, how="anti")
        .with_columns(
            cand_label=pl.lit(0, dtype=pl.Int8),
            cand_splice_type=pl.col("cand_argmax_type"),
        )
    )
    print(f"  negative pool (base>{args.min_base}, not real): {negpool.height:,}")

    negatives = base_matched_sample(positives, negpool, args.neg_ratio, args.n_bins, args.seed)
    print(f"  negatives sampled (base-matched, ~{args.neg_ratio}x): {negatives.height:,}")

    # ── Assemble + report base-match quality ─────────────────────────────
    drop = ["cand_argmax_type"]
    table = pl.concat([positives.drop(drop), negatives.drop(drop)], how="diagonal_relaxed")

    print("\nBase-score match check (median cand_base_score; should be close):")
    for t in ("donor", "acceptor"):
        pm = positives.filter(pl.col("cand_splice_type") == t)["cand_base_score"].median()
        nm = negatives.filter(pl.col("cand_splice_type") == t)["cand_base_score"].median()
        print(f"  {t:8s}: pos median={pm:.4f}  neg median={nm:.4f}")
    print(f"\nTotal candidates: {table.height:,} "
          f"({table['cand_label'].sum():,} real / {(table['cand_label']==0).sum():,} artifact)")
    print(f"  by type: {table.group_by('cand_splice_type').len().sort('cand_splice_type').to_dicts()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out)
    print(f"\nWrote {args.out}  ({table.width} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
