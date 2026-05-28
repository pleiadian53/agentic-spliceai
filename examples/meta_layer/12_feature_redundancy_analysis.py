#!/usr/bin/env python
"""Feature redundancy / modality-orthogonality analysis for the M2 meta-layer.

Question this answers (before building any new fusion architecture):
    Do the auxiliary modalities (conservation, epigenetic, junction, RBP)
    carry signal that is *complementary* to the base splice site scores —
    especially where the base scores are uncertain — or are they largely
    *redundant*?

The answer determines whether a staged/grouped-mixing inductive bias
(e.g. grouped Conv1d, late fusion) could help, or whether the meta-layer
ceiling is a data/label property no fusion schedule will move.

Three tests:
  1. Cross-modality Spearman correlation — are base scores already
     correlated with the auxiliary modalities? (high → early mixing fine)
  2. "Where does the base model fail, and do aux modalities rescue?" —
     at TRUE splice sites the base model is uncertain about (correct-class
     prob < threshold), do aux modalities discriminate them from negatives?
     This is the crux: complementary-where-it-counts.
  3. Incremental discrimination — logistic AUC(base) vs AUC(base + modality
     group). Lift = orthogonal signal each modality adds beyond base scores.

Substrate: the Phase-5A full-genome feature parquets
(``data/mane/GRCh38/openspliceai_eval/analysis_sequences/analysis_sequences_chr*.parquet``,
116 columns). Modality redundancy is a feature-level property, so the MANE
analysis set is a valid proxy for the M2 (Ensembl) question.

Usage:
    python examples/meta_layer/12_feature_redundancy_analysis.py \
        --chromosomes chr22 chr21 chr19 --base-uncertain-threshold 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Modality channel groups (column names in the 116-col parquet).
MODALITIES: dict[str, list[str]] = {
    "base": ["donor_prob", "acceptor_prob", "neither_prob"],
    "conservation": ["phylop_score", "phastcons_score"],
    "epigenetic": [
        "h3k4me3_max_across_tissues", "h3k36me3_max_across_tissues",
        "atac_max_across_tissues", "dnase_max_across_tissues",
    ],
    "junction": ["junction_log1p", "junction_has_support"],
    "rbp": ["rbp_n_bound", "rbp_max_signal", "rbp_n_sr_proteins"],
}
ALL_COLS = [c for cols in MODALITIES.values() for c in cols]
DATA_DIR = Path("data/mane/GRCh38/openspliceai_eval/analysis_sequences")


def load(chromosomes: list[str]) -> pl.DataFrame:
    frames = []
    for chrom in chromosomes:
        p = DATA_DIR / f"analysis_sequences_{chrom}.parquet"
        if not p.exists():
            print(f"  WARN: {p} missing, skipping")
            continue
        frames.append(pl.read_parquet(p, columns=ALL_COLS + ["splice_type"]))
        print(f"  loaded {chrom}: {frames[-1].height:,} rows")
    df = pl.concat(frames)
    # is_splice + correct-class base prob per row
    df = df.with_columns(
        is_splice=pl.col("splice_type").is_in(["donor", "acceptor"]).cast(pl.Int8),
        base_correct_prob=pl.when(pl.col("splice_type") == "donor")
        .then(pl.col("donor_prob"))
        .when(pl.col("splice_type") == "acceptor")
        .then(pl.col("acceptor_prob"))
        .otherwise(pl.col("neither_prob")),
    )
    return df


def test1_correlation(df: pl.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TEST 1 — Cross-modality Spearman correlation")
    print("=" * 70)
    X = df.select(ALL_COLS).to_numpy()
    rho, _ = spearmanr(X)
    # Block-level summary: mean |rho| between base channels and each aux group
    base_idx = [ALL_COLS.index(c) for c in MODALITIES["base"]]
    print("\nMean |Spearman rho| between base scores and each aux modality:")
    for mod, cols in MODALITIES.items():
        if mod == "base":
            continue
        idx = [ALL_COLS.index(c) for c in cols]
        block = np.abs(rho[np.ix_(base_idx, idx)])
        print(f"  base ↔ {mod:14s}: mean|rho|={block.mean():.3f}  max|rho|={block.max():.3f}")
    print("\n(low |rho| → base scores and that modality are linearly independent →")
    print(" staged/grouped mixing has room to learn complementary features)")


def test2_rescue(df: pl.DataFrame, thresh: float) -> None:
    print("\n" + "=" * 70)
    print(f"TEST 2 — Where the base model fails, do aux modalities rescue?")
    print(f"        (base-uncertain TRUE site = correct-class prob < {thresh})")
    print("=" * 70)
    pos = df.filter(pl.col("is_splice") == 1)
    neg = df.filter(pl.col("is_splice") == 0)
    n_pos = pos.height
    uncertain = pos.filter(pl.col("base_correct_prob") < thresh)
    n_unc = uncertain.height
    print(f"\nTrue splice sites: {n_pos:,}")
    print(f"  base-uncertain (the ones M2 must rescue): {n_unc:,} ({n_unc/n_pos*100:.1f}%)")
    if n_unc < 30:
        print("  too few base-uncertain sites for a stable AUC — skipping rescue test")
        return

    # For each aux modality channel: can it alone discriminate
    #   {base-uncertain TRUE sites} from {negatives}?
    # AUC > 0.5 means the modality lights up exactly where the base model fails.
    neg_s = neg.sample(n=min(neg.height, 50_000), seed=42)
    print(f"\nPer-channel AUC discriminating base-uncertain TRUE sites vs negatives")
    print(f"  (vs. the same channel's AUC on ALL true sites, for reference):")
    for mod, cols in MODALITIES.items():
        if mod == "base":
            continue
        for c in cols:
            unc_v = uncertain[c].to_numpy()
            neg_v = neg_s[c].to_numpy()
            all_pos_v = pos[c].to_numpy()
            y_unc = np.r_[np.ones(len(unc_v)), np.zeros(len(neg_v))]
            x_unc = np.r_[unc_v, neg_v]
            auc_unc = roc_auc_score(y_unc, x_unc)
            y_all = np.r_[np.ones(len(all_pos_v)), np.zeros(len(neg_v))]
            x_all = np.r_[all_pos_v, neg_v]
            auc_all = roc_auc_score(y_all, x_all)
            flag = "  <- rescue signal" if auc_unc > 0.60 else ""
            print(f"  {c:28s}: AUC_uncertain={auc_unc:.3f}  AUC_all={auc_all:.3f}{flag}")
    print("\n(AUC_uncertain >> 0.5 → that modality carries signal exactly where the")
    print(" base model is uncertain → genuinely complementary → grouped/staged mixing")
    print(" could let the model exploit it. AUC_uncertain ≈ 0.5 → no rescue signal.)")


def _auc_logreg(X: np.ndarray, y: np.ndarray, seed: int = 42) -> float:
    n = len(y)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(X[tr]), y[tr])
    return roc_auc_score(y[te], clf.decision_function(sc.transform(X[te])))


def test3_incremental(df: pl.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("TEST 3 — Incremental discrimination beyond base scores")
    print("=" * 70)
    # Balance the problem a bit: all positives + 10x negatives
    pos = df.filter(pl.col("is_splice") == 1)
    neg = df.filter(pl.col("is_splice") == 0).sample(
        n=min(df.filter(pl.col("is_splice") == 0).height, pos.height * 10), seed=42
    )
    sub = pl.concat([pos, neg])
    y = sub["is_splice"].to_numpy()

    base_X = sub.select(MODALITIES["base"]).to_numpy()
    auc_base = _auc_logreg(base_X, y)
    print(f"\nAUC(base scores only):           {auc_base:.4f}")
    print("Lift from adding each modality group to base scores:")
    for mod, cols in MODALITIES.items():
        if mod == "base":
            continue
        X = sub.select(MODALITIES["base"] + cols).to_numpy()
        auc = _auc_logreg(X, y)
        lift = auc - auc_base
        flag = "  <- complementary" if lift > 0.01 else "  (redundant)"
        print(f"  base + {mod:14s}: AUC={auc:.4f}  lift={lift:+.4f}{flag}")
    all_X = sub.select(ALL_COLS).to_numpy()
    auc_all = _auc_logreg(all_X, y)
    print(f"  base + ALL modalities: AUC={auc_all:.4f}  lift={auc_all-auc_base:+.4f}")
    print("\n(lift > ~0.01 → that modality adds discrimination the base scores lack →")
    print(" complementary. lift ≈ 0 → redundant with base scores.)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chromosomes", nargs="+", default=["chr22", "chr21", "chr19"])
    ap.add_argument("--base-uncertain-threshold", type=float, default=0.5)
    args = ap.parse_args()

    print(f"Loading {args.chromosomes} ...")
    df = load(args.chromosomes)
    print(f"\nTotal: {df.height:,} positions  "
          f"({df['is_splice'].sum():,} splice sites, "
          f"{df['is_splice'].sum()/df.height*100:.2f}%)")

    test1_correlation(df)
    test2_rescue(df, args.base_uncertain_threshold)
    test3_incremental(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
