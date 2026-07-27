#!/usr/bin/env python
"""M3-R Phase 2 — anti-circular eval: does candidate reranking beat the base model?

The real test of the Tier 2 reframe. The Phase 1 AUC (0.90) is on the training
label distribution; this scores the trained M3-R against **independent** truth on
**held-out** chromosomes, head-to-head with base-score ranking on the *same*
candidate set.

Per test-chrom gene: the candidate pool = feature-parquet positions
(base prob > --min-base, typed by argmax(donor, acceptor)). Rank them by (a) base
score and (b) M3-R P(real). Both are rasterized into `[L, 3]` arrays with **-inf
fill** (so the precision@k denominator stays candidate-native and the novelty
post-filter fires for free) and scored through the **unchanged** `_m3_novel_eval`
metric library — the exact machinery the Tier 0 harness uses. Win condition:
M3-R's reranking beats base on D1/D1_hiconf/D2 precision@k / recall@k.

(Absolute numbers are NOT comparable to Tier 0 — that scored a dense pool, this
scores the parquet-sampled pool — but base-vs-M3-R here is apples-to-apples.)

Usage:
    python examples/meta_layer/15_evaluate_candidate_classifier.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb  # before torch (macOS libomp clash)
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment  # noqa: E402
setup_example_environment()
sys.path.insert(0, str(Path(__file__).parent))
import _m3_novel_eval as m3eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FEATURE_DIR = REPO / "data/mane/GRCh38/openspliceai_eval/analysis_sequences"
D1_TRUTH = REPO / "data/encode_longread/GRCh38/longread_truth_novel.parquet"
D2_ANCHORS = REPO / "data/mane/GRCh38/m3_labels/disease_anchors.parquet"
ANNOTATION = REPO / "data/mane/GRCh38/m3_labels/annotation_mask.parquet"
KS = (5, 10, 20)
MAX_K = max(KS)


def rasterize(pos, score, is_donor, gene) -> np.ndarray:
    """Scatter per-candidate scores into a [L,3] array, -inf elsewhere.

    donor→col0, acceptor→col1 (under gene.strand, per the harness convention);
    duplicate (position,type) collapsed by max. -inf fill keeps the precision@k
    denominator candidate-native (see _m3_novel_eval.top_novel_candidates).
    """
    L = gene.end - gene.start
    arr = np.full((L, 3), -np.inf, dtype=np.float64)
    rel = pos - gene.start
    v = (rel >= 0) & (rel < L)
    rel, sc, col = rel[v], score[v], np.where(is_donor[v], 0, 1)
    np.maximum.at(arr, (rel, col), sc)
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", type=Path, default=REPO / "output/meta_layer/m3_eval_d1/eval_genes.parquet")
    ap.add_argument("--model-dir", type=Path, default=REPO / "output/meta_layer/m3r_candidate_refiner")
    ap.add_argument("--output-dir", type=Path, default=REPO / "output/meta_layer/m3_eval_d1")
    ap.add_argument("--min-base", type=float, default=0.01, help="candidate pool: max(donor,acceptor) prob >")
    args = ap.parse_args()

    booster = xgb.Booster()
    booster.load_model(str(args.model_dir / "m3r_xgb.ubj"))
    feats = json.loads((args.model_dir / "features.json").read_text())

    # ── Universe (test-chrom gene intervals) ─────────────────────────────
    uni = pl.read_parquet(args.universe)
    genes = [m3eval.GeneInterval(r["gene_id"], r["chrom"], int(r["start"]), int(r["end"]), r["strand"])
             for r in uni.iter_rows(named=True)]
    test_chroms = sorted({g.chrom for g in genes})
    print(f"universe: {len(genes)} genes on {test_chroms}")

    # ── Candidate pool: test-chrom feature-parquet positions ─────────────
    frames = []
    for c in test_chroms:
        p = FEATURE_DIR / f"analysis_sequences_chr{c}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p))
    cand = pl.concat(frames, how="diagonal")
    cand = cand.with_columns(
        pl.col("chrom").cast(pl.String).str.replace_all(r"^chr", "").alias("chrom"),
        pl.col("position").cast(pl.Int64),
        pl.max_horizontal("donor_prob", "acceptor_prob").alias("cand_base_score"),
        (pl.col("donor_prob") >= pl.col("acceptor_prob")).alias("is_donor"),
    ).filter(pl.col("cand_base_score") > args.min_base)
    print(f"candidate positions (base>{args.min_base}): {cand.height:,}")

    # M3-R score for every candidate (batch)
    X = np.nan_to_num(cand.select(feats).fill_null(0).to_numpy().astype(np.float32))
    cand = cand.with_columns(pl.Series("m3r", booster.predict(xgb.DMatrix(X, feature_names=feats))))

    # per-chrom sorted index for O(log n) per-gene slicing
    idx = {}
    for key, sub in cand.group_by("chrom"):
        c = key[0] if isinstance(key, tuple) else key
        s = sub.sort("position")
        idx[str(c)] = (s["position"].to_numpy(), s["cand_base_score"].to_numpy(),
                       s["m3r"].to_numpy(), s["is_donor"].to_numpy())

    # ── Truth + annotation indices (reuse Tier 0 loaders verbatim) ───────
    annotation = m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(ANNOTATION, keep_chroms_bare=test_chroms))
    truth_sets = {
        "D1": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(D1_TRUTH, keep_chroms_bare=test_chroms)),
        "D1_hiconf": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(D1_TRUTH, keep_chroms_bare=test_chroms, min_biosamples=2)),
        "D2": m3eval.SiteIndex.from_df(m3eval.load_sites_parquet(D2_ANCHORS, keep_chroms_bare=test_chroms, is_novel_only=True)),
    }
    MODELS = ["base", "M3-R"]
    accs = {ts: {m: {"donor": m3eval.KMetricAccumulator(KS), "acceptor": m3eval.KMetricAccumulator(KS)}
                 for m in MODELS} for ts in truth_sets}

    # ── Score every gene ─────────────────────────────────────────────────
    n_scored = 0
    for gene in genes:
        chrom_idx = idx.get(gene.chrom)
        if chrom_idx is None:
            continue
        pos_all, base_all, m3r_all, isd_all = chrom_idx
        lo = np.searchsorted(pos_all, gene.start, "left")
        hi = np.searchsorted(pos_all, gene.end, "left")
        if hi <= lo:
            continue
        pos, base_s, m3r_s, isd = pos_all[lo:hi], base_all[lo:hi], m3r_all[lo:hi], isd_all[lo:hi]
        arr_base = rasterize(pos, base_s, isd, gene)
        arr_m3r = rasterize(pos, m3r_s, isd, gene)
        for ts, tidx in truth_sets.items():
            m3eval.evaluate_gene(arr_base, gene, annotation, tidx, accs[ts]["base"], MAX_K)
            m3eval.evaluate_gene(arr_m3r, gene, annotation, tidx, accs[ts]["M3-R"], MAX_K)
        n_scored += 1
    print(f"scored {n_scored} genes")

    # ── Aggregate + report ───────────────────────────────────────────────
    results = {}
    for ts in truth_sets:
        per_model = {}
        for m in MODELS:
            d = accs[ts][m]["donor"].result()
            a = accs[ts][m]["acceptor"].result()
            per_model[m] = {"donor": d, "acceptor": a, "combined": m3eval.combine_type_results(d, a, KS)}
        results[ts] = per_model
        n = next(iter(per_model.values()))["combined"]["n_genes_with_truth"]
        print(f"\n{'='*70}\n{ts}: base vs M3-R (per-gene macro, donor+acceptor; {n} gene×type w/ truth)\n{'='*70}")
        print(m3eval.format_comparison_table(per_model, KS))

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    payload = {"models": MODELS, "ks": list(KS), "min_base": args.min_base,
               "n_genes_scored": n_scored, "test_chroms": test_chroms, "results": results}
    (out / "m3r_eval_metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nResults -> {out / 'm3r_eval_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
