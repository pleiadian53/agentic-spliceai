#!/usr/bin/env python
"""Train the M3-R candidate refiner (Tier 2, Phase 1).

Binary classifier over the base-score-matched candidate table
(``candidate_labels.parquet``, built by
``examples/data_preparation/m3/11_build_candidate_labels.py``): given a
base-proposed splice-site candidate, is it a **real cryptic site or an artifact**?
Because the negatives are base-score-matched to the positives, the base score
cannot separate the classes — the model must use the multimodal evidence, which
is exactly the hypothesis Tier 0 could not confirm genome-wide.

Reuses the M1-P plumbing: ``data_utils`` loaders/splitter/feature-selection, the
leakage-safe **gene-level SpliceAI split** (test = chr1/3/5/7/9, the Tier 0
universe), SHAP via ``pred_contribs``. Runs locally on the M1 laptop.

Modes:
  --diagnostic-only : Phase-0 logistic AUC(base) vs AUC(base+modality) lift.
  (default)         : train XGBoost + held-out AUC/PR-AUC + SHAP-by-modality +
                      leave-one-modality-out ablation; save model + metrics.

Usage:
    python examples/meta_layer/14_train_candidate_refiner.py
    python examples/meta_layer/14_train_candidate_refiner.py --diagnostic-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
# Import xgboost BEFORE any torch-adjacent module: on macOS, if torch's bundled
# libomp loads first, libxgboost resolves OpenMP to it and fails on a missing
# symbol (___kmpc_dispatch_deinit). Loading xgboost first avoids the clash.
import xgboost as xgb
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment  # noqa: E402
setup_example_environment()

from agentic_spliceai.splice_engine.meta_layer.training.data_utils import (  # noqa: E402
    get_gene_split, split_dataframe, get_feature_columns, MODALITY_COLUMNS,
)

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]  # examples/meta_layer/<file> -> repo root
CAND = REPO / "data/mane/GRCh38/m3_labels/candidate_labels.parquet"
OUT = REPO / "output/meta_layer/m3r_candidate_refiner"

# Never features: the raw base probabilities (used to propose + base-match
# candidates) and the candidate bookkeeping columns.
EXTRA_EXCLUDE = {"donor_prob", "acceptor_prob", "neither_prob",
                 "cand_label", "cand_base_score", "cand_splice_type"}
# Modalities scored for lift/ablation (junction is the label-side modality, dropped).
AUX_MODALITIES = ["conservation", "epigenetic", "rbp_eclip", "chrom_access", "genomic"]


def _Xy(df: pl.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.nan_to_num(df.select(feats).fill_null(0).to_numpy().astype(np.float32))
    y = df["cand_label"].to_numpy().astype(int)
    return X, y


def _spliceai_split(df: pl.DataFrame):
    """Gene-level SpliceAI split; prefix bare chrom so the preset matches."""
    dfx = df.with_columns(("chr" + pl.col("chrom").cast(pl.String)).alias("chrom"))
    split = get_gene_split(dfx, preset="spliceai", val_fraction=0.1)
    return split_dataframe(df, split)  # split by gene_id on the original df


def _auc_logreg(Xtr, ytr, Xte, yte) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    return roc_auc_score(yte, clf.decision_function(sc.transform(Xte)))


def run_diagnostic(train, test) -> dict:
    """Phase-0 gate: does each multimodal group add discrimination beyond base?"""
    base = [c for c in MODALITY_COLUMNS["base_scores"] if c in train.columns]
    Xtr_b, ytr = _Xy(train, base); Xte_b, yte = _Xy(test, base)
    a_base = _auc_logreg(Xtr_b, ytr, Xte_b, yte)
    print(f"\n{'='*62}\nPhase-0 diagnostic (held-out chr1/3/5/7/9, real vs artifact)\n{'='*62}")
    print(f"AUC(base_scores, {len(base)} feats): {a_base:.4f}")
    out = {"base": a_base, "lift": {}}
    for mod in AUX_MODALITIES:
        cols = base + [c for c in MODALITY_COLUMNS[mod] if c in train.columns]
        Xtr, _ = _Xy(train, cols); Xte, _ = _Xy(test, cols)
        a = _auc_logreg(Xtr, ytr, Xte, yte)
        out["lift"][mod] = a - a_base
        flag = "  <- complementary" if a - a_base > 0.01 else "  (redundant)"
        print(f"  base + {mod:13s}: AUC={a:.4f}  lift={a-a_base:+.4f}{flag}")
    allcols = base + sum([[c for c in MODALITY_COLUMNS[m] if c in train.columns] for m in AUX_MODALITIES], [])
    Xtr, _ = _Xy(train, allcols); Xte, _ = _Xy(test, allcols)
    out["base_plus_all"] = _auc_logreg(Xtr, ytr, Xte, yte)
    print(f"  base + ALL multimodal: AUC={out['base_plus_all']:.4f}  lift={out['base_plus_all']-a_base:+.4f}")
    return out


def _train_xgb(Xtr, ytr, Xva, yva, feats, seed=42):
    w = np.where(ytr == 1, (ytr == 0).sum() / max(1, (ytr == 1).sum()), 1.0)
    m = xgb.XGBClassifier(
        n_estimators=600, max_depth=6, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, objective="binary:logistic", eval_metric="aucpr",
        early_stopping_rounds=30, tree_method="hist", n_jobs=-1, random_state=seed,
    )
    m.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=CAND)
    ap.add_argument("--output-dir", type=Path, default=OUT)
    ap.add_argument("--diagnostic-only", action="store_true")
    ap.add_argument("--no-ablation", action="store_true", help="skip leave-one-modality-out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from sklearn.metrics import roc_auc_score, average_precision_score

    df = pl.read_parquet(args.candidates)
    feats = get_feature_columns(df, exclude_modalities=["junction"], extra_exclude=EXTRA_EXCLUDE)
    train, val, test = _spliceai_split(df)
    print(f"candidates: {df.height:,} ({df['cand_label'].sum():,} real / "
          f"{(df['cand_label']==0).sum():,} artifact) | {len(feats)} features")
    print(f"split: train={train.height:,} val={val.height:,} test={test.height:,} (test=chr1/3/5/7/9)")

    diag = run_diagnostic(train, test)
    if args.diagnostic_only:
        return 0

    # ── Train ────────────────────────────────────────────────────────────
    Xtr, ytr = _Xy(train, feats); Xva, yva = _Xy(val, feats); Xte, yte = _Xy(test, feats)
    model = _train_xgb(Xtr, ytr, Xva, yva, feats)
    p = model.predict_proba(Xte)[:, 1]
    auc, prauc = roc_auc_score(yte, p), average_precision_score(yte, p)
    print(f"\n{'='*62}\nM3-R XGBoost (held-out test)\n{'='*62}")
    print(f"  AUC={auc:.4f}  PR-AUC={prauc:.4f}  (best_iteration={model.best_iteration})")

    # ── SHAP by modality ──────────────────────────────────────────────────
    booster = model.get_booster()
    sv = booster.predict(xgb.DMatrix(Xte, feature_names=feats), pred_contribs=True)[:, :-1]
    imp = np.abs(sv).mean(0)
    col2mod = {c: m for m, cols in MODALITY_COLUMNS.items() for c in cols}
    modimp = defaultdict(float)
    for c, i in zip(feats, imp):
        modimp[col2mod.get(c, "other")] += float(i)
    tot = sum(modimp.values()) or 1.0
    shap_by_mod = {m: v / tot for m, v in sorted(modimp.items(), key=lambda x: -x[1])}
    print("SHAP importance by modality (share of total):")
    for m, v in shap_by_mod.items():
        print(f"  {m:13s}: {v*100:5.1f}%")
    nonbase = 1.0 - shap_by_mod.get("base_scores", 0.0)
    print(f"  -> non-base share: {nonbase*100:.1f}%")

    # ── Leave-one-modality-out ablation ──────────────────────────────────
    ablation = {}
    if not args.no_ablation:
        print("\nLeave-one-modality-out (test AUC drop vs full):")
        for mod in AUX_MODALITIES + ["base_scores"]:
            drop = set(MODALITY_COLUMNS[mod])
            f2 = [c for c in feats if c not in drop]
            m2 = _train_xgb(train.select(f2).fill_null(0).to_numpy().astype(np.float32), ytr,
                            val.select(f2).fill_null(0).to_numpy().astype(np.float32), yva, f2)
            a2 = roc_auc_score(yte, m2.predict_proba(
                np.nan_to_num(test.select(f2).fill_null(0).to_numpy().astype(np.float32)))[:, 1])
            ablation[mod] = auc - a2
            print(f"  drop {mod:13s}: AUC={a2:.4f}  drop={auc-a2:+.4f}")

    # ── Save ─────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save the raw booster (not the sklearn wrapper — avoids an _estimator_type
    # quirk on save, and Phase 2 loads via Booster + DMatrix anyway).
    booster.save_model(str(args.output_dir / "m3r_xgb.ubj"))
    (args.output_dir / "features.json").write_text(json.dumps(feats, indent=2))
    metrics = {
        "n_candidates": df.height, "n_real": int(df["cand_label"].sum()),
        "n_features": len(feats), "test_chroms": ["1", "3", "5", "7", "9"],
        "n_train": train.height, "n_val": val.height, "n_test": test.height,
        "test_auc": auc, "test_pr_auc": prauc, "best_iteration": int(model.best_iteration),
        "diagnostic": diag, "shap_by_modality": shap_by_mod,
        "nonbase_shap_share": nonbase, "ablation_auc_drop": ablation,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nSaved model + metrics -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
