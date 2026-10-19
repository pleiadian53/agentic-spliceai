#!/usr/bin/env python
"""M4 benchmark: predict a held-out regulator's splicing response from k examples.

Task (HepG2 RBP knockdown, A3SS/A5SS): given an event, a knocked-down regulator R,
and features available before the knockdown outcome, predict ``dpsi``, the signed
change in usage of the event's long form. Each regulator is a *task*; the question
is how well a method adapts to a regulator it was not trained on, from k labeled
events of that regulator (k = 0 is zero-shot).

Regulators are split into ``--folds`` folds; each fold is held out in turn. Folds are
grouped by **control batch** (knockdowns sharing control libraries share control noise
in their ΔPSI; splitting a batch leaks it). Every
method is scored on the **same rows**: the held-out regulator's events in held-out
genes (SpliceAI chromosomes 1/3/5/7/9). Shots are drawn only from the regulator's
events in training genes, so no test gene is ever seen under R.

Two settings differ only in what is known about the *test genes*:

* ``S3`` (headline) — held-out regulators AND held-out genes: nothing about the test
  genes' knockdown responses is available. The pooled model trains on training
  regulators × training genes.
* ``S2`` — held-out regulators on genes already profiled under other knockdowns
  (a new assay on known targets). The pooled model may also train on training
  regulators × test genes, and the ``volatility`` prior (an event's mean ΔPSI under
  the training regulators) is available.

Methods (the benchmark ladder):

| method | k | what it is |
|---|---|---|
| ``zero`` | – | predict 0; the magnitude floor |
| ``volatility`` | – | S2 only: event's mean ΔPSI under training regulators |
| ``pooled`` | 0 | XGBoost on site ⊕ regulator-binding features, all training regulators |
| ``xgb`` | k | XGBoost fit on R's k shots only |
| ``tabpfn`` | k | TabPFN in-context on R's k shots (no gradient steps) |
| ``pooled_tabpfn`` | k | TabPFN on R's k shots, predicting the residual of ``pooled`` |

Metrics per (regulator, method, k, draw) from ``_m4_common.regression_metrics``:
Spearman/Pearson and weighted MAE on all events; direction accuracy, sign AUROC
and MAE on the prespecified significant events. Summaries are macro-averaged over
regulators (each regulator is one task).

Outputs (``--out-dir``): ``per_regulator.parquet``, ``summary.csv``,
``learning_curves.png``, ``config.json``.

Usage:
    # quick configuration: 2 folds, 20 regulators, local methods only
    python examples/meta_layer/17_m4_fewshot_benchmark.py \\
        --folds 2 --regulators 20 --methods zero pooled xgb --k 16 64

    # full S3 ladder (TabPFN v2 by default: free, Apache-2.0-based, no license gate)
    python examples/meta_layer/17_m4_fewshot_benchmark.py --setting S3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

# xgboost must load before torch (torch is imported lazily by tabpfn): if torch's
# libomp loads first, libxgboost fails on ___kmpc_dispatch_deinit (macOS).
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _m4_common as m4  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "output/meta_layer/m4_fewshot"
SCORE_COLS = ["spearman", "pearson", "mae_w", "dir_acc_sig", "sign_auroc_sig", "mae_sig"]


def build_table(feature_set: str) -> tuple[pl.DataFrame, list[str]]:
    """Events (labels, split, weights) joined to geometry + regulator features."""
    ev = m4.assign_gene_split(m4.load_events())
    geo, geo_cols = m4.geometry_features(ev, pl.read_parquet(m4.SITE_FEATURES))
    reg, reg_cols = m4.regulator_features(ev, pl.read_parquet(m4.REGULATOR_FEATURES))
    if feature_set == "compact":  # geometry + long−short deltas (+ partner context)
        geo_cols = [c for c in geo_cols if not c.startswith(("long__", "short__"))]
    feats = geo_cols + ["baseline_psi_loo", "n_other_kd"] + reg_cols
    leaked = set(feats) & m4.OUTCOME_COLS
    if leaked:
        raise ValueError(f"outcome columns in the feature set: {sorted(leaked)}")
    table = (
        ev.select("event_key", "rbp", "control_batch", "gene_id", "split", "dpsi", "weight",
                  "significant", "baseline_psi_loo", "n_other_kd")
        .join(geo, on="event_key", how="left")
        .join(reg, on=["event_key", "rbp"], how="left")
    )
    return table, feats


def _xgb(n_estimators: int, depth: int, seed: int) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(n_estimators=n_estimators, max_depth=depth, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                            random_state=seed, n_jobs=-1)


# TabPFN v2 (Hollmann et al., Nature 2025) is the default: its weights are under the
# Prior Labs License 1.1 (Apache 2.0 + "Built with PriorLabs-TabPFN" attribution) and
# download without a license gate. v2.5 and later are free only for non-commercial,
# non-production use, require license acceptance on the Prior Labs account, and need
# TABPFN_TOKEN; select them with --tabpfn-version.
TABPFN_DEFAULT_VERSION = "v2"


def _tabpfn(seed: int, version: str = TABPFN_DEFAULT_VERSION):
    """TabPFN regressor on CUDA when present (pods), else CPU (MPS needs torch >= 2.6)."""
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TabPFNRegressor.create_default_for_version(
        ModelVersion(version), device=device, random_state=seed)


def _tabpfn_ready(version: str) -> str | None:
    """None if TabPFN can fit here; otherwise the reason (download, license, ...)."""
    try:
        rng = np.random.default_rng(0)
        _tabpfn(0, version).fit(rng.random((16, 3)), rng.random(16))
    except Exception as exc:  # noqa: BLE001 — any failure means: run without TabPFN
        return f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return None


def _init_tabpfn_worker(threads: int) -> None:
    """Pool initializer: split the CPU threads across workers (preprocessing is CPU-bound)."""
    import torch

    torch.set_num_threads(max(1, threads))


def _tabpfn_predict(xs: np.ndarray, target: np.ndarray, xt: np.ndarray, seed: int,
                    version: str) -> np.ndarray:
    """One in-context TabPFN fit on the shots, predicting the regulator's test rows."""
    return _tabpfn(seed, version).fit(xs, target).predict(xt)


def run_tabpfn_jobs(jobs: list[dict], tests: dict[str, np.ndarray], version: str,
                    workers: int) -> list[tuple[dict, np.ndarray]]:
    """Run TabPFN jobs serially or on a bounded process pool.

    A TabPFN v2 call (8 ensemble members, 229 features, ~2,000 query rows) takes 7-10 s,
    almost all of it single-threaded CPU preprocessing, so a full benchmark (~5,000 calls)
    is ~12 h serially. Workers use the ``spawn`` start method (CUDA-safe), and at most
    ``2 * workers`` jobs are in flight so each regulator's test matrix is pickled only
    while it is needed.
    """
    if workers <= 1:
        return [(j, _tabpfn_predict(j["xs"], j["target"], tests[j["rbp"]], j["seed"], version))
                for j in jobs]
    import multiprocessing as mp
    import os
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

    threads = (os.cpu_count() or workers) // workers
    out: list[tuple[dict, np.ndarray]] = []
    pending: dict = {}
    t0, n_done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=_init_tabpfn_worker, initargs=(threads,)) as ex:
        it = iter(jobs)
        while True:
            while len(pending) < 2 * workers:
                j = next(it, None)
                if j is None:
                    break
                fut = ex.submit(_tabpfn_predict, j["xs"], j["target"], tests[j["rbp"]],
                                j["seed"], version)
                pending[fut] = j
            if not pending:
                break
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                out.append((pending.pop(fut), fut.result()))
                n_done += 1
                if n_done % 250 == 0 or n_done == len(jobs):
                    rate = n_done / (time.time() - t0)
                    print(f"  TabPFN {n_done:,}/{len(jobs):,} jobs "
                          f"({rate:.2f}/s, ~{(len(jobs) - n_done) / rate / 60:.0f} min left)",
                          flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["S2", "S3"], default="S3")
    ap.add_argument("--methods", nargs="+",
                    default=["zero", "volatility", "pooled", "xgb", "tabpfn", "pooled_tabpfn"])
    ap.add_argument("--k", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--draws", type=int, default=3, help="shot resamples per (regulator, k)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fold-grouping", choices=["control_batch", "none"], default="control_batch",
                    help="keep regulators that share control samples in one fold (default); "
                         "'none' reproduces the leaky v0 folds")
    ap.add_argument("--regulators", type=int, default=0,
                    help="evaluate only the N held-out regulators with the most significant "
                         "test events (0 = all)")
    ap.add_argument("--feature-set", choices=["compact", "full"], default="compact")
    ap.add_argument("--tabpfn-version", default=TABPFN_DEFAULT_VERSION,
                    help="TabPFN model version (v2 default; v2.5+ are non-commercial and gated)")
    ap.add_argument("--tabpfn-workers", type=int, default=1,
                    help="parallel TabPFN worker processes (1 = serial; ~8 on a GPU pod)")
    ap.add_argument("--pooled-max-rows", type=int, default=400_000,
                    help="uniformly subsample pooled training rows (event weights still apply in the fit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    methods = list(args.methods)
    if args.setting == "S3" and "volatility" in methods:
        methods.remove("volatility")  # needs test genes' responses under other regulators
    if {"tabpfn", "pooled_tabpfn"} & set(methods):
        reason = _tabpfn_ready(args.tabpfn_version)
        if reason:
            print(f"TabPFN {args.tabpfn_version} unavailable ({reason}) → skipping tabpfn / "
                  "pooled_tabpfn.")
            methods = [x for x in methods if x not in ("tabpfn", "pooled_tabpfn")]
    out_dir = args.out_dir or DEFAULT_OUT / f"{args.setting.lower()}_{args.feature_set}"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    table, feats = build_table(args.feature_set)
    print(f"{table.height:,} event×regulator rows, {len(feats)} features ({args.feature_set}); "
          f"methods={methods}, setting={args.setting}")
    groups = (dict(table.select("rbp", "control_batch").unique().iter_rows())
              if args.fold_grouping == "control_batch" else None)
    folds = m4.regulator_folds(table["rbp"].unique().to_list(), args.folds, args.seed, groups)
    table = table.with_columns(pl.col("rbp").replace_strict(folds).alias("fold"))

    test_rows = table.filter(pl.col("split") == "test")
    sig_per_reg = test_rows.group_by("rbp").agg(pl.col("significant").sum().alias("n_sig"))
    chosen = sig_per_reg.sort("n_sig", descending=True)["rbp"].to_list()
    if args.regulators:
        chosen = chosen[: args.regulators]
    chosen_set = set(chosen)

    records = []
    tabpfn_jobs: list[dict] = []
    reg_ctx: dict[str, dict] = {}     # per regulator: fold, y, w, sig, pooled prediction
    reg_test: dict[str, np.ndarray] = {}  # per regulator: test feature matrix (stored once)
    for fold in range(args.folds):
        held = [r for r, f in folds.items() if f == fold and r in chosen_set]
        if not held:
            continue
        tf = time.time()
        train_regs = table.filter(pl.col("fold") != fold)
        pool_genes = ["train", "val"] if args.setting == "S3" else ["train", "val", "test"]
        pool = train_regs.filter(pl.col("split").is_in(pool_genes))

        pooled_model = None
        if {"pooled", "pooled_tabpfn"} & set(methods):
            if pool.height > args.pooled_max_rows:
                pool = pool.sample(n=args.pooled_max_rows, seed=args.seed)
            pooled_model = _xgb(400, 6, args.seed).fit(
                pool.select(feats).to_numpy(), pool["dpsi"].to_numpy(),
                sample_weight=pool["weight"].to_numpy())
        volatility = None
        if "volatility" in methods:
            volatility = train_regs.group_by("event_key").agg(
                pl.col("dpsi").mean().alias("vol"))

        for rbp in held:
            r_rows = table.filter(pl.col("rbp") == rbp)
            test = r_rows.filter(pl.col("split") == "test")
            shots_pool = r_rows.filter(pl.col("split") == "train")
            if test.height < 3:
                continue
            Xt = test.select(feats).to_numpy()
            y, w, sig = (test["dpsi"].to_numpy(), test["weight"].to_numpy(),
                         test["significant"].to_numpy())
            base_pred = pooled_model.predict(Xt) if pooled_model is not None else None

            ctx = {"fold": fold, "rbp": rbp, "y": y, "w": w, "sig": sig, "base": base_pred}
            if {"tabpfn", "pooled_tabpfn"} & set(methods):
                reg_ctx[rbp], reg_test[rbp] = ctx, Xt

            def record(method: str, k: int, draw: int, yhat: np.ndarray, ctx=ctx) -> None:
                records.append({"fold": ctx["fold"], "rbp": ctx["rbp"], "method": method,
                                "k": k, "draw": draw,
                                **m4.regression_metrics(ctx["y"], yhat, ctx["w"], ctx["sig"])})

            if "zero" in methods:
                record("zero", 0, 0, np.zeros_like(y))
            if volatility is not None:
                v = test.join(volatility, on="event_key", how="left")["vol"].fill_null(0.0)
                record("volatility", 0, 0, v.to_numpy())
            if base_pred is not None and "pooled" in methods:
                record("pooled", 0, 0, base_pred)

            for k in args.k:
                if shots_pool.height < k:
                    continue
                for draw in range(args.draws):
                    shots = shots_pool.sample(n=k, seed=args.seed * 1000 + draw)
                    Xs, ys = shots.select(feats).to_numpy(), shots["dpsi"].to_numpy()
                    if "xgb" in methods:
                        mdl = _xgb(100, 3, draw).fit(Xs, ys)
                        record("xgb", k, draw, mdl.predict(Xt))
                    if "tabpfn" in methods:
                        tabpfn_jobs.append({"rbp": rbp, "method": "tabpfn", "k": k,
                                            "draw": draw, "seed": draw, "xs": Xs, "target": ys})
                    if "pooled_tabpfn" in methods and pooled_model is not None:
                        tabpfn_jobs.append({"rbp": rbp, "method": "pooled_tabpfn", "k": k,
                                            "draw": draw, "seed": draw, "xs": Xs,
                                            "target": ys - pooled_model.predict(Xs)})
        print(f"  fold {fold}: {len(held)} held-out regulators, pooled on {pool.height:,} rows "
              f"({time.time() - tf:.0f}s)")

    if tabpfn_jobs:
        del table, test_rows  # free the event table before starting TabPFN workers
        import gc

        gc.collect()
        print(f"TabPFN {args.tabpfn_version}: {len(tabpfn_jobs):,} jobs on "
              f"{args.tabpfn_workers} worker(s)", flush=True)
        for job, yhat in run_tabpfn_jobs(tabpfn_jobs, reg_test, args.tabpfn_version,
                                         args.tabpfn_workers):
            c = reg_ctx[job["rbp"]]
            pred = c["base"] + yhat if job["method"] == "pooled_tabpfn" else yhat
            records.append({"fold": c["fold"], "rbp": job["rbp"], "method": job["method"],
                            "k": job["k"], "draw": job["draw"],
                            **m4.regression_metrics(c["y"], pred, c["w"], c["sig"])})

    per = pl.DataFrame(records)
    per.write_parquet(out_dir / "per_regulator.parquet")
    # average draws within a regulator first, then macro over regulators
    by_reg = per.group_by(["method", "k", "rbp"]).agg([pl.col(c).mean() for c in SCORE_COLS
                                                       if c in per.columns])
    summary = by_reg.group_by(["method", "k"]).agg(
        pl.len().alias("n_regulators"),
        *[pl.col(c).mean().alias(c) for c in SCORE_COLS if c in by_reg.columns],
        *[pl.col(c).std().alias(f"{c}_sd") for c in ("spearman", "dir_acc_sig")
          if c in by_reg.columns],
    ).sort(["method", "k"])
    summary.write_csv(out_dir / "summary.csv")
    _plot(summary, out_dir / "learning_curves.png", args.setting)
    (out_dir / "config.json").write_text(json.dumps(
        vars(args) | {"methods_run": methods, "out_dir": str(out_dir), "features": feats,
                      "n_regulators_evaluated": len(chosen),
                      "runtime_seconds": round(time.time() - t0, 1)}, indent=2, default=str))
    with pl.Config(tbl_rows=50, tbl_cols=12, float_precision=3):
        print(summary)
    print(f"\n→ {out_dir}  ({time.time() - t0:.0f}s)")
    return 0


# One fixed colour per method, so the same method looks the same in every figure.
METHOD_COLORS = {"zero": "#7f7f7f", "volatility": "#d62728", "pooled": "#1f77b4",
                 "xgb": "#ff7f0e", "tabpfn": "#2ca02c", "pooled_tabpfn": "#9467bd"}


def _plot(summary: pl.DataFrame, path: Path, setting: str) -> None:
    """Learning curves over k for Spearman and direction accuracy.

    k-shot methods are lines with markers; k = 0 methods (zero, volatility, pooled)
    are dashed horizontal references in their own colour.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, label in ((axes[0], "spearman", "Spearman ρ (all events)"),
                              (axes[1], "dir_acc_sig", "Direction accuracy (significant)")):
        if metric not in summary.columns:
            continue
        for method in summary["method"].unique().sort().to_list():
            s = summary.filter((pl.col("method") == method)
                               & pl.col(metric).is_not_null()).sort("k")
            if s.height == 0:  # e.g. Spearman is undefined for constant `zero`
                continue
            color = METHOD_COLORS.get(method)
            if s.height == 1 and s["k"][0] == 0:
                ax.axhline(s[metric][0], ls="--", lw=1.5, color=color,
                           label=f"{method} (k=0)")
            else:
                ax.plot(s["k"], s[metric], marker="o", color=color, label=method)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("k (labeled events of the held-out regulator)")
        ax.set_title(label)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"M4 held-out-regulator benchmark ({setting})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    sys.exit(main())
