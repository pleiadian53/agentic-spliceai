#!/usr/bin/env python
"""Meta-Layer Example 2: Probability Calibration Analysis.

Compares probability calibration between the base model (OpenSpliceAI)
and the XGBoost meta-layer. Answers: "Do the predicted probabilities
actually match observed frequencies?"

Uses calibration utilities from splice_engine.eval.calibration.

Background
----------
A model is **well-calibrated** if, among all positions where it predicts
P(donor) = 0.8, approximately 80% are truly donor sites.

Good calibration matters for:
  - Threshold decisions (which positions to call as splice sites)
  - Score fusion across models (multimodal combination)
  - Clinical confidence reporting (variant interpretation)

Note: High accuracy (discrimination) does NOT imply good calibration.
A model can perfectly rank positions but still output poorly calibrated
probabilities.

Usage:
    python 02_calibration_analysis.py
    python 02_calibration_analysis.py --n-bins 20 --plot
    python 02_calibration_analysis.py --test-chroms chr21 chr22 --plot
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Import xgboost BEFORE agentic_spliceai to avoid libomp conflict
# with PyTorch on macOS. See dev/errors/dyld-library-path-torch-import.md
import xgboost as xgb  # noqa: E402 (must precede torch-importing packages)
import numpy as np
import polars as pl

from agentic_spliceai.splice_engine.eval.calibration import (
    compare_calibration,
    print_calibration_comparison,
)

log = logging.getLogger(__name__)


# ── Column definitions ────────────────────────────────────────────────

LABEL_COL = "splice_type"
LABEL_ENCODING = {"donor": 0, "acceptor": 1, "neither": 2, "": 2}

EXCLUDE_COLS = {
    "splice_type", "pred_type", "true_position", "predicted_position",
    "is_correct", "error_type",
    "gene_id", "gene_name", "transcript_id", "gene_type",
    "chrom", "strand", "position", "absolute_position",
    "window_start", "window_end", "transcript_count",
    "gene_start", "gene_end", "sequence",
}


# ── Helpers ───────────────────────────────────────────────────────────


def load_data(input_dir: Path, chroms: list[str]) -> pl.DataFrame:
    """Load analysis_sequences parquets for specified chromosomes."""
    frames = []
    for chrom in chroms:
        for ext in ("parquet", "tsv"):
            path = input_dir / f"analysis_sequences_{chrom}.{ext}"
            if path.exists():
                if ext == "parquet":
                    frames.append(pl.read_parquet(path))
                else:
                    frames.append(pl.read_csv(path, separator="\t"))
                break

    if not frames:
        raise FileNotFoundError(f"No data for {chroms} in {input_dir}")

    # Schema alignment
    if len(frames) > 1:
        common = set(frames[0].columns)
        for f in frames[1:]:
            common &= set(f.columns)
        max_cols = max(len(f.columns) for f in frames)
        if len(common) < max_cols:
            ordered = [c for c in frames[0].columns if c in common]
            frames = [f.select(ordered) for f in frames]

    return pl.concat(frames)


def prepare_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and label vector y."""
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in (
            pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32,
        )
    ]
    X = np.nan_to_num(df.select(feature_cols).to_numpy().astype(np.float32))
    labels = df[LABEL_COL].to_list()
    y = np.array([LABEL_ENCODING.get(str(l).lower(), 2) for l in labels])
    return X, y, feature_cols


def train_meta_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> "xgboost.XGBClassifier":
    """Train XGBoost meta-layer model."""
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weight_map = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weights = np.array([weight_map[yi] for yi in y_train])

    model = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=20,
        tree_method="hist", random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(
        X_train, y_train, sample_weight=sample_weights,
        eval_set=[(X_test, y_test)], verbose=False,
    )
    return model


def plot_reliability_diagrams(
    comparisons: list,
    output_dir: Path,
) -> None:
    """Plot reliability diagrams for each class."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, comp in enumerate(comparisons):
        ax = axes[idx]

        # Perfect calibration
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

        # Base model
        base_rc = comp.base_reliability
        if base_rc.predicted:
            ax.plot(
                base_rc.predicted, base_rc.observed,
                "o-", color="#e74c3c", label="Base (OpenSpliceAI)",
                markersize=6, linewidth=1.5,
            )

        # Meta-layer
        meta_rc = comp.meta_reliability
        if meta_rc.predicted:
            ax.plot(
                meta_rc.predicted, meta_rc.observed,
                "s-", color="#3572a5", label="Meta (XGBoost)",
                markersize=6, linewidth=1.5,
            )

        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_title(comp.class_name.capitalize())
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle("Reliability Diagrams — Base Model vs Meta-Layer", fontsize=13)
    plt.tight_layout()

    path = output_dir / "reliability_diagrams.png"
    fig.savefig(path, dpi=150)
    print(f"\n  Reliability diagrams saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibration Analysis: Base Model vs Meta-Layer",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--train-chroms", nargs="+", default=["chr19"])
    parser.add_argument("--test-chroms", nargs="+", default=["chr21", "chr22"])
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve input
    if args.input_dir is not None:
        input_dir = args.input_dir
    else:
        from agentic_spliceai.splice_engine.resources import get_model_resources
        resources = get_model_resources("openspliceai")
        registry = resources.get_registry()
        input_dir = (
            registry.get_base_model_eval_dir("openspliceai") / "analysis_sequences"
        )

    print("=" * 70)
    print("Probability Calibration Analysis")
    print("Base Model (OpenSpliceAI) vs Meta-Layer (XGBoost)")
    print("=" * 70)

    t0 = time.time()

    # Load data
    print(f"\n  Train: {', '.join(args.train_chroms)}")
    df_train = load_data(input_dir, args.train_chroms)
    print(f"    {df_train.height:,} positions")

    print(f"  Test:  {', '.join(args.test_chroms)}")
    df_test = load_data(input_dir, args.test_chroms)
    print(f"    {df_test.height:,} positions")

    # Prepare features
    X_train, y_train, feat_names = prepare_xy(df_train)
    X_test, y_test, test_feat_names = prepare_xy(df_test)

    if set(feat_names) != set(test_feat_names):
        common = [f for f in feat_names if f in set(test_feat_names)]
        X_train = X_train[:, [feat_names.index(f) for f in common]]
        X_test = X_test[:, [test_feat_names.index(f) for f in common]]
        feat_names = common

    print(f"  Features: {len(feat_names)}")

    # Base model probabilities
    base_probs = {
        "donor": df_test["donor_prob"].to_numpy(),
        "acceptor": df_test["acceptor_prob"].to_numpy(),
        "neither": df_test["neither_prob"].to_numpy(),
    }

    # Train meta-layer
    print(f"\n  Training XGBoost meta-layer...")
    model = train_meta_model(X_train, y_train, X_test, y_test)
    meta_proba = model.predict_proba(X_test)
    print(f"  Best iteration: {model.best_iteration}/500")

    # Calibration comparison using library utilities
    comparisons = []
    for class_idx, class_name in [(0, "donor"), (1, "acceptor"), (2, "neither")]:
        y_binary = (y_test == class_idx).astype(int)
        comp = compare_calibration(
            y_true=y_binary,
            base_prob=base_probs[class_name],
            meta_prob=meta_proba[:, class_idx],
            class_name=class_name,
            n_bins=args.n_bins,
        )
        comparisons.append(comp)

    # Print results using library formatter
    print(f"\n{'=' * 70}")
    print(f"Calibration Results ({args.n_bins} bins)")
    print(f"{'=' * 70}")
    print_calibration_comparison(comparisons)

    # Reliability curve detail tables
    print(f"{'=' * 70}")
    print("Reliability Curves (per-bin detail)")
    print(f"{'=' * 70}")

    for comp in comparisons:
        base_rc = comp.base_reliability
        meta_rc = comp.meta_reliability

        print(f"\n  {comp.class_name.upper()}:")
        print(f"  {'Bin':>4s}  {'Pred(base)':>10s} {'Obs(base)':>10s} "
              f"{'Gap':>8s}  {'Pred(meta)':>10s} {'Obs(meta)':>10s} "
              f"{'Gap':>8s}  {'N':>8s}")

        max_rows = max(len(base_rc.predicted), len(meta_rc.predicted))
        for i in range(max_rows):
            bp = f"{base_rc.predicted[i]:.4f}" if i < len(base_rc.predicted) else ""
            bo = f"{base_rc.observed[i]:.4f}" if i < len(base_rc.observed) else ""
            bg = (
                f"{abs(base_rc.observed[i] - base_rc.predicted[i]):.4f}"
                if i < len(base_rc.predicted) else ""
            )
            mp = f"{meta_rc.predicted[i]:.4f}" if i < len(meta_rc.predicted) else ""
            mo = f"{meta_rc.observed[i]:.4f}" if i < len(meta_rc.observed) else ""
            mg = (
                f"{abs(meta_rc.observed[i] - meta_rc.predicted[i]):.4f}"
                if i < len(meta_rc.predicted) else ""
            )
            bn = (
                f"{base_rc.counts[i]:,}" if i < len(base_rc.counts) else ""
            )

            print(f"  {i + 1:4d}  {bp:>10s} {bo:>10s} {bg:>8s}  "
                  f"{mp:>10s} {mo:>10s} {mg:>8s}  {bn:>8s}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Plot
    if args.plot:
        output_dir = args.output_dir or Path("output/calibration")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_reliability_diagrams(comparisons, output_dir)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
