#!/usr/bin/env python
"""Meta-Layer Example 1: XGBoost Tabular Baseline.

Trains an XGBoost classifier on feature-engineered analysis_sequences
artifacts. Uses chromosome-split validation (train on some chromosomes,
test on held-out chromosomes) to avoid within-gene leakage.

By default, uses the SpliceAI chromosome split (Jaganathan et al., 2019):
  Train: chr2, 4, 6, 8, 10-22, X, Y
  Test:  chr1, 3, 5, 7, 9

A 10% random holdout from training genes is used for early stopping
(validation set), so the test set is never seen during training.

Usage:
    # Full genome with SpliceAI split (default)
    python 01_xgboost_baseline.py

    # Custom chromosome split (overrides the default)
    python 01_xgboost_baseline.py \\
        --train-chroms chr19 --test-chroms chr21 chr22

    # Custom input directory
    python 01_xgboost_baseline.py --input-dir /path/to/analysis_sequences/

    # With feature importance plot
    python 01_xgboost_baseline.py --plot --output-dir output/m1_fullgenome
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Import xgboost BEFORE agentic_spliceai to avoid libomp conflict
# with PyTorch on macOS. See dev/errors/dyld-library-path-torch-import.md
import xgboost as xgb  # noqa: E402 (must precede torch-importing packages)
import numpy as np

log = logging.getLogger(__name__)


# ── Training ─────────────────────────────────────────────────────────


LABEL_DECODING = {0: "donor", 1: "acceptor", 2: "neither"}


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 20,
) -> "xgboost.XGBClassifier":
    """Train XGBoost classifier with early stopping on validation set."""
    # Compute class weights (inverse frequency)
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    weight_map = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    sample_weights = np.array([weight_map[yi] for yi in y_train])

    log.info(
        "Class distribution (train): %s",
        {LABEL_DECODING.get(c, c): int(cnt) for c, cnt in zip(classes, counts)},
    )

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    log.info("Best iteration: %d / %d", model.best_iteration, n_estimators)
    return model


# ── Evaluation ─────────────────────────────────���─────────────────────


def evaluate(
    model: "xgboost.XGBClassifier",
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    top_k: int = 25,
) -> dict:
    """Evaluate model and report metrics + feature importance."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    target_names = ["donor", "acceptor", "neither"]
    report = classification_report(y_test, y_pred, target_names=target_names)

    # Per-class PR-AUC (one-vs-rest)
    pr_aucs = {}
    for i, name in enumerate(target_names):
        binary_true = (y_test == i).astype(int)
        if binary_true.sum() > 0:
            pr_aucs[name] = average_precision_score(binary_true, y_proba[:, i])
        else:
            pr_aucs[name] = float("nan")

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Feature importance
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = [
        (feature_names[i], float(importances[i]))
        for i in sorted_idx[:top_k]
    ]

    # Print results
    print("\n" + "=" * 70)
    print("XGBoost Meta-Layer Baseline — Evaluation Results")
    print("=" * 70)

    print(f"\n  Accuracy: {accuracy:.4f}")
    print(f"\n  PR-AUC (per class):")
    for name, auc in pr_aucs.items():
        print(f"    {name:12s}: {auc:.4f}")

    print(f"\n  Classification Report:\n{report}")

    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"  {'':12s} {'donor':>8s} {'acceptor':>8s} {'neither':>8s}")
    for i, name in enumerate(target_names):
        print(f"  {name:12s} {cm[i, 0]:8d} {cm[i, 1]:8d} {cm[i, 2]:8d}")

    print(f"\n  Top {top_k} Feature Importances:")
    for rank, (fname, imp) in enumerate(top_features, 1):
        print(f"    {rank:2d}. {fname:40s} {imp:.4f}")

    return {
        "accuracy": float(accuracy),
        "pr_aucs": {k: float(v) for k, v in pr_aucs.items()},
        "confusion_matrix": cm.tolist(),
        "top_features": top_features,
        "classification_report": report,
    }


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_k: int = 25,
    output_path: Optional[Path] = None,
) -> None:
    """Plot top feature importances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plot")
        return

    sorted_idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in sorted_idx]
    values = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(names)), values[::-1], color="#3572a5")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost Meta-Layer Baseline — Top {top_k} Features")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"\n  Feature importance plot saved to: {output_path}")
    else:
        plt.show()


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="XGBoost Meta-Layer Baseline (M1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Path to analysis_sequences directory. "
             "Default: auto-resolve from openspliceai registry.",
    )
    parser.add_argument(
        "--train-chroms",
        nargs="+",
        default=None,
        help="Training chromosomes (overrides default SpliceAI split). "
             "Example: --train-chroms chr19 chr20",
    )
    parser.add_argument(
        "--test-chroms",
        nargs="+",
        default=None,
        help="Test chromosomes (overrides default SpliceAI split). "
             "Example: --test-chroms chr21 chr22",
    )
    parser.add_argument(
        "--val-fraction", type=float, default=0.1,
        help="Fraction of training genes held out for validation / early "
             "stopping (default: 0.1)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=500,
        help="Max boosting rounds (default: 500)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Max tree depth (default: 6)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1,
        help="Learning rate (default: 0.1)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate feature importance plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for saving model and results",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Imports (after xgboost to avoid libomp conflict) ─────────────
    from agentic_spliceai.splice_engine.meta_layer.training.data_utils import (
        load_analysis_sequences,
        resolve_input_dir,
        get_gene_split,
        split_dataframe,
        get_feature_columns,
        prepare_features_and_labels,
    )
    from agentic_spliceai.splice_engine.eval.splitting import (
        SPLICEAI_TRAIN_CHROMS,
        SPLICEAI_TEST_CHROMS,
    )

    # ── Resolve input directory ─────────────────────────���────────────
    try:
        input_dir = resolve_input_dir(args.input_dir)
    except FileNotFoundError as e:
        print(str(e))
        return 1

    # ── Determine chromosomes ────────────────────────────────────────
    # Default: SpliceAI split.  Custom overrides take precedence.
    using_custom_split = args.train_chroms is not None or args.test_chroms is not None
    if using_custom_split:
        if args.train_chroms is None or args.test_chroms is None:
            print("ERROR: --train-chroms and --test-chroms must both be specified.")
            return 1
        train_chroms = args.train_chroms
        test_chroms = args.test_chroms
        all_chroms = sorted(set(train_chroms) | set(test_chroms))
        split_desc = f"Custom: train={train_chroms}, test={test_chroms}"
    else:
        train_chroms = sorted(SPLICEAI_TRAIN_CHROMS)
        test_chroms = sorted(SPLICEAI_TEST_CHROMS)
        all_chroms = sorted(SPLICEAI_TRAIN_CHROMS | SPLICEAI_TEST_CHROMS)
        split_desc = "SpliceAI (Jaganathan et al., 2019)"

    print("=" * 70)
    print("XGBoost Meta-Layer Baseline (M1)")
    print("=" * 70)
    print(f"  Input:     {input_dir}")
    print(f"  Split:     {split_desc}")
    print(f"  Train:     {len(train_chroms)} chromosomes")
    print(f"  Test:      {len(test_chroms)} chromosomes ({', '.join(test_chroms)})")
    print(f"  Val frac:  {args.val_fraction:.0%} of training genes")

    # ── Load data ─────────────────────────���──────────────────────────
    t0 = time.time()

    # Load ALL chromosomes, then split by gene membership.
    print(f"\n  Loading all {len(all_chroms)} chromosomes...")
    df_all = load_analysis_sequences(input_dir, all_chroms)
    print(f"    {df_all.height:,} positions, {df_all.width} columns")

    # Gene-level split: assigns genes to train/val/test by chromosome
    if using_custom_split:
        gene_split = get_gene_split(
            df_all,
            preset="custom",
            val_fraction=args.val_fraction,
            custom_train_chroms=set(train_chroms),
            custom_test_chroms=set(test_chroms),
        )
    else:
        gene_split = get_gene_split(
            df_all,
            preset="spliceai",
            val_fraction=args.val_fraction,
        )

    print(gene_split.summary())

    # Split into train/val/test DataFrames
    df_train, df_val, df_test = split_dataframe(df_all, gene_split)
    del df_all

    print(f"    Train: {df_train.height:,} positions ({gene_split.n_train} genes)")
    print(f"    Val:   {df_val.height:,} positions ({gene_split.n_val} genes)")
    print(f"    Test:  {df_test.height:,} positions ({gene_split.n_test} genes)")

    # Determine features from training data
    feature_cols = get_feature_columns(df_train)
    print(f"    Features: {len(feature_cols)}")

    # Extract arrays and free DataFrames
    X_train, y_train, feature_names = prepare_features_and_labels(
        df_train, feature_cols=feature_cols,
    )
    X_val, y_val, _ = prepare_features_and_labels(
        df_val, feature_cols=feature_cols,
    )
    X_test, y_test, _ = prepare_features_and_labels(
        df_test, feature_cols=feature_cols,
    )
    del df_train, df_val, df_test

    # ── Train ─────────────────────────��──────────────────────────────
    print(f"\n  Training XGBoost (max {args.n_estimators} rounds, depth={args.max_depth})...")
    print(f"  Early stopping on validation set ({X_val.shape[0]:,} positions)")
    model = train_xgboost(
        X_train, y_train,
        X_val, y_val,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # ── Evaluate on held-out test set ───────────────────────────────��
    results = evaluate(model, X_test, y_test, feature_names)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Save outputs ─────────────────────────���───────────────────────
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("output/meta_layer/m1_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "xgb_m1_baseline.ubj"
    model.get_booster().save_model(str(model_path))
    print(f"\n  Model saved: {model_path}")

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    serializable = {
        "split": split_desc,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "best_iteration": int(model.best_iteration),
        "accuracy": results["accuracy"],
        "pr_aucs": results["pr_aucs"],
        "confusion_matrix": results["confusion_matrix"],
        "top_features": results["top_features"],
    }
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # Save gene split for reproducibility
    split_path = output_dir / "gene_split.json"
    with open(split_path, "w") as f:
        json.dump(gene_split.to_dict(), f, indent=2)
    print(f"  Gene split saved: {split_path}")

    if args.plot:
        plot_path = output_dir / "feature_importance.png"
        plot_feature_importance(
            feature_names, model.feature_importances_,
            output_path=plot_path,
        )

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
