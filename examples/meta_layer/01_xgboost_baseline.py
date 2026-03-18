#!/usr/bin/env python
"""Meta-Layer Example 1: XGBoost Tabular Baseline.

Trains an XGBoost classifier on feature-engineered analysis_sequences
artifacts. Uses chromosome-split validation (train on some chromosomes,
test on held-out chromosomes) to avoid within-gene leakage.

This establishes the baseline that any neural meta-layer must beat.

Usage:
    # Default: train on chr19, test on chr21+chr22
    python 01_xgboost_baseline.py

    # Custom chromosome split
    python 01_xgboost_baseline.py \\
        --train-chroms chr1 chr2 chr3 \\
        --test-chroms chr21 chr22

    # Custom input directory
    python 01_xgboost_baseline.py --input-dir /path/to/analysis_sequences/

    # With feature importance plot
    python 01_xgboost_baseline.py --plot

Example output:
    Train: chr19 (120,432 positions, 1,345 genes)
    Test:  chr21+chr22 (38,291 positions, 566 genes)
    Features: 50 tabular columns

    === Classification Report (3-class: donor/acceptor/neither) ===
    PR-AUC (donor):    0.XX
    PR-AUC (acceptor): 0.XX
    Accuracy:          XX.X%

    === Top 20 Feature Importances ===
    1. donor_prob             0.XXX
    2. acceptor_prob          0.XXX
    ...
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ── Column categories ────────────────────────────────────────────────
# Mirror feature_schema.py leakage/metadata definitions.

LABEL_COL = "splice_type"

LEAKAGE_COLS = {
    "splice_type",
    "pred_type",
    "true_position",
    "predicted_position",
    "is_correct",
    "error_type",
}

METADATA_COLS = {
    "gene_id",
    "gene_name",
    "transcript_id",
    "gene_type",
    "chrom",
    "strand",
    "position",
    "absolute_position",
    "window_start",
    "window_end",
    "transcript_count",
    "gene_start",
    "gene_end",
}

# Non-numeric columns that can't be used as features
NON_NUMERIC_COLS = {
    "sequence",
}

EXCLUDE_COLS = LEAKAGE_COLS | METADATA_COLS | NON_NUMERIC_COLS | {LABEL_COL}

LABEL_ENCODING = {"donor": 0, "acceptor": 1, "neither": 2, "": 2}


# ── Data loading ─────────────────────────────────────────────────────


def load_analysis_sequences(
    input_dir: Path,
    chromosomes: list[str],
) -> pl.DataFrame:
    """Load analysis_sequences parquets for specified chromosomes."""
    frames = []
    for chrom in chromosomes:
        path = input_dir / f"analysis_sequences_{chrom}.parquet"
        if not path.exists():
            # Try TSV fallback
            path = input_dir / f"analysis_sequences_{chrom}.tsv"
            if not path.exists():
                log.warning("No data for %s at %s", chrom, input_dir)
                continue
            frames.append(pl.read_csv(path, separator="\t"))
        else:
            frames.append(pl.read_parquet(path))
        log.info("Loaded %s: %d positions", chrom, frames[-1].height)

    if not frames:
        raise FileNotFoundError(
            f"No analysis_sequences found in {input_dir} "
            f"for chromosomes: {chromosomes}"
        )

    # Align schemas: use column intersection (different chromosomes may have
    # been generated with different modality configs)
    if len(frames) > 1:
        common_cols = set(frames[0].columns)
        for f in frames[1:]:
            common_cols &= set(f.columns)
        max_cols = max(len(f.columns) for f in frames)
        if len(common_cols) < max_cols:
            dropped = max_cols - len(common_cols)
            log.info(
                "Schema alignment: keeping %d common columns (dropped %d)",
                len(common_cols),
                dropped,
            )
            # Preserve column order from first frame
            ordered_cols = [c for c in frames[0].columns if c in common_cols]
            frames = [f.select(ordered_cols) for f in frames]

    return pl.concat(frames)


def prepare_features_and_labels(
    df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract feature matrix X and label vector y from DataFrame.

    Returns
    -------
    X : np.ndarray of shape [n_samples, n_features]
    y : np.ndarray of shape [n_samples] with values in {0, 1, 2}
    feature_names : list of str
    """
    # Select numeric feature columns (exclude leakage, metadata, labels)
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and df[c].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int8, pl.UInt32)
    ]

    if not feature_cols:
        raise ValueError(
            f"No valid feature columns found. Available: {df.columns}"
        )

    log.info("Using %d feature columns", len(feature_cols))

    X = df.select(feature_cols).to_numpy().astype(np.float32)

    # Encode labels
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in data")

    labels = df[LABEL_COL].to_list()
    y = np.array([LABEL_ENCODING.get(str(l).lower(), 2) for l in labels])

    return X, y, feature_cols


# ── Training ─────────────────────────────────────────────────────────


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    early_stopping_rounds: int = 20,
) -> "xgboost.XGBClassifier":
    """Train XGBoost classifier with early stopping on test set."""
    import xgboost as xgb

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
        tree_method="hist",  # fast histogram-based
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    log.info(
        "Best iteration: %d / %d",
        model.best_iteration,
        n_estimators,
    )

    return model


LABEL_DECODING = {0: "donor", 1: "acceptor", 2: "neither"}


# ── Evaluation ───────────────────────────────────────────────────────


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

    # Classification report
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
        "accuracy": accuracy,
        "pr_aucs": pr_aucs,
        "confusion_matrix": cm,
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
        description="XGBoost Meta-Layer Baseline",
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
        default=["chr19"],
        help="Training chromosomes (default: chr19)",
    )
    parser.add_argument(
        "--test-chroms",
        nargs="+",
        default=["chr21", "chr22"],
        help="Test chromosomes (default: chr21 chr22)",
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

    # ── Resolve input directory ───────────────────────────────────────
    if args.input_dir is not None:
        input_dir = args.input_dir
    else:
        from agentic_spliceai.splice_engine.resources import get_model_resources
        resources = get_model_resources("openspliceai")
        registry = resources.get_registry()
        input_dir = registry.get_base_model_eval_dir("openspliceai") / "precomputed" / ".." / "analysis_sequences"
        input_dir = input_dir.resolve()
        if not input_dir.exists():
            # Try sibling of precomputed
            alt = registry.get_base_model_eval_dir("openspliceai") / "analysis_sequences"
            if alt.exists():
                input_dir = alt
            else:
                print(f"Cannot find analysis_sequences directory.")
                print(f"Tried: {input_dir}")
                print(f"Run feature engineering first:")
                print(f"  python examples/features/06_multimodal_genome_workflow.py "
                      f"--chromosomes {' '.join(args.train_chroms + args.test_chroms)}")
                return 1

    print("=" * 70)
    print("XGBoost Meta-Layer Baseline")
    print("=" * 70)
    print(f"  Input:     {input_dir}")
    print(f"  Train:     {', '.join(args.train_chroms)}")
    print(f"  Test:      {', '.join(args.test_chroms)}")

    # ── Load data ─────────────────────────────────────────────────────
    t0 = time.time()

    print(f"\n  Loading training data...")
    df_train = load_analysis_sequences(input_dir, args.train_chroms)
    print(f"    {df_train.height:,} positions, {df_train.width} columns")

    print(f"  Loading test data...")
    df_test = load_analysis_sequences(input_dir, args.test_chroms)
    print(f"    {df_test.height:,} positions, {df_test.width} columns")

    # ── Prepare features ──────────────────────────────────────────────
    X_train, y_train, feature_names = prepare_features_and_labels(df_train)
    X_test, y_test, test_feature_names = prepare_features_and_labels(df_test)

    # Align features: use intersection of train and test feature columns
    if set(feature_names) != set(test_feature_names):
        common = [f for f in feature_names if f in set(test_feature_names)]
        train_idx = [feature_names.index(f) for f in common]
        test_idx = [test_feature_names.index(f) for f in common]
        X_train = X_train[:, train_idx]
        X_test = X_test[:, test_idx]
        log.info(
            "Feature alignment: %d train, %d test → %d common",
            len(feature_names), len(test_feature_names), len(common),
        )
        feature_names = common

    # Handle NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    n_train_genes = df_train["gene_id"].n_unique() if "gene_id" in df_train.columns else "?"
    n_test_genes = df_test["gene_id"].n_unique() if "gene_id" in df_test.columns else "?"

    print(f"\n  Features:  {len(feature_names)}")
    print(f"  Train:     {X_train.shape[0]:,} positions, {n_train_genes} genes")
    print(f"  Test:      {X_test.shape[0]:,} positions, {n_test_genes} genes")

    # Free DataFrames
    del df_train, df_test

    # ── Train ─────────────────────────────────────────────────────────
    print(f"\n  Training XGBoost (max {args.n_estimators} rounds, depth={args.max_depth})...")
    model = train_xgboost(
        X_train, y_train, X_test, y_test,
        feature_names,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # ── Evaluate ──────────────────────────────────────────────────────
    results = evaluate(model, X_test, y_test, feature_names)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # ── Save outputs ──────────────────────────────────────────────────
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("output/meta_layer_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "xgb_baseline.ubj"
    model.get_booster().save_model(str(model_path))
    print(f"\n  Model saved: {model_path}")

    # Save feature importance
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
