#!/usr/bin/env python3
"""Modality ablation analysis for the XGBoost meta-layer baseline.

Measures the contribution of each feature modality by training models
with one modality removed at a time (leave-one-out ablation), and
compares against the full-stack and base-only baselines.

Outputs:
  - Per-modality importance (grouped feature importance)
  - Ablation table (accuracy/PR-AUC drop when each modality is removed)
  - Feature importance plots (top-N features, grouped by modality)
  - PR curve comparison (base-only vs full-stack)

Usage:
    python 03_modality_ablation.py \\
        --input-dir data/mane/GRCh38/openspliceai_eval/analysis_sequences \\
        --train-chroms chr19 chr20 --test-chroms chr21 chr22 \\
        --output-dir output/modality_ablation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ── Modality → column prefix mapping ────────────────────────────────
# Each modality produces columns with distinctive prefixes or names.
MODALITY_COLUMNS: Dict[str, List[str]] = {
    "base_scores": [
        "donor_score", "acceptor_score", "neither_score",
        "context_score_m2", "context_score_m1", "context_score_p1", "context_score_p2",
        "relative_donor_probability", "splice_probability", "donor_acceptor_diff",
        "splice_neither_diff", "donor_acceptor_logodds", "splice_neither_logodds",
        "probability_entropy", "context_neighbor_mean", "context_asymmetry", "context_max",
        "donor_diff_m1", "donor_diff_m2", "donor_diff_p1", "donor_diff_p2",
        "donor_surge_ratio", "donor_is_local_peak", "donor_weighted_context",
        "donor_peak_height_ratio", "donor_second_derivative", "donor_signal_strength",
        "donor_context_diff_ratio",
        "acceptor_diff_m1", "acceptor_diff_m2", "acceptor_diff_p1", "acceptor_diff_p2",
        "acceptor_surge_ratio", "acceptor_is_local_peak", "acceptor_weighted_context",
        "acceptor_peak_height_ratio", "acceptor_second_derivative", "acceptor_signal_strength",
        "acceptor_context_diff_ratio",
        "donor_acceptor_peak_ratio", "type_signal_difference",
        "score_difference_ratio", "signal_strength_ratio",
    ],
    "genomic": [
        "relative_gene_position", "distance_to_gene_start",
        "distance_to_gene_end", "gc_content",
    ],
    "conservation": [
        "phylop_score", "phylop_context_mean", "phylop_context_max", "phylop_context_std",
        "phastcons_score", "phastcons_context_mean", "phastcons_context_max",
        "phastcons_context_std", "conservation_contrast",
    ],
    "epigenetic": [
        "h3k36me3_max_across_tissues", "h3k36me3_mean_across_tissues",
        "h3k36me3_tissue_breadth", "h3k36me3_variance",
        "h3k36me3_context_mean", "h3k36me3_exon_intron_ratio",
        "h3k4me3_max_across_tissues", "h3k4me3_mean_across_tissues",
        "h3k4me3_tissue_breadth", "h3k4me3_variance",
        "h3k4me3_context_mean", "h3k4me3_exon_intron_ratio",
    ],
    "junction": [
        "junction_log1p", "junction_has_support", "junction_n_partners",
        "junction_max_reads", "junction_entropy", "junction_is_annotated",
        "junction_tissue_breadth", "junction_tissue_max", "junction_tissue_mean",
        "junction_tissue_variance", "junction_psi", "junction_psi_variance",
    ],
}

# Columns that are never features
EXCLUDE_COLS = {
    "gene_id", "gene_name", "chrom", "position", "strand",
    "gene_start", "gene_end", "donor_prob", "acceptor_prob", "neither_prob",
    "splice_type", "transcript_id", "transcript_count",
    "sequence", "window_start", "window_end",
}

LABEL_COL = "splice_type"
CLASSES = ["donor", "acceptor", "neither"]


def load_data(
    input_dir: Path,
    chroms: List[str],
) -> pl.DataFrame:
    """Load and concatenate analysis_sequences parquet files."""
    dfs = []
    for chrom in chroms:
        path = input_dir / f"analysis_sequences_{chrom}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pl.read_parquet(path)
        dfs.append(df)
        logger.info("Loaded %s: %d positions", chrom, df.height)
    return pl.concat(dfs)


def get_feature_columns(df: pl.DataFrame) -> List[str]:
    """Get all valid feature columns (exclude metadata, labels, strings)."""
    feature_cols = []
    for col in df.columns:
        if col in EXCLUDE_COLS:
            continue
        if df[col].dtype in (pl.Utf8, pl.String):
            continue
        feature_cols.append(col)
    return sorted(feature_cols)


def feature_to_modality(col: str) -> str:
    """Map a feature column to its modality."""
    for modality, cols in MODALITY_COLUMNS.items():
        if col in cols:
            return modality
    return "other"


def prepare_xy(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract X, y arrays from DataFrame."""
    X = df.select(feature_cols).to_numpy().astype(np.float32)
    # Handle NaN in features
    X = np.nan_to_num(X, nan=0.0)

    y_raw = df[LABEL_COL].to_list()
    label_map = {"donor": 0, "acceptor": 1}
    y = np.array([label_map.get(v, 2) for v in y_raw])
    return X, y


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    max_rounds: int = 500,
) -> Dict:
    """Train XGBoost and return evaluation metrics + feature importances."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
        "seed": 42,
        "verbosity": 0,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )

    # Predictions
    y_proba = model.predict(dtest)
    y_pred = y_proba.argmax(axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)

    # PR-AUC per class
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    pr_aucs = {}
    for i, cls in enumerate(CLASSES):
        pr_aucs[cls] = average_precision_score(y_test_bin[:, i], y_proba[:, i])

    # Feature importance (gain)
    importance = model.get_score(importance_type="gain")

    # Also collect weight and cover for multi-method comparison
    importance_weight = model.get_score(importance_type="weight")
    importance_cover = model.get_score(importance_type="cover")

    return {
        "accuracy": acc,
        "pr_auc": pr_aucs,
        "macro_pr_auc": np.mean(list(pr_aucs.values())),
        "feature_importance": importance,
        "importance_weight": importance_weight,
        "importance_cover": importance_cover,
        "best_iteration": model.best_iteration,
        "y_proba": y_proba,
        "y_pred": y_pred,
        "model": model,
    }


def run_ablation(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    all_features: List[str],
    output_dir: Path,
) -> Dict:
    """Run leave-one-out modality ablation."""
    results = {}

    # Full model
    logger.info("Training FULL model (%d features)...", len(all_features))
    X_train, y_train = prepare_xy(train_df, all_features)
    X_test, y_test = prepare_xy(test_df, all_features)
    full_result = train_and_evaluate(X_train, y_train, X_test, y_test, all_features)
    results["full_stack"] = {
        "accuracy": full_result["accuracy"],
        "pr_auc": full_result["pr_auc"],
        "macro_pr_auc": full_result["macro_pr_auc"],
        "n_features": len(all_features),
        "best_iteration": full_result["best_iteration"],
    }
    full_importance = full_result["feature_importance"]

    # Base-only model (base_scores + genomic only)
    base_features = [
        f for f in all_features
        if feature_to_modality(f) in ("base_scores", "genomic")
    ]
    logger.info("Training BASE-ONLY model (%d features)...", len(base_features))
    X_train_base, _ = prepare_xy(train_df, base_features)
    X_test_base, _ = prepare_xy(test_df, base_features)
    base_result = train_and_evaluate(
        X_train_base, y_train, X_test_base, y_test, base_features
    )
    results["base_only"] = {
        "accuracy": base_result["accuracy"],
        "pr_auc": base_result["pr_auc"],
        "macro_pr_auc": base_result["macro_pr_auc"],
        "n_features": len(base_features),
        "best_iteration": base_result["best_iteration"],
    }

    # Leave-one-out ablation for each modality
    for modality in MODALITY_COLUMNS:
        modality_cols = set(MODALITY_COLUMNS[modality])
        ablated_features = [f for f in all_features if f not in modality_cols]

        if len(ablated_features) == len(all_features):
            logger.info("Skipping %s (no columns in feature set)", modality)
            continue

        n_removed = len(all_features) - len(ablated_features)
        logger.info(
            "Training WITHOUT %s (%d features removed, %d remaining)...",
            modality, n_removed, len(ablated_features),
        )
        X_train_abl, _ = prepare_xy(train_df, ablated_features)
        X_test_abl, _ = prepare_xy(test_df, ablated_features)
        abl_result = train_and_evaluate(
            X_train_abl, y_train, X_test_abl, y_test, ablated_features
        )
        results[f"without_{modality}"] = {
            "accuracy": abl_result["accuracy"],
            "pr_auc": abl_result["pr_auc"],
            "macro_pr_auc": abl_result["macro_pr_auc"],
            "n_features": len(ablated_features),
            "n_removed": n_removed,
            "best_iteration": abl_result["best_iteration"],
        }

    return results, full_importance, full_result, base_result


def compute_modality_importance(
    importance: Dict[str, float],
    feature_cols: List[str],
) -> Dict[str, float]:
    """Aggregate feature importance by modality."""
    modality_totals: Dict[str, float] = {}
    total = sum(importance.values()) if importance else 1.0

    for col in feature_cols:
        mod = feature_to_modality(col)
        gain = importance.get(col, 0.0)
        modality_totals[mod] = modality_totals.get(mod, 0.0) + gain

    # Normalize to fractions
    return {k: v / total for k, v in sorted(modality_totals.items(), key=lambda x: -x[1])}


def compute_shap_importance(
    model: xgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    n_sample: int = 10000,
) -> Dict[str, float]:
    """Compute Tree SHAP values using XGBoost's native pred_contribs.

    XGBoost computes exact Tree SHAP without the `shap` package.
    For multiclass, returns shape (n_samples, n_features+1, n_classes).

    Parameters
    ----------
    model : xgb.Booster
        Trained XGBoost model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels (for per-class analysis).
    feature_names : list of str
        Feature names matching X_test columns.
    output_dir : Path
        Directory for SHAP output files.
    n_sample : int
        Subsample size for SHAP computation (full test set may be slow).

    Returns
    -------
    dict
        Feature name → mean |SHAP| across all classes and samples.
    """
    logger.info("Computing Tree SHAP values (n=%d)...", min(n_sample, len(y_test)))

    # Subsample for speed
    if len(y_test) > n_sample:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(y_test), n_sample, replace=False)
        X_sub = X_test[idx]
        y_sub = y_test[idx]
    else:
        X_sub = X_test
        y_sub = y_test

    dmat = xgb.DMatrix(X_sub, feature_names=feature_names)

    # pred_contribs returns shape (n_samples, n_features + 1) per class
    # For multiclass with 3 classes: (n_samples, n_features + 1, 3)
    # The last column (+1) is the bias term
    shap_values = model.predict(dmat, pred_contribs=True)
    # shape: (n_samples, n_features + 1, n_classes) for multiclass
    # or (n_samples, n_features + 1) for binary

    # XGBoost multiclass pred_contribs shape: (n_samples, n_classes, n_features+1)
    # Last column per class is the bias term.
    n_feat = len(feature_names)
    if shap_values.ndim == 2:
        # Binary — shape (n_samples, n_features+1)
        shap_no_bias = shap_values[:, :n_feat]
        mean_abs_shap = np.abs(shap_no_bias).mean(axis=0)
    elif shap_values.ndim == 3:
        # Multiclass — shape (n_samples, n_classes, n_features+1)
        shap_no_bias = shap_values[:, :, :n_feat]  # (n, classes, features)
        # Mean |SHAP| across samples and classes
        mean_abs_shap = np.abs(shap_no_bias).mean(axis=(0, 1))
    else:
        logger.warning("Unexpected SHAP shape: %s", shap_values.shape)
        return {}

    # Build importance dict
    shap_importance = {
        feature_names[i]: float(mean_abs_shap[i])
        for i in range(n_feat)
    }

    # Save overall SHAP importance
    shap_csv_path = output_dir / "shap_importance.csv"
    sorted_shap = sorted(shap_importance.items(), key=lambda x: -x[1])
    with open(shap_csv_path, "w") as f:
        f.write("feature,modality,mean_abs_shap\n")
        for feat, val in sorted_shap:
            f.write(f"{feat},{feature_to_modality(feat)},{val:.6f}\n")
    logger.info("Saved SHAP importance to %s", shap_csv_path)

    # Per-class SHAP for donor/acceptor/neither
    if shap_values.ndim == 3:
        for cls_idx, cls_name in enumerate(CLASSES):
            cls_shap = np.abs(shap_no_bias[:, cls_idx, :]).mean(axis=0)
            cls_path = output_dir / f"shap_importance_{cls_name}.csv"
            cls_sorted = sorted(
                zip(feature_names, cls_shap), key=lambda x: -x[1]
            )
            with open(cls_path, "w") as f:
                f.write("feature,modality,mean_abs_shap\n")
                for feat, val in cls_sorted:
                    f.write(f"{feat},{feature_to_modality(feat)},{val:.6f}\n")

        # SHAP for misclassified positions (where model is wrong)
        pred = model.predict(dmat).argmax(axis=1)
        fn_mask = (y_sub != pred)
        if fn_mask.sum() > 10:
            fn_shap = np.abs(shap_no_bias[fn_mask]).mean(axis=(0, 1))
            fn_path = output_dir / "shap_importance_misclassified.csv"
            fn_sorted = sorted(
                zip(feature_names, fn_shap), key=lambda x: -x[1]
            )
            with open(fn_path, "w") as f:
                f.write("feature,modality,mean_abs_shap\n")
                for feat, val in fn_sorted:
                    f.write(f"{feat},{feature_to_modality(feat)},{val:.6f}\n")
            logger.info(
                "Saved SHAP for %d misclassified positions to %s",
                fn_mask.sum(), fn_path,
            )

    return shap_importance


def plot_shap(
    shap_importance: Dict[str, float],
    gain_importance: Dict[str, float],
    feature_names: List[str],
    output_dir: Path,
) -> None:
    """Plot SHAP-based feature importance and compare with XGBoost gain."""
    import matplotlib.pyplot as plt

    colors = {
        "base_scores": "#3572a5",
        "junction": "#e74c3c",
        "conservation": "#27ae60",
        "epigenetic": "#f39c12",
        "genomic": "#9b59b6",
        "other": "#95a5a6",
    }

    # ── 1. SHAP importance (top 30) ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 30
    sorted_shap = sorted(shap_importance.items(), key=lambda x: -x[1])[:top_n]
    feat_names = [f[0] for f in sorted_shap]
    feat_vals = [f[1] for f in sorted_shap]
    feat_colors = [colors.get(feature_to_modality(f), "#95a5a6") for f in feat_names]

    ax.barh(range(len(feat_names)), feat_vals, color=feat_colors,
            edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features — Tree SHAP Importance")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=m)
                       for m, c in colors.items()
                       if any(feature_to_modality(f) == m for f in feat_names)]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "shap_feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved shap_feature_importance.png")

    # ── 2. SHAP vs Gain ranking comparison ───────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get top 30 by either method
    shap_rank = {f: i for i, (f, _) in enumerate(
        sorted(shap_importance.items(), key=lambda x: -x[1])
    )}
    gain_rank = {f: i for i, (f, _) in enumerate(
        sorted(gain_importance.items(), key=lambda x: -x[1])
    )}

    common = set(shap_rank.keys()) & set(gain_rank.keys())
    top_features = sorted(common, key=lambda f: shap_rank[f])[:40]

    x_vals = [gain_rank.get(f, len(gain_rank)) for f in top_features]
    y_vals = [shap_rank.get(f, len(shap_rank)) for f in top_features]
    point_colors = [colors.get(feature_to_modality(f), "#95a5a6") for f in top_features]

    ax.scatter(x_vals, y_vals, c=point_colors, s=60, edgecolors="black", alpha=0.8, zorder=3)

    # Label top 15 by SHAP
    for f, x, y in zip(top_features[:15], x_vals[:15], y_vals[:15]):
        short_name = f[:25] + "..." if len(f) > 25 else f
        ax.annotate(short_name, (x, y), fontsize=6.5, xytext=(5, 3),
                    textcoords="offset points")

    max_rank = max(max(x_vals), max(y_vals)) + 2
    ax.plot([0, max_rank], [0, max_rank], "k--", alpha=0.3, label="Perfect agreement")
    ax.set_xlabel("XGBoost Gain Rank")
    ax.set_ylabel("SHAP Rank")
    ax.set_title("Feature Ranking: SHAP vs XGBoost Gain")
    ax.set_xlim(-1, min(max_rank, 50))
    ax.set_ylim(-1, min(max_rank, 50))
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "shap_vs_gain_ranking.png", dpi=150)
    plt.close()
    logger.info("Saved shap_vs_gain_ranking.png")

    # ── 3. Modality importance: SHAP vs Gain ─────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # SHAP-based modality importance
    shap_mod = {}
    total_shap = sum(shap_importance.values())
    for feat, val in shap_importance.items():
        mod = feature_to_modality(feat)
        shap_mod[mod] = shap_mod.get(mod, 0.0) + val / total_shap

    for ax_idx, (title, mod_dict) in enumerate([
        ("XGBoost Gain", compute_modality_importance(gain_importance, list(gain_importance.keys()))),
        ("Tree SHAP", shap_mod),
    ]):
        ax = axes[ax_idx]
        mods = sorted(mod_dict.keys(), key=lambda m: -mod_dict[m])
        vals = [mod_dict[m] * 100 for m in mods]
        bar_colors = [colors.get(m, "#95a5a6") for m in mods]
        bars = ax.barh(mods, vals, color=bar_colors, edgecolor="black", alpha=0.85)
        ax.set_xlabel("% of total importance")
        ax.set_title(f"Modality Importance — {title}")
        ax.invert_yaxis()
        for bar, val in zip(bars, vals):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "modality_importance_shap_vs_gain.png", dpi=150)
    plt.close()
    logger.info("Saved modality_importance_shap_vs_gain.png")


def plot_results(
    ablation_results: Dict,
    modality_importance: Dict[str, float],
    feature_importance: Dict[str, float],
    full_result: Dict,
    base_result: Dict,
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate all plots."""
    import matplotlib.pyplot as plt

    # ── 1. Modality importance bar chart ─────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    mods = list(modality_importance.keys())
    vals = [modality_importance[m] * 100 for m in mods]
    colors = {
        "base_scores": "#3572a5",
        "junction": "#e74c3c",
        "conservation": "#27ae60",
        "epigenetic": "#f39c12",
        "genomic": "#9b59b6",
        "other": "#95a5a6",
    }
    bar_colors = [colors.get(m, "#95a5a6") for m in mods]
    bars = ax.barh(mods, vals, color=bar_colors, edgecolor="black", alpha=0.85)
    ax.set_xlabel("Feature Importance (% of total gain)")
    ax.set_title("Modality Importance — XGBoost Meta-Layer")
    ax.invert_yaxis()
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "modality_importance.png", dpi=150)
    plt.close()
    logger.info("Saved modality_importance.png")

    # ── 2. Ablation table as bar chart ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    models = []
    accs = []
    for name, res in sorted(ablation_results.items()):
        models.append(name.replace("without_", "w/o "))
        accs.append(res["accuracy"] * 100)

    y_pos = range(len(models))
    ax = axes[0]
    bars = ax.barh(y_pos, accs, color=["#3572a5" if "full" in m else
                                        "#27ae60" if "base" in m else
                                        "#e74c3c" for m in models],
                   edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Ablation: Accuracy")
    ax.set_xlim(min(accs) - 0.5, max(accs) + 0.3)
    ax.invert_yaxis()
    for bar, val in zip(bars, accs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}%", va="center", fontsize=9)

    # Macro PR-AUC comparison
    macro_aucs = [ablation_results[name.replace("w/o ", "without_")
                                    if "w/o " in name else name]["macro_pr_auc"]
                  for name in [m.replace("w/o ", "without_") if "w/o " in m else m
                               for m in models]]
    ax = axes[1]
    bars = ax.barh(y_pos, macro_aucs, color=["#3572a5" if "full" in m else
                                              "#27ae60" if "base" in m else
                                              "#e74c3c" for m in models],
                   edgecolor="black", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel("Macro PR-AUC")
    ax.set_title("Ablation: Macro PR-AUC")
    ax.set_xlim(min(macro_aucs) - 0.005, max(macro_aucs) + 0.003)
    ax.invert_yaxis()
    for bar, val in zip(bars, macro_aucs):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "ablation_comparison.png", dpi=150)
    plt.close()
    logger.info("Saved ablation_comparison.png")

    # ── 3. Top-N feature importance (colored by modality) ────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = 30
    sorted_feats = sorted(feature_importance.items(), key=lambda x: -x[1])[:top_n]
    feat_names = [f[0] for f in sorted_feats]
    feat_vals = [f[1] for f in sorted_feats]
    feat_colors = [colors.get(feature_to_modality(f), "#95a5a6") for f in feat_names]

    ax.barh(range(len(feat_names)), feat_vals, color=feat_colors,
            edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("XGBoost Gain")
    ax.set_title(f"Top {top_n} Features — Colored by Modality")
    ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, edgecolor="black", label=m)
                       for m, c in colors.items() if m in modality_importance]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_by_modality.png", dpi=150)
    plt.close()
    logger.info("Saved feature_importance_by_modality.png")

    # ── 4. PR curves: base vs full-stack ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    for i, (cls, ax) in enumerate(zip(["donor", "acceptor"], axes)):
        # Full model
        precision_f, recall_f, _ = precision_recall_curve(
            y_test_bin[:, i], full_result["y_proba"][:, i]
        )
        ap_f = full_result["pr_auc"][cls]

        # Base model
        precision_b, recall_b, _ = precision_recall_curve(
            y_test_bin[:, i], base_result["y_proba"][:, i]
        )
        ap_b = base_result["pr_auc"][cls]

        ax.plot(recall_f, precision_f, color="#e74c3c", lw=2,
                label=f"Full-stack (AP={ap_f:.4f})")
        ax.plot(recall_b, precision_b, color="#3572a5", lw=2, linestyle="--",
                label=f"Base-only (AP={ap_b:.4f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR Curve — {cls.capitalize()}")
        ax.legend(loc="lower left")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / "pr_base_vs_fullstack.png", dpi=150)
    plt.close()
    logger.info("Saved pr_base_vs_fullstack.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("data/mane/GRCh38/openspliceai_eval/analysis_sequences"),
    )
    parser.add_argument("--train-chroms", nargs="+", default=["chr19", "chr20"])
    parser.add_argument("--test-chroms", nargs="+", default=["chr21", "chr22"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/modality_ablation"),
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Load data
    train_df = load_data(args.input_dir, args.train_chroms)
    test_df = load_data(args.input_dir, args.test_chroms)

    all_features = get_feature_columns(train_df)
    logger.info("Total feature columns: %d", len(all_features))

    # Run ablation
    ablation_results, full_importance, full_result, base_result = run_ablation(
        train_df, test_df, all_features, args.output_dir
    )

    # Compute modality-level importance
    modality_importance = compute_modality_importance(full_importance, all_features)

    # Prepare test arrays for SHAP and plots
    X_test, y_test = prepare_xy(test_df, all_features)

    # Print results
    print("\n" + "=" * 70)
    print("Modality Ablation Analysis")
    print("=" * 70)

    print("\n  Modality Importance (% of total XGBoost gain):")
    for mod, frac in modality_importance.items():
        print(f"    {mod:20s}  {frac*100:6.1f}%")

    print("\n  Ablation Results:")
    print(f"    {'Model':<25s} {'Acc':>8s} {'PR-AUC(D)':>10s} {'PR-AUC(A)':>10s} {'Macro':>8s} {'#Feat':>6s}")
    print("    " + "-" * 68)
    full = ablation_results["full_stack"]
    print(f"    {'FULL STACK':<25s} {full['accuracy']*100:>7.2f}% "
          f"{full['pr_auc']['donor']:>10.4f} {full['pr_auc']['acceptor']:>10.4f} "
          f"{full['macro_pr_auc']:>8.4f} {full['n_features']:>6d}")
    base = ablation_results["base_only"]
    print(f"    {'BASE ONLY':<25s} {base['accuracy']*100:>7.2f}% "
          f"{base['pr_auc']['donor']:>10.4f} {base['pr_auc']['acceptor']:>10.4f} "
          f"{base['macro_pr_auc']:>8.4f} {base['n_features']:>6d}")
    for name in sorted(ablation_results):
        if name.startswith("without_"):
            res = ablation_results[name]
            delta_acc = (res["accuracy"] - full["accuracy"]) * 100
            label = name.replace("without_", "w/o ")
            print(f"    {label:<25s} {res['accuracy']*100:>7.2f}% "
                  f"{res['pr_auc']['donor']:>10.4f} {res['pr_auc']['acceptor']:>10.4f} "
                  f"{res['macro_pr_auc']:>8.4f} {res['n_features']:>6d}  "
                  f"(Δacc={delta_acc:+.2f}%)")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 70)

    # Save results
    results_path = args.output_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)
    logger.info("Saved %s", results_path)

    # Save modality importance
    mod_imp_path = args.output_dir / "modality_importance.json"
    with open(mod_imp_path, "w") as f:
        json.dump(modality_importance, f, indent=2)

    # Save feature importance
    feat_imp_path = args.output_dir / "feature_importance.csv"
    sorted_feats = sorted(full_importance.items(), key=lambda x: -x[1])
    with open(feat_imp_path, "w") as f:
        f.write("feature,modality,gain\n")
        for feat, gain in sorted_feats:
            f.write(f"{feat},{feature_to_modality(feat)},{gain:.4f}\n")

    # SHAP analysis
    shap_importance = compute_shap_importance(
        full_result["model"], X_test, y_test, all_features, args.output_dir,
    )

    if shap_importance:
        shap_mod = {}
        total_shap = sum(shap_importance.values())
        for feat, val in shap_importance.items():
            mod = feature_to_modality(feat)
            shap_mod[mod] = shap_mod.get(mod, 0.0) + val / total_shap

        print("\n  SHAP-based Modality Importance:")
        for mod in sorted(shap_mod, key=lambda m: -shap_mod[m]):
            print(f"    {mod:20s}  {shap_mod[mod]*100:6.1f}%")

        # Compare top features: SHAP vs Gain
        shap_top10 = sorted(shap_importance.items(), key=lambda x: -x[1])[:10]
        gain_top10 = sorted(full_importance.items(), key=lambda x: -x[1])[:10]
        print("\n  Top 10 Features — SHAP vs Gain:")
        print(f"    {'Rank':>4s}  {'SHAP':^30s}  {'Gain':^30s}")
        for i in range(10):
            sf, sv = shap_top10[i] if i < len(shap_top10) else ("", 0)
            gf, gv = gain_top10[i] if i < len(gain_top10) else ("", 0)
            print(f"    {i+1:>4d}  {sf:<30s}  {gf:<30s}")

    # Plots
    if not args.no_plot:
        plot_results(
            ablation_results, modality_importance, full_importance,
            full_result, base_result, y_test, args.output_dir,
        )
        if shap_importance:
            plot_shap(
                shap_importance, full_importance, all_features, args.output_dir,
            )

    print(f"\n  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
