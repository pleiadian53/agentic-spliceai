#!/usr/bin/env python3
"""
Example: Evaluate Trained Exon Classifier

Loads a checkpoint from 03_train_classifier.py, evaluates on embeddings,
and reports per-nucleotide and per-gene metrics.

Usage:
    python examples/04_evaluate_classifier.py \
        --checkpoint /tmp/ckpt/best_model.pt \
        --embeddings /tmp/emb/synth.h5 \
        --labels /tmp/emb/synth.labels.npz \
        --window-size 1024

    # Save results to JSON
    python examples/04_evaluate_classifier.py \
        --checkpoint /tmp/ckpt/best_model.pt \
        --embeddings /tmp/emb/synth.h5 \
        --labels /tmp/emb/synth.labels.npz \
        --window-size 1024 \
        --output results/eval.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np

# Add foundation_models to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ExonClassifier.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to best_model.pt checkpoint",
    )
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to HDF5 embeddings file",
    )
    parser.add_argument(
        "--labels", type=str, required=True,
        help="Path to .labels.npz file",
    )
    parser.add_argument(
        "--window-size", type=int, default=1024,
        help="Window size (must match training; default: 1024)",
    )
    parser.add_argument(
        "--step-size", type=int, default=None,
        help="Step size for windowing (default: window_size / 2)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Classification threshold (default: 0.5)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Optional JSON output path for metrics",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (default: cpu)",
    )

    args = parser.parse_args()

    if args.step_size is None:
        args.step_size = args.window_size // 2

    t0 = time.time()

    # Load model
    import torch
    from foundation_models.evo2 import ExonClassifier
    from foundation_models.utils.chunking import window_embeddings

    logger.info("Loading checkpoint: %s", args.checkpoint)
    classifier = ExonClassifier.load_checkpoint(args.checkpoint, device=args.device)
    logger.info("Model: %s", classifier)

    # Load data
    emb_file = h5py.File(args.embeddings, "r")
    labels_data = np.load(args.labels)
    gene_ids = list(emb_file.keys())
    logger.info("Evaluating on %d genes", len(gene_ids))

    # Collect predictions per gene
    all_probs = []
    all_labels = []
    per_gene = {}

    for gene_id in gene_ids:
        if gene_id not in labels_data:
            logger.warning("No labels for %s, skipping", gene_id)
            continue

        emb = emb_file[gene_id][:]
        lbl = labels_data[gene_id]
        min_len = min(len(emb), len(lbl))
        emb = emb[:min_len]
        lbl = lbl[:min_len]

        windows = window_embeddings(
            embeddings=emb,
            labels=lbl,
            gene_id=gene_id,
            window_size=args.window_size,
            step_size=args.step_size,
        )

        if not windows:
            logger.warning(
                "%s: no windows (seq_len=%d < window_size=%d)",
                gene_id, min_len, args.window_size,
            )
            continue

        # Predict
        win_emb = torch.tensor(
            np.stack([w[0] for w in windows]), dtype=torch.float32,
        ).to(args.device)
        win_lbl = np.concatenate([w[1] for w in windows])

        probs = classifier.predict(win_emb).flatten()
        all_probs.append(probs)
        all_labels.append(win_lbl)

        # Per-gene AUROC
        from sklearn.metrics import roc_auc_score
        try:
            gene_auroc = roc_auc_score(win_lbl, probs)
        except ValueError:
            gene_auroc = float("nan")

        per_gene[gene_id] = {
            "auroc": round(gene_auroc, 4),
            "n_windows": len(windows),
            "seq_len": min_len,
            "exon_fraction": round(float(lbl.mean()), 4),
        }

    emb_file.close()

    if not all_probs:
        logger.error("No predictions generated. Check data and window_size.")
        sys.exit(1)

    # Overall metrics
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    all_probs_flat = np.concatenate(all_probs)
    all_labels_flat = np.concatenate(all_labels)
    preds = (all_probs_flat >= args.threshold).astype(int)

    overall = {
        "auroc": round(float(roc_auc_score(all_labels_flat, all_probs_flat)), 4),
        "auprc": round(float(average_precision_score(all_labels_flat, all_probs_flat)), 4),
        "accuracy": round(float(accuracy_score(all_labels_flat, preds)), 4),
        "f1": round(float(f1_score(all_labels_flat, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(all_labels_flat, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(all_labels_flat, preds, zero_division=0)), 4),
        "threshold": args.threshold,
    }

    dataset_info = {
        "n_genes": len(per_gene),
        "n_positions": int(len(all_probs_flat)),
        "exon_fraction": round(float(all_labels_flat.mean()), 4),
        "window_size": args.window_size,
        "step_size": args.step_size,
    }

    model_info = {
        "architecture": classifier.architecture,
        "input_dim": classifier.input_dim,
        "hidden_dim": classifier.hidden_dim,
        "num_layers": classifier.num_layers,
        "checkpoint": str(args.checkpoint),
    }

    elapsed = time.time() - t0

    # Print report
    print()
    print("=" * 60)
    print("Exon Classifier Evaluation Report")
    print("=" * 60)
    print()

    print("Overall Metrics:")
    print(f"  AUROC:     {overall['auroc']:.4f}")
    print(f"  AUPRC:     {overall['auprc']:.4f}")
    print(f"  Accuracy:  {overall['accuracy']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  Threshold: {overall['threshold']}")
    print()

    print("Dataset:")
    print(f"  Genes:          {dataset_info['n_genes']}")
    print(f"  Total positions: {dataset_info['n_positions']:,}")
    print(f"  Exon fraction:  {dataset_info['exon_fraction']:.4f}")
    print()

    print("Model:")
    print(f"  Architecture: {model_info['architecture']}")
    print(f"  Input dim:    {model_info['input_dim']}")
    print(f"  Hidden dim:   {model_info['hidden_dim']}")
    print()

    print("Per-Gene AUROC:")
    sorted_genes = sorted(per_gene.items(), key=lambda x: x[1]["auroc"], reverse=True)
    for gene_id, metrics in sorted_genes:
        auroc_str = f"{metrics['auroc']:.4f}" if not np.isnan(metrics["auroc"]) else "  N/A "
        print(
            f"  {gene_id}: AUROC={auroc_str} "
            f"(windows={metrics['n_windows']}, "
            f"exon_frac={metrics['exon_fraction']:.2f})"
        )
    print()
    print(f"Evaluation time: {elapsed:.1f}s")
    print()

    # Save JSON if requested
    if args.output:
        results = {
            "overall": overall,
            "per_gene": per_gene,
            "dataset": dataset_info,
            "model": model_info,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved metrics to %s", output_path)


if __name__ == "__main__":
    main()
