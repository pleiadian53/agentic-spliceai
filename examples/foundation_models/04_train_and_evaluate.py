#!/usr/bin/env python3
"""
Example: Train and Evaluate ExonClassifier from Pre-Extracted Embeddings

Loads HDF5 embeddings and NPZ labels, trains an ExonClassifier with early
stopping, and evaluates with per-gene metrics. Includes a resource check
to verify training data fits in memory.

Usage:
    # From synthetic pipeline output
    python examples/foundation_models/04_train_and_evaluate.py \
        --embeddings /tmp/fm_demo/embeddings.h5 \
        --labels /tmp/fm_demo/embeddings.labels.npz \
        --output /tmp/fm_demo/model/

    # With specific architecture and parameters
    python examples/foundation_models/04_train_and_evaluate.py \
        --embeddings /tmp/fm_demo/embeddings.h5 \
        --labels /tmp/fm_demo/embeddings.labels.npz \
        --output /tmp/fm_demo/model/ \
        --architecture cnn \
        --window-size 1024 \
        --epochs 50

    # On remote GPU via SkyPilot
    sky launch foundation_models/configs/skypilot/train_classifier_a40.yaml
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ExonClassifier from embeddings.",
    )

    # Data
    parser.add_argument(
        "--embeddings", type=str, required=True,
        help="Path to HDF5 embeddings file",
    )
    parser.add_argument(
        "--labels", type=str, required=True,
        help="Path to .labels.npz file",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for checkpoint and metrics",
    )

    # Model
    parser.add_argument(
        "--architecture", type=str, default="mlp",
        choices=["linear", "mlp", "cnn", "lstm"],
        help="Classifier architecture (default: mlp)",
    )
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Hidden layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")

    # Windowing
    parser.add_argument("--window-size", type=int, default=1024, help="Window size (bp)")
    parser.add_argument(
        "--step-size", type=int, default=None,
        help="Step size (default: window_size / 2)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of genes for validation",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument(
        "--skip-resource-check", action="store_true",
        help="Skip memory feasibility check",
    )

    args = parser.parse_args()

    if args.step_size is None:
        args.step_size = args.window_size // 2

    import h5py
    import numpy as np
    import torch

    from foundation_models.evo2 import ExonClassifier
    from foundation_models.utils.chunking import window_embeddings
    from foundation_models.utils.resources import estimate_classifier_training

    t0 = time.time()

    # ------------------------------------------------------------------
    # Load data and create windows
    # ------------------------------------------------------------------
    logger.info("Loading data...")
    emb_file = h5py.File(args.embeddings, "r")
    labels_data = np.load(args.labels)
    gene_ids = list(emb_file.keys())
    hidden_dim = int(emb_file.attrs.get("hidden_dim", 0))

    # Split genes: last val_fraction for validation
    n_val = max(1, int(len(gene_ids) * args.val_fraction))
    val_genes = set(gene_ids[-n_val:])

    train_windows = []
    val_windows = []

    for gene_id in gene_ids:
        if gene_id not in labels_data:
            logger.warning("No labels for %s, skipping", gene_id)
            continue

        emb = emb_file[gene_id][:]
        lbl = labels_data[gene_id]
        min_len = min(len(emb), len(lbl))

        windows = window_embeddings(
            embeddings=emb[:min_len],
            labels=lbl[:min_len],
            gene_id=gene_id,
            window_size=args.window_size,
            step_size=args.step_size,
        )

        if gene_id in val_genes:
            val_windows.extend(windows)
        else:
            train_windows.extend(windows)

    emb_file.close()

    logger.info(
        "Windows: %d train, %d val (from %d genes)",
        len(train_windows), len(val_windows), len(gene_ids),
    )

    if not train_windows:
        logger.error("No training windows. Check data and window_size.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resource check
    # ------------------------------------------------------------------
    total_windows = len(train_windows) + len(val_windows)

    if not args.skip_resource_check:
        result = estimate_classifier_training(
            n_windows=total_windows,
            window_size=args.window_size,
            hidden_dim=hidden_dim or args.hidden_dim,
        )
        if not result["feasible"]:
            print()
            print("RESOURCE CHECK FAILED")
            for note in result["notes"]:
                print(f"  {note}")
            print()
            print("Use --skip-resource-check to force, or reduce data size.")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Stack tensors
    # ------------------------------------------------------------------
    train_emb = torch.tensor(np.stack([w[0] for w in train_windows]), dtype=torch.float32)
    train_lbl = torch.tensor(np.stack([w[1] for w in train_windows]), dtype=torch.float32)
    val_emb = torch.tensor(np.stack([w[0] for w in val_windows]), dtype=torch.float32)
    val_lbl = torch.tensor(np.stack([w[1] for w in val_windows]), dtype=torch.float32)

    input_dim = train_emb.shape[-1]

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"Training ExonClassifier ({args.architecture})")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {args.hidden_dim}")
    print(f"  Window size:  {args.window_size}")
    print(f"  Windows:      {len(train_windows)} train, {len(val_windows)} val")
    print(f"  Device:       {device}")
    print("=" * 60)
    print()

    classifier = ExonClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        architecture=args.architecture,
        dropout=args.dropout,
    )

    history = classifier.fit(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        verbose=True,
        checkpoint_dir=str(output_dir),
        patience=args.patience,
        lr_schedule=True,
    )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    print()

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    probs = classifier.predict(val_emb).flatten()
    labels_flat = val_lbl.numpy().flatten()
    preds = (probs >= 0.5).astype(int)

    try:
        auroc = float(roc_auc_score(labels_flat, probs))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(labels_flat, probs))
    except ValueError:
        auprc = float("nan")

    overall = {
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
        "accuracy": round(float(accuracy_score(labels_flat, preds)), 4),
        "f1": round(float(f1_score(labels_flat, preds, zero_division=0)), 4),
        "precision": round(float(precision_score(labels_flat, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(labels_flat, preds, zero_division=0)), 4),
    }

    elapsed = time.time() - t0

    print("Overall Metrics:")
    print(f"  AUROC:     {overall['auroc']:.4f}")
    print(f"  AUPRC:     {overall['auprc']:.4f}")
    print(f"  Accuracy:  {overall['accuracy']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print()

    training_info = {
        "best_epoch": history["best_epoch"] + 1,
        "best_val_auroc": round(history["best_val_auroc"], 4),
        "stopped_early": history["stopped_early"],
        "architecture": args.architecture,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "window_size": args.window_size,
        "n_train_windows": len(train_windows),
        "n_val_windows": len(val_windows),
    }

    print(f"Training:")
    print(f"  Best epoch:     {training_info['best_epoch']}")
    print(f"  Best val AUROC: {training_info['best_val_auroc']:.4f}")
    print(f"  Early stopped:  {training_info['stopped_early']}")
    print(f"  Time:           {elapsed:.1f}s")
    print()

    # Save results
    results = {
        "overall": overall,
        "training": training_info,
    }
    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Artifacts:")
    print(f"  Checkpoint: {output_dir / 'best_model.pt'}")
    print(f"  Metrics:    {metrics_path}")
    print()


if __name__ == "__main__":
    main()
