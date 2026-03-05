#!/usr/bin/env python3
"""
Example: End-to-End Training Pipeline with Synthetic Data

Demonstrates the full workflow — generate synthetic embeddings, train an
ExonClassifier, and evaluate — without needing Evo2 or a GPU. Always works
on any hardware.

Usage:
    python examples/foundation_models/02_synthetic_training_pipeline.py \
        --output /tmp/fm_demo/

    # With custom parameters
    python examples/foundation_models/02_synthetic_training_pipeline.py \
        --output /tmp/fm_demo/ \
        --n-genes 10 \
        --architecture cnn \
        --epochs 30
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
        description="End-to-end training pipeline with synthetic data.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="/tmp/fm_demo/",
        help="Output directory (default: /tmp/fm_demo/)",
    )
    parser.add_argument(
        "--n-genes", type=int, default=5,
        help="Number of synthetic genes (default: 5)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Embedding dimension (default: 256; real Evo2 7B uses 2560)",
    )
    parser.add_argument(
        "--architecture", type=str, default="mlp",
        choices=["linear", "mlp", "cnn", "lstm"],
        help="Classifier architecture (default: mlp)",
    )
    parser.add_argument(
        "--window-size", type=int, default=512,
        help="Training window size (default: 512)",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs (default: 20)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    import numpy as np
    import torch

    from foundation_models.evo2 import ExonClassifier
    from foundation_models.utils import (
        save_synthetic_embeddings,
        window_embeddings,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic embeddings
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 1: Generate Synthetic Embeddings")
    print("=" * 60)
    print()

    emb_path = output_dir / "embeddings.h5"

    # Generate Gaussian-sampled embeddings in HDF5 format (matches
    # Evo2Embedder cache layout) so downstream training works unchanged.
    hdf5_path, labels_dict = save_synthetic_embeddings(
        output_path=emb_path,
        n_genes=args.n_genes,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    labels_path = hdf5_path.with_suffix(".labels.npz")
    logger.info("Saved embeddings: %s", hdf5_path)
    logger.info("Saved labels: %s", labels_path)

    # ------------------------------------------------------------------
    # Step 2: Window embeddings and train classifier
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Step 2: Train ExonClassifier ({args.architecture})")
    print("=" * 60)
    print()

    import h5py

    emb_file = h5py.File(hdf5_path, "r")
    labels_data = np.load(labels_path)
    gene_ids = list(emb_file.keys())

    # Split genes: last 20% for validation
    n_val = max(1, int(len(gene_ids) * 0.2))
    val_genes = set(gene_ids[-n_val:])
    step_size = args.window_size // 2

    train_windows = []
    val_windows = []

    for gene_id in gene_ids:
        emb = emb_file[gene_id][:]  # HDF5 random access: reads one gene without loading the entire file
        # The wording "lazy loading" is slightly loose since [:] 
        # eagerly loads the full dataset — it's more accurately 
        # random access (seek to one dataset without reading others)

        lbl = labels_data[gene_id]
        min_len = min(len(emb), len(lbl))

        # Slide fixed-size windows over embeddings + labels with 50% overlap
        windows = window_embeddings(
            embeddings=emb[:min_len],
            labels=lbl[:min_len],
            gene_id=gene_id,
            window_size=args.window_size,
            step_size=step_size,
        )

        if gene_id in val_genes:
            val_windows.extend(windows)
        else:
            train_windows.extend(windows)

    emb_file.close()

    logger.info(
        "Windows: %d train, %d val (size=%d, step=%d)",
        len(train_windows), len(val_windows), args.window_size, step_size,
    )

    if not train_windows:
        logger.error("No training windows generated. Increase n_genes or decrease window_size.")
        sys.exit(1)

    train_emb = torch.tensor(np.stack([w[0] for w in train_windows]), dtype=torch.float32)
    train_lbl = torch.tensor(np.stack([w[1] for w in train_windows]), dtype=torch.float32)
    val_emb = torch.tensor(np.stack([w[0] for w in val_windows]), dtype=torch.float32)
    val_lbl = torch.tensor(np.stack([w[1] for w in val_windows]), dtype=torch.float32)

    ckpt_dir = output_dir / "checkpoints"
    classifier = ExonClassifier(
        input_dim=args.hidden_dim,
        hidden_dim=min(256, args.hidden_dim),
        num_layers=2,
        architecture=args.architecture,
        dropout=0.1,
    )

    history = classifier.fit(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        epochs=args.epochs,
        batch_size=32,
        lr=1e-3,
        device="cpu",
        verbose=True,
        checkpoint_dir=str(ckpt_dir),
        patience=10,
        lr_schedule=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Step 3: Evaluate")
    print("=" * 60)
    print()

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        roc_auc_score,
    )

    # Predict on validation set
    probs = classifier.predict(val_emb).flatten()
    labels_flat = val_lbl.numpy().flatten()
    preds = (probs >= 0.5).astype(int)

    try:
        auroc = roc_auc_score(labels_flat, probs)
    except ValueError:
        auroc = float("nan")

    try:
        auprc = average_precision_score(labels_flat, probs)
    except ValueError:
        auprc = float("nan")

    metrics = {
        "auroc": round(float(auroc), 4),
        "auprc": round(float(auprc), 4),
        "accuracy": round(float(accuracy_score(labels_flat, preds)), 4),
        "f1": round(float(f1_score(labels_flat, preds, zero_division=0)), 4),
        "best_epoch": history["best_epoch"] + 1,
        "best_val_auroc": round(history["best_val_auroc"], 4),
        "stopped_early": history["stopped_early"],
    }

    elapsed = time.time() - t0

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print(f"  AUROC:         {metrics['auroc']:.4f}")
    print(f"  AUPRC:         {metrics['auprc']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f}")
    print(f"  F1:            {metrics['f1']:.4f}")
    print(f"  Best epoch:    {metrics['best_epoch']}")
    print(f"  Early stopped: {metrics['stopped_early']}")
    print()
    print(f"  Time:          {elapsed:.1f}s")
    print()
    print("Artifacts:")
    print(f"  Embeddings:  {hdf5_path}")
    print(f"  Labels:      {labels_path}")
    print(f"  Checkpoint:  {ckpt_dir / 'best_model.pt'}")
    print(f"  Metrics:     {metrics_path}")
    print()
    print("NOTE: AUROC ~0.5 is expected for synthetic (random) embeddings.")
    print("      With real Evo2 embeddings, expect AUROC > 0.9.")
    print()


if __name__ == "__main__":
    main()
