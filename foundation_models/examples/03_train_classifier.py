#!/usr/bin/env python3
"""
Example: Train Exon Classifier from Pre-Extracted Embeddings

Loads HDF5 embeddings and NPZ labels produced by 02_extract_embeddings.py,
generates fixed-size training windows, and trains an ExonClassifier.
Supports checkpoint saving, early stopping, and LR scheduling.

Usage:
    python examples/03_train_classifier.py \
        --embeddings /tmp/emb/synth.h5 \
        --labels /tmp/emb/synth.labels.npz \
        --output /tmp/ckpt/ \
        --architecture mlp \
        --window-size 1024 \
        --epochs 20 \
        --patience 10
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# Add foundation_models to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_data(
    embeddings_path: str,
    labels_path: str,
    window_size: int,
    step_size: int,
    min_exon_fraction: float,
    val_fraction: float,
) -> tuple:
    """Load embeddings + labels and split into train/val by gene."""
    from foundation_models.utils.chunking import window_embeddings

    emb_file = h5py.File(embeddings_path, "r")
    labels_data = np.load(labels_path)
    hidden_dim = int(emb_file.attrs.get("hidden_dim", 0))

    gene_ids = list(emb_file.keys())
    logger.info("Found %d genes (hidden_dim=%d)", len(gene_ids), hidden_dim)

    # Split genes into train/val
    n_val = max(1, int(len(gene_ids) * val_fraction))
    # Use deterministic split based on gene order
    val_genes = set(gene_ids[-n_val:])
    train_genes = set(gene_ids[:-n_val])

    logger.info("Train genes: %d, Val genes: %d", len(train_genes), len(val_genes))

    train_windows = []
    val_windows = []

    for gene_id in gene_ids:
        if gene_id not in labels_data:
            logger.warning("No labels for %s, skipping", gene_id)
            continue

        emb = emb_file[gene_id][:]
        lbl = labels_data[gene_id]

        # Ensure consistent lengths
        min_len = min(len(emb), len(lbl))
        emb = emb[:min_len]
        lbl = lbl[:min_len]

        windows = window_embeddings(
            embeddings=emb,
            labels=lbl,
            gene_id=gene_id,
            window_size=window_size,
            step_size=step_size,
            min_exon_fraction=min_exon_fraction,
        )

        if gene_id in val_genes:
            val_windows.extend(windows)
        else:
            train_windows.extend(windows)

    emb_file.close()

    logger.info(
        "Windows: %d train, %d val (window_size=%d, step=%d)",
        len(train_windows), len(val_windows), window_size, step_size,
    )

    if not train_windows:
        logger.error("No training windows generated. Check data and window_size.")
        sys.exit(1)

    # Stack into tensors
    train_emb = torch.tensor(
        np.stack([w[0] for w in train_windows]), dtype=torch.float32,
    )
    train_lbl = torch.tensor(
        np.stack([w[1] for w in train_windows]), dtype=torch.float32,
    )

    val_emb = None
    val_lbl = None
    if val_windows:
        val_emb = torch.tensor(
            np.stack([w[0] for w in val_windows]), dtype=torch.float32,
        )
        val_lbl = torch.tensor(
            np.stack([w[1] for w in val_windows]), dtype=torch.float32,
        )

    return train_emb, train_lbl, val_emb, val_lbl, hidden_dim


def main():
    parser = argparse.ArgumentParser(
        description="Train ExonClassifier from pre-extracted embeddings.",
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

    # Output
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Checkpoint directory",
    )

    # Model architecture
    parser.add_argument(
        "--architecture", type=str, default="mlp",
        choices=["linear", "mlp", "cnn", "lstm"],
        help="Classifier architecture (default: mlp)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=256,
        help="Classifier hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--num-layers", type=int, default=2,
        help="Number of hidden layers (default: 2)",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout probability (default: 0.1)",
    )

    # Windowing
    parser.add_argument(
        "--window-size", type=int, default=1024,
        help="Window size in bp (default: 1024)",
    )
    parser.add_argument(
        "--step-size", type=int, default=None,
        help="Step size for windowing (default: window_size / 2)",
    )
    parser.add_argument(
        "--min-exon-fraction", type=float, default=0.0,
        help="Skip windows below this exon fraction (default: 0.0)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument(
        "--val-fraction", type=float, default=0.2,
        help="Fraction of genes for validation (default: 0.2)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cpu, cuda, mps (default: auto)",
    )

    args = parser.parse_args()

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

    # Resolve step size
    if args.step_size is None:
        args.step_size = args.window_size // 2

    t0 = time.time()

    # Load and window data
    logger.info("Loading data...")
    train_emb, train_lbl, val_emb, val_lbl, input_dim = load_data(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        window_size=args.window_size,
        step_size=args.step_size,
        min_exon_fraction=args.min_exon_fraction,
        val_fraction=args.val_fraction,
    )

    logger.info(
        "Train: %s, Val: %s",
        list(train_emb.shape),
        list(val_emb.shape) if val_emb is not None else "None",
    )

    # Memory estimate
    train_mb = train_emb.nelement() * 4 / (1024 * 1024)
    val_mb = val_emb.nelement() * 4 / (1024 * 1024) if val_emb is not None else 0
    logger.info("Memory: train=%.0f MB, val=%.0f MB", train_mb, val_mb)

    # Create classifier
    from foundation_models.evo2 import ExonClassifier

    classifier = ExonClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        architecture=args.architecture,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in classifier.parameters())
    logger.info("Model: %s (%s params)", classifier, f"{n_params:,}")

    # Train
    print()
    print("=" * 60)
    print(f"Training ExonClassifier ({args.architecture})")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {args.hidden_dim}")
    print(f"  Window size:  {args.window_size}")
    print(f"  Device:       {device}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Patience:     {args.patience}")
    print("=" * 60)
    print()

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
        checkpoint_dir=args.output,
        patience=args.patience,
        lr_schedule=True,
    )

    elapsed = time.time() - t0

    # Summary
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"  Best epoch:     {history['best_epoch'] + 1}")
    print(f"  Best val AUROC: {history['best_val_auroc']:.4f}")
    print(f"  Early stopped:  {history['stopped_early']}")
    print(f"  Time:           {elapsed:.1f}s")
    print(f"  Checkpoint:     {Path(args.output) / 'best_model.pt'}")
    print()
    print("Next step: evaluate the classifier")
    print(f"  python examples/04_evaluate_classifier.py \\")
    print(f"      --checkpoint {Path(args.output) / 'best_model.pt'} \\")
    print(f"      --embeddings {args.embeddings} \\")
    print(f"      --labels {args.labels} \\")
    print(f"      --window-size {args.window_size}")
    print()


if __name__ == "__main__":
    main()
