"""
Dense Per-Nucleotide Splice Site Predictor

Trains a lightweight prediction head on frozen foundation model embeddings
to produce per-nucleotide splice site scores: P(donor), P(acceptor),
P(neither) — matching SpliceAI/OpenSpliceAI's output format.

Supports multiple embedding models:
  - **Evo2** (7B/40B) — causal DNA LM (Apache 2.0)
  - **SpliceBERT** (19.4M) — bidirectional BERT (AGPL-3.0)
  - **HyenaDNA** (1.6M-300M) — causal Hyena operator (Apache 2.0)

Pipeline:
  1. Generate or load per-gene embeddings (windowed)
  2. Build 3-class splice labels (0=none, 1=acceptor, 2=donor)
  3. Train SpliceClassifier with focal loss on windowed data
  4. Evaluate on held-out genes

Usage:
    # Mock mode (no GPU — validates entire pipeline)
    python 06_dense_splice_predictor.py \\
        --foundation-model splicebert --mock --n-genes 5 \\
        -o /tmp/dense-test/

    # SpliceBERT on GPU
    python 06_dense_splice_predictor.py \\
        --foundation-model splicebert --n-genes 20 \\
        -o /workspace/output/dense-splicebert/

    # Evo2 on A40+
    python 06_dense_splice_predictor.py \\
        --foundation-model evo2 --n-genes 20 \\
        -o /workspace/output/dense-evo2/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_window_config(
    model_name: str,
    max_context: int,
) -> Tuple[int, int]:
    """Return (window_size, step_size) based on model capabilities.

    Smaller models get smaller windows to match their embedding context.
    """
    if max_context <= 1024:
        # SpliceBERT: 512 bp windows
        return 512, 256
    elif max_context <= 8192:
        return 4096, 2048
    else:
        # Evo2, HyenaDNA: 8192 bp windows
        return 8192, 4096


# -----------------------------------------------------------------------
# Step 1: Generate windowed data (mock or real)
# -----------------------------------------------------------------------

def prepare_mock_data(
    n_genes: int,
    hidden_dim: int,
    window_size: int,
    step_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic windowed embeddings + splice labels.

    Returns:
        (embeddings, labels, gene_ids) where:
        - embeddings: [N_windows, window_size, hidden_dim]
        - labels: [N_windows, window_size] (0/1/2)
        - gene_ids: list of gene IDs per window
    """
    from foundation_models.utils.synthetic import generate_synthetic_splice_data

    emb_dict, lbl_dict = generate_synthetic_splice_data(
        n_genes=n_genes, hidden_dim=hidden_dim, seed=seed,
    )

    # Window each gene
    all_emb = []
    all_lbl = []
    all_genes = []

    for gene_id in sorted(emb_dict.keys()):
        emb = emb_dict[gene_id]   # [seq_len, hidden_dim]
        lbl = lbl_dict[gene_id]   # [seq_len]
        seq_len = len(lbl)

        pos = 0
        while pos + window_size <= seq_len:
            all_emb.append(emb[pos:pos + window_size])
            all_lbl.append(lbl[pos:pos + window_size])
            all_genes.append(gene_id)
            pos += step_size

    embeddings = np.stack(all_emb, axis=0)  # [N, W, H]
    labels = np.stack(all_lbl, axis=0)      # [N, W]

    n_splice = (labels > 0).sum()
    n_total = labels.size
    logger.info(
        "Mock data: %d windows (size=%d, step=%d) from %d genes",
        len(embeddings), window_size, step_size, n_genes,
    )
    logger.info(
        "  Splice sites: %d / %d positions (%.4f%%)",
        n_splice, n_total, n_splice / n_total * 100 if n_total else 0,
    )

    return embeddings, labels, all_genes


# -----------------------------------------------------------------------
# Step 2: Train / Val / Test split
# -----------------------------------------------------------------------

def split_by_gene(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gene_ids: List[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split windowed data by gene (no data leakage).

    Returns dict with keys 'train', 'val', 'test', each containing
    (embeddings, labels) arrays.
    """
    rng = np.random.RandomState(seed)
    unique_genes = sorted(set(gene_ids))
    rng.shuffle(unique_genes)

    n_test = max(1, int(len(unique_genes) * test_fraction))
    n_val = max(1, int(len(unique_genes) * val_fraction))

    test_genes = set(unique_genes[:n_test])
    val_genes = set(unique_genes[n_test:n_test + n_val])
    train_genes = set(unique_genes[n_test + n_val:])

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gene_arr = np.array(gene_ids)

    for name, gene_set in [
        ("train", train_genes), ("val", val_genes), ("test", test_genes),
    ]:
        mask = np.isin(gene_arr, list(gene_set))
        splits[name] = (embeddings[mask], labels[mask])
        n_splice = (labels[mask] > 0).sum()
        logger.info(
            "  %s: %d windows (%d genes), %d splice sites",
            name, mask.sum(), len(gene_set), n_splice,
        )

    return splits


# -----------------------------------------------------------------------
# Step 3: Train and evaluate
# -----------------------------------------------------------------------

def train_and_evaluate(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    output_dir: Path,
    architecture: str = "dilated_cnn",
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 20,
    focal_gamma: float = 2.0,
) -> Dict:
    """Train SpliceClassifier and evaluate on test set.

    Returns:
        Results dict with training history and test metrics.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    from foundation_models.classifiers.splice_classifier import SpliceClassifier

    device = get_device()

    train_emb, train_lbl = splits["train"]
    val_emb, val_lbl = splits["val"]
    test_emb, test_lbl = splits["test"]

    # Convert to tensors
    train_emb_t = torch.tensor(train_emb, dtype=torch.float32)
    train_lbl_t = torch.tensor(train_lbl, dtype=torch.long)
    val_emb_t = torch.tensor(val_emb, dtype=torch.float32)
    val_lbl_t = torch.tensor(val_lbl, dtype=torch.long)

    # Create classifier
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    classifier = SpliceClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
    )

    n_params = sum(p.numel() for p in classifier.parameters())

    print()
    print("=" * 60)
    print("Training SpliceClassifier (dense splice site prediction)")
    print("=" * 60)
    print(f"  Architecture: {architecture}")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Windows:      {len(train_emb)} train, {len(val_emb)} val, {len(test_emb)} test")
    print(f"  Device:       {device}")
    print(f"  Focal gamma:  {focal_gamma}")
    print(f"  Hyperparams:  lr={lr}, batch={batch_size}, wd={weight_decay}")
    print("=" * 60)
    print()

    # Train
    history = classifier.fit(
        train_emb_t, train_lbl_t,
        val_emb_t, val_lbl_t,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=str(model_dir),
        patience=patience,
        focal_gamma=focal_gamma,
    )

    # Evaluate on test set
    classifier.to(device)
    test_emb_t = torch.tensor(test_emb, dtype=torch.float32).to(device)

    preds = classifier.predict(test_emb_t)

    # Per-class metrics (acceptor + donor)
    test_lbl_flat = test_lbl.flatten()
    results: Dict = {"history": history}

    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        y_true = (test_lbl_flat == cls_idx).astype(np.int32)
        y_score = preds[f"{cls_name}_prob"].flatten()

        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        else:
            auroc = float("nan")
            auprc = float("nan")

        results[cls_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "n_positive": int(y_true.sum()),
            "n_total": int(len(y_true)),
        }

    # Average across splice classes
    aurocs = [
        results[c]["auroc"] for c in ("acceptor", "donor")
        if not np.isnan(results[c]["auroc"])
    ]
    auprcs = [
        results[c]["auprc"] for c in ("acceptor", "donor")
        if not np.isnan(results[c]["auprc"])
    ]
    results["mean_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    results["mean_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")

    # Save results
    results_path = model_dir / "test_results.json"
    serializable = {
        k: v for k, v in results.items()
        if k != "history"
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print()
    print("Test Metrics (held-out genes):")
    for cls_name in ("acceptor", "donor"):
        m = results[cls_name]
        print(f"  {cls_name:10s}: AUROC={m['auroc']:.4f}, "
              f"AUPRC={m['auprc']:.4f} "
              f"({m['n_positive']} sites / {m['n_total']} positions)")
    print(f"  {'mean':10s}: AUROC={results['mean_auroc']:.4f}, "
          f"AUPRC={results['mean_auprc']:.4f}")
    print()

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    from foundation_models.base import (
        get_model_metadata,
        list_available_models,
    )

    available_models = list_available_models()

    parser = argparse.ArgumentParser(
        description="Dense per-nucleotide splice site predictor using "
                    "foundation model embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument("--foundation-model", type=str, default="splicebert",
                        choices=available_models,
                        help=f"Foundation model (default: splicebert; "
                             f"available: {', '.join(available_models)})")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Model-specific variant")

    # Data
    parser.add_argument("--n-genes", type=int, default=10,
                        help="Number of genes (mock mode: synthetic; "
                             "real mode: max genes to process)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Window size in bp (default: model-dependent)")
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size in bp (default: window_size // 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")

    # Classifier architecture
    parser.add_argument("--architecture", type=str, default="dilated_cnn",
                        choices=["dilated_cnn", "mlp", "linear"],
                        help="Prediction head architecture (default: dilated_cnn)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Prediction head hidden dim (default: 128)")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")

    # Mock mode
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data (no GPU or FASTA needed)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Get model metadata (lightweight — no model loading)
    model_kwargs: dict = {}
    if args.foundation_model == "evo2":
        model_kwargs["model_size"] = args.model_size or "7b"
    elif args.foundation_model == "hyenadna":
        model_kwargs["model_size"] = args.model_size or "medium-160k"
    elif args.foundation_model == "splicebert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size

    meta = get_model_metadata(args.foundation_model, **model_kwargs)
    hidden_dim = meta.hidden_dim

    # Resolve window config
    if args.window_size is not None:
        window_size = args.window_size
        step_size = args.step_size or window_size // 2
    else:
        window_size, step_size = get_default_window_config(
            meta.name, meta.max_context,
        )
        if args.step_size is not None:
            step_size = args.step_size

    print()
    print("=" * 70)
    print("Dense Splice Site Predictor (Foundation Model Embeddings)")
    print("=" * 70)
    print(f"  Model:        {'MOCK (synthetic)' if args.mock else meta.name}")
    print(f"  Type:         {meta.model_type}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Window:       {window_size} bp (step: {step_size})")
    print(f"  Genes:        {args.n_genes}")
    print(f"  Focal gamma:  {args.focal_gamma}")
    print(f"  Seed:         {args.seed}")
    print("=" * 70)
    print()

    # Step 1: Prepare data
    if args.mock:
        logger.info("Step 1/2: Generating synthetic windowed data...")
        embeddings, labels, gene_ids = prepare_mock_data(
            n_genes=args.n_genes,
            hidden_dim=hidden_dim,
            window_size=window_size,
            step_size=step_size,
            seed=args.seed,
        )
    else:
        logger.error(
            "Real data mode not yet implemented in Phase 1. Use --mock "
            "for pipeline validation."
        )
        sys.exit(1)

    # Step 2: Split and train
    logger.info("Splitting by gene...")
    splits = split_by_gene(
        embeddings, labels, gene_ids, seed=args.seed,
    )

    logger.info("Step 2/2: Training classifier...")
    results = train_and_evaluate(
        splits=splits,
        input_dim=hidden_dim,
        output_dir=output_dir,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
    )

    elapsed = time.time() - t0

    print("=" * 70)
    print("Complete")
    print("=" * 70)
    print(f"  Model:        {meta.name}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Output:       {output_dir}")
    print(f"  Checkpoint:   {output_dir / 'model' / 'best_model.pt'}")
    print(f"  Mean AUROC:   {results['mean_auroc']:.4f}")
    print(f"  Mean AUPRC:   {results['mean_auprc']:.4f}")
    print(f"  Time:         {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
