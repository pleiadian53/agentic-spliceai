#!/usr/bin/env python3
"""
Streaming Extract-and-Train: Per-Chromosome Embedding + Classifier Training

Extracts Evo2 embeddings and trains an exon classifier in a streaming fashion,
processing one chromosome at a time to avoid storing hundreds of GB of embeddings.

Pipeline per epoch:
    for chrom in training_chromosomes:
        1. Extract embeddings for chrom genes → temp HDF5
        2. Window embeddings + labels → training batches
        3. Train one pass over the chromosome's batches
        4. Delete temp HDF5 → free disk

Evaluation:
    After training, extracts test chromosome embeddings (one at a time)
    and evaluates the classifier.

Splitting:
    Uses SpliceAI's chromosome holdout split by default:
    - Train: even chromosomes (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    - Test:  odd chromosomes (1, 3, 5, 7, 9)
    - Val:   10% of training genes (for early stopping)

Storage:
    Peak disk usage = one chromosome's embeddings (~20-50 GB for chr1).
    See docs/embedding-storage-challenge.md for details.

Usage:
    # All MANE genes, streaming mode
    python examples/foundation_models/04_extract_and_train.py \
        --all-genes \
        --output /workspace/output/ \
        --model evo2 --model-size 7b \
        --architecture mlp --epochs 5 --chunk-size 4096

    # Specific chromosomes only
    python examples/foundation_models/04_extract_and_train.py \
        --chromosomes 21 22 \
        --output /workspace/output/ \
        --model evo2

    # On remote GPU via SkyPilot
    python examples/foundation_models/ops_run_pipeline.py --execute \
        --cluster sky-c0ec-pleiadian53 --no-teardown \
        -- python examples/foundation_models/04_extract_and_train.py \
             --all-genes --output /workspace/output/ \
             --model evo2 --model-size 7b --split-preset spliceai \
             --homology-filter --architecture mlp --epochs 5
"""

import json
import logging
import random
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming extract-and-train: per-chromosome embedding + classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Gene selection
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument("--genes", type=str, nargs="+", help="Gene symbols")
    gene_group.add_argument("--chromosomes", type=str, nargs="+", help="Chromosomes")
    gene_group.add_argument("--all-genes", action="store_true", help="All MANE genes")

    # Shared
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "evo2", "hyenadna"])
    parser.add_argument("--model-size", type=str, default=None)

    # Extraction
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Sequence chunk size for embedding extraction")
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--skip-resource-check", action="store_true")

    # Training
    parser.add_argument("--architecture", type=str, default="mlp",
                        choices=["linear", "mlp", "cnn", "lstm"])
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Classifier hidden dim (not embedding dim)")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of full passes over all training chromosomes")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (in epochs)")
    parser.add_argument("--device", type=str, default="auto")

    # Splitting
    parser.add_argument("--split-preset", type=str, default="spliceai",
                        choices=["spliceai", "even_odd", "naive"])
    parser.add_argument("--homology-filter", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)

    return parser.parse_args()


def _resolve_backend(model: str, model_size: str | None) -> tuple[str, str]:
    """Resolve model backend and size."""
    import torch

    if model == "auto":
        backend = "evo2" if torch.cuda.is_available() else "hyenadna"
    else:
        backend = model

    if backend == "evo2" and not torch.cuda.is_available():
        logger.error("Evo2 requires CUDA but no CUDA device found.")
        sys.exit(1)

    if model_size is None:
        model_size = "7b" if backend == "evo2" else "medium-160k"

    return backend, model_size


def _resolve_device(device_arg: str) -> str:
    """Resolve device string."""
    import torch
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_arg


def _get_chromosome_split(
    chromosomes: list[str],
    preset: str,
) -> tuple[list[str], list[str]]:
    """Split chromosomes into train and test sets.

    Returns (train_chroms, test_chroms) as chromosome strings without 'chr' prefix.
    """
    if preset in ("spliceai", "even_odd"):
        train_chroms, test_chroms = [], []
        for c in chromosomes:
            c_clean = c.replace("chr", "")
            if c_clean in ("X", "Y"):
                train_chroms.append(c_clean)
            elif int(c_clean) % 2 == 0:
                train_chroms.append(c_clean)
            else:
                test_chroms.append(c_clean)
        return train_chroms, test_chroms
    else:
        return [c.replace("chr", "") for c in chromosomes], []


def _extract_chrom_to_hdf5(
    chrom: str,
    embedder: object,
    hdf5_path: Path,
    data_dir: Path,
    chunk_size: int,
    overlap: int,
    backend: str,
    model_size: str,
) -> tuple[Path, dict]:
    """Extract embeddings for one chromosome, write HDF5, return labels dict."""
    import h5py
    import numpy as np
    import torch

    from agentic_spliceai.splice_engine.base_layer.data.preparation import prepare_gene_data
    from foundation_models.utils.chunking import (
        build_exon_labels,
        chunk_sequence,
        stitch_embeddings,
    )

    chrom_label = chrom if chrom.startswith("chr") else f"chr{chrom}"

    # Load gene sequences for this chromosome
    gene_data = prepare_gene_data(
        chromosomes=[chrom],
        build="GRCh38",
        annotation_source="mane",
    )

    if gene_data.is_empty():
        logger.warning("No genes found for %s", chrom_label)
        return hdf5_path, {}

    logger.info("Processing %d genes on %s", len(gene_data), chrom_label)

    hidden_dim = embedder.model.hidden_dim

    # Load splice sites for labels
    splice_tsv = data_dir / "splice_sites_enhanced.tsv"
    splice_parquet = data_dir / "splice_sites_enhanced.parquet"
    splice_sites_df = None
    if splice_parquet.exists():
        import pandas as pd
        splice_sites_df = pd.read_parquet(splice_parquet)
    elif splice_tsv.exists():
        import pandas as pd
        splice_sites_df = pd.read_csv(splice_tsv, sep="\t")

    labels_dict = {}

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["model"] = f"{backend}-{model_size}"
        hf.attrs["hidden_dim"] = hidden_dim

        for i, row in enumerate(gene_data.iter_rows(named=True), 1):
            gene_id = row["gene_id"]
            gene_name = row["gene_name"]
            sequence = row["sequence"]
            gene_start = int(row["start"])
            seq_len = len(sequence)

            if i % 50 == 1 or i == len(gene_data):
                logger.info(
                    "  [%d/%d] %s / %s (len=%d)",
                    i, len(gene_data), gene_name, gene_id, seq_len,
                )

            chunks = chunk_sequence(
                sequence=sequence,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            chunk_embs = []
            for chunk in chunks:
                emb = embedder.encode(chunk.sequence)
                if hasattr(emb, "cpu"):
                    emb = emb.cpu()
                chunk_embs.append(emb)

            full_emb = stitch_embeddings(
                chunks=chunks,
                chunk_embeddings=chunk_embs,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )

            hf.create_dataset(gene_id, data=full_emb, compression="gzip")

            del chunk_embs, full_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Build labels
            if splice_sites_df is not None:
                labels = build_exon_labels(
                    gene_id=gene_id,
                    gene_start=gene_start,
                    gene_sequence_length=seq_len,
                    splice_sites_df=splice_sites_df,
                )
                labels_dict[gene_id] = labels

    logger.info("Extracted %d genes for %s -> %s", len(gene_data), chrom_label, hdf5_path)
    return hdf5_path, labels_dict


def _load_chrom_windows(
    hdf5_path: Path,
    labels_dict: dict,
    window_size: int,
    step_size: int,
) -> list[tuple]:
    """Load embeddings from HDF5, build windows, return (emb, label, gene_id) tuples."""
    import h5py

    from foundation_models.utils.chunking import window_embeddings

    if not hdf5_path.exists():
        return []

    windows = []
    with h5py.File(hdf5_path, "r") as hf:
        for gene_id in hf.keys():
            if gene_id not in labels_dict:
                continue

            emb = hf[gene_id][:]
            lbl = labels_dict[gene_id]
            min_len = min(len(emb), len(lbl))

            gene_windows = window_embeddings(
                embeddings=emb[:min_len],
                labels=lbl[:min_len],
                gene_id=gene_id,
                window_size=window_size,
                step_size=step_size,
            )
            windows.extend(gene_windows)

    return windows


def _windows_to_tensors(windows: list[tuple]) -> tuple:
    """Convert (emb, label, gene_id) tuples to stacked tensors."""
    import numpy as np
    import torch

    if not windows:
        return None, None

    emb = torch.tensor(np.stack([w[0] for w in windows]), dtype=torch.float32)
    lbl = torch.tensor(np.stack([w[1] for w in windows]), dtype=torch.float32)
    return emb, lbl


def _load_embedder(backend: str, model_size: str) -> object:
    """Load the foundation model embedder (once, reused across all chromosomes)."""
    if backend == "evo2":
        from foundation_models.evo2 import Evo2Embedder
        logger.info("Loading Evo2 %s...", model_size)
        return Evo2Embedder(model_size=model_size)
    else:
        from foundation_models.hyenadna import HyenaDNAEmbedder
        logger.info("Loading HyenaDNA %s...", model_size)
        return HyenaDNAEmbedder(model_size=model_size)


def main() -> None:
    args = _parse_args()

    import numpy as np
    import torch

    from foundation_models.evo2 import ExonClassifier

    t0 = time.time()
    backend, model_size = _resolve_backend(args.model, args.model_size)
    device = _resolve_device(args.device)

    output_dir = Path(args.output)
    embeddings_dir = output_dir / "embeddings"
    model_dir = output_dir / "model"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/mane/GRCh38/")

    # --genes mode: fall back to simple sequential pipeline
    if args.genes:
        logger.info("--genes mode: using sequential extract-then-train")
        _run_simple_mode(args, output_dir)
        return

    # Determine chromosomes
    if args.chromosomes:
        chromosomes = args.chromosomes
    else:
        chromosomes = [str(c) for c in range(1, 23)] + ["X", "Y"]

    # ---------------------------------------------------------------
    # Chromosome split
    # ---------------------------------------------------------------
    train_chroms, test_chroms = _get_chromosome_split(chromosomes, args.split_preset)
    step_size = args.window_size // 2

    model_label = f"Evo2 {model_size}" if backend == "evo2" else f"HyenaDNA {model_size}"

    print()
    print("=" * 70)
    print("Streaming Extract-and-Train Pipeline")
    print("=" * 70)
    print(f"  Model:        {model_label}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Chunk size:   {args.chunk_size}")
    print(f"  Window size:  {args.window_size}")
    print(f"  Split:        {args.split_preset}")
    print(f"  Train chroms: {', '.join(train_chroms)}")
    print(f"  Test chroms:  {', '.join(test_chroms) or '(none)'}")
    print(f"  Device:       {device}")
    print("=" * 70)
    print()

    # Save config
    config = {
        "model": f"{backend}-{model_size}",
        "architecture": args.architecture,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "window_size": args.window_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "split_preset": args.split_preset,
        "train_chroms": train_chroms,
        "test_chroms": test_chroms,
        "chunk_size": args.chunk_size,
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # ---------------------------------------------------------------
    # Load foundation model (once, shared across all chromosomes)
    # ---------------------------------------------------------------
    embedder = _load_embedder(backend, model_size)

    # ---------------------------------------------------------------
    # Initialize classifier (input_dim discovered from first extraction)
    # ---------------------------------------------------------------
    classifier = None
    optimizer = None
    input_dim = None

    best_val_loss = float("inf")
    patience_counter = 0
    epoch_history = []

    # ---------------------------------------------------------------
    # Training loop: epochs × chromosomes
    # ---------------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        epoch_train_loss = 0.0
        epoch_train_batches = 0
        epoch_val_windows = []

        print()
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        for chrom_idx, chrom in enumerate(train_chroms, 1):
            chrom_label = f"chr{chrom}"
            chrom_t0 = time.time()

            logger.info(
                "--- Epoch %d, Chromosome %s (%d/%d) ---",
                epoch, chrom_label, chrom_idx, len(train_chroms),
            )

            # Step 1: Extract embeddings for this chromosome
            hdf5_path = embeddings_dir / f"embeddings_{chrom_label}.h5"
            hdf5_path, labels_dict = _extract_chrom_to_hdf5(
                chrom=chrom,
                embedder=embedder,
                hdf5_path=hdf5_path,
                data_dir=data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                backend=backend,
                model_size=model_size,
            )

            if not hdf5_path.exists() or not labels_dict:
                logger.warning("No data for %s, skipping", chrom_label)
                continue

            # Step 2: Load windows
            windows = _load_chrom_windows(
                hdf5_path=hdf5_path,
                labels_dict=labels_dict,
                window_size=args.window_size,
                step_size=step_size,
            )

            if not windows:
                logger.warning("No windows for %s, skipping", chrom_label)
                if hdf5_path.exists():
                    hdf5_path.unlink()
                continue

            # Discover input_dim and initialize classifier from first chromosome
            if input_dim is None:
                input_dim = windows[0][0].shape[-1]
                classifier = ExonClassifier(
                    input_dim=input_dim,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    architecture=args.architecture,
                    dropout=args.dropout,
                )
                classifier.to(device)
                optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
                logger.info(
                    "Classifier: input_dim=%d, arch=%s, device=%s",
                    input_dim, args.architecture, device,
                )

            # Split this chromosome's windows into train + val
            rng = random.Random(args.split_seed + epoch * 100 + chrom_idx)
            rng.shuffle(windows)
            n_val = max(1, int(len(windows) * args.val_fraction))
            val_windows_chrom = windows[:n_val]
            train_windows_chrom = windows[n_val:]

            epoch_val_windows.extend(val_windows_chrom)

            # Step 3: Train on this chromosome's windows
            train_emb, train_lbl = _windows_to_tensors(train_windows_chrom)

            if train_emb is not None:
                dataset = torch.utils.data.TensorDataset(train_emb, train_lbl)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=args.batch_size, shuffle=True,
                )

                classifier.train()
                chrom_loss = 0.0
                chrom_batches = 0

                for batch_emb, batch_lbl in loader:
                    batch_emb = batch_emb.to(device)
                    batch_lbl = batch_lbl.to(device)

                    optimizer.zero_grad()
                    logits = classifier(batch_emb).squeeze(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits, batch_lbl,
                    )
                    loss.backward()
                    optimizer.step()

                    chrom_loss += loss.item()
                    chrom_batches += 1

                epoch_train_loss += chrom_loss
                epoch_train_batches += chrom_batches

                avg_loss = chrom_loss / max(chrom_batches, 1)
                del train_emb, train_lbl, dataset, loader
            else:
                avg_loss = float("nan")

            # Step 4: Delete embeddings to free disk
            h5_size_gb = hdf5_path.stat().st_size / (1024**3) if hdf5_path.exists() else 0
            if hdf5_path.exists():
                hdf5_path.unlink()

            chrom_elapsed = time.time() - chrom_t0
            logger.info(
                "✓ %s: %d train + %d val windows, loss=%.4f, freed %.1f GB, %.0fs",
                chrom_label, len(train_windows_chrom), len(val_windows_chrom),
                avg_loss, h5_size_gb, chrom_elapsed,
            )

            # Free memory
            del windows, train_windows_chrom, val_windows_chrom, labels_dict
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # End of epoch: validate
        avg_epoch_loss = epoch_train_loss / max(epoch_train_batches, 1)

        val_loss = float("inf")
        val_auroc = float("nan")
        if epoch_val_windows and classifier is not None:
            val_emb, val_lbl = _windows_to_tensors(epoch_val_windows)
            classifier.eval()
            with torch.no_grad():
                val_logits = classifier(val_emb.to(device)).squeeze(-1)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    val_logits, val_lbl.to(device),
                ).item()

                from sklearn.metrics import roc_auc_score
                val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
                val_labels_np = val_lbl.numpy().flatten()
                try:
                    val_auroc = float(roc_auc_score(val_labels_np, val_probs))
                except ValueError:
                    val_auroc = float("nan")

            del val_emb, val_lbl, val_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        epoch_elapsed = time.time() - epoch_t0
        epoch_history.append({
            "epoch": epoch,
            "train_loss": round(avg_epoch_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_auroc": round(val_auroc, 4),
            "time_min": round(epoch_elapsed / 60, 1),
        })

        logger.info(
            "Epoch %d: train_loss=%.4f, val_loss=%.4f, val_auroc=%.4f (%.1f min)",
            epoch, avg_epoch_loss, val_loss, val_auroc, epoch_elapsed / 60,
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": classifier.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_auroc": val_auroc,
                "config": config,
            }, model_dir / "best_model.pt")
            logger.info("✓ New best model saved (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping: no improvement for %d epochs", args.patience)
                break

        del epoch_val_windows

    # ---------------------------------------------------------------
    # Evaluation on held-out test chromosomes
    # ---------------------------------------------------------------
    test_metrics = {}
    if test_chroms and classifier is not None:
        print()
        print("=" * 60)
        print("Evaluation on Test Chromosomes")
        print("=" * 60)
        print()

        # Load best checkpoint
        ckpt_path = model_dir / "best_model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            classifier.load_state_dict(ckpt["model_state_dict"])
        classifier.eval()

        all_test_probs = []
        all_test_labels = []

        for chrom in test_chroms:
            chrom_label = f"chr{chrom}"
            logger.info("Evaluating on %s...", chrom_label)

            hdf5_path = embeddings_dir / f"embeddings_{chrom_label}.h5"
            hdf5_path, labels_dict = _extract_chrom_to_hdf5(
                chrom=chrom,
                embedder=embedder,
                hdf5_path=hdf5_path,
                data_dir=data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                backend=backend,
                model_size=model_size,
            )

            if not hdf5_path.exists() or not labels_dict:
                continue

            windows = _load_chrom_windows(
                hdf5_path=hdf5_path,
                labels_dict=labels_dict,
                window_size=args.window_size,
                step_size=step_size,
            )

            if windows:
                test_emb, test_lbl = _windows_to_tensors(windows)
                with torch.no_grad():
                    logits = classifier(test_emb.to(device)).squeeze(-1)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    labels_np = test_lbl.numpy().flatten()

                all_test_probs.extend(probs)
                all_test_labels.extend(labels_np)
                logger.info("  %s: %d windows", chrom_label, len(windows))
                del test_emb, test_lbl, windows

            # Clean up
            h5_size_gb = hdf5_path.stat().st_size / (1024**3) if hdf5_path.exists() else 0
            if hdf5_path.exists():
                hdf5_path.unlink()
            logger.info("  Freed %.1f GB", h5_size_gb)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Compute test metrics
        if all_test_probs:
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
            )

            all_test_probs_np = np.array(all_test_probs)
            all_test_labels_np = np.array(all_test_labels)
            preds = (all_test_probs_np >= 0.5).astype(int)

            try:
                auroc = float(roc_auc_score(all_test_labels_np, all_test_probs_np))
            except ValueError:
                auroc = float("nan")
            try:
                auprc = float(average_precision_score(all_test_labels_np, all_test_probs_np))
            except ValueError:
                auprc = float("nan")

            test_metrics = {
                "auroc": round(auroc, 4),
                "auprc": round(auprc, 4),
                "accuracy": round(float(accuracy_score(all_test_labels_np, preds)), 4),
                "f1": round(float(f1_score(all_test_labels_np, preds, zero_division=0)), 4),
                "precision": round(float(precision_score(all_test_labels_np, preds, zero_division=0)), 4),
                "recall": round(float(recall_score(all_test_labels_np, preds, zero_division=0)), 4),
                "n_test_chroms": len(test_chroms),
                "n_test_windows": len(all_test_probs),
            }

            print()
            print("Test Metrics (held-out chromosomes):")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    print(f"  {k:12s}: {v:.4f}")
                else:
                    print(f"  {k:12s}: {v}")
            print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    elapsed = time.time() - t0

    results = {
        "training": {
            "epochs_completed": len(epoch_history),
            "best_val_loss": round(best_val_loss, 4),
            "history": epoch_history,
        },
        "config": config,
    }
    if test_metrics:
        results["test"] = test_metrics

    metrics_path = model_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
    print(f"  Model:       {model_dir / 'best_model.pt'}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Epochs:      {len(epoch_history)}")
    if test_metrics:
        print(f"  Test AUROC:  {test_metrics.get('auroc', 'N/A')}")
    print(f"  Total time:  {elapsed / 60:.1f} min")
    print()


def _run_simple_mode(args, output_dir: Path) -> None:
    """Fallback for --genes mode: extract all then train (small gene sets only)."""
    import subprocess

    embeddings_dir = output_dir / "embeddings"
    model_dir = output_dir / "model"

    extract_cmd = [
        sys.executable, "examples/foundation_models/02_embedding_extraction.py",
        "--output", str(embeddings_dir),
        "--model", args.model,
        "--chunk-size", str(args.chunk_size),
        "--overlap", str(args.overlap),
        "--genes",
    ] + args.genes

    if args.model_size:
        extract_cmd.extend(["--model-size", args.model_size])
    if args.skip_resource_check:
        extract_cmd.append("--skip-resource-check")

    logger.info("Extracting embeddings for %d genes...", len(args.genes))
    result = subprocess.run(extract_cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)

    train_cmd = [
        sys.executable, "examples/foundation_models/03_train_and_evaluate.py",
        "--output", str(model_dir),
        "--embeddings-dir", str(embeddings_dir),
        "--architecture", args.architecture,
        "--hidden-dim", str(args.hidden_dim),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--split-preset", "naive",
        "--device", args.device,
    ]

    logger.info("Training classifier...")
    result = subprocess.run(train_cmd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
