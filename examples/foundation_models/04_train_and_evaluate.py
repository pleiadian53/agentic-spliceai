#!/usr/bin/env python3
"""
Example: Train and Evaluate ExonClassifier from Pre-Extracted Embeddings

Loads HDF5 embeddings and NPZ labels, trains an ExonClassifier with early
stopping, and evaluates with per-gene metrics.

Gene-level splitting follows SpliceAI's methodology:
  - Chromosome holdout: odd chromosomes (1,3,5,7,9) held out for testing
  - Validation: 10% of training-chromosome genes for early stopping
  - Homology filtering: test genes with paralogs in training are removed

Supports both single-file and per-chromosome directory input.

Usage:
    # From single HDF5 file (small gene set, naive split)
    python examples/foundation_models/04_train_and_evaluate.py \
        --embeddings /tmp/fm_demo/embeddings.h5 \
        --labels /tmp/fm_demo/embeddings.labels.npz \
        --output /tmp/fm_demo/model/

    # From per-chromosome directory with SpliceAI split (recommended)
    python examples/foundation_models/04_train_and_evaluate.py \
        --embeddings-dir /workspace/output/embeddings/ \
        --output /workspace/output/model/ \
        --split-preset spliceai \
        --homology-filter \
        --architecture mlp --epochs 50

    # Custom split
    python examples/foundation_models/04_train_and_evaluate.py \
        --embeddings-dir /workspace/output/embeddings/ \
        --output /workspace/output/model/ \
        --split-preset spliceai \
        --architecture cnn --epochs 50

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


def _load_embeddings_and_labels(
    embeddings_path: str | None,
    labels_path: str | None,
    embeddings_dir: str | None,
) -> tuple:
    """Load embeddings and labels from single file or directory.

    Returns:
        Tuple of (h5_files_list, labels_data_dict, hidden_dim, gene_ids).
        h5_files_list contains (h5py.File, gene_ids_in_file) tuples.
    """
    import h5py
    import numpy as np

    h5_files = []
    all_labels = {}
    hidden_dim = 0
    all_gene_ids = []

    if embeddings_dir:
        emb_dir = Path(embeddings_dir)
        h5_paths = sorted(emb_dir.glob("embeddings_chr*.h5"))
        if not h5_paths:
            single = emb_dir / "embeddings.h5"
            if single.exists():
                h5_paths = [single]
            else:
                logger.error("No HDF5 files found in %s", emb_dir)
                sys.exit(1)

        for npz_path in sorted(emb_dir.glob("embeddings_chr*.labels.npz")):
            all_labels.update(dict(np.load(npz_path)))
        single_labels = emb_dir / "embeddings.labels.npz"
        if single_labels.exists():
            all_labels.update(dict(np.load(single_labels)))

        for hp in h5_paths:
            hf = h5py.File(hp, "r")
            file_genes = list(hf.keys())
            if not hidden_dim:
                hidden_dim = int(hf.attrs.get("hidden_dim", 0))
            h5_files.append((hf, file_genes))
            all_gene_ids.extend(file_genes)
            logger.info("Loaded %s: %d genes", hp.name, len(file_genes))

    else:
        hf = h5py.File(embeddings_path, "r")
        file_genes = list(hf.keys())
        hidden_dim = int(hf.attrs.get("hidden_dim", 0))
        h5_files.append((hf, file_genes))
        all_gene_ids.extend(file_genes)

        if labels_path:
            all_labels = dict(np.load(labels_path))

    logger.info("Total: %d genes, %d with labels", len(all_gene_ids), len(all_labels))
    return h5_files, all_labels, hidden_dim, all_gene_ids


def _build_gene_chrom_map_from_h5(
    h5_files: list,
    embeddings_dir: str | None,
) -> dict[str, str]:
    """Infer gene->chromosome mapping from per-chromosome HDF5 filenames.

    For per-chromosome files (embeddings_chr1.h5, embeddings_chr2.h5, ...),
    the chromosome is embedded in the filename. For single files, we fall
    back to loading from annotation data.
    """
    gene_chrom = {}

    if embeddings_dir:
        emb_dir = Path(embeddings_dir)
        for hp in sorted(emb_dir.glob("embeddings_chr*.h5")):
            # Extract chromosome from filename: embeddings_chr17.h5 -> chr17
            chrom = hp.stem.replace("embeddings_", "")  # "chr17"
            import h5py
            with h5py.File(hp, "r") as hf:
                for gene_id in hf.keys():
                    gene_chrom[gene_id] = chrom

    if not gene_chrom:
        # Fallback: load from GTF annotation
        try:
            from agentic_spliceai.splice_engine.eval.splitting import (
                gene_chromosomes_from_gtf,
            )
            from agentic_spliceai.splice_engine.resources import get_genomic_registry

            registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
            gtf_path = str(registry.get_gtf_path(validate=True))
            gene_chrom = gene_chromosomes_from_gtf(gtf_path)
            logger.info("Loaded gene->chromosome mapping from GTF (%d genes)", len(gene_chrom))
        except Exception as e:
            logger.warning("Could not load gene->chromosome mapping: %s", e)

    return gene_chrom


def _naive_split(gene_ids: list[str], val_fraction: float, seed: int) -> tuple:
    """Fallback: random gene split when chromosome info is unavailable."""
    import random
    rng = random.Random(seed)
    shuffled = sorted(gene_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    val_genes = set(shuffled[:n_val])
    train_genes = set(shuffled[n_val:])
    return train_genes, val_genes, set()  # No test set in naive mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate ExonClassifier from embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data (two modes: single file or directory)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--embeddings", type=str,
        help="Path to single HDF5 embeddings file",
    )
    data_group.add_argument(
        "--embeddings-dir", type=str,
        help="Directory containing per-chromosome HDF5 + labels files",
    )
    parser.add_argument(
        "--labels", type=str, default=None,
        help="Path to .labels.npz file (required with --embeddings, auto-found with --embeddings-dir)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for checkpoint and metrics",
    )

    # Splitting strategy
    parser.add_argument(
        "--split-preset", type=str, default="auto",
        choices=["auto", "spliceai", "even_odd", "naive"],
        help="Gene splitting strategy. 'auto' uses spliceai preset when chromosome "
             "info is available, falls back to naive. Default: auto",
    )
    parser.add_argument(
        "--homology-filter", action="store_true",
        help="Remove test genes with paralogs in training set (name-based detection)",
    )
    parser.add_argument(
        "--split-seed", type=int, default=42,
        help="Random seed for val split (default: 42)",
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
        "--val-fraction", type=float, default=0.1,
        help="Fraction of training genes for validation (default: 0.1, matching SpliceAI)",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument(
        "--skip-resource-check", action="store_true",
        help="Skip memory feasibility check",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit to first N genes (for quick validation runs)",
    )

    args = parser.parse_args()

    if args.embeddings and not args.labels:
        auto_labels = Path(args.embeddings).with_suffix(".labels.npz")
        if auto_labels.exists():
            args.labels = str(auto_labels)
        else:
            logger.error("--labels required when using --embeddings (not found at %s)", auto_labels)
            sys.exit(1)

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
    # Load data
    # ------------------------------------------------------------------
    logger.info("Loading data...")
    h5_files, labels_data, hidden_dim, gene_ids = _load_embeddings_and_labels(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        embeddings_dir=args.embeddings_dir,
    )

    if args.max_genes and len(gene_ids) > args.max_genes:
        logger.info("Limiting to first %d genes (of %d)", args.max_genes, len(gene_ids))
        gene_ids = gene_ids[:args.max_genes]

    # ------------------------------------------------------------------
    # Gene-level splitting
    # ------------------------------------------------------------------
    gene_id_set = set(gene_ids)
    use_chromosome_split = False

    # Try to get chromosome info for proper splitting
    gene_chrom_map = _build_gene_chrom_map_from_h5(h5_files, args.embeddings_dir)

    # Filter to genes we actually have embeddings for
    gene_chrom_map = {g: c for g, c in gene_chrom_map.items() if g in gene_id_set}

    # Decide split strategy
    if args.split_preset == "naive":
        use_chromosome_split = False
    elif args.split_preset == "auto":
        use_chromosome_split = len(gene_chrom_map) >= len(gene_ids) * 0.8
        if use_chromosome_split:
            logger.info("Auto-detected chromosome info: using spliceai preset")
        else:
            logger.info("Chromosome info unavailable: falling back to naive split")
    else:
        use_chromosome_split = True

    if use_chromosome_split:
        from agentic_spliceai.splice_engine.eval.splitting import build_gene_split

        gene_families = None
        if args.homology_filter:
            from agentic_spliceai.splice_engine.data.homology import (
                detect_gene_families_by_name,
            )
            # For Ensembl gene IDs, we need gene names for family detection.
            # The HDF5 files store gene_ids (ENSG...), so we need a mapping.
            # For now, use gene IDs directly — name-based detection works on
            # gene symbols, so we load them from annotation if available.
            try:
                from agentic_spliceai.splice_engine.eval.splitting import (
                    gene_chromosomes_from_gtf,
                )
                from agentic_spliceai.splice_engine.resources import get_genomic_registry

                registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
                gtf_path = str(registry.get_gtf_path(validate=True))
                from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
                    extract_gene_annotations,
                )
                import polars as pl
                annot_df = extract_gene_annotations(gtf_path, verbosity=0)
                # Build gene_id -> gene_name mapping
                id_to_name = dict(zip(
                    annot_df["gene_id"].to_list(),
                    annot_df["gene_name"].to_list(),
                ))
                gene_names = [id_to_name.get(g, g) for g in gene_chrom_map.keys()]
                name_families = detect_gene_families_by_name(gene_names)
                # Map back to gene_ids
                name_to_ids = {}
                for gid, gname in id_to_name.items():
                    name_to_ids.setdefault(gname, []).append(gid)
                gene_families = {}
                for gname, family in name_families.items():
                    for gid in name_to_ids.get(gname, []):
                        if gid in gene_chrom_map:
                            gene_families[gid] = family
                logger.info("Homology filter: %d genes in %d families",
                            len(gene_families), len(set(gene_families.values())))
            except Exception as e:
                logger.warning("Could not load gene names for homology filter: %s", e)
                gene_families = None

        preset = "spliceai" if args.split_preset == "auto" else args.split_preset
        split = build_gene_split(
            gene_chromosomes=gene_chrom_map,
            preset=preset,
            val_fraction=args.val_fraction,
            seed=args.split_seed,
            gene_families=gene_families,
            exclude_test_paralogs=args.homology_filter,
        )

        print()
        print(split.summary())
        print()

        train_genes = split.train_genes
        val_genes = split.val_genes
        test_genes = split.test_genes

        # Save split for reproducibility
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        split_path = output_dir / "gene_split.json"
        with open(split_path, "w") as f:
            json.dump(split.to_dict(), f, indent=2)
        logger.info("Saved gene split: %s", split_path)

    else:
        train_genes, val_genes, test_genes = _naive_split(
            gene_ids, args.val_fraction, args.split_seed,
        )
        logger.info("Naive split: %d train, %d val (no test set)", len(train_genes), len(val_genes))

    # ------------------------------------------------------------------
    # Build windows per split
    # ------------------------------------------------------------------
    gene_to_h5 = {}
    for hf, file_genes in h5_files:
        for gid in file_genes:
            gene_to_h5[gid] = hf

    train_windows = []
    val_windows = []
    test_windows = []
    n_skipped = 0

    for gene_id in gene_ids:
        if gene_id not in labels_data or gene_id not in gene_to_h5:
            n_skipped += 1
            continue

        # Determine which split this gene belongs to
        if gene_id in train_genes:
            target = train_windows
        elif gene_id in val_genes:
            target = val_windows
        elif gene_id in test_genes:
            target = test_windows
        else:
            n_skipped += 1
            continue

        hf = gene_to_h5[gene_id]
        emb = hf[gene_id][:]
        lbl = labels_data[gene_id]
        min_len = min(len(emb), len(lbl))

        windows = window_embeddings(
            embeddings=emb[:min_len],
            labels=lbl[:min_len],
            gene_id=gene_id,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        target.extend(windows)

    # Close HDF5 files
    for hf, _ in h5_files:
        hf.close()

    if n_skipped:
        logger.info("Skipped %d genes (missing labels/embeddings or not in split)", n_skipped)

    logger.info(
        "Windows: %d train, %d val, %d test (from %d genes)",
        len(train_windows), len(val_windows), len(test_windows), len(gene_ids),
    )

    if not train_windows:
        logger.error("No training windows. Check data and window_size.")
        sys.exit(1)

    if not val_windows:
        logger.error("No validation windows. Check split configuration.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Resource check
    # ------------------------------------------------------------------
    total_windows = len(train_windows) + len(val_windows) + len(test_windows)

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
            print("Use --skip-resource-check to force, or reduce with --max-genes N.")
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
    # Train (using val set for early stopping)
    # ------------------------------------------------------------------
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_train_genes = len(train_genes)
    n_val_genes = len(val_genes)
    n_test_genes = len(test_genes)

    print()
    print("=" * 60)
    print(f"Training ExonClassifier ({args.architecture})")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {args.hidden_dim}")
    print(f"  Window size:  {args.window_size}")
    print(f"  Windows:      {len(train_windows)} train, {len(val_windows)} val, {len(test_windows)} test")
    print(f"  Genes:        {n_train_genes} train, {n_val_genes} val, {n_test_genes} test")
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
    # Evaluate on VALIDATION set (for early stopping metrics)
    # ------------------------------------------------------------------
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    def _compute_metrics(emb_tensor, lbl_tensor, set_name):
        probs = classifier.predict(emb_tensor).flatten()
        labels_flat = lbl_tensor.numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        try:
            auroc = float(roc_auc_score(labels_flat, probs))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(labels_flat, probs))
        except ValueError:
            auprc = float("nan")

        metrics = {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "accuracy": round(float(accuracy_score(labels_flat, preds)), 4),
            "f1": round(float(f1_score(labels_flat, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(labels_flat, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(labels_flat, preds, zero_division=0)), 4),
        }

        print(f"{set_name} Metrics:")
        print(f"  AUROC:     {metrics['auroc']:.4f}")
        print(f"  AUPRC:     {metrics['auprc']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print()

        return metrics

    print()
    print("=" * 60)
    print("Evaluation")
    print("=" * 60)
    print()

    val_metrics = _compute_metrics(val_emb, val_lbl, "Validation")

    # ------------------------------------------------------------------
    # Evaluate on TEST set (held-out, not used for early stopping)
    # ------------------------------------------------------------------
    test_metrics = {}
    if test_windows:
        test_emb = torch.tensor(np.stack([w[0] for w in test_windows]), dtype=torch.float32)
        test_lbl = torch.tensor(np.stack([w[1] for w in test_windows]), dtype=torch.float32)
        test_metrics = _compute_metrics(test_emb, test_lbl, "Test (held-out)")
    else:
        print("No test set available (use --split-preset spliceai for proper evaluation)")
        print()

    elapsed = time.time() - t0

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
        "n_test_windows": len(test_windows),
        "n_train_genes": n_train_genes,
        "n_val_genes": n_val_genes,
        "n_test_genes": n_test_genes,
        "split_preset": args.split_preset,
        "homology_filter": args.homology_filter,
        "split_seed": args.split_seed,
    }

    print(f"Training:")
    print(f"  Best epoch:     {training_info['best_epoch']}")
    print(f"  Best val AUROC: {training_info['best_val_auroc']:.4f}")
    print(f"  Early stopped:  {training_info['stopped_early']}")
    print(f"  Time:           {elapsed / 60:.1f} min")
    print()

    # Save results
    results = {
        "validation": val_metrics,
        "training": training_info,
    }
    if test_metrics:
        results["test"] = test_metrics

    metrics_path = output_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Artifacts:")
    print(f"  Checkpoint:  {output_dir / 'best_model.pt'}")
    print(f"  Metrics:     {metrics_path}")
    if (output_dir / "gene_split.json").exists():
        print(f"  Gene split:  {output_dir / 'gene_split.json'}")
    print()


if __name__ == "__main__":
    main()
