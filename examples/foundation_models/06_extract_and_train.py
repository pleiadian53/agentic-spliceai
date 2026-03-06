#!/usr/bin/env python3
"""
Combined Workflow: Extract Embeddings + Train Classifier

Runs embedding extraction (if not already available) followed by classifier
training and evaluation. Designed for end-to-end GPU pipeline execution.

Bypass logic:
  - If embeddings HDF5 file(s) already exist in the output directory,
    extraction is skipped and training proceeds directly.
  - Use --force-extract to re-extract even if files exist.

Usage:
    # All MANE genes: extract + train (skips extraction if already done)
    python examples/foundation_models/06_extract_and_train.py \
        --all-genes \
        --output /workspace/output/ \
        --model evo2 --model-size 7b \
        --architecture mlp --epochs 50

    # Specific chromosomes
    python examples/foundation_models/06_extract_and_train.py \
        --chromosomes 21 22 \
        --output /workspace/output/ \
        --model evo2

    # Specific genes (quick test)
    python examples/foundation_models/06_extract_and_train.py \
        --genes BRCA1 TP53 BRCA2 RB1 CFTR \
        --output /workspace/output/ \
        --model evo2

    # Force re-extraction
    python examples/foundation_models/06_extract_and_train.py \
        --all-genes --force-extract \
        --output /workspace/output/

    # On remote GPU via the task runner
    python examples/foundation_models/05_run_pipeline.py --execute \
        -- python examples/foundation_models/06_extract_and_train.py \
             --all-genes --output /workspace/output/ \
             --model evo2 --architecture mlp --epochs 50
"""

import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Combined workflow: extract embeddings + train classifier.",
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
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract embeddings even if they exist")
    parser.add_argument("--skip-resource-check", action="store_true")

    # Training
    parser.add_argument("--architecture", type=str, default="mlp",
                        choices=["linear", "mlp", "cnn", "lstm"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-genes", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")

    # Evaluation splitting
    parser.add_argument("--split-preset", type=str, default="auto",
                        choices=["auto", "spliceai", "even_odd", "naive"],
                        help="Gene split strategy (default: auto)")
    parser.add_argument("--homology-filter", action="store_true",
                        help="Remove test genes with paralogs in training")
    parser.add_argument("--split-seed", type=int, default=42)

    # Workflow control
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract embeddings, skip training")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train (embeddings must already exist)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    embeddings_dir = output_dir / "embeddings"
    model_dir = output_dir / "model"

    t0 = time.time()

    print()
    print("=" * 70)
    print("Foundation Model Pipeline: Extract + Train")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 1: Embedding Extraction
    # ------------------------------------------------------------------
    per_chromosome = args.all_genes or args.chromosomes is not None

    # Check if embeddings already exist
    if per_chromosome:
        existing_h5 = sorted(embeddings_dir.glob("embeddings_chr*.h5"))
    else:
        existing_h5 = list(embeddings_dir.glob("embeddings.h5"))

    embeddings_exist = len(existing_h5) > 0
    skip_extraction = embeddings_exist and not args.force_extract and not args.extract_only

    if args.train_only:
        if not embeddings_exist:
            logger.error("--train-only specified but no embeddings found in %s", embeddings_dir)
            sys.exit(1)
        skip_extraction = True

    if skip_extraction:
        print(f"[Step 1/2] SKIP — Embeddings already exist ({len(existing_h5)} files)")
        print(f"  Directory: {embeddings_dir}")
        print(f"  (Use --force-extract to re-extract)")
        print()
    else:
        print(f"[Step 1/2] Extracting embeddings...")
        print()

        extract_cmd = [
            sys.executable, "examples/foundation_models/03_embedding_extraction.py",
            "--output", str(embeddings_dir),
            "--model", args.model,
            "--chunk-size", str(args.chunk_size),
            "--overlap", str(args.overlap),
        ]

        if args.model_size:
            extract_cmd.extend(["--model-size", args.model_size])
        if args.skip_resource_check:
            extract_cmd.append("--skip-resource-check")
        if embeddings_exist and not args.force_extract:
            extract_cmd.append("--resume")

        # Gene selection
        if args.genes:
            extract_cmd.extend(["--genes"] + args.genes)
        elif args.chromosomes:
            extract_cmd.extend(["--chromosomes"] + args.chromosomes)
        else:
            extract_cmd.append("--all-genes")

        logger.info("Running: %s", " ".join(extract_cmd))
        result = subprocess.run(extract_cmd, check=False)
        if result.returncode != 0:
            logger.error("Embedding extraction failed (exit code %d)", result.returncode)
            sys.exit(result.returncode)

    if args.extract_only:
        elapsed = time.time() - t0
        print()
        print(f"Extraction complete ({elapsed / 60:.1f} min). Skipping training (--extract-only).")
        print()
        return

    # ------------------------------------------------------------------
    # Step 2: Train + Evaluate
    # ------------------------------------------------------------------
    print()
    print(f"[Step 2/2] Training classifier...")
    print()

    train_cmd = [
        sys.executable, "examples/foundation_models/04_train_and_evaluate.py",
        "--output", str(model_dir),
        "--architecture", args.architecture,
        "--hidden-dim", str(args.hidden_dim),
        "--num-layers", str(args.num_layers),
        "--dropout", str(args.dropout),
        "--window-size", str(args.window_size),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--patience", str(args.patience),
        "--val-fraction", str(args.val_fraction),
        "--device", args.device,
    ]

    if args.skip_resource_check:
        train_cmd.append("--skip-resource-check")
    if args.max_genes:
        train_cmd.extend(["--max-genes", str(args.max_genes)])

    # Splitting strategy
    train_cmd.extend(["--split-preset", args.split_preset])
    train_cmd.extend(["--split-seed", str(args.split_seed)])
    if args.homology_filter:
        train_cmd.append("--homology-filter")

    # Use --embeddings-dir for per-chromosome, single file otherwise
    if per_chromosome:
        train_cmd.extend(["--embeddings-dir", str(embeddings_dir)])
    else:
        train_cmd.extend([
            "--embeddings", str(embeddings_dir / "embeddings.h5"),
            "--labels", str(embeddings_dir / "embeddings.labels.npz"),
        ])

    logger.info("Running: %s", " ".join(train_cmd))
    result = subprocess.run(train_cmd, check=False)
    if result.returncode != 0:
        logger.error("Training failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
    print(f"  Embeddings: {embeddings_dir}")
    print(f"  Model:      {model_dir}")
    print(f"  Metrics:    {model_dir / 'eval_metrics.json'}")
    print(f"  Total time: {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
