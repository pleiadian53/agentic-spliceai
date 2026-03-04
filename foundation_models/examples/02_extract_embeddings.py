#!/usr/bin/env python3
"""
Example: Extract Evo2 Embeddings for Training

Extract per-nucleotide Evo2 embeddings for a set of genes and save to HDF5.
Supports a --synthetic mode that generates random embeddings matching the
real data shapes, enabling end-to-end pipeline validation without loading
the full Evo2 model (~14 GB).

Usage:
    # Synthetic mode (no Evo2 needed -- local-first testing)
    python examples/02_extract_embeddings.py \
        --synthetic --output /tmp/emb/synth.h5 --n-genes 5

    # Real mode (requires Evo2 model download)
    python examples/02_extract_embeddings.py \
        --data-dir data/mane/GRCh38/ \
        --chromosomes 22 \
        --max-genes 5 \
        --output /tmp/emb/chr22.h5
"""

import argparse
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


def extract_real_embeddings(args: argparse.Namespace) -> None:
    """Extract real Evo2 embeddings from gene sequences."""
    import pandas as pd

    from foundation_models.evo2 import Evo2Config, Evo2Embedder
    from foundation_models.utils.chunking import (
        build_exon_labels,
        load_gene_sequences_parquet,
    )

    data_dir = Path(args.data_dir)

    # Load splice sites for exon labels
    splice_sites_path = data_dir / "splice_sites_enhanced.tsv"
    if not splice_sites_path.exists():
        logger.error(
            "splice_sites_enhanced.tsv not found in %s. "
            "Run: python examples/data_preparation/04_generate_ground_truth.py "
            "--output %s",
            data_dir, data_dir,
        )
        sys.exit(1)

    logger.info("Loading splice sites from %s", splice_sites_path)
    splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")

    # Load gene sequences for requested chromosomes
    all_genes = []
    for chrom in args.chromosomes:
        parquet_path = data_dir / f"gene_sequence_{chrom}.parquet"
        if not parquet_path.exists():
            logger.warning("Parquet not found: %s (skipping)", parquet_path)
            continue
        df = load_gene_sequences_parquet(str(parquet_path))
        all_genes.append(df)
        logger.info("Loaded %d genes from chr%s", len(df), chrom)

    if not all_genes:
        logger.error("No gene sequences found. Check --data-dir and --chromosomes.")
        sys.exit(1)

    genes_df = pd.concat(all_genes, ignore_index=True)
    if args.max_genes and args.max_genes < len(genes_df):
        genes_df = genes_df.head(args.max_genes)
        logger.info("Limiting to %d genes (--max-genes)", args.max_genes)

    # Initialize Evo2
    logger.info("Loading Evo2 model...")
    config = Evo2Config(device="auto", quantize=True)
    embedder = Evo2Embedder(config=config)
    hidden_dim = embedder.model.hidden_dim
    logger.info("Evo2 loaded (hidden_dim=%d)", hidden_dim)

    # Extract embeddings gene by gene
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_dict = {}

    with h5py.File(output_path, "w") as f:
        f.attrs["model_size"] = config.model_size
        f.attrs["hidden_dim"] = hidden_dim

        for idx, row in genes_df.iterrows():
            gene_id = row["gene_id"]
            gene_name = row.get("gene_name", gene_id)
            seq = row["sequence"]

            logger.info(
                "[%d/%d] %s (%s) - %d bp",
                idx + 1, len(genes_df), gene_name, gene_id, len(seq),
            )

            # Extract embedding
            emb = embedder.encode(seq)
            if hasattr(emb, "numpy"):
                emb = emb.numpy()

            f.create_dataset(
                gene_id, data=emb, compression="gzip", compression_opts=4,
            )

            # Build exon labels
            exon_labels = build_exon_labels(
                gene_id=gene_id,
                gene_start=int(row["start"]),
                gene_sequence_length=len(seq),
                splice_sites_df=splice_sites_df,
            )
            labels_dict[gene_id] = exon_labels

    # Save labels
    labels_path = output_path.with_suffix(".labels.npz")
    np.savez_compressed(labels_path, **labels_dict)

    logger.info("Saved embeddings to %s", output_path)
    logger.info("Saved labels to %s", labels_path)


def extract_synthetic_embeddings(args: argparse.Namespace) -> None:
    """Generate synthetic embeddings for pipeline testing."""
    from foundation_models.utils.synthetic import save_synthetic_embeddings

    output_path = Path(args.output)
    hidden_dim = args.hidden_dim or 2560

    logger.info(
        "Generating synthetic embeddings: %d genes, hidden_dim=%d",
        args.n_genes, hidden_dim,
    )

    hdf5_path, labels = save_synthetic_embeddings(
        output_path=output_path,
        n_genes=args.n_genes,
        hidden_dim=hidden_dim,
        seed=args.seed,
    )

    # Print summary
    labels_path = hdf5_path.with_suffix(".labels.npz")
    hdf5_size = hdf5_path.stat().st_size / (1024 * 1024)

    print()
    print("=" * 60)
    print("Synthetic Embeddings Generated")
    print("=" * 60)
    print(f"  Genes:      {args.n_genes}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  HDF5:       {hdf5_path} ({hdf5_size:.1f} MB)")
    print(f"  Labels:     {labels_path}")
    print()

    with h5py.File(hdf5_path, "r") as f:
        for gene_id in f.keys():
            shape = f[gene_id].shape
            exon_frac = labels[gene_id].mean()
            print(f"  {gene_id}: {shape[0]:,} bp, exon fraction={exon_frac:.2f}")

    print()
    print("Next step: train a classifier")
    print(f"  python examples/03_train_classifier.py \\")
    print(f"      --embeddings {hdf5_path} \\")
    print(f"      --labels {labels_path} \\")
    print(f"      --output /tmp/ckpt/ --epochs 20")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract Evo2 embeddings for exon classifier training.",
    )

    # Mode selection
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic embeddings (no Evo2 needed)",
    )

    # Real mode options
    parser.add_argument(
        "--data-dir", type=str,
        help="Data directory with gene_sequence_*.parquet and splice_sites_enhanced.tsv",
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=["22"],
        help="Chromosome numbers to process (default: 22)",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit to N genes for quick testing",
    )

    # Synthetic mode options
    parser.add_argument(
        "--n-genes", type=int, default=5,
        help="Number of synthetic genes to generate (default: 5)",
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=None,
        help="Embedding dimension (default: 2560 for Evo2 7B)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output HDF5 path (labels saved as .labels.npz alongside)",
    )

    args = parser.parse_args()

    t0 = time.time()

    if args.synthetic:
        extract_synthetic_embeddings(args)
    else:
        if not args.data_dir:
            parser.error("--data-dir is required in real mode (or use --synthetic)")
        extract_real_embeddings(args)

    elapsed = time.time() - t0
    logger.info("Done in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
