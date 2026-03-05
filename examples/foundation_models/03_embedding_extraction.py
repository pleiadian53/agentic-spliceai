#!/usr/bin/env python3
"""
Example: Extract Foundation Model Embeddings for Gene Sequences

Loads gene sequences from prepared data, extracts per-nucleotide embeddings
using Evo2 (CUDA) or HyenaDNA (MPS/CPU), and builds exon labels.

Requirements:
    - MANE reference data in data/mane/GRCh38/ (FASTA + GTF)
    - For Evo2: CUDA GPU + ``pip install evo2``
    - For HyenaDNA: Any device (MPS, CPU, CUDA)

Usage:
    # HyenaDNA on local Mac (MPS) — default if no CUDA
    python examples/foundation_models/03_embedding_extraction.py \
        --genes BRCA1 TP53 \
        --model hyenadna \
        --output /tmp/fm_demo/embeddings/

    # Evo2 on GPU machine
    python examples/foundation_models/03_embedding_extraction.py \
        --genes BRCA1 TP53 \
        --model evo2 \
        --output /tmp/fm_demo/embeddings/

    # Evo2 40B (requires A100 80GB+)
    python examples/foundation_models/03_embedding_extraction.py \
        --genes BRCA1 \
        --model evo2 --model-size 40b \
        --output /tmp/fm_demo/embeddings/

    # On remote GPU via SkyPilot
    sky launch foundation_models/configs/skypilot/extract_embeddings_a40.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # picks up HF_TOKEN from .env

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract foundation model embeddings (Evo2 or HyenaDNA).",
    )
    parser.add_argument(
        "--genes", type=str, nargs="+", required=True,
        help="Gene symbols to process (e.g., BRCA1 TP53)",
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output directory for HDF5 embeddings + labels",
    )
    parser.add_argument(
        "--model", type=str, default="auto", choices=["auto", "evo2", "hyenadna"],
        help="Foundation model backend: 'evo2' (CUDA only), 'hyenadna' (any device), "
             "or 'auto' (evo2 if CUDA available, else hyenadna). Default: auto",
    )
    parser.add_argument(
        "--model-size", type=str, default=None,
        help="Model variant. Evo2: '7b' or '40b' (default: 7b). "
             "HyenaDNA: 'tiny-1k', 'small-32k', 'medium-160k', etc. "
             "(default: medium-160k)",
    )
    parser.add_argument(
        "--skip-resource-check", action="store_true",
        help="Skip resource feasibility check (not recommended)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Sequence chunk size (default: 8192)",
    )
    parser.add_argument(
        "--overlap", type=int, default=256,
        help="Chunk overlap for stitching (default: 256)",
    )

    args = parser.parse_args()

    import torch

    # ------------------------------------------------------------------
    # Resolve model backend
    # ------------------------------------------------------------------
    if args.model == "auto":
        backend = "evo2" if torch.cuda.is_available() else "hyenadna"
        print(f"Auto-detected backend: {backend}"
              f" ({'CUDA available' if backend == 'evo2' else 'no CUDA, using HyenaDNA'})")
    else:
        backend = args.model

    if backend == "evo2" and not torch.cuda.is_available():
        print("ERROR: Evo2 requires CUDA but no CUDA device found.")
        print("  Use --model hyenadna for MPS/CPU, or run on a GPU machine.")
        sys.exit(1)

    # Set default model size per backend
    if args.model_size is None:
        args.model_size = "7b" if backend == "evo2" else "medium-160k"

    # ------------------------------------------------------------------
    # Resource check (Evo2 only)
    # ------------------------------------------------------------------
    if backend == "evo2" and not args.skip_resource_check:
        from foundation_models.utils.resources import estimate_embedding_extraction

        print()
        print("Checking resource feasibility...")
        result = estimate_embedding_extraction(
            model_size=args.model_size,
            n_genes=len(args.genes),
        )

        if not result["feasible"]:
            print()
            print("RESOURCE CHECK FAILED")
            print()
            for note in result["notes"]:
                print(f"  {note}")
            print()
            print("Options:")
            print("  1. Use --skip-resource-check to force (may OOM)")
            print("  2. Use --model hyenadna (no GPU required)")
            print("  3. Use a remote GPU:")
            print("     sky launch foundation_models/configs/skypilot/"
                  f"extract_embeddings_{'a100' if args.model_size == '40b' else 'a40'}.yaml")
            sys.exit(1)
        else:
            print(f"  OK: {result['notes'][0]}")
            print()

    # ------------------------------------------------------------------
    # Load gene sequences
    # ------------------------------------------------------------------
    t0 = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = f"Evo2 {args.model_size}" if backend == "evo2" else f"HyenaDNA {args.model_size}"
    print("=" * 60)
    print(f"Extracting {model_label} Embeddings")
    print(f"  Genes: {', '.join(args.genes)}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    from foundation_models.utils.chunking import (
        build_exon_labels,
        chunk_sequence,
        stitch_embeddings,
    )
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        prepare_gene_data,
    )

    # Load gene sequences via the main data pipeline (handles gene lookup,
    # GTF parsing, and FASTA extraction from data/mane/GRCh38/)
    logger.info("Preparing gene sequences for %s", ", ".join(args.genes))
    gene_data = prepare_gene_data(
        genes=args.genes,
        build="GRCh38",
        annotation_source="mane",
    )

    if gene_data.is_empty():
        logger.error("No gene sequences found for %s.", ", ".join(args.genes))
        sys.exit(1)

    logger.info("Loaded %d genes", len(gene_data))
    data_dir = Path("data/mane/GRCh38/")

    # ------------------------------------------------------------------
    # Load foundation model
    # ------------------------------------------------------------------
    if backend == "evo2":
        from foundation_models.evo2 import Evo2Embedder
        logger.info("Loading Evo2 %s...", args.model_size)
        embedder = Evo2Embedder(model_size=args.model_size)
    else:
        from foundation_models.hyenadna import HyenaDNAEmbedder
        logger.info("Loading HyenaDNA %s...", args.model_size)
        embedder = HyenaDNAEmbedder(model_size=args.model_size)

    import h5py
    import numpy as np
    import pandas as pd

    hdf5_path = output_dir / "embeddings.h5"
    labels_dict = {}
    hidden_dim = embedder.model.hidden_dim

    # Pre-load splice site annotations (if available) for exon label building
    splice_tsv = data_dir / "splice_sites_enhanced.tsv"
    splice_sites_df = pd.read_csv(splice_tsv, sep="\t") if splice_tsv.exists() else None

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["model"] = f"{backend}-{args.model_size}"
        hf.attrs["hidden_dim"] = hidden_dim

        for row in gene_data.iter_rows(named=True):
            gene_id = row["gene_id"]
            gene_name = row["gene_name"]
            sequence = row["sequence"]
            gene_start = int(row["start"])
            seq_len = len(sequence)

            logger.info("Processing %s / %s (len=%d)", gene_name, gene_id, seq_len)

            # Chunk the sequence
            chunks = chunk_sequence(
                sequence=sequence,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            # Extract embeddings per chunk
            chunk_embs = []
            for chunk in chunks:
                emb = embedder.encode(chunk.sequence)
                chunk_embs.append(emb)

            # Stitch into full-length embeddings
            full_emb = stitch_embeddings(
                chunks=chunks,
                chunk_embeddings=chunk_embs,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )

            hf.create_dataset(gene_id, data=full_emb, compression="gzip")

            # Build exon labels
            if splice_sites_df is not None:
                labels = build_exon_labels(
                    gene_id=gene_id,
                    gene_start=gene_start,
                    gene_sequence_length=seq_len,
                    splice_sites_df=splice_sites_df,
                )
                labels_dict[gene_id] = labels
            else:
                logger.warning("No splice_sites_enhanced.tsv found; skipping labels")

    # Save labels
    if labels_dict:
        labels_path = hdf5_path.with_suffix(".labels.npz")
        np.savez(labels_path, **labels_dict)
        logger.info("Saved labels: %s", labels_path)

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("Embedding Extraction Complete")
    print("=" * 60)
    print(f"  Genes processed: {len(gene_data)}")
    print(f"  Embeddings:      {hdf5_path}")
    if labels_dict:
        print(f"  Labels:          {labels_path}")
    print(f"  Time:            {elapsed:.1f}s")
    print()
    print("Next step: train classifier")
    print(f"  python examples/foundation_models/04_train_and_evaluate.py \\")
    print(f"      --embeddings {hdf5_path} \\")
    print(f"      --labels {labels_path} \\")
    print(f"      --output {output_dir / 'model'}/")
    print()


if __name__ == "__main__":
    main()
