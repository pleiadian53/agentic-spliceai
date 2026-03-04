#!/usr/bin/env python3
"""
Example: Extract Evo2 Embeddings for Gene Sequences

Loads gene sequences from prepared data, extracts per-nucleotide Evo2
embeddings, and builds exon labels. Includes a resource check that aborts
gracefully if Evo2 won't fit in available memory.

Requirements:
    - Prepared gene data (run: agentic-spliceai-prepare --genes <genes>)
    - Sufficient VRAM for Evo2 (see 01_resource_check.py)

Usage:
    # Check resources first
    python examples/foundation_models/01_resource_check.py --task embedding

    # Extract embeddings for specific genes
    python examples/foundation_models/03_embedding_extraction.py \
        --genes BRCA1 TP53 \
        --output /tmp/fm_demo/embeddings/

    # Use Evo2 40B (requires A100 80GB+)
    python examples/foundation_models/03_embedding_extraction.py \
        --genes BRCA1 \
        --model-size 40b \
        --output /tmp/fm_demo/embeddings/

    # On remote GPU via SkyPilot
    sky launch foundation_models/configs/skypilot/extract_embeddings_a40.yaml
"""

import argparse
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
        description="Extract Evo2 embeddings with resource check.",
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
        "--model-size", type=str, default="7b", choices=["7b", "40b"],
        help="Evo2 model size (default: 7b)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directory with prepared gene data (parquet + TSV). "
             "If omitted, uses default data path from registry.",
    )
    parser.add_argument(
        "--skip-resource-check", action="store_true",
        help="Skip resource feasibility check (not recommended)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=8192,
        help="Sequence chunk size for Evo2 (default: 8192)",
    )
    parser.add_argument(
        "--overlap", type=int, default=256,
        help="Chunk overlap for stitching (default: 256)",
    )

    args = parser.parse_args()

    from foundation_models.utils.resources import estimate_embedding_extraction

    # ------------------------------------------------------------------
    # Resource check
    # ------------------------------------------------------------------
    if not args.skip_resource_check:
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
            print("  2. Use a remote GPU:")
            print("     sky launch foundation_models/configs/skypilot/"
                  f"extract_embeddings_{'a100' if args.model_size == '40b' else 'a40'}.yaml")
            print("  3. Check available profiles:")
            print("     python examples/foundation_models/01_resource_check.py --list-hardware")
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

    print("=" * 60)
    print(f"Extracting Evo2 {args.model_size} Embeddings")
    print(f"  Genes: {', '.join(args.genes)}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    from foundation_models.utils.chunking import (
        build_exon_labels,
        chunk_sequence,
        load_gene_sequences_parquet,
        stitch_embeddings,
    )

    # Load gene sequences
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try default MANE path
        data_dir = Path("data/mane/GRCh38/")
        if not data_dir.exists():
            logger.error(
                "Default data directory not found: %s. "
                "Run: agentic-spliceai-prepare --genes %s",
                data_dir, " ".join(args.genes),
            )
            sys.exit(1)

    logger.info("Loading gene sequences from %s", data_dir)
    gene_data = load_gene_sequences_parquet(
        parquet_path=data_dir / "gene_sequences.parquet",
        gene_symbols=args.genes,
    )

    if not gene_data:
        logger.error("No gene sequences found. Run data preparation first.")
        sys.exit(1)

    logger.info("Loaded %d genes", len(gene_data))

    # ------------------------------------------------------------------
    # Load Evo2 and extract embeddings
    # ------------------------------------------------------------------
    from foundation_models.evo2 import Evo2Embedder

    logger.info("Loading Evo2 %s...", args.model_size)
    embedder = Evo2Embedder(model_size=args.model_size)

    import h5py
    import numpy as np

    hdf5_path = output_dir / "embeddings.h5"
    labels_dict = {}

    with h5py.File(hdf5_path, "w") as hf:
        hf.attrs["model"] = f"evo2-{args.model_size}"
        hf.attrs["hidden_dim"] = embedder.hidden_dim

        for gene_id, seq_info in gene_data.items():
            logger.info("Processing %s (len=%d)", gene_id, len(seq_info["sequence"]))

            # Chunk the sequence
            chunks = chunk_sequence(
                sequence=seq_info["sequence"],
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )

            # Extract embeddings per chunk
            chunk_embeddings = []
            for chunk in chunks:
                emb = embedder.embed(chunk.sequence)
                chunk_embeddings.append(emb)

            # Stitch into full-length embeddings
            full_emb = stitch_embeddings(
                chunk_embeddings=chunk_embeddings,
                chunks=chunks,
                total_length=len(seq_info["sequence"]),
            )

            hf.create_dataset(gene_id, data=full_emb, compression="gzip")

            # Build exon labels
            splice_tsv = data_dir / "splice_sites_enhanced.tsv"
            if splice_tsv.exists():
                labels = build_exon_labels(
                    gene_id=gene_id,
                    seq_length=len(seq_info["sequence"]),
                    splice_tsv_path=splice_tsv,
                )
                labels_dict[gene_id] = labels
            else:
                logger.warning("No splice_sites_enhanced.tsv found; skipping labels for %s", gene_id)

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
