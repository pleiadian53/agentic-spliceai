#!/usr/bin/env python3
"""
Example: Extract Foundation Model Embeddings for Gene Sequences

Loads gene sequences from prepared data, extracts per-nucleotide embeddings
using Evo2 (CUDA) or HyenaDNA (MPS/CPU), and builds exon labels.

Supports processing specific genes, entire chromosomes, or all MANE genes.
For large runs, outputs per-chromosome HDF5 files and supports resumption.

Requirements:
    - MANE reference data in data/mane/GRCh38/ (FASTA + GTF)
    - For Evo2: CUDA GPU + ``pip install evo2``
    - For HyenaDNA: Any device (MPS, CPU, CUDA)

Usage:
    # Specific genes (quick test)
    python examples/foundation_models/02_embedding_extraction.py \
        --genes BRCA1 TP53 \
        --model evo2 \
        --output /workspace/output/embeddings/

    # Entire chromosome(s)
    python examples/foundation_models/02_embedding_extraction.py \
        --chromosomes 21 22 \
        --model evo2 \
        --output /workspace/output/embeddings/

    # All MANE genes (produces per-chromosome HDF5 files)
    python examples/foundation_models/02_embedding_extraction.py \
        --all-genes \
        --model evo2 \
        --output /workspace/output/embeddings/

    # Resume after interruption (skips already-extracted genes)
    python examples/foundation_models/02_embedding_extraction.py \
        --all-genes --resume \
        --model evo2 \
        --output /workspace/output/embeddings/

    # On remote GPU via SkyPilot
    python examples/foundation_models/ops_run_pipeline.py --execute \
        -- python examples/foundation_models/02_embedding_extraction.py \
             --all-genes --model evo2 --output /workspace/output/embeddings/
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


def _get_splice_sites_df(data_dir: Path) -> "pd.DataFrame | None":
    """Load splice sites annotation if available."""
    import pandas as pd

    splice_tsv = data_dir / "splice_sites_enhanced.tsv"
    splice_parquet = data_dir / "splice_sites_enhanced.parquet"
    if splice_parquet.exists():
        return pd.read_parquet(splice_parquet)
    if splice_tsv.exists():
        return pd.read_csv(splice_tsv, sep="\t")
    return None


def _extract_genes_to_hdf5(
    gene_data: "pl.DataFrame",
    embedder: object,
    hdf5_path: Path,
    data_dir: Path,
    chunk_size: int,
    overlap: int,
    resume: bool,
    backend: str,
    model_size: str,
) -> dict:
    """Extract embeddings for genes and write to an HDF5 file.

    Returns:
        Dict mapping gene_id -> exon labels array (for genes with labels).
    """
    import h5py
    import numpy as np
    import pandas as pd

    from foundation_models.utils.chunking import (
        build_exon_labels,
        chunk_sequence,
        stitch_embeddings,
    )

    hidden_dim = embedder.model.hidden_dim
    splice_sites_df = _get_splice_sites_df(data_dir)
    labels_dict = {}

    # Check which genes are already done (for resume)
    existing_genes = set()
    if resume and hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as hf:
            existing_genes = set(hf.keys())
        if existing_genes:
            logger.info("Resume: %d genes already extracted in %s", len(existing_genes), hdf5_path)

    mode = "a" if resume and hdf5_path.exists() else "w"

    with h5py.File(hdf5_path, mode) as hf:
        if mode == "w":
            hf.attrs["model"] = f"{backend}-{model_size}"
            hf.attrs["hidden_dim"] = hidden_dim

        n_total = len(gene_data)
        n_skipped = 0
        n_processed = 0

        for row in gene_data.iter_rows(named=True):
            gene_id = row["gene_id"]
            gene_name = row["gene_name"]
            sequence = row["sequence"]
            gene_start = int(row["start"])
            seq_len = len(sequence)

            if gene_id in existing_genes:
                n_skipped += 1
                continue

            n_processed += 1
            logger.info(
                "[%d/%d] Processing %s / %s (len=%d)",
                n_processed + n_skipped, n_total, gene_name, gene_id, seq_len,
            )

            chunks = chunk_sequence(
                sequence=sequence,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            chunk_embs = []
            for chunk in chunks:
                emb = embedder.encode(chunk.sequence)
                # Move to CPU immediately to free GPU memory for the next chunk
                if hasattr(emb, 'cpu'):
                    emb = emb.cpu()
                chunk_embs.append(emb)

            full_emb = stitch_embeddings(
                chunks=chunks,
                chunk_embeddings=chunk_embs,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
            )

            hf.create_dataset(gene_id, data=full_emb, compression="gzip")

            # Free GPU memory between genes
            import torch as _torch
            del chunk_embs, full_emb
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

            if splice_sites_df is not None:
                labels = build_exon_labels(
                    gene_id=gene_id,
                    gene_start=gene_start,
                    gene_sequence_length=seq_len,
                    splice_sites_df=splice_sites_df,
                )
                labels_dict[gene_id] = labels

    if n_skipped:
        logger.info("Skipped %d already-extracted genes", n_skipped)
    logger.info("Extracted embeddings for %d genes -> %s", n_processed, hdf5_path)

    return labels_dict


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract foundation model embeddings (Evo2 or HyenaDNA).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Gene selection (mutually exclusive)
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument(
        "--genes", type=str, nargs="+",
        help="Gene symbols to process (e.g., BRCA1 TP53)",
    )
    gene_group.add_argument(
        "--chromosomes", type=str, nargs="+",
        help="Chromosomes to process (e.g., 21 22 X). Outputs per-chromosome HDF5.",
    )
    gene_group.add_argument(
        "--all-genes", action="store_true",
        help="Process all MANE genes (~19K). Outputs per-chromosome HDF5 files.",
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
        "--chunk-size", type=int, default=4096,
        help="Sequence chunk size (default: 4096). Evo2 7b needs ~5 GB/1K tokens "
             "for activations; 4096 fits A40 (48 GB), 8192 needs A100 (80 GB).",
    )
    parser.add_argument(
        "--overlap", type=int, default=256,
        help="Chunk overlap for stitching (default: 256)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip genes already extracted in existing HDF5 files",
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

    if args.model_size is None:
        args.model_size = "7b" if backend == "evo2" else "medium-160k"

    # ------------------------------------------------------------------
    # Determine gene selection mode
    # ------------------------------------------------------------------
    per_chromosome = args.all_genes or args.chromosomes is not None

    # For --all-genes, get list of all chromosomes
    chromosomes = None
    if args.chromosomes:
        chromosomes = args.chromosomes
    elif args.all_genes:
        chromosomes = [str(c) for c in range(1, 23)] + ["X", "Y"]

    # ------------------------------------------------------------------
    # Resource check (Evo2 only)
    # ------------------------------------------------------------------
    if backend == "evo2" and not args.skip_resource_check:
        from foundation_models.utils.resources import estimate_embedding_extraction

        n_genes_est = len(args.genes) if args.genes else (800 if args.chromosomes else 19000)
        print()
        print("Checking resource feasibility...")
        result = estimate_embedding_extraction(
            model_size=args.model_size,
            n_genes=n_genes_est,
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
            if per_chromosome:
                print(f"  Output: per-chromosome HDF5 files (estimated {result['output_hdf5_gb']:.1f} GB total)")
            print()

    # ------------------------------------------------------------------
    # Load foundation model (once, reuse across chromosomes)
    # ------------------------------------------------------------------
    t0 = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_label = f"Evo2 {args.model_size}" if backend == "evo2" else f"HyenaDNA {args.model_size}"
    print("=" * 60)
    print(f"Extracting {model_label} Embeddings")
    if args.genes:
        print(f"  Genes: {', '.join(args.genes)}")
    elif args.chromosomes:
        print(f"  Chromosomes: {', '.join(args.chromosomes)}")
    else:
        print(f"  Mode: all MANE genes (~19K)")
    print(f"  Output: {output_dir}")
    if args.resume:
        print(f"  Resume: enabled (skipping already-extracted genes)")
    print("=" * 60)
    print()

    if backend == "evo2":
        from foundation_models.evo2 import Evo2Embedder
        logger.info("Loading Evo2 %s...", args.model_size)
        embedder = Evo2Embedder(model_size=args.model_size)
    else:
        from foundation_models.hyenadna import HyenaDNAEmbedder
        logger.info("Loading HyenaDNA %s...", args.model_size)
        embedder = HyenaDNAEmbedder(model_size=args.model_size)

    import numpy as np

    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        prepare_gene_data,
    )

    data_dir = Path("data/mane/GRCh38/")
    all_labels = {}

    # ------------------------------------------------------------------
    # Per-chromosome mode: one HDF5 file per chromosome
    # ------------------------------------------------------------------
    if per_chromosome:
        total_genes_all = 0
        total_bytes_all = 0
        chrom_t0 = time.time()

        for chrom_idx, chrom in enumerate(chromosomes, 1):
            chrom_label = chrom if chrom.startswith("chr") else f"chr{chrom}"
            hdf5_path = output_dir / f"embeddings_{chrom_label}.h5"
            labels_path = output_dir / f"embeddings_{chrom_label}.labels.npz"

            logger.info(
                "=== Chromosome %s (%d/%d) ===",
                chrom_label, chrom_idx, len(chromosomes),
            )
            gene_data = prepare_gene_data(
                chromosomes=[chrom],
                build="GRCh38",
                annotation_source="mane",
            )

            if gene_data.is_empty():
                logger.warning("No genes found for %s, skipping", chrom_label)
                continue

            total_genes_all += len(gene_data)
            logger.info("Processing %d genes on %s", len(gene_data), chrom_label)

            labels_dict = _extract_genes_to_hdf5(
                gene_data=gene_data,
                embedder=embedder,
                hdf5_path=hdf5_path,
                data_dir=data_dir,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                resume=args.resume,
                backend=backend,
                model_size=args.model_size,
            )

            if labels_dict:
                # Merge with any existing labels (for resume)
                if args.resume and labels_path.exists():
                    existing = dict(np.load(labels_path))
                    existing.update(labels_dict)
                    labels_dict = existing
                np.savez(labels_path, **labels_dict)
                logger.info("Saved labels: %s", labels_path)

            all_labels.update(labels_dict)

            # Per-chromosome summary
            if hdf5_path.exists():
                h5_size = hdf5_path.stat().st_size
                total_bytes_all += h5_size
                chrom_elapsed = time.time() - chrom_t0
                logger.info(
                    "✓ %s done: %d genes, %.1f GB (cumulative: %d genes, %.1f GB, %.0f min)",
                    chrom_label,
                    len(gene_data),
                    h5_size / (1024**3),
                    total_genes_all,
                    total_bytes_all / (1024**3),
                    chrom_elapsed / 60,
                )

    # ------------------------------------------------------------------
    # Single-file mode: specific genes -> one HDF5 file
    # ------------------------------------------------------------------
    else:
        hdf5_path = output_dir / "embeddings.h5"

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

        labels_dict = _extract_genes_to_hdf5(
            gene_data=gene_data,
            embedder=embedder,
            hdf5_path=hdf5_path,
            data_dir=data_dir,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            resume=args.resume,
            backend=backend,
            model_size=args.model_size,
        )

        if labels_dict:
            labels_path = hdf5_path.with_suffix(".labels.npz")
            np.savez(labels_path, **labels_dict)
            logger.info("Saved labels: %s", labels_path)

        all_labels.update(labels_dict)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("Embedding Extraction Complete")
    print("=" * 60)
    print(f"  Output dir:  {output_dir}")
    print(f"  Time:        {elapsed / 60:.1f} min")

    if per_chromosome:
        h5_files = sorted(output_dir.glob("embeddings_chr*.h5"))
        total_size = sum(f.stat().st_size for f in h5_files)
        print(f"  HDF5 files:  {len(h5_files)} ({total_size / (1024**3):.1f} GB total)")
        for f in h5_files:
            print(f"    {f.name} ({f.stat().st_size / (1024**3):.1f} GB)")
    else:
        hdf5_path = output_dir / "embeddings.h5"
        print(f"  Embeddings:  {hdf5_path}")
        if all_labels:
            print(f"  Labels:      {hdf5_path.with_suffix('.labels.npz')}")

    print()
    print("Next step: train classifier")
    if per_chromosome:
        print(f"  python examples/foundation_models/03_train_and_evaluate.py \\")
        print(f"      --embeddings-dir {output_dir} \\")
        print(f"      --output {output_dir / 'model'}/")
    else:
        print(f"  python examples/foundation_models/03_train_and_evaluate.py \\")
        print(f"      --embeddings {output_dir / 'embeddings.h5'} \\")
        print(f"      --labels {output_dir / 'embeddings.labels.npz'} \\")
        print(f"      --output {output_dir / 'model'}/")
    print()


if __name__ == "__main__":
    main()
