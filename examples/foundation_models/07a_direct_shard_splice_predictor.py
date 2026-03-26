"""
Genome-Scale Splice Site Predictor — Direct-to-Shard Pipeline

Variant of 07_genome_scale_splice_predictor.py that writes embeddings directly
to shard files during extraction, never creating per-gene HDF5 cache.  This
eliminates the disk space problem where gzip-compressed float32 embeddings
still consume ~1.4x the raw size (gzip compresses poorly on dense floats).

Disk usage: only the uncompressed training shards exist on disk.  Each
chromosome's embeddings are extracted gene-by-gene, accumulated in a memory
buffer, and flushed to shard files when the buffer fills (~4 GB).  The buffer
is then freed before processing the next batch of genes.

Three-phase pipeline:
  Phase 1: Extract embeddings → direct to shard files (no per-gene cache)
  Phase 2: Train SpliceClassifier from shards (streaming DataLoader)
  Phase 3: Evaluate on held-out chromosomes (live model inference)

Usage:
    # Mock mode (pipeline validation, no GPU needed)
    python 07a_direct_shard_splice_predictor.py \\
        --mock --n-genes 10 -o /tmp/07a-mock/

    # Multi-chromosome with train/test split
    python 07a_direct_shard_splice_predictor.py \\
        --foundation-model splicebert --chromosomes 1,2,3,22 \\
        -o /workspace/output/splice_classifier/splicebert-multi-chrom/

    # Full genome on GPU pod
    python 07a_direct_shard_splice_predictor.py \\
        --foundation-model splicebert --chromosomes all \\
        -o /workspace/output/genome-splicebert/
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical chromosomes for --chromosomes all
CANONICAL_CHROMS = [str(c) for c in range(1, 23)] + ["X", "Y"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GeneEntry:
    """Metadata for a cached gene."""

    gene_id: str
    gene_name: str
    chrom: str
    start: int
    end: int
    strand: str
    n_windows: int
    n_splice_sites: int
    hdf5_path: str


# ---------------------------------------------------------------------------
# Helpers (reused from 06_dense_splice_predictor.py)
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_chrom_key(fasta: Any, chrom: str) -> Optional[str]:
    """Resolve chromosome name against FASTA index (handles chr prefix)."""
    if chrom in fasta:
        return chrom
    bare = chrom.lstrip("chr") if chrom.startswith("chr") else chrom
    if bare in fasta:
        return bare
    prefixed = f"chr{bare}"
    if prefixed in fasta:
        return prefixed
    return None


def _default_overlap(max_context: int) -> int:
    """Sensible overlap for chunked embedding extraction."""
    return min(128, max_context // 8)


def _normalize_chrom(chrom: str) -> str:
    """Normalize chromosome to bare form (no chr prefix) for comparison."""
    return chrom.lstrip("chr") if chrom.startswith("chr") else chrom


def get_default_window_config(max_context: int) -> Tuple[int, int]:
    """Return (window_size, step_size) based on model max_context."""
    if max_context <= 1024:
        return 512, 256
    elif max_context <= 8192:
        return 4096, 2048
    else:
        return 8192, 4096


# ---------------------------------------------------------------------------
# BPE upsampling (reused from 06)
# ---------------------------------------------------------------------------

def _bpe_to_nucleotide_embeddings(
    model: Any,
    sequence: str,
) -> torch.Tensor:
    """Upsample BPE token embeddings to per-nucleotide via replication."""
    token_emb = model.encode(sequence)
    hidden_dim = token_emb.shape[-1]

    try:
        inputs = model.tokenizer(
            sequence, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = inputs["offset_mapping"][0].tolist()
        offsets = offsets[1:-1]  # strip [CLS], [SEP]
    except Exception:
        n_tokens = token_emb.shape[0]
        seq_len = len(sequence)
        nuc_emb = torch.zeros(seq_len, hidden_dim, dtype=token_emb.dtype,
                              device=token_emb.device)
        for i in range(n_tokens):
            start = int(i * seq_len / n_tokens)
            end = int((i + 1) * seq_len / n_tokens)
            nuc_emb[start:end] = token_emb[i]
        return nuc_emb

    seq_len = len(sequence)
    nuc_emb = torch.zeros(seq_len, hidden_dim, dtype=token_emb.dtype,
                          device=token_emb.device)
    n_mapped = min(len(offsets), token_emb.shape[0])
    for i in range(n_mapped):
        start, end = offsets[i]
        if start < seq_len and end > start:
            end = min(end, seq_len)
            nuc_emb[start:end] = token_emb[i]
    return nuc_emb


# ---------------------------------------------------------------------------
# Dense gene embedding extraction (reused from 06)
# ---------------------------------------------------------------------------

def extract_dense_gene_embeddings(
    model: Any,
    gene_sequence: str,
    max_context: int,
    tokenization: str = "character",
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Extract per-nucleotide embeddings for an entire gene.

    Chunks the gene if it exceeds *max_context*, encodes each chunk,
    stitches back to ``[gene_len, hidden_dim]``.

    Returns:
        ``np.ndarray`` of shape ``[gene_len, hidden_dim]``, float32.
    """
    from foundation_models.utils.chunking import chunk_sequence, stitch_embeddings

    gene_len = len(gene_sequence)
    hidden_dim = model.metadata().hidden_dim
    ovlp = overlap if overlap is not None else _default_overlap(max_context)

    if gene_len <= max_context:
        with torch.no_grad():
            if tokenization == "bpe":
                emb = _bpe_to_nucleotide_embeddings(model, gene_sequence)
            else:
                emb = model.encode(gene_sequence)
        if hasattr(emb, "cpu"):
            emb = emb.detach().cpu().float().numpy()
        if emb.shape[0] >= gene_len:
            return emb[:gene_len]
        padded = np.zeros((gene_len, hidden_dim), dtype=np.float32)
        padded[:emb.shape[0]] = emb
        return padded

    chunks = chunk_sequence(gene_sequence, chunk_size=max_context, overlap=ovlp)
    chunk_embeddings = []

    for chunk in chunks:
        with torch.no_grad():
            if tokenization == "bpe":
                emb = _bpe_to_nucleotide_embeddings(model, chunk.sequence)
            else:
                emb = model.encode(chunk.sequence)
        if hasattr(emb, "cpu"):
            emb = emb.detach().cpu().float().numpy()
        chunk_len = len(chunk.sequence)
        if emb.shape[0] < chunk_len:
            padded = np.zeros((chunk_len, hidden_dim), dtype=np.float32)
            padded[:emb.shape[0]] = emb
            emb = padded
        elif emb.shape[0] > chunk_len:
            emb = emb[:chunk_len]
        chunk_embeddings.append(emb)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stitch_embeddings(chunks, chunk_embeddings, gene_len, hidden_dim)


# ---------------------------------------------------------------------------
# Phase 1: Extract embeddings → direct to shard files (no per-gene cache)
# ---------------------------------------------------------------------------

_DEFAULT_SHARD_MEMORY_BUDGET = 4 * 1024**3  # 4 GB


def _flush_shard(
    emb_buf: np.ndarray,
    lbl_buf: np.ndarray,
    n_filled: int,
    shard_dir: Path,
    split_name: str,
    shard_num: int,
    t0: float,
) -> Path:
    """Write a filled buffer to an uncompressed shard file."""
    shard_path = shard_dir / f"{split_name}_shard_{shard_num:03d}.h5"
    with h5py.File(shard_path, "w") as f:
        f.create_dataset("embeddings", data=emb_buf[:n_filled])
        f.create_dataset("labels", data=lbl_buf[:n_filled])
    elapsed = time.time() - t0
    logger.info(
        "  Flushed %s_%03d: %d windows (%.1f s)",
        split_name, shard_num, n_filled, elapsed,
    )
    return shard_path


def extract_to_shards(
    chromosomes: List[str],
    gene_split: Dict[str, str],
    train_genes: Set[str],
    val_genes: Set[str],
    test_genes: Set[str],
    model_name: str,
    model_kwargs: dict,
    max_context: int,
    tokenization: str,
    hidden_dim: int,
    window_size: int,
    step_size: int,
    output_dir: Path,
    build: str = "GRCh38_MANE",
    max_gene_length: int = 200_000,
    shard_memory_budget: int = _DEFAULT_SHARD_MEMORY_BUDGET,
) -> List[GeneEntry]:
    """Extract embeddings and write directly to shard files.

    Never writes per-gene HDF5 cache.  Windows are accumulated in a memory
    buffer (~4 GB) and flushed to shard files when full.  Peak disk usage
    is only the uncompressed shard files themselves.

    Train and val windows go to separate shard series (``train_shard_*.h5``,
    ``val_shard_*.h5``).  Test gene embeddings are NOT stored — Phase 3
    evaluation runs the model live.

    Returns manifest of all processed genes (for split bookkeeping).
    """
    import pandas as pd
    import polars as pl
    from pyfaidx import Fasta

    from foundation_models.base import load_embedding_model
    from foundation_models.utils.chunking import build_splice_labels

    # --- Resolve data paths ---
    release = "1.3" if "MANE" in build.upper() else None
    try:
        from agentic_spliceai.splice_engine.resources import get_genomic_registry
        registry = get_genomic_registry(build=build, release=release)
        gtf_path = str(registry.get_gtf_path(validate=True))
        fasta_path = str(registry.get_fasta_path(validate=True))
        data_dir = Path(fasta_path).parent
    except Exception as exc:
        logger.error("Cannot resolve data paths for build=%s: %s", build, exc)
        sys.exit(1)

    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        load_gene_annotations,
    )

    # --- Load splice sites once ---
    splice_sites_df = None
    for fname in ("splice_sites_enhanced.parquet", "splice_sites_enhanced.tsv"):
        path = data_dir / fname
        if path.exists():
            if fname.endswith(".parquet"):
                splice_sites_df = pd.read_parquet(path)
            else:
                splice_sites_df = pd.read_csv(path, sep="\t")
            logger.info("Loaded splice sites from %s (%d rows)", path, len(splice_sites_df))
            break

    if splice_sites_df is None:
        logger.error("Splice site annotations not found in %s", data_dir)
        sys.exit(1)

    # --- Load foundation model once ---
    logger.info("Loading foundation model '%s'...", model_name)
    model = load_embedding_model(model_name, **model_kwargs)
    meta = model.metadata()
    logger.info("Model loaded: %s (hidden_dim=%d)", meta.name, meta.hidden_dim)

    # --- Open FASTA ---
    fasta = Fasta(fasta_path)
    logger.info("Using FASTA: %s", fasta_path)

    # --- Prepare shard buffers ---
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    bytes_per_window = window_size * hidden_dim * 4  # float32
    buf_capacity = max(100, shard_memory_budget // bytes_per_window)
    buf_mem_gb = buf_capacity * bytes_per_window / 1e9
    logger.info(
        "Shard buffer: %d windows (%.1f GB), window=%d, hidden=%d",
        buf_capacity, buf_mem_gb, window_size, hidden_dim,
    )

    # Separate buffers for train and val
    train_emb_buf = np.empty((buf_capacity, window_size, hidden_dim), dtype=np.float32)
    train_lbl_buf = np.empty((buf_capacity, window_size), dtype=np.int8)
    train_filled = 0
    train_shard_num = 0

    val_emb_buf = np.empty((buf_capacity, window_size, hidden_dim), dtype=np.float32)
    val_lbl_buf = np.empty((buf_capacity, window_size), dtype=np.int8)
    val_filled = 0
    val_shard_num = 0

    t0 = time.time()

    # --- Process each chromosome ---
    manifest: List[GeneEntry] = []
    total_genes = 0
    total_windows = 0
    total_splice = 0
    total_skipped = 0

    for chrom in chromosomes:
        chrom_t0 = time.time()
        chrom_bare = _normalize_chrom(chrom)
        chrom_chr = f"chr{chrom_bare}" if not chrom.startswith("chr") else chrom

        gene_data = load_gene_annotations(
            gtf_path=gtf_path, chromosomes=[chrom_chr], verbosity=0,
        )
        if gene_data.height == 0:
            gene_data = load_gene_annotations(
                gtf_path=gtf_path, chromosomes=[chrom_bare], verbosity=0,
            )
        if gene_data.height == 0:
            logger.warning("  %s: no genes found, skipping", chrom_chr)
            continue

        gene_data = gene_data.with_columns(
            (pl.col("end") - pl.col("start")).alias("gene_length"),
        ).filter(pl.col("gene_length") <= max_gene_length)

        chrom_genes = 0
        chrom_windows = 0
        chrom_splice = 0
        n_chrom_genes = gene_data.height

        for gene_idx, row in enumerate(gene_data.iter_rows(named=True), 1):
            gene_id = row["gene_id"]
            gene_name = row["gene_name"]
            g_chrom = str(row.get("chrom") or row.get("seqname", ""))
            start = int(row["start"])
            end = int(row["end"])
            strand = str(row.get("strand", "+"))
            gene_len = end - start

            # Determine split assignment
            if gene_id in test_genes:
                split_label = "test"
            elif gene_id in val_genes:
                split_label = "val"
            elif gene_id in train_genes:
                split_label = "train"
            else:
                # Gene not in any split (shouldn't happen but skip gracefully)
                total_skipped += 1
                continue

            # Resolve chromosome in FASTA
            chrom_key = _resolve_chrom_key(fasta, g_chrom)
            if chrom_key is None:
                total_skipped += 1
                continue

            gene_seq = str(fasta[chrom_key][start:end]).upper()
            if len(gene_seq) < window_size:
                total_skipped += 1
                continue

            # Build splice labels
            labels = build_splice_labels(
                gene_id=gene_id,
                gene_start=start,
                gene_sequence_length=len(gene_seq),
                splice_sites_df=splice_sites_df,
            )
            n_splice = int((labels > 0).sum())

            # Skip embedding extraction for test genes (Phase 3 runs model live)
            if split_label == "test":
                manifest.append(GeneEntry(
                    gene_id=gene_id, gene_name=gene_name, chrom=chrom_chr,
                    start=start, end=end, strand=strand,
                    n_windows=0, n_splice_sites=n_splice,
                    hdf5_path="",  # No cache — Phase 3 runs live
                ))
                chrom_genes += 1
                chrom_splice += n_splice
                del labels, gene_seq
                continue

            # Extract dense embeddings
            gene_emb = extract_dense_gene_embeddings(
                model=model,
                gene_sequence=gene_seq,
                max_context=max_context,
                tokenization=tokenization,
                overlap=None,
            )

            # Window and write to buffer
            n_gene_windows = 0
            pos = 0
            while pos + window_size <= len(gene_seq):
                win_emb = gene_emb[pos:pos + window_size]
                win_lbl = labels[pos:pos + window_size]

                if split_label == "train":
                    train_emb_buf[train_filled] = win_emb
                    train_lbl_buf[train_filled] = win_lbl
                    train_filled += 1
                    if train_filled >= buf_capacity:
                        _flush_shard(
                            train_emb_buf, train_lbl_buf, train_filled,
                            shard_dir, "train", train_shard_num, t0,
                        )
                        train_shard_num += 1
                        train_filled = 0
                else:  # val
                    val_emb_buf[val_filled] = win_emb
                    val_lbl_buf[val_filled] = win_lbl
                    val_filled += 1
                    if val_filled >= buf_capacity:
                        _flush_shard(
                            val_emb_buf, val_lbl_buf, val_filled,
                            shard_dir, "val", val_shard_num, t0,
                        )
                        val_shard_num += 1
                        val_filled = 0

                n_gene_windows += 1
                pos += step_size

            if gene_idx % 10 == 1 or gene_idx == n_chrom_genes:
                logger.info(
                    "  %s [%d/%d] %s: %d windows, %d splice sites (%d bp) [%s]",
                    chrom_chr, gene_idx, n_chrom_genes, gene_name,
                    n_gene_windows, n_splice, gene_len, split_label,
                )

            manifest.append(GeneEntry(
                gene_id=gene_id, gene_name=gene_name, chrom=chrom_chr,
                start=start, end=end, strand=strand,
                n_windows=n_gene_windows, n_splice_sites=n_splice,
                hdf5_path="",  # No per-gene cache
            ))

            chrom_genes += 1
            chrom_windows += n_gene_windows
            chrom_splice += n_splice

            # Free memory
            del gene_emb, labels, gene_seq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_genes += chrom_genes
        total_windows += chrom_windows
        total_splice += chrom_splice

        chrom_elapsed = time.time() - chrom_t0
        logger.info(
            "  %s: %d genes, %d windows, %d splice sites (%.1f min)",
            chrom_chr, chrom_genes, chrom_windows, chrom_splice,
            chrom_elapsed / 60,
        )

        # Save manifest incrementally after each chromosome so --resume can
        # recover after a mid-run crash (e.g., disk-full during a later chrom).
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as _mf:
            json.dump([asdict(e) for e in manifest], _mf, indent=2)

    fasta.close()

    # Flush remaining buffers
    if train_filled > 0:
        _flush_shard(
            train_emb_buf, train_lbl_buf, train_filled,
            shard_dir, "train", train_shard_num, t0,
        )
    if val_filled > 0:
        _flush_shard(
            val_emb_buf, val_lbl_buf, val_filled,
            shard_dir, "val", val_shard_num, t0,
        )

    # Free buffers
    del train_emb_buf, train_lbl_buf, val_emb_buf, val_lbl_buf

    logger.info(
        "Extraction complete: %d genes, %d windows, %d splice sites (%d skipped)",
        total_genes, total_windows, total_splice, total_skipped,
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([asdict(e) for e in manifest], f, indent=2)
    logger.info("Manifest saved: %s (%d genes)", manifest_path, len(manifest))

    return manifest


# ---------------------------------------------------------------------------
# Phase 1 (mock): synthetic data for pipeline validation
# ---------------------------------------------------------------------------

def extract_mock_embeddings(
    n_genes: int,
    hidden_dim: int,
    window_size: int,
    step_size: int,
    output_dir: Path,
    seed: int = 42,
) -> List[GeneEntry]:
    """Generate synthetic data and save to HDF5 cache (same format as real)."""
    from foundation_models.utils.synthetic import generate_synthetic_splice_data

    cache_dir = output_dir / "cache"
    emb_dict, lbl_dict = generate_synthetic_splice_data(
        n_genes=n_genes, hidden_dim=hidden_dim, seed=seed,
    )

    manifest: List[GeneEntry] = []
    # Assign synthetic genes to chromosomes for split testing
    chroms = [f"chr{(i % 22) + 1}" for i in range(n_genes)]

    for idx, gene_id in enumerate(sorted(emb_dict.keys())):
        emb = emb_dict[gene_id]
        lbl = lbl_dict[gene_id]
        chrom = chroms[idx]

        chrom_dir = cache_dir / chrom
        chrom_dir.mkdir(parents=True, exist_ok=True)

        # Window
        win_embs, win_lbls = [], []
        pos = 0
        while pos + window_size <= len(lbl):
            win_embs.append(emb[pos:pos + window_size])
            win_lbls.append(lbl[pos:pos + window_size])
            pos += step_size

        if not win_embs:
            continue

        emb_arr = np.stack(win_embs, axis=0)
        lbl_arr = np.stack(win_lbls, axis=0)
        n_splice = int((lbl > 0).sum())

        h5_path = chrom_dir / f"{gene_id}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("embeddings", data=emb_arr)
            f.create_dataset("labels", data=lbl_arr)
            f.attrs["gene_id"] = gene_id
            f.attrs["gene_name"] = gene_id
            f.attrs["chrom"] = chrom
            f.attrs["start"] = 0
            f.attrs["end"] = len(lbl)
            f.attrs["strand"] = "+"
            f.attrs["n_splice_sites"] = n_splice
            f.attrs["hidden_dim"] = emb.shape[1]
            f.attrs["window_size"] = window_size

        manifest.append(GeneEntry(
            gene_id=gene_id, gene_name=gene_id, chrom=chrom,
            start=0, end=len(lbl), strand="+",
            n_windows=len(win_embs), n_splice_sites=n_splice,
            hdf5_path=str(h5_path),
        ))

    logger.info(
        "Mock data: %d genes, %d windows cached",
        len(manifest), sum(e.n_windows for e in manifest),
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump([asdict(e) for e in manifest], f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# HDF5 Window Dataset
# ---------------------------------------------------------------------------

from foundation_models.data import (
    HDF5WindowDataset,
    ShardedWindowDataset,
    repack_into_shards,
)


# ---------------------------------------------------------------------------
# Phase 2: Train
# ---------------------------------------------------------------------------

def train_classifier(
    manifest: List[GeneEntry],
    train_genes: Set[str],
    val_genes: Set[str],
    input_dim: int,
    output_dir: Path,
    architecture: str = "dilated_cnn",
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 64,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 20,
    focal_gamma: float = 2.0,
    use_shards: bool = False,
    seed: int = 42,
) -> Any:
    """Train SpliceClassifier on cached embeddings via streaming DataLoaders.

    Never loads the full dataset into RAM — streams batches from HDF5 cache.
    When ``use_shards=True``, packs per-gene HDF5 files into large contiguous
    shard files first for 10-50x faster I/O.
    """
    from torch.utils.data import DataLoader

    from foundation_models.classifiers.losses import compute_class_weights_from_counts
    from foundation_models.classifiers.splice_classifier import SpliceClassifier

    device = get_device()
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Build datasets — use shards if requested, otherwise per-gene HDF5
    train_dataset: torch.utils.data.Dataset
    val_dataset: Optional[torch.utils.data.Dataset] = None

    shard_dir = output_dir / "shards"
    if use_shards:
        # Check for existing shards first.  Supports both standard naming
        # (train_shard_000.h5) and per-chrom naming from --stream-chroms
        # (train_chr1_shard_000.h5).
        existing_train = sorted(shard_dir.glob("train*shard_*.h5"))
        existing_val = sorted(shard_dir.glob("val*shard_*.h5"))

        if existing_train:
            logger.info("Using existing shards from %s", shard_dir)
        else:
            logger.info("Packing training shards...")
            repack_into_shards(
                manifest, train_genes, output_dir, "train", seed=seed,
            )
            if val_genes:
                logger.info("Packing validation shards...")
                repack_into_shards(
                    manifest, val_genes, output_dir, "val", seed=seed,
                )
            existing_train = sorted(shard_dir.glob("train*shard_*.h5"))
            existing_val = sorted(shard_dir.glob("val*shard_*.h5"))

        logger.info("Building training dataset (sharded)...")
        train_dataset = ShardedWindowDataset(existing_train)
        if existing_val:
            logger.info("Building validation dataset (sharded)...")
            val_dataset = ShardedWindowDataset(existing_val)
    else:
        logger.info("Building training dataset...")
        train_dataset = HDF5WindowDataset(manifest, train_genes)
        if val_genes:
            logger.info("Building validation dataset...")
            val_dataset = HDF5WindowDataset(manifest, val_genes)

    # DataLoaders — ShardedWindowDataset is fork-safe (lazy per-PID handles),
    # so num_workers=2 allows prefetching to overlap with GPU compute.
    # HDF5WindowDataset uses an LRU handle cache that is NOT fork-safe, so
    # num_workers=0 is kept for that path.
    cuda_available = torch.cuda.is_available()
    _nw = 2 if (use_shards and cuda_available) else 0
    # Use "spawn" context (not default "fork") to avoid Polars/h5py deadlocks
    # when forking workers after Polars thread pools have been initialized in
    # Phase 1.  spawn is slightly slower to start but safe on Linux.
    _mp_ctx = "spawn" if _nw > 0 else None
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=_nw, pin_memory=cuda_available,
        persistent_workers=(_nw > 0),
        multiprocessing_context=_mp_ctx,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=_nw, pin_memory=cuda_available,
            persistent_workers=(_nw > 0),
            multiprocessing_context=_mp_ctx,
        )

    # Class weights from label scan (no full array in memory)
    class_weights = compute_class_weights_from_counts(train_dataset.class_counts)

    n_train_splice = int(train_dataset.class_counts[1] + train_dataset.class_counts[2])
    n_val_splice = 0
    if val_dataset is not None:
        n_val_splice = int(val_dataset.class_counts[1] + val_dataset.class_counts[2])

    classifier = SpliceClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
    )
    n_params = sum(p.numel() for p in classifier.parameters())

    print()
    print("=" * 60)
    print("Training SpliceClassifier (genome-scale, streaming)")
    print("=" * 60)
    print(f"  Architecture: {architecture}")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Train:        {len(train_dataset)} windows ({n_train_splice} splice sites)")
    if val_dataset is not None:
        print(f"  Val:          {len(val_dataset)} windows ({n_val_splice} splice sites)")
    print(f"  Device:       {device}")
    print(f"  Focal gamma:  {focal_gamma}")
    print(f"  Hyperparams:  lr={lr}, batch={batch_size}, wd={weight_decay}")
    print("=" * 60)
    print()

    history = classifier.fit_streaming(
        train_loader,
        val_loader,
        class_weights=class_weights,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=str(model_dir),
        patience=patience,
        focal_gamma=focal_gamma,
    )

    # Save training history
    history_path = model_dir / "training_history.json"
    serializable = {
        k: v for k, v in history.items()
        if isinstance(v, (list, int, float, str, bool))
    }
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # Clean up training file handles (keep val alive for calibration)
    train_dataset.close()

    return classifier, val_dataset, val_loader


# ---------------------------------------------------------------------------
# Phase 3: Evaluate with SpliceAI-standard metrics
# ---------------------------------------------------------------------------

def _compute_cls_metrics(
    true_flat: np.ndarray,
    scores: np.ndarray,
    cls_idx: int,
) -> Dict[str, Any]:
    """Compute AUROC/AUPRC/F1 for a single splice class."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = (true_flat == cls_idx).astype(np.int32)
    n_positive = int(y_true.sum())
    n_total = len(y_true)

    if n_positive > 0 and n_positive < n_total:
        auroc = float(roc_auc_score(y_true, scores))
        auprc = float(average_precision_score(y_true, scores))
    else:
        auroc = float("nan")
        auprc = float("nan")

    pred_positive = (scores >= 0.5).astype(np.int32)
    tp = int(((pred_positive == 1) & (y_true == 1)).sum())
    fp = int(((pred_positive == 1) & (y_true == 0)).sum())
    fn = int(((pred_positive == 0) & (y_true == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "auroc": auroc, "auprc": auprc,
        "precision": precision, "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "n_positive": n_positive, "n_total": n_total,
    }


def _evaluate_gene(
    entry: GeneEntry,
    classifier: Any,
    device: str,
    window_size: int,
    step_size: int,
    *,
    _model: Any = None,
    _fasta: Any = None,
    _splice_sites_df: Any = None,
    max_context: int = 512,
    tokenization: str = "nucleotide",
) -> Optional[tuple]:
    """Evaluate a single gene and return (true, donor_scores, acceptor_scores).

    Returns None if the gene should be skipped (too short, missing from FASTA, etc.).
    """
    if entry.hdf5_path:
        # Mock mode or cached embeddings
        with h5py.File(entry.hdf5_path, "r") as f:
            emb = f["embeddings"][:]
            lbl = f["labels"][:]
        emb_t = torch.tensor(emb, dtype=torch.float32).to(device)
        preds = classifier.predict(emb_t)
        return (lbl.flatten(), preds["donor_prob"].flatten(), preds["acceptor_prob"].flatten())

    # Live inference path
    from foundation_models.utils.chunking import build_splice_labels

    chrom_key = _resolve_chrom_key(_fasta, entry.chrom)
    if chrom_key is None:
        return None

    gene_seq = str(_fasta[chrom_key][entry.start:entry.end]).upper()
    gene_len = len(gene_seq)
    if gene_len < window_size:
        return None

    labels = build_splice_labels(
        gene_id=entry.gene_id,
        gene_start=entry.start,
        gene_sequence_length=gene_len,
        splice_sites_df=_splice_sites_df,
    )

    gene_emb = extract_dense_gene_embeddings(
        model=_model,
        gene_sequence=gene_seq,
        max_context=max_context,
        tokenization=tokenization,
        overlap=None,
    )

    gene_true, gene_donor, gene_acceptor = [], [], []
    pos = 0
    while pos + window_size <= gene_len:
        win_emb = gene_emb[pos:pos + window_size]
        win_lbl = labels[pos:pos + window_size]
        emb_t = torch.tensor(win_emb[np.newaxis], dtype=torch.float32).to(device)
        preds = classifier.predict(emb_t)
        gene_true.append(win_lbl)
        gene_donor.append(preds["donor_prob"].flatten())
        gene_acceptor.append(preds["acceptor_prob"].flatten())
        pos += step_size

    del gene_emb, gene_seq, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not gene_true:
        return None

    return (
        np.concatenate(gene_true),
        np.concatenate(gene_donor),
        np.concatenate(gene_acceptor),
    )


def evaluate_on_test_set(
    classifier: Any,
    manifest: List[GeneEntry],
    test_genes: Set[str],
    output_dir: Path,
    model_name: str = "",
    model_kwargs: Optional[Dict] = None,
    build: str = "GRCh38_MANE",
    window_size: int = 512,
    step_size: int = 256,
    max_context: int = 512,
    tokenization: str = "nucleotide",
) -> Dict:
    """Evaluate trained classifier on test chromosomes.

    Processes one chromosome at a time to bound memory usage. Each chromosome's
    predictions are concatenated, metrics computed, and then freed before moving
    to the next chromosome. Per-chromosome and aggregate metrics are saved.

    For real-data runs (07a), test genes have ``hdf5_path=""`` because Phase 1
    never writes per-gene cache.  This function runs the foundation model **live**
    on each test gene (load FASTA → encode → classify window-by-window).

    For mock runs, test genes have valid ``hdf5_path`` from ``extract_mock_embeddings``,
    so they are read from the cached HDF5 files (no model reload needed).
    """
    device = get_device()
    classifier.to(device)
    classifier.eval()

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Determine if any test genes require live inference (hdf5_path == "")
    test_entries = [e for e in manifest if e.gene_id in test_genes]
    needs_live_inference = any(not e.hdf5_path for e in test_entries)

    # Lazy-load live inference resources only when needed
    _model = None
    _fasta = None
    _splice_sites_df = None

    if needs_live_inference and model_name:
        import pandas as pd
        from pyfaidx import Fasta
        from foundation_models.base import load_embedding_model

        release = "1.3" if "MANE" in build.upper() else None
        try:
            from agentic_spliceai.splice_engine.resources import get_genomic_registry
            registry = get_genomic_registry(build=build, release=release)
            _fasta_path = str(registry.get_fasta_path(validate=True))
            data_dir = Path(_fasta_path).parent
        except Exception as exc:
            logger.error("Phase 3: Cannot resolve data paths for build=%s: %s", build, exc)
            return {}

        for fname in ("splice_sites_enhanced.parquet", "splice_sites_enhanced.tsv"):
            path = data_dir / fname
            if path.exists():
                if fname.endswith(".parquet"):
                    _splice_sites_df = pd.read_parquet(path)
                else:
                    _splice_sites_df = pd.read_csv(path, sep="\t")
                logger.info("Phase 3: Loaded splice sites (%d rows)", len(_splice_sites_df))
                break

        if _splice_sites_df is None:
            logger.error("Phase 3: Splice site annotations not found — skipping evaluation")
            return {}

        logger.info("Phase 3: Loading foundation model '%s' for live inference...", model_name)
        _model = load_embedding_model(model_name, **(model_kwargs or {}))
        _fasta = Fasta(_fasta_path)
        logger.info("Phase 3: Model loaded, FASTA opened (%s)", _fasta_path)

    elif needs_live_inference and not model_name:
        logger.error(
            "Phase 3: Test genes have no cached embeddings but model_name is empty. "
            "Cannot run live inference — skipping evaluation."
        )
        return {}

    # --- Per-chromosome evaluation (memory-bounded) ---
    # Group test entries by chromosome
    chrom_groups: Dict[str, List[GeneEntry]] = {}
    for entry in test_entries:
        chrom_groups.setdefault(entry.chrom, []).append(entry)

    # Natural sort chromosomes (chr1, chr3, chr5, chr7, chr9)
    sorted_chroms = sorted(
        chrom_groups.keys(),
        key=lambda c: (int(c.lstrip("chrXYMT")) if c.lstrip("chr").isdigit() else 999, c),
    )

    per_chrom_results: Dict[str, Dict] = {}
    total_genes = 0
    total_positions = 0

    for chrom in sorted_chroms:
        entries = chrom_groups[chrom]
        chrom_true, chrom_donor, chrom_acceptor = [], [], []
        chrom_genes = 0
        t_chrom = time.time()

        for entry in entries:
            result = _evaluate_gene(
                entry, classifier, device, window_size, step_size,
                _model=_model, _fasta=_fasta,
                _splice_sites_df=_splice_sites_df,
                max_context=max_context, tokenization=tokenization,
            )
            if result is None:
                continue

            gene_true, gene_donor, gene_acceptor = result
            chrom_true.append(gene_true)
            chrom_donor.append(gene_donor)
            chrom_acceptor.append(gene_acceptor)
            chrom_genes += 1

            if chrom_genes % 50 == 0:
                logger.info(
                    "  Phase 3 [%s]: %d / %d genes...",
                    chrom, chrom_genes, len(entries),
                )

        if not chrom_true:
            logger.warning("  Phase 3 [%s]: no evaluable genes — skipping", chrom)
            continue

        # Concatenate and compute metrics for this chromosome
        true_flat = np.concatenate(chrom_true)
        donor_flat = np.concatenate(chrom_donor)
        acceptor_flat = np.concatenate(chrom_acceptor)

        # Free the per-gene arrays immediately
        del chrom_true, chrom_donor, chrom_acceptor

        chrom_metrics: Dict[str, Any] = {
            "n_genes": chrom_genes,
            "n_positions": len(true_flat),
        }
        for cls_name, cls_idx, scores in [
            ("acceptor", 1, acceptor_flat),
            ("donor", 2, donor_flat),
        ]:
            chrom_metrics[cls_name] = _compute_cls_metrics(true_flat, scores, cls_idx)

        # Mean across splice classes
        for metric in ("auroc", "auprc", "f1"):
            vals = [
                chrom_metrics[c][metric] for c in ("acceptor", "donor")
                if not (isinstance(chrom_metrics[c][metric], float)
                        and np.isnan(chrom_metrics[c][metric]))
            ]
            chrom_metrics[f"mean_{metric}"] = (
                float(np.mean(vals)) if vals else float("nan")
            )

        per_chrom_results[chrom] = chrom_metrics
        total_genes += chrom_genes
        total_positions += len(true_flat)

        elapsed_chrom = time.time() - t_chrom
        logger.info(
            "  Phase 3 [%s]: %d genes, %d positions, "
            "AUPRC=%.4f (acc=%.4f, don=%.4f) — %.1f min",
            chrom, chrom_genes, len(true_flat),
            chrom_metrics["mean_auprc"],
            chrom_metrics["acceptor"]["auprc"],
            chrom_metrics["donor"]["auprc"],
            elapsed_chrom / 60,
        )

        # Save per-chromosome metrics incrementally (survives crashes)
        with open(eval_dir / f"test_metrics_{chrom}.json", "w") as f:
            json.dump(chrom_metrics, f, indent=2, default=str)

        # Free chromosome arrays
        del true_flat, donor_flat, acceptor_flat
        import gc
        gc.collect()

    if _fasta is not None:
        _fasta.close()

    if not per_chrom_results:
        logger.warning("No test genes found — skipping evaluation")
        return {}

    # --- Aggregate results across chromosomes ---
    results: Dict[str, Any] = {
        "n_test_genes": total_genes,
        "n_test_positions": total_positions,
        "per_chromosome": per_chrom_results,
    }

    # Weighted-average aggregate metrics (weighted by n_positions per chromosome)
    for cls_name in ("acceptor", "donor"):
        agg: Dict[str, float] = {}
        for metric in ("auroc", "auprc", "precision", "recall", "f1"):
            weighted_sum = 0.0
            weight_sum = 0
            for _chrom, cm in per_chrom_results.items():
                if cls_name not in cm:
                    continue
                val = cm[cls_name][metric]
                if isinstance(val, float) and np.isnan(val):
                    continue
                w = cm["n_positions"]
                weighted_sum += val * w
                weight_sum += w
            agg[metric] = weighted_sum / weight_sum if weight_sum > 0 else float("nan")

        # Aggregate counts
        agg["tp"] = sum(cm.get(cls_name, {}).get("tp", 0) for cm in per_chrom_results.values())
        agg["fp"] = sum(cm.get(cls_name, {}).get("fp", 0) for cm in per_chrom_results.values())
        agg["fn"] = sum(cm.get(cls_name, {}).get("fn", 0) for cm in per_chrom_results.values())
        agg["n_positive"] = sum(
            cm.get(cls_name, {}).get("n_positive", 0) for cm in per_chrom_results.values()
        )
        agg["n_total"] = total_positions
        results[cls_name] = agg

    for metric in ("auroc", "auprc", "f1"):
        vals = [
            results[c][metric] for c in ("acceptor", "donor")
            if not (isinstance(results[c][metric], float) and np.isnan(results[c][metric]))
        ]
        results[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")

    # Save aggregate
    with open(eval_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print
    print()
    print("Test Metrics (held-out chromosomes):")
    print(f"  Genes: {total_genes}, Positions: {total_positions:,}")
    print()
    print("  Per-chromosome breakdown:")
    for chrom in sorted_chroms:
        if chrom not in per_chrom_results:
            continue
        cm = per_chrom_results[chrom]
        print(f"    {chrom:6s}: {cm['n_genes']:4d} genes, {cm['n_positions']:>10,} pos, "
              f"AUPRC={cm['mean_auprc']:.4f} "
              f"(acc={cm['acceptor']['auprc']:.4f}, don={cm['donor']['auprc']:.4f})")
    print()
    print("  Aggregate (position-weighted):")
    for cls_name in ("acceptor", "donor"):
        m = results[cls_name]
        print(f"    {cls_name:10s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, "
              f"F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}) "
              f"[{m['n_positive']} sites]")
    print(f"    {'mean':10s}: AUROC={results['mean_auroc']:.4f}, "
          f"AUPRC={results['mean_auprc']:.4f}, F1={results['mean_f1']:.4f}")
    print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from foundation_models.base import get_model_metadata, list_available_models

    available_models = list_available_models()

    parser = argparse.ArgumentParser(
        description="Genome-scale splice site predictor using foundation model "
                    "embeddings with SpliceAI chromosome-split evaluation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument("--foundation-model", type=str, default="splicebert",
                        choices=available_models,
                        help=f"Foundation model (default: splicebert)")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Model variant (e.g., 7b, 40b, medium-160k)")

    # Data
    parser.add_argument("--chromosomes", nargs="+", type=str, default=None,
                        help="Chromosomes to process (e.g., 22, or 'all')")
    parser.add_argument("--split", type=str, default="spliceai",
                        choices=["spliceai", "even_odd", "balanced", "custom"],
                        help=(
                            "Chromosome split preset (default: spliceai). "
                            "'balanced' auto-assigns by gene count so the larger "
                            "chromosomes always go to train — recommended for subset runs. "
                            "'custom' requires --train-chromosomes and --test-chromosomes."
                        ))
    parser.add_argument("--train-chromosomes", nargs="+", type=str, default=None,
                        metavar="CHROM",
                        help=(
                            "Explicit train chromosomes, e.g. --train-chromosomes 3 22. "
                            "Implies --split custom."
                        ))
    parser.add_argument("--test-chromosomes", nargs="+", type=str, default=None,
                        metavar="CHROM",
                        help=(
                            "Explicit test chromosomes, e.g. --test-chromosomes 1 5. "
                            "Implies --split custom."
                        ))
    parser.add_argument("--build", type=str, default="GRCh38_MANE",
                        help="Genomic build (default: GRCh38_MANE)")
    parser.add_argument("--max-gene-length", type=int, default=200_000,
                        help="Skip genes longer than this (default: 200000)")
    parser.add_argument("--n-genes", type=int, default=20,
                        help="Number of synthetic genes in mock mode")
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--step-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, required=True)

    # Classifier
    parser.add_argument("--architecture", type=str, default="dilated_cnn",
                        choices=["dilated_cnn", "mlp", "linear"])
    parser.add_argument("--hidden-dim", type=int, default=128)

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # Mode
    parser.add_argument("--mock", action="store_true",
                        help="Synthetic data for pipeline validation")
    parser.add_argument("--resume", action="store_true",
                        help="Skip Phase 1 if shard files already exist")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt or directory containing it. "
                             "Loads a pre-trained classifier instead of training.")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip Phase 1+2, run only Phase 3 evaluation. "
                             "Requires --checkpoint.")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- Model metadata ---
    model_kwargs: dict = {}
    if args.foundation_model == "evo2":
        model_kwargs["model_size"] = args.model_size or "7b"
    elif args.foundation_model == "hyenadna":
        model_kwargs["model_size"] = args.model_size or "medium-160k"
    elif args.foundation_model == "splicebert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size
    elif args.foundation_model == "dnabert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size

    meta = get_model_metadata(args.foundation_model, **model_kwargs)
    hidden_dim = meta.hidden_dim

    if args.window_size is not None:
        window_size = args.window_size
        step_size = args.step_size or window_size // 2
    else:
        window_size, step_size = get_default_window_config(meta.max_context)
        if args.step_size is not None:
            step_size = args.step_size

    # Resolve chromosomes — support both space-separated and comma-separated
    # e.g., --chromosomes 1 2 3 22  OR  --chromosomes 1,2,3,22
    if args.chromosomes is not None:
        expanded = []
        for item in args.chromosomes:
            expanded.extend(item.split(","))
        args.chromosomes = [c.strip() for c in expanded if c.strip()]

    # Resolve --train-chromosomes / --test-chromosomes (implies --split custom)
    def _expand_chrom_list(raw: List[str]) -> Set[str]:
        result = []
        for item in raw:
            result.extend(item.split(","))
        # Normalise to chr-prefix
        out = set()
        for c in result:
            c = c.strip()
            if c:
                out.add(c if c.startswith("chr") else f"chr{c}")
        return out

    custom_train_chroms: Optional[Set[str]] = None
    custom_test_chroms: Optional[Set[str]] = None
    if args.train_chromosomes is not None or args.test_chromosomes is not None:
        if args.train_chromosomes is None or args.test_chromosomes is None:
            logger.error(
                "--train-chromosomes and --test-chromosomes must both be specified together"
            )
            sys.exit(1)
        custom_train_chroms = _expand_chrom_list(args.train_chromosomes)
        custom_test_chroms = _expand_chrom_list(args.test_chromosomes)
        overlap = custom_train_chroms & custom_test_chroms
        if overlap:
            logger.error(
                "Chromosomes appear in both --train-chromosomes and --test-chromosomes: %s",
                ", ".join(sorted(overlap)),
            )
            sys.exit(1)
        args.split = "custom"
        logger.info(
            "Custom split: train=%s, test=%s",
            ", ".join(sorted(custom_train_chroms)),
            ", ".join(sorted(custom_test_chroms)),
        )

    if args.mock:
        chromosomes = []  # Not used in mock
    elif args.chromosomes is None or len(args.chromosomes) == 0:
        logger.error("Specify --chromosomes (e.g., 22 or all) or use --mock")
        sys.exit(1)
    elif args.chromosomes == ["all"]:
        chromosomes = CANONICAL_CHROMS
    else:
        chromosomes = args.chromosomes

    # --- Header ---
    print()
    print("=" * 70)
    print("Genome-Scale Splice Site Predictor — Direct-to-Shard Pipeline")
    print("=" * 70)
    print(f"  Model:        {'MOCK' if args.mock else meta.name}")
    print(f"  Type:         {meta.model_type}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Window:       {window_size} bp (step: {step_size})")
    if args.mock:
        print(f"  Data:         Synthetic ({args.n_genes} genes)")
    else:
        print(f"  Chromosomes:  {', '.join(chromosomes)}")
    if custom_train_chroms is not None:
        _split_str = (
            f"custom  (train={', '.join(sorted(custom_train_chroms))}"
            f"  test={', '.join(sorted(custom_test_chroms or {}))})"
        )
    else:
        _split_str = args.split
    print(f"  Split:        {_split_str}")
    print(f"  Focal gamma:  {args.focal_gamma}")
    print(f"  Resume:       {args.resume}")
    print("=" * 70)
    print()

    # ===================================================================
    # Pre-compute chromosome split (needed before extraction)
    # ===================================================================
    from agentic_spliceai.splice_engine.eval.splitting import build_gene_split

    if args.mock:
        # Mock split handled in Phase 1 below
        split = None
    else:
        # Discover genes across all chromosomes (metadata only)
        logger.info("Discovering genes across %d chromosomes...", len(chromosomes))
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            load_gene_annotations,
        )
        from agentic_spliceai.splice_engine.resources import get_genomic_registry

        _build = args.build
        _release = "1.3" if "MANE" in _build.upper() else None
        _registry = get_genomic_registry(build=_build, release=_release)
        _gtf_path = str(_registry.get_gtf_path(validate=True))

        # Build gene info dict with coordinates (needed for live inference in Phase 3)
        all_gene_info: Dict[str, GeneEntry] = {}
        for chrom in chromosomes:
            chrom_chr = f"chr{chrom}" if not chrom.startswith("chr") else chrom
            chrom_bare = _normalize_chrom(chrom)
            genes_df = load_gene_annotations(_gtf_path, chromosomes=[chrom_chr], verbosity=0)
            if genes_df.height == 0:
                genes_df = load_gene_annotations(_gtf_path, chromosomes=[chrom_bare], verbosity=0)
            for row in genes_df.iter_rows(named=True):
                chrom_val = str(row.get("chrom") or row.get("seqname", ""))
                if not chrom_val.startswith("chr"):
                    chrom_val = f"chr{chrom_val}"
                all_gene_info[row["gene_id"]] = GeneEntry(
                    gene_id=row["gene_id"],
                    gene_name=row.get("gene_name", row["gene_id"]),
                    chrom=chrom_val,
                    start=row["start"],
                    end=row["end"],
                    strand=row.get("strand", "+"),
                    n_windows=0,
                    n_splice_sites=0,
                    hdf5_path="",
                )
        all_gene_chroms = {gid: e.chrom for gid, e in all_gene_info.items()}

        split = build_gene_split(
            all_gene_chroms, preset=args.split, val_fraction=0.1, seed=args.seed,
            custom_train_chroms=custom_train_chroms,
            custom_test_chroms=custom_test_chroms,
        )
        logger.info(
            "  Split: %d train, %d val, %d test genes",
            len(split.train_genes), len(split.val_genes), len(split.test_genes),
        )

    # ===================================================================
    # Eval-only mode: skip Phase 1+2, jump straight to Phase 3
    # ===================================================================
    if args.eval_only:
        if not args.checkpoint:
            logger.error("--eval-only requires --checkpoint")
            sys.exit(1)

        # For mock mode, generate synthetic data to get manifest + split
        if args.mock:
            mock_manifest = extract_mock_embeddings(
                n_genes=args.n_genes, hidden_dim=hidden_dim,
                window_size=window_size, step_size=step_size,
                output_dir=output_dir, seed=args.seed,
            )
            gene_chromosomes = {e.gene_id: e.chrom for e in mock_manifest}
            split = build_gene_split(
                gene_chromosomes, preset=args.split, val_fraction=0.1, seed=args.seed,
                custom_train_chroms=custom_train_chroms,
                custom_test_chroms=custom_test_chroms,
            )
            test_manifest = [e for e in mock_manifest if e.gene_id in split.test_genes]
        elif split is None:
            logger.error("--eval-only requires --chromosomes (for test gene discovery)")
            sys.exit(1)
        else:
            # Real data: build test manifest from GTF gene info
            test_manifest = [
                all_gene_info[g] for g in split.test_genes if g in all_gene_info
            ]

        from foundation_models.classifiers.splice_classifier import SpliceClassifier

        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best_model.pt"
        device = get_device()
        classifier = SpliceClassifier.load_model(ckpt_path, device=device)

        # Load temperature if saved separately and not already in checkpoint
        temp_path = ckpt_path.parent / "temperature.pt"
        if temp_path.exists() and classifier.temperature is None:
            classifier.temperature = nn.Parameter(
                torch.load(temp_path, map_location=device, weights_only=True)
            )
            logger.info("Loaded calibration temperature from %s", temp_path)
        logger.info(
            "Eval-only mode: %d test genes from %d test chromosomes",
            len(test_manifest), len({e.chrom for e in test_manifest}),
        )

        # Phase 3 only
        logger.info("Phase 3/3: Evaluating on test chromosomes...")
        results = evaluate_on_test_set(
            classifier=classifier,
            manifest=test_manifest,
            test_genes=split.test_genes,
            output_dir=output_dir,
            model_name=args.foundation_model if not args.mock else "",
            model_kwargs=model_kwargs,
            build=args.build,
            window_size=window_size,
            step_size=step_size,
            max_context=meta.max_context,
            tokenization=meta.tokenization,
        )

        elapsed = time.time() - t0
        print()
        print("=" * 70)
        print("Eval-only Complete")
        print("=" * 70)
        print(f"  Model:        {meta.name}")
        print(f"  Checkpoint:   {ckpt_path}")
        print(f"  Test genes:   {len(test_manifest)}")
        if results:
            print(f"  Mean AUROC:   {results.get('mean_auroc', float('nan')):.4f}")
            print(f"  Mean AUPRC:   {results.get('mean_auprc', float('nan')):.4f}")
            print(f"  Mean F1:      {results.get('mean_f1', float('nan')):.4f}")
        print(f"  Time:         {elapsed / 60:.1f} min")
        print()
        sys.exit(0)

    # ===================================================================
    # Phase 1: Extract → direct to shard files
    # ===================================================================
    shard_dir = output_dir / "shards"
    existing_shards = sorted(shard_dir.glob("train*shard_*.h5")) if shard_dir.exists() else []

    if args.mock:
        logger.info("Phase 1/3: Generating synthetic cached data...")
        manifest = extract_mock_embeddings(
            n_genes=args.n_genes,
            hidden_dim=hidden_dim,
            window_size=window_size,
            step_size=step_size,
            output_dir=output_dir,
            seed=args.seed,
        )
        # Build split from mock manifest
        gene_chromosomes = {e.gene_id: e.chrom for e in manifest}
        split = build_gene_split(
            gene_chromosomes, preset=args.split, val_fraction=0.1, seed=args.seed,
            custom_train_chroms=custom_train_chroms,
            custom_test_chroms=custom_test_chroms,
        )
    elif args.resume and existing_shards:
        logger.info("Phase 1/3: Using existing shards from %s (%d files)", shard_dir, len(existing_shards))
        # Load manifest from prior run
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = [GeneEntry(**e) for e in json.load(f)]
            logger.info("  Loaded manifest: %d genes", len(manifest))
        else:
            # Manifest missing (e.g. disk-full crash before final save).
            # Reconstruct a shard-only manifest: no per-gene entries, but
            # ShardedWindowDataset only needs the shard files, not the manifest.
            # Phase 3 will rediscover test genes from the GTF split.
            logger.warning(
                "manifest.json not found — reconstructing from shard files. "
                "Phase 2 will train on existing shards. "
                "Re-run without --resume to also extract any missing chromosomes."
            )
            manifest = []  # empty — Phase 2 uses shard files directly
            # Add test gene entries so Phase 3 can run live inference.
            # These have hdf5_path="" which triggers the live inference path.
            # Phase 2 filters by train/val genes so these are excluded from training.
            if all_gene_info:
                test_entries = [
                    all_gene_info[g] for g in split.test_genes
                    if g in all_gene_info
                ]
                manifest.extend(test_entries)
                logger.info(
                    "  Added %d test gene entries for Phase 3 live inference",
                    len(test_entries),
                )
    else:
        logger.info("Phase 1/3: Extracting embeddings → direct to shards...")
        manifest = extract_to_shards(
            chromosomes=chromosomes,
            gene_split=all_gene_chroms,
            train_genes=split.train_genes,
            val_genes=split.val_genes,
            test_genes=split.test_genes,
            model_name=args.foundation_model,
            model_kwargs=model_kwargs,
            max_context=meta.max_context,
            tokenization=meta.tokenization,
            hidden_dim=hidden_dim,
            window_size=window_size,
            step_size=step_size,
            output_dir=output_dir,
            build=args.build,
            max_gene_length=args.max_gene_length,
        )

    # When manifest is empty (crash-recovery resume), check that shards exist
    # instead. train_classifier scans the shard directory directly.
    shard_dir = output_dir / "shards"
    existing_train_shards = sorted(shard_dir.glob("train*shard_*.h5")) if shard_dir.exists() else []
    if not manifest and not existing_train_shards:
        logger.error("No genes processed and no shards found — nothing to train on.")
        sys.exit(1)

    # ===================================================================
    # Phase 2: Train from shards (or load checkpoint)
    # ===================================================================
    val_dataset = None
    val_loader = None

    if args.checkpoint:
        # Skip training — load pre-trained classifier from checkpoint
        from foundation_models.classifiers.splice_classifier import SpliceClassifier

        ckpt_path = Path(args.checkpoint)
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "best_model.pt"
        device = get_device()
        classifier = SpliceClassifier.load_model(ckpt_path, device=device)
        logger.info("Phase 2/3: Loaded classifier from %s (skipping training)", ckpt_path)

        # Build val loader for calibration if shards exist
        existing_val_shards = sorted(
            shard_dir.glob("val*shard_*.h5")
        ) if shard_dir.exists() else []
        if existing_val_shards:
            from foundation_models.data.datasets import ShardedWindowDataset
            val_dataset = ShardedWindowDataset(existing_val_shards)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                pin_memory=torch.cuda.is_available(),
            )
            logger.info(
                "  Loaded %d val shards (%d windows) for calibration",
                len(existing_val_shards), len(val_dataset),
            )
    else:
        logger.info("Phase 2/3: Training from shards...")

        logger.info(
            "  Split: %d train, %d val, %d test genes",
            len(split.train_genes), len(split.val_genes), len(split.test_genes),
        )

        n_train_win = sum(e.n_windows for e in manifest if e.gene_id in split.train_genes)
        n_val_win = sum(e.n_windows for e in manifest if e.gene_id in split.val_genes)
        n_test_genes = sum(1 for e in manifest if e.gene_id in split.test_genes)
        if manifest:
            logger.info(
                "  Windows: %d train, %d val | Test genes: %d (live inference)",
                n_train_win, n_val_win, n_test_genes,
            )
        else:
            logger.info(
                "  Windows: from %d shard files (manifest missing) | Test genes: %d (live inference)",
                len(existing_train_shards), len(split.test_genes),
            )

        classifier, val_dataset, val_loader = train_classifier(
            manifest=manifest,
            train_genes=split.train_genes,
            val_genes=split.val_genes,
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
            use_shards=True,  # Always use shards in 07a
            seed=args.seed,
        )

    # ===================================================================
    # Phase 2.5: Calibrate probabilities (temperature scaling)
    # ===================================================================
    cal_metrics: Optional[Dict] = None
    if val_loader is not None:
        logger.info("Phase 2.5: Calibrating probabilities (temperature scaling)...")
        device = get_device()
        cal_metrics = classifier.calibrate(val_loader, device=device)

        # Save temperature vector
        temp_path = output_dir / "model" / "temperature.pt"
        torch.save(classifier.temperature.data.cpu(), temp_path)
        logger.info("  Temperature saved: %s", temp_path)

        # Re-save checkpoint with temperature included
        model_dir = output_dir / "model"
        classifier._save_checkpoint(
            str(model_dir), epoch=-1,
            metrics={"calibration": cal_metrics},
        )
    else:
        logger.warning("No validation data — skipping calibration")

    # Clean up val dataset file handles
    if val_dataset is not None:
        val_dataset.close()

    # ===================================================================
    # Phase 3: Evaluate
    # ===================================================================
    logger.info("Phase 3/3: Evaluating on test chromosomes...")

    results = evaluate_on_test_set(
        classifier=classifier,
        manifest=manifest,
        test_genes=split.test_genes,
        output_dir=output_dir,
        model_name=args.foundation_model if not args.mock else "",
        model_kwargs=model_kwargs,
        build=args.build,
        window_size=window_size,
        step_size=step_size,
        max_context=meta.max_context,
        tokenization=meta.tokenization,
    )

    elapsed = time.time() - t0

    print("=" * 70)
    print("Complete")
    print("=" * 70)
    print(f"  Model:        {meta.name}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Output:       {output_dir}")
    print(f"  Cache:        {output_dir / 'cache'}")
    print(f"  Checkpoint:   {output_dir / 'model' / 'best_model.pt'}")
    if cal_metrics:
        print(f"  ECE (before): {cal_metrics['ece_before']:.4f}")
        print(f"  ECE (after):  {cal_metrics['ece_after']:.4f}")
        temps = cal_metrics["temperature"]
        print(f"  Temperature:  [{temps[0]:.3f}, {temps[1]:.3f}, {temps[2]:.3f}]")
    if results:
        print(f"  Mean AUROC:   {results.get('mean_auroc', float('nan')):.4f}")
        print(f"  Mean AUPRC:   {results.get('mean_auprc', float('nan')):.4f}")
        print(f"  Mean F1:      {results.get('mean_f1', float('nan')):.4f}")
    print(f"  Time:         {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
