"""
Genome-Scale Splice Site Predictor (Foundation Model Embeddings)

Production-grade per-nucleotide splice site prediction using frozen foundation
model embeddings with SpliceAI-standard chromosome-split evaluation.

Three-phase pipeline:
  Phase 1: Extract per-gene embeddings → cache to HDF5 (resumable)
  Phase 2: Train SpliceClassifier on windowed embeddings (focal loss)
  Phase 3: Evaluate on held-out chromosomes with SpliceAI metrics

Supports all registered foundation models: Evo2, SpliceBERT, HyenaDNA, DNABERT-2.

Usage:
    # Mock mode (pipeline validation, no GPU needed)
    python 07_genome_scale_splice_predictor.py \\
        --mock --n-genes 10 -o /tmp/07-mock/

    # Single chromosome (quick local test)
    python 07_genome_scale_splice_predictor.py \\
        --foundation-model splicebert --chromosomes 22 \\
        -o /tmp/07-chr22/

    # Full genome on GPU pod
    python 07_genome_scale_splice_predictor.py \\
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
# Phase 1: Extract & cache embeddings to HDF5
# ---------------------------------------------------------------------------

def extract_and_cache_embeddings(
    chromosomes: List[str],
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
    resume: bool = False,
) -> List[GeneEntry]:
    """Extract per-gene embeddings, window, and cache to HDF5.

    Returns manifest of cached genes.
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
        logger.error(
            "Ensure data is prepared. See: "
            "python examples/data_preparation/04_generate_ground_truth.py "
            "--output data/mane/GRCh38/"
        )
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
        logger.error(
            "Splice site annotations not found in %s. Generate with:\n"
            "  python examples/data_preparation/04_generate_ground_truth.py "
            "--output %s", data_dir, data_dir,
        )
        sys.exit(1)

    # --- Load foundation model once ---
    logger.info("Loading foundation model '%s'...", model_name)
    model = load_embedding_model(model_name, **model_kwargs)
    meta = model.metadata()
    logger.info("Model loaded: %s (hidden_dim=%d)", meta.name, meta.hidden_dim)

    # --- Open FASTA ---
    fasta = Fasta(fasta_path)
    logger.info("Using FASTA: %s", fasta_path)

    # --- Process each chromosome ---
    cache_dir = output_dir / "cache"
    manifest: List[GeneEntry] = []
    total_genes = 0
    total_windows = 0
    total_splice = 0
    total_skipped = 0
    total_cached = 0

    for chrom_i, chrom in enumerate(chromosomes, 1):
        chrom_t0 = time.time()
        chrom_bare = _normalize_chrom(chrom)
        chrom_chr = f"chr{chrom_bare}" if not chrom.startswith("chr") else chrom
        chrom_dir = cache_dir / chrom_chr
        chrom_dir.mkdir(parents=True, exist_ok=True)

        # Load genes for this chromosome
        gene_data = load_gene_annotations(
            gtf_path=gtf_path, chromosomes=[chrom_chr], verbosity=0,
        )
        if gene_data.height == 0:
            # Try without chr prefix
            gene_data = load_gene_annotations(
                gtf_path=gtf_path, chromosomes=[chrom_bare], verbosity=0,
            )
        if gene_data.height == 0:
            logger.warning("  %s: no genes found, skipping", chrom_chr)
            continue

        # Filter by gene length
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

            h5_path = chrom_dir / f"{gene_id}.h5"

            # Resume: skip if already cached
            if resume and h5_path.exists():
                try:
                    with h5py.File(h5_path, "r") as f:
                        n_win = f["embeddings"].shape[0]
                        n_spl = int(f.attrs.get("n_splice_sites", 0))
                    manifest.append(GeneEntry(
                        gene_id=gene_id, gene_name=gene_name, chrom=chrom_chr,
                        start=start, end=end, strand=strand,
                        n_windows=n_win, n_splice_sites=n_spl,
                        hdf5_path=str(h5_path),
                    ))
                    chrom_genes += 1
                    chrom_windows += n_win
                    chrom_splice += n_spl
                    total_cached += 1
                    if gene_idx % 50 == 1:
                        logger.info(
                            "  %s [%d/%d] %s: cached (%d windows)",
                            chrom_chr, gene_idx, n_chrom_genes, gene_name, n_win,
                        )
                    continue
                except Exception:
                    pass  # Corrupted cache — re-extract

            # Resolve chromosome in FASTA
            chrom_key = _resolve_chrom_key(fasta, g_chrom)
            if chrom_key is None:
                total_skipped += 1
                continue

            # Extract sequence
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

            # Extract dense embeddings
            gene_emb = extract_dense_gene_embeddings(
                model=model,
                gene_sequence=gene_seq,
                max_context=max_context,
                tokenization=tokenization,
                overlap=None,
            )

            # Window
            win_embs = []
            win_lbls = []
            pos = 0
            while pos + window_size <= len(gene_seq):
                win_embs.append(gene_emb[pos:pos + window_size])
                win_lbls.append(labels[pos:pos + window_size])
                pos += step_size

            if not win_embs:
                total_skipped += 1
                del gene_emb, labels, gene_seq
                continue

            emb_arr = np.stack(win_embs, axis=0)  # [n_win, W, H]
            lbl_arr = np.stack(win_lbls, axis=0)  # [n_win, W]

            if gene_idx % 10 == 1 or gene_idx == n_chrom_genes:
                logger.info(
                    "  %s [%d/%d] %s: %d windows, %d splice sites (%d bp)",
                    chrom_chr, gene_idx, n_chrom_genes, gene_name,
                    len(win_embs), n_splice, gene_len,
                )

            # Save to HDF5
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("embeddings", data=emb_arr)
                f.create_dataset("labels", data=lbl_arr)
                f.attrs["gene_id"] = gene_id
                f.attrs["gene_name"] = gene_name
                f.attrs["chrom"] = chrom_chr
                f.attrs["start"] = start
                f.attrs["end"] = end
                f.attrs["strand"] = strand
                f.attrs["n_splice_sites"] = n_splice
                f.attrs["hidden_dim"] = hidden_dim
                f.attrs["window_size"] = window_size

            manifest.append(GeneEntry(
                gene_id=gene_id, gene_name=gene_name, chrom=chrom_chr,
                start=start, end=end, strand=strand,
                n_windows=len(win_embs), n_splice_sites=n_splice,
                hdf5_path=str(h5_path),
            ))

            chrom_genes += 1
            chrom_windows += len(win_embs)
            chrom_splice += n_splice

            # Free memory
            del gene_emb, labels, gene_seq, emb_arr, lbl_arr, win_embs, win_lbls
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_genes += chrom_genes
        total_windows += chrom_windows
        total_splice += chrom_splice

        chrom_elapsed = time.time() - chrom_t0
        logger.info(
            "  %s: %d genes, %d windows, %d splice sites (%.1f min)%s",
            chrom_chr, chrom_genes, chrom_windows, chrom_splice,
            chrom_elapsed / 60,
            f" ({total_cached} from cache)" if total_cached else "",
        )

    fasta.close()

    logger.info(
        "Extraction complete: %d genes, %d windows, %d splice sites "
        "(%d skipped, %d from cache)",
        total_genes, total_windows, total_splice, total_skipped, total_cached,
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
        # Check for existing shards first
        existing_train = sorted(shard_dir.glob("train_shard_*.h5"))
        existing_val = sorted(shard_dir.glob("val_shard_*.h5"))

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
            existing_train = sorted(shard_dir.glob("train_shard_*.h5"))
            existing_val = sorted(shard_dir.glob("val_shard_*.h5"))

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

    # DataLoaders — num_workers=0 because HDF5 file handles are not
    # fork-safe; the LRU cache already makes single-worker I/O fast.
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=torch.cuda.is_available(),
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

def evaluate_on_test_set(
    classifier: Any,
    manifest: List[GeneEntry],
    test_genes: Set[str],
    output_dir: Path,
) -> Dict:
    """Evaluate trained classifier on test chromosomes.

    Computes per-class AUROC/AUPRC and site-level precision/recall/F1.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = get_device()
    classifier.to(device)
    classifier.eval()

    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Collect per-gene predictions
    all_true = []
    all_donor_score = []
    all_acceptor_score = []
    n_test_genes = 0

    for entry in manifest:
        if entry.gene_id not in test_genes:
            continue
        n_test_genes += 1

        with h5py.File(entry.hdf5_path, "r") as f:
            emb = f["embeddings"][:]  # [n_win, W, H]
            lbl = f["labels"][:]      # [n_win, W]

        emb_t = torch.tensor(emb, dtype=torch.float32).to(device)
        preds = classifier.predict(emb_t)

        all_true.append(lbl.flatten())
        all_donor_score.append(preds["donor_prob"].flatten())
        all_acceptor_score.append(preds["acceptor_prob"].flatten())

        del emb, lbl, emb_t

    if not all_true:
        logger.warning("No test genes found — skipping evaluation")
        return {}

    true_flat = np.concatenate(all_true)
    donor_flat = np.concatenate(all_donor_score)
    acceptor_flat = np.concatenate(all_acceptor_score)

    results: Dict[str, Any] = {
        "n_test_genes": n_test_genes,
        "n_test_positions": len(true_flat),
    }

    # Per-class metrics
    for cls_name, cls_idx, scores in [
        ("acceptor", 1, acceptor_flat),
        ("donor", 2, donor_flat),
    ]:
        y_true = (true_flat == cls_idx).astype(np.int32)
        n_positive = int(y_true.sum())
        n_total = len(y_true)

        if n_positive > 0 and n_positive < n_total:
            auroc = float(roc_auc_score(y_true, scores))
            auprc = float(average_precision_score(y_true, scores))
        else:
            auroc = float("nan")
            auprc = float("nan")

        # Site-level precision/recall/F1 at threshold 0.5
        pred_positive = (scores >= 0.5).astype(np.int32)
        tp = int(((pred_positive == 1) & (y_true == 1)).sum())
        fp = int(((pred_positive == 1) & (y_true == 0)).sum())
        fn = int(((pred_positive == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[cls_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "n_positive": n_positive,
            "n_total": n_total,
        }

    # Mean across splice classes
    for metric in ("auroc", "auprc", "f1"):
        vals = [
            results[c][metric] for c in ("acceptor", "donor")
            if not (isinstance(results[c][metric], float) and np.isnan(results[c][metric]))
        ]
        results[f"mean_{metric}"] = float(np.mean(vals)) if vals else float("nan")

    # Save
    with open(eval_dir / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print
    print()
    print("Test Metrics (held-out chromosomes):")
    print(f"  Genes: {n_test_genes}, Positions: {len(true_flat):,}")
    for cls_name in ("acceptor", "donor"):
        m = results[cls_name]
        print(f"  {cls_name:10s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.4f}, "
              f"F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}) "
              f"[{m['n_positive']} sites]")
    print(f"  {'mean':10s}: AUROC={results['mean_auroc']:.4f}, "
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
                        choices=["spliceai", "even_odd"],
                        help="Chromosome split preset (default: spliceai)")
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
                        help="Skip genes with existing HDF5 cache")
    parser.add_argument("--shard", action="store_true",
                        help="Pack per-gene HDF5 into shard files for fast I/O")

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

    # Resolve chromosomes
    if args.mock:
        chromosomes = []  # Not used in mock
    elif args.chromosomes is None:
        logger.error("Specify --chromosomes (e.g., 22 or all) or use --mock")
        sys.exit(1)
    elif args.chromosomes == ["all"]:
        chromosomes = CANONICAL_CHROMS
    else:
        chromosomes = args.chromosomes

    # --- Header ---
    print()
    print("=" * 70)
    print("Genome-Scale Splice Site Predictor (Foundation Model Embeddings)")
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
    print(f"  Split:        {args.split}")
    print(f"  Focal gamma:  {args.focal_gamma}")
    print(f"  Resume:       {args.resume}")
    print("=" * 70)
    print()

    # ===================================================================
    # Phase 1: Extract & cache
    # ===================================================================
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
    else:
        logger.info("Phase 1/3: Extracting and caching embeddings...")
        manifest = extract_and_cache_embeddings(
            chromosomes=chromosomes,
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
            resume=args.resume,
        )

    if not manifest:
        logger.error("No genes processed — nothing to train on.")
        sys.exit(1)

    # ===================================================================
    # Phase 2: Chromosome-based split + train
    # ===================================================================
    logger.info("Phase 2/3: Splitting genes and training...")

    from agentic_spliceai.splice_engine.eval.splitting import build_gene_split

    gene_chromosomes = {e.gene_id: e.chrom for e in manifest}
    split = build_gene_split(
        gene_chromosomes, preset=args.split, val_fraction=0.1, seed=args.seed,
    )

    logger.info(
        "  Split: %d train, %d val, %d test genes",
        len(split.train_genes), len(split.val_genes), len(split.test_genes),
    )

    n_train_win = sum(e.n_windows for e in manifest if e.gene_id in split.train_genes)
    n_val_win = sum(e.n_windows for e in manifest if e.gene_id in split.val_genes)
    n_test_win = sum(e.n_windows for e in manifest if e.gene_id in split.test_genes)
    logger.info(
        "  Windows: %d train, %d val, %d test", n_train_win, n_val_win, n_test_win,
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
        use_shards=args.shard,
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
