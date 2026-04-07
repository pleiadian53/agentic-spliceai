"""
Sequence-level PyTorch Dataset for M*-S meta-layer training.

Provides windowed training samples with:
    - One-hot DNA sequence ``[4, W + context]``
    - Base model scores ``[W, 3]``
    - Dense multimodal features ``[C, W]``
    - Per-position labels ``[W]``

Gene data is stored on disk as ``.npz`` files (one per gene) and loaded
on-the-fly during training.  This bounds peak memory to ~1 gene at a
time, regardless of the total number of training genes.

Label encoding follows the meta-layer convention:
    0 = donor, 1 = acceptor, 2 = neither

Note: ``build_splice_labels()`` in ``foundation_models/utils/chunking.py``
uses a different encoding (0=none, 1=acceptor, 2=donor).  The mapping
is applied once during cache construction.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# build_splice_labels: 0=none, 1=acceptor, 2=donor
# Meta-layer:          0=donor, 1=acceptor, 2=neither
_CHUNKING_TO_META = np.array([2, 1, 0], dtype=np.int64)

_ONEHOT_MAP = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


# ---------------------------------------------------------------------------
# Gene index (lightweight metadata, held in RAM)
# ---------------------------------------------------------------------------


@dataclass
class GeneIndexEntry:
    """Lightweight metadata for one cached gene (no arrays in RAM)."""
    gene_id: str
    npz_path: Path
    length: int
    n_splice_sites: int
    splice_positions: np.ndarray  # small int array (typically <100 positions)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SequenceLevelDataset(Dataset):
    """Window-level dataset with disk-backed gene cache.

    Gene data (sequence, base scores, features, labels) is stored on disk
    as ``.npz`` files.  Only one gene is loaded into memory at a time
    during ``__getitem__``, bounding peak RAM to ~1 gene regardless of
    the total training set size.

    Parameters
    ----------
    gene_index : list of GeneIndexEntry
        Lightweight index of cached genes (from ``build_gene_cache``).
    window_size : int
        Output window length (default 5001).
    context_padding : int
        Extra sequence context for CNN receptive field (default 400).
    splice_bias : float
        Fraction of windows centered on a splice site (default 0.5).
    samples_per_epoch : int
        Number of windows per epoch (default 50000).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        gene_index: List[GeneIndexEntry],
        window_size: int = 5001,
        context_padding: int = 400,
        splice_bias: float = 0.5,
        samples_per_epoch: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.gene_index = gene_index
        self.window_size = window_size
        self.context_padding = context_padding
        self.splice_bias = splice_bias
        self.samples_per_epoch = samples_per_epoch
        self._rng = np.random.RandomState(seed)

        min_length = window_size + context_padding
        self._genes_with_splices: List[int] = []
        self._valid_genes: List[int] = []

        for i, entry in enumerate(gene_index):
            if entry.length < min_length:
                continue
            self._valid_genes.append(i)
            if entry.n_splice_sites > 0:
                self._genes_with_splices.append(i)

        logger.info(
            "SequenceLevelDataset: %d genes (%d with splice sites), "
            "%d samples/epoch, window=%d+%d",
            len(self._valid_genes),
            len(self._genes_with_splices),
            samples_per_epoch,
            window_size,
            context_padding,
        )

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        W = self.window_size
        ctx = self.context_padding

        use_splice = (
            self._rng.random() < self.splice_bias
            and len(self._genes_with_splices) > 0
        )

        if use_splice:
            gene_idx = self._rng.choice(self._genes_with_splices)
            entry = self.gene_index[gene_idx]
            # Load gene data from disk
            gene = _load_gene_npz(entry.npz_path)
            sp = entry.splice_positions
            center = self._rng.choice(sp)
            jitter = self._rng.randint(-W // 4, W // 4 + 1)
            out_start = max(0, center - W // 2 + jitter)
        else:
            gene_idx = self._rng.choice(self._valid_genes)
            entry = self.gene_index[gene_idx]
            gene = _load_gene_npz(entry.npz_path)
            out_start = self._rng.randint(0, max(1, entry.length - W))

        out_start = max(0, min(out_start, entry.length - W))
        out_end = out_start + W

        # Sequence with context padding
        seq_start = max(0, out_start - ctx // 2)
        seq_end = min(entry.length, out_end + (ctx - ctx // 2))
        seq_onehot = _one_hot_encode(gene["sequence"][seq_start:seq_end])

        total_len = W + ctx
        if seq_onehot.shape[1] < total_len:
            padded = np.zeros((4, total_len), dtype=np.float32)
            offset = (total_len - seq_onehot.shape[1]) // 2
            padded[:, offset:offset + seq_onehot.shape[1]] = seq_onehot
            seq_onehot = padded

        base_scores = gene["base_scores"][out_start:out_end]
        mm_features = gene["mm_features"][out_start:out_end]
        labels = gene["labels"][out_start:out_end]

        return {
            "sequence": torch.from_numpy(seq_onehot),
            "base_scores": torch.from_numpy(base_scores),
            "mm_features": torch.from_numpy(mm_features.T.copy()),  # [C, W]
            "labels": torch.from_numpy(labels),
        }


def _load_gene_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load one gene's data from a .npz file."""
    data = np.load(path, allow_pickle=True)
    return {
        "sequence": str(data["sequence"]),
        "base_scores": data["base_scores"],
        "mm_features": data["mm_features"],
        "labels": data["labels"],
    }


def _one_hot_encode(seq: str) -> np.ndarray:
    """One-hot encode DNA to [4, L] float32."""
    L = len(seq)
    out = np.zeros((4, L), dtype=np.float32)
    for i, nt in enumerate(seq.upper()):
        enc = _ONEHOT_MAP.get(nt, [0, 0, 0, 0])
        out[0, i] = enc[0]
        out[1, i] = enc[1]
        out[2, i] = enc[2]
        out[3, i] = enc[3]
    return out


def _one_hot_from_uint8(arr: np.ndarray) -> np.ndarray:
    """One-hot encode uint8 DNA (A=0,C=1,G=2,T=3,N=4) to [4, L] float32."""
    L = len(arr)
    out = np.zeros((4, L), dtype=np.float32)
    for ch in range(4):
        out[ch, arr == ch] = 1.0
    return out


# ---------------------------------------------------------------------------
# Shard-backed dataset (per-chromosome HDF5 files)
# ---------------------------------------------------------------------------


@dataclass
class ShardGeneIndexEntry:
    """Gene metadata for shard-backed dataset."""
    gene_id: str
    shard_path: Path
    length: int
    n_splice_sites: int
    splice_positions: np.ndarray


class ShardedSequenceLevelDataset(Dataset):
    """Window-level dataset backed by per-chromosome HDF5 shards.

    Gene data is stored in HDF5 files (one per chromosome), enabling
    random-access slice reads — only the training window is read from
    disk, not the full gene.  Fork-safe: each worker process gets its
    own ``h5py.File`` handles via a per-PID cache.

    Parameters
    ----------
    shard_index_path : Path
        Path to ``shard_index.json`` produced by
        :func:`~shard_packing.pack_gene_cache_to_shards`.
    window_size : int
        Output window length (default 5001).
    context_padding : int
        Extra sequence context for CNN receptive field (default 400).
    splice_bias : float
        Fraction of windows centered on a splice site (default 0.5).
    samples_per_epoch : int
        Number of windows per epoch (default 50000).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        shard_index_path: Path,
        window_size: int = 5001,
        context_padding: int = 400,
        splice_bias: float = 0.5,
        samples_per_epoch: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.context_padding = context_padding
        self.splice_bias = splice_bias
        self.samples_per_epoch = samples_per_epoch
        self._rng = np.random.RandomState(seed)

        # Per-process HDF5 handle cache (fork-safe)
        self._handles: Dict[int, Dict[str, "h5py.File"]] = {}

        # Load shard index
        self.gene_index = self._load_index(shard_index_path)

        min_length = window_size + context_padding
        self._genes_with_splices: List[int] = []
        self._valid_genes: List[int] = []

        for i, entry in enumerate(self.gene_index):
            if entry.length < min_length:
                continue
            self._valid_genes.append(i)
            if entry.n_splice_sites > 0:
                self._genes_with_splices.append(i)

        logger.info(
            "ShardedSequenceLevelDataset: %d genes (%d with splice sites), "
            "%d samples/epoch, window=%d+%d",
            len(self._valid_genes),
            len(self._genes_with_splices),
            samples_per_epoch,
            window_size,
            context_padding,
        )

    @staticmethod
    def _load_index(path: Path) -> List[ShardGeneIndexEntry]:
        """Load shard_index.json into a list of gene entries."""
        with open(path) as f:
            data = json.load(f)
        entries = []
        for gene_id, meta in data["genes"].items():
            entries.append(ShardGeneIndexEntry(
                gene_id=gene_id,
                shard_path=Path(meta["shard_file"]),
                length=meta["length"],
                n_splice_sites=meta["n_splice_sites"],
                splice_positions=np.array(meta["splice_positions"], dtype=np.int64),
            ))
        return entries

    def _get_handle(self, shard_path: Path) -> "h5py.File":
        """Get or create an HDF5 handle for the current worker process."""
        import h5py

        pid = os.getpid()
        if pid not in self._handles:
            self._handles[pid] = {}
        key = str(shard_path)
        if key not in self._handles[pid]:
            self._handles[pid][key] = h5py.File(key, "r")
        return self._handles[pid][key]

    def close(self) -> None:
        """Close all HDF5 handles across all worker processes."""
        for pid_handles in self._handles.values():
            for fh in pid_handles.values():
                try:
                    fh.close()
                except Exception:
                    pass
        self._handles.clear()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        W = self.window_size
        ctx = self.context_padding

        use_splice = (
            self._rng.random() < self.splice_bias
            and len(self._genes_with_splices) > 0
        )

        if use_splice:
            gene_idx = self._rng.choice(self._genes_with_splices)
            entry = self.gene_index[gene_idx]
            sp = entry.splice_positions
            center = self._rng.choice(sp)
            jitter = self._rng.randint(-W // 4, W // 4 + 1)
            out_start = max(0, center - W // 2 + jitter)
        else:
            gene_idx = self._rng.choice(self._valid_genes)
            entry = self.gene_index[gene_idx]
            out_start = self._rng.randint(0, max(1, entry.length - W))

        out_start = max(0, min(out_start, entry.length - W))
        out_end = out_start + W

        # Open shard and read gene group
        h5 = self._get_handle(entry.shard_path)
        grp = h5[entry.gene_id]

        # Sequence with context padding (slice read from HDF5)
        seq_start = max(0, out_start - ctx // 2)
        seq_end = min(entry.length, out_end + (ctx - ctx // 2))
        seq_uint8 = grp["sequence"][seq_start:seq_end]
        seq_onehot = _one_hot_from_uint8(seq_uint8)

        total_len = W + ctx
        if seq_onehot.shape[1] < total_len:
            padded = np.zeros((4, total_len), dtype=np.float32)
            offset = (total_len - seq_onehot.shape[1]) // 2
            padded[:, offset:offset + seq_onehot.shape[1]] = seq_onehot
            seq_onehot = padded

        # Slice reads: only the window, not the full gene
        base_scores = grp["base_scores"][out_start:out_end]
        mm_features = grp["mm_features"][out_start:out_end]
        labels = grp["labels"][out_start:out_end]

        return {
            "sequence": torch.from_numpy(seq_onehot),
            "base_scores": torch.from_numpy(base_scores),
            "mm_features": torch.from_numpy(np.ascontiguousarray(mm_features.T)),
            "labels": torch.from_numpy(labels),
        }


# ---------------------------------------------------------------------------
# Cache building (disk-backed)
# ---------------------------------------------------------------------------


def build_gene_cache(
    gene_ids: List[str],
    splice_sites_df: "pandas.DataFrame",
    fasta_path: str,
    base_scores_dir: Path,
    feature_extractor: "DenseFeatureExtractor",
    gene_annotations: "polars.DataFrame",
    cache_dir: Path = Path("/tmp/gene_cache"),
) -> List[GeneIndexEntry]:
    """Build disk-backed gene cache for training.

    Genes are grouped by chromosome so that each prediction parquet is
    loaded only once, rather than once per gene.  This reduces peak
    memory and avoids redundant I/O on large parquets (~1 GB each).

    Each gene is saved as a ``.npz`` file containing sequence, base
    scores, multimodal features, and labels.  Returns a lightweight
    index (gene_id, path, length, splice positions) that fits easily
    in RAM even for 20K+ genes.

    Parameters
    ----------
    gene_ids:
        Gene IDs to include.
    splice_sites_df:
        DataFrame from ``splice_sites_enhanced.tsv`` (pandas).
    fasta_path:
        Path to reference FASTA (indexed with .fai).
    base_scores_dir:
        Directory with ``predictions_{chrom}.parquet``.
    feature_extractor:
        Initialized ``DenseFeatureExtractor``.
    gene_annotations:
        Polars DataFrame with gene_id, chrom, start, end, strand.
    cache_dir:
        Directory for ``.npz`` files.  Existing files are reused
        (resume-safe).

    Returns
    -------
    List of ``GeneIndexEntry`` (lightweight, ~1 KB per gene in RAM).
    """
    import pyfaidx
    import polars as pl
    from foundation_models.utils.chunking import build_splice_labels

    cache_dir.mkdir(parents=True, exist_ok=True)
    fasta = pyfaidx.Fasta(fasta_path)
    index: List[GeneIndexEntry] = []

    # ── Phase 1: resolve gene annotations in a single pass ────────
    gene_to_row: Dict[str, dict] = {}
    for gene_id in gene_ids:
        gene_row = gene_annotations.filter(pl.col("gene_id") == gene_id)
        if gene_row.height == 0:
            gene_row = gene_annotations.filter(pl.col("gene_name") == gene_id)
        if gene_row.height > 0:
            gene_to_row[gene_id] = gene_row.row(0, named=True)

    # ── Phase 2: scan cached .npz files (resume) ─────────────────
    # Separate already-cached genes from those that need building.
    genes_to_build: List[str] = []
    for gene_id in gene_ids:
        npz_path = cache_dir / f"{gene_id}.npz"
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                labels = data["labels"]
                sp = np.where(labels != 2)[0]
                index.append(GeneIndexEntry(
                    gene_id=gene_id,
                    npz_path=npz_path,
                    length=int(data["length"]),
                    n_splice_sites=len(sp),
                    splice_positions=sp,
                ))
                continue
            except Exception:
                pass  # Corrupted file, regenerate
        if gene_id in gene_to_row:
            genes_to_build.append(gene_id)
        else:
            logger.warning("Gene %s not found in annotations, skipping", gene_id)

    if not genes_to_build:
        logger.info("Gene cache complete: %d genes in %s (all resumed)", len(index), cache_dir)
        return index

    logger.info(
        "Cache: %d resumed, %d to build",
        len(index), len(genes_to_build),
    )

    # ── Phase 3: group uncached genes by chromosome ───────────────
    chrom_to_genes: Dict[str, List[str]] = defaultdict(list)
    for gene_id in genes_to_build:
        chrom = gene_to_row[gene_id]["chrom"]
        chrom_to_genes[chrom].append(gene_id)

    # ── Phase 4: iterate per chromosome, sub-batched ───────────────
    import gc
    import os

    def _log_mem(tag: str) -> None:
        try:
            rss_kb = int(open(f"/proc/{os.getpid()}/status").read()
                         .split("VmRSS:")[1].split("kB")[0].strip())
            logger.info("[MEM] %s: %.1f GB RSS", tag, rss_kb / 1e6)
        except Exception:
            pass  # /proc not available (macOS)

    n_done = len(index)
    n_total = len(gene_ids)
    sub_batch_size = 200  # gc.collect() after every N genes per chromosome

    for chrom, chrom_gene_ids in chrom_to_genes.items():
        _log_mem(f"{chrom} start ({len(chrom_gene_ids)} genes)")

        # Collect gene ranges for bulk parquet load
        gene_ranges: List[Tuple[str, int, int, int]] = []
        for gid in chrom_gene_ids:
            row = gene_to_row[gid]
            start, end = int(row["start"]), int(row["end"])
            gene_ranges.append((gid, start, end, end - start))

        # Load base scores for ALL genes on this chromosome in one read
        chrom_scores = _load_chrom_base_scores(base_scores_dir, chrom, gene_ranges)

        # Process each gene on this chromosome
        batch_count = 0
        for gid, start, end, _ in gene_ranges:
            npz_path = cache_dir / f"{gid}.npz"

            # Extract sequence
            try:
                fasta_chrom = chrom.replace("chr", "") if chrom not in fasta.keys() else chrom
                sequence = str(fasta[fasta_chrom][start:end]).upper()
            except Exception as e:
                logger.warning("FASTA failed for %s: %s", gid, e)
                continue

            gene_len = len(sequence)

            # Labels
            raw_labels = build_splice_labels(gid, start, gene_len, splice_sites_df)
            labels = _CHUNKING_TO_META[raw_labels].astype(np.int64)

            # Base model scores (from pre-loaded chromosome data)
            base_scores = chrom_scores.get(gid)
            if base_scores is None or base_scores.shape[0] != gene_len:
                base_scores = np.full((gene_len, 3), 1.0 / 3, dtype=np.float32)

            # Dense multimodal features
            mm_features = feature_extractor.extract_window(chrom, start, end)

            # Save to disk
            np.savez_compressed(
                npz_path,
                sequence=np.array(sequence, dtype=object),
                base_scores=base_scores,
                mm_features=mm_features,
                labels=labels,
                length=np.array(gene_len),
            )

            # Add to index
            sp = np.where(labels != 2)[0]
            index.append(GeneIndexEntry(
                gene_id=gid,
                npz_path=npz_path,
                length=gene_len,
                n_splice_sites=len(sp),
                splice_positions=sp,
            ))

            del sequence, base_scores, mm_features, labels, raw_labels
            n_done += 1
            batch_count += 1

            if n_done % 50 == 0:
                logger.info("Cached %d / %d genes", n_done, n_total)

            # Reclaim Polars/Arrow/numpy buffers between sub-batches
            if batch_count % sub_batch_size == 0:
                gc.collect()

        # Free chromosome-level data before next chromosome
        del chrom_scores
        gc.collect()
        _log_mem(f"{chrom} done")

    logger.info("Gene cache complete: %d genes in %s", len(index), cache_dir)
    return index


def _load_chrom_base_scores(
    base_scores_dir: Path,
    chrom: str,
    gene_ranges: List[Tuple[str, int, int, int]],
) -> Dict[str, np.ndarray]:
    """Load base model scores for all genes on one chromosome.

    Reads the prediction parquet **once**, converts to numpy, then
    slices per gene.  Returns ``{gene_id: [gene_len, 3]}`` dict.

    Parameters
    ----------
    base_scores_dir:
        Directory with ``predictions_{chrom}.parquet``.
    chrom:
        Chromosome name (e.g. ``chr17``).
    gene_ranges:
        List of ``(gene_id, start, end, gene_len)`` tuples for all
        genes on this chromosome.
    """
    import polars as pl

    path = base_scores_dir / f"predictions_{chrom}.parquet"
    if not path.exists():
        path = base_scores_dir / f"predictions_{chrom.replace('chr', '')}.parquet"
    if not path.exists():
        # Try adding chr prefix (Ensembl uses bare "1", parquets may use "chr1")
        bare = chrom.replace("chr", "")
        path = base_scores_dir / f"predictions_chr{bare}.parquet"
    if not path.exists():
        logger.warning("No predictions for %s, using uniform prior", chrom)
        return {
            gid: np.full((glen, 3), 1.0 / 3, dtype=np.float32)
            for gid, _, _, glen in gene_ranges
        }

    # Single read: filter to the span covering all genes on this chrom
    min_start = min(s for _, s, _, _ in gene_ranges)
    max_end = max(e for _, _, e, _ in gene_ranges)

    try:
        df = pl.scan_parquet(path).filter(
            (pl.col("position") >= min_start) & (pl.col("position") < max_end)
        ).select(
            ["position", "donor_prob", "acceptor_prob", "neither_prob"]
        ).collect()
    except Exception as e:
        logger.warning("Failed to read predictions for %s: %s", chrom, e)
        return {
            gid: np.full((glen, 3), 1.0 / 3, dtype=np.float32)
            for gid, _, _, glen in gene_ranges
        }

    if df.height == 0:
        return {
            gid: np.full((glen, 3), 1.0 / 3, dtype=np.float32)
            for gid, _, _, glen in gene_ranges
        }

    # Convert to numpy once
    all_pos = df["position"].to_numpy()
    all_donor = df["donor_prob"].to_numpy().astype(np.float32)
    all_acc = df["acceptor_prob"].to_numpy().astype(np.float32)
    all_neither = df["neither_prob"].to_numpy().astype(np.float32)
    del df  # free Polars DataFrame

    # Slice per gene
    result: Dict[str, np.ndarray] = {}
    for gid, start, end, glen in gene_ranges:
        gene_len = end - start
        scores = np.full((gene_len, 3), 1.0 / 3, dtype=np.float32)
        mask = (all_pos >= start) & (all_pos < end)
        if mask.any():
            pos = all_pos[mask] - start
            valid = (pos >= 0) & (pos < gene_len)
            scores[pos[valid], 0] = all_donor[mask][valid]
            scores[pos[valid], 1] = all_acc[mask][valid]
            scores[pos[valid], 2] = all_neither[mask][valid]
        result[gid] = scores

    return result
