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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

    Each gene is saved as a ``.npz`` file containing sequence, base
    scores, multimodal features, and labels.  Returns a lightweight
    index (gene_id, path, length, splice positions) that fits easily
    in RAM even for 20K+ genes.

    Peak memory: ~1 gene at a time (the current gene being processed).

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

    for i, gene_id in enumerate(gene_ids):
        npz_path = cache_dir / f"{gene_id}.npz"

        # Resume: skip if already cached
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

        # Look up gene coordinates
        gene_row = gene_annotations.filter(pl.col("gene_id") == gene_id)
        if gene_row.height == 0:
            gene_row = gene_annotations.filter(pl.col("gene_name") == gene_id)
        if gene_row.height == 0:
            logger.warning("Gene %s not found, skipping", gene_id)
            continue

        row = gene_row.row(0, named=True)
        chrom = row["chrom"]
        start, end = int(row["start"]), int(row["end"])

        # Extract sequence
        try:
            fasta_chrom = chrom.replace("chr", "") if chrom not in fasta.keys() else chrom
            sequence = str(fasta[fasta_chrom][start:end]).upper()
        except Exception as e:
            logger.warning("FASTA failed for %s: %s", gene_id, e)
            continue

        gene_len = len(sequence)

        # Labels
        raw_labels = build_splice_labels(gene_id, start, gene_len, splice_sites_df)
        labels = _CHUNKING_TO_META[raw_labels].astype(np.int64)

        # Base model scores
        base_scores = _load_base_scores(base_scores_dir, chrom, start, end, gene_len)

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

        # Add to index (only metadata in RAM)
        sp = np.where(labels != 2)[0]
        index.append(GeneIndexEntry(
            gene_id=gene_id,
            npz_path=npz_path,
            length=gene_len,
            n_splice_sites=len(sp),
            splice_positions=sp,
        ))

        # Free arrays immediately
        del sequence, base_scores, mm_features, labels, raw_labels

        if (i + 1) % 100 == 0:
            logger.info("Cached %d / %d genes", i + 1, len(gene_ids))

    logger.info("Gene cache complete: %d genes in %s", len(index), cache_dir)
    return index


def _load_base_scores(
    base_scores_dir: Path,
    chrom: str,
    start: int,
    end: int,
    gene_len: int,
) -> np.ndarray:
    """Load base model scores from prediction parquets.  [gene_len, 3]."""
    import polars as pl

    path = base_scores_dir / f"predictions_{chrom}.parquet"
    if not path.exists():
        path = base_scores_dir / f"predictions_{chrom.replace('chr', '')}.parquet"
    if not path.exists():
        return np.full((gene_len, 3), 1.0 / 3, dtype=np.float32)

    try:
        df = pl.scan_parquet(path).filter(
            (pl.col("position") >= start) & (pl.col("position") < end)
        ).select(["position", "donor_prob", "acceptor_prob", "neither_prob"]).collect()

        scores = np.full((gene_len, 3), 1.0 / 3, dtype=np.float32)
        if df.height > 0:
            pos = df["position"].to_numpy() - start
            valid = (pos >= 0) & (pos < gene_len)
            scores[pos[valid], 0] = df["donor_prob"].to_numpy()[valid]
            scores[pos[valid], 1] = df["acceptor_prob"].to_numpy()[valid]
            scores[pos[valid], 2] = df["neither_prob"].to_numpy()[valid]
        return scores
    except Exception as e:
        logger.warning("Base scores failed for %s: %s", chrom, e)
        return np.full((gene_len, 3), 1.0 / 3, dtype=np.float32)
