"""
Sequence-level PyTorch Dataset for M*-S meta-layer training.

Provides windowed training samples with:
    - One-hot DNA sequence ``[4, W + context]``
    - Base model scores ``[W, 3]``
    - Dense multimodal features ``[C, W]``
    - Per-position labels ``[W]``

Each sample is a genomic window extracted from a pre-cached gene.
Windows are sampled with bias toward splice sites to mitigate the
extreme class imbalance (~99.4% "neither").

Label encoding follows the meta-layer convention:
    0 = donor, 1 = acceptor, 2 = neither

Note: ``build_splice_labels()`` in ``foundation_models/utils/chunking.py``
uses a different encoding (0=none, 1=acceptor, 2=donor).  The mapping
is applied once during cache construction.  If we later unify the
conventions across the codebase, the ``_CHUNKING_TO_META`` map can be
replaced with an identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass
class GeneCacheEntry:
    """Pre-computed data for a single gene."""
    gene_id: str
    chrom: str
    start: int
    end: int
    strand: str
    sequence: str
    base_scores: np.ndarray   # [L, 3] float32
    mm_features: np.ndarray   # [L, C] float32
    labels: np.ndarray        # [L] int64

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def splice_positions(self) -> np.ndarray:
        """Indices where labels != 2 (donor or acceptor)."""
        return np.where(self.labels != 2)[0]


class SequenceLevelDataset(Dataset):
    """Window-level dataset for training sequence-level meta models.

    Parameters
    ----------
    gene_cache : list of GeneCacheEntry
        Pre-computed gene data.
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
        gene_cache: List[GeneCacheEntry],
        window_size: int = 5001,
        context_padding: int = 400,
        splice_bias: float = 0.5,
        samples_per_epoch: int = 50_000,
        seed: int = 42,
    ) -> None:
        self.gene_cache = gene_cache
        self.window_size = window_size
        self.context_padding = context_padding
        self.splice_bias = splice_bias
        self.samples_per_epoch = samples_per_epoch
        self._rng = np.random.RandomState(seed)

        min_length = window_size + context_padding
        self._genes_with_splices: List[int] = []
        self._splice_positions_by_gene: Dict[int, np.ndarray] = {}
        self._valid_genes: List[int] = []

        for i, gene in enumerate(gene_cache):
            if gene.length < min_length:
                continue
            self._valid_genes.append(i)
            sp = gene.splice_positions
            if len(sp) > 0:
                self._genes_with_splices.append(i)
                self._splice_positions_by_gene[i] = sp

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
            gene = self.gene_cache[gene_idx]
            sp = self._splice_positions_by_gene[gene_idx]
            center = self._rng.choice(sp)
            jitter = self._rng.randint(-W // 4, W // 4 + 1)
            out_start = max(0, center - W // 2 + jitter)
        else:
            gene_idx = self._rng.choice(self._valid_genes)
            gene = self.gene_cache[gene_idx]
            out_start = self._rng.randint(0, max(1, gene.length - W))

        out_start = max(0, min(out_start, gene.length - W))
        out_end = out_start + W

        # Sequence with context padding
        seq_start = max(0, out_start - ctx // 2)
        seq_end = min(gene.length, out_end + (ctx - ctx // 2))
        seq_onehot = _one_hot_encode(gene.sequence[seq_start:seq_end])

        total_len = W + ctx
        if seq_onehot.shape[1] < total_len:
            padded = np.zeros((4, total_len), dtype=np.float32)
            offset = (total_len - seq_onehot.shape[1]) // 2
            padded[:, offset:offset + seq_onehot.shape[1]] = seq_onehot
            seq_onehot = padded

        base_scores = gene.base_scores[out_start:out_end]
        mm_features = gene.mm_features[out_start:out_end]
        labels = gene.labels[out_start:out_end]

        return {
            "sequence": torch.from_numpy(seq_onehot),
            "base_scores": torch.from_numpy(base_scores),
            "mm_features": torch.from_numpy(mm_features.T.copy()),  # [C, W]
            "labels": torch.from_numpy(labels),
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
# Cache building
# ---------------------------------------------------------------------------


def build_gene_cache(
    gene_ids: List[str],
    splice_sites_df: "pandas.DataFrame",
    fasta_path: str,
    base_scores_dir: Path,
    feature_extractor: "DenseFeatureExtractor",
    gene_annotations: "polars.DataFrame",
) -> List[GeneCacheEntry]:
    """Build pre-computed gene cache for training.

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
    """
    import pyfaidx
    import polars as pl
    from foundation_models.utils.chunking import build_splice_labels

    fasta = pyfaidx.Fasta(fasta_path)
    cache: List[GeneCacheEntry] = []

    for gene_id in gene_ids:
        gene_row = gene_annotations.filter(pl.col("gene_id") == gene_id)
        if gene_row.height == 0:
            gene_row = gene_annotations.filter(pl.col("gene_name") == gene_id)
        if gene_row.height == 0:
            logger.warning("Gene %s not found, skipping", gene_id)
            continue

        row = gene_row.row(0, named=True)
        chrom = row["chrom"]
        start, end = int(row["start"]), int(row["end"])
        strand = row.get("strand", "+")

        try:
            fasta_chrom = chrom.replace("chr", "") if chrom not in fasta.keys() else chrom
            sequence = str(fasta[fasta_chrom][start:end]).upper()
        except Exception as e:
            logger.warning("FASTA failed for %s: %s", gene_id, e)
            continue

        gene_len = len(sequence)

        # Labels (convert from chunking encoding to meta-layer encoding)
        raw_labels = build_splice_labels(gene_id, start, gene_len, splice_sites_df)
        labels = _CHUNKING_TO_META[raw_labels].astype(np.int64)

        # Base model scores
        base_scores = _load_base_scores(base_scores_dir, chrom, start, end, gene_len)

        # Dense multimodal features
        mm_features = feature_extractor.extract_window(chrom, start, end)

        cache.append(GeneCacheEntry(
            gene_id=gene_id, chrom=chrom, start=start, end=end,
            strand=strand, sequence=sequence,
            base_scores=base_scores, mm_features=mm_features, labels=labels,
        ))

        if len(cache) % 100 == 0:
            logger.info("Cached %d / %d genes", len(cache), len(gene_ids))

    logger.info("Gene cache complete: %d genes", len(cache))
    return cache


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
