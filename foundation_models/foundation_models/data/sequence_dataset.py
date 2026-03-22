"""
Sequence-based datasets for fine-tuning foundation models.

Unlike :mod:`foundation_models.data.datasets` which stores pre-computed
embeddings, these datasets store raw DNA sequences + labels.  This is
required for fine-tuning because embeddings change every training step.

Storage is ~500x smaller than embedding datasets: a 512-char DNA window
is 512 bytes vs 512 × 512 × 4 = 1 MB for SpliceBERT embeddings.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


class SequenceWindowDataset(torch.utils.data.Dataset):
    """Dataset that yields (DNA string, label tensor) windows from FASTA.

    Reads raw DNA sequences from a FASTA file via pyfaidx and splice site
    labels from ``splice_sites_enhanced.tsv``.  Each sample is a fixed-size
    window of DNA text + per-nucleotide labels (0=neither, 1=acceptor,
    2=donor).

    DNA windows are small (~512 bytes each) so the index + labels can fit
    entirely in memory for genome-scale datasets (~500 MB for full genome).

    Args:
        gene_entries: List of gene dicts with keys: gene_id, chrom, start,
            end, strand.  Typically from a manifest.
        gene_set: Gene IDs to include in this dataset.
        fasta_path: Path to genome FASTA file (indexed with samtools faidx).
        splice_sites_df: DataFrame with columns: gene_id, position, splice_type.
        window_size: Window size in nucleotides.
        step_size: Step between windows (for overlapping windows).
    """

    def __init__(
        self,
        gene_entries: List[Dict[str, Any]],
        gene_set: Set[str],
        fasta_path: str,
        splice_sites_df: Any,
        window_size: int = 512,
        step_size: int = 256,
    ) -> None:
        from foundation_models.utils.chunking import build_splice_labels

        self._fasta_path = str(fasta_path)
        self._fasta = None  # Lazy-loaded
        self._window_size = window_size

        # Build flat index: (chrom_key, window_start_genomic, labels_slice)
        self._windows: List[Tuple[str, int, np.ndarray]] = []
        self._class_counts = np.zeros(3, dtype=np.int64)

        t0 = time.time()
        n_genes = 0

        for entry in gene_entries:
            gene_id = entry.get("gene_id") or getattr(entry, "gene_id", None)
            if gene_id not in gene_set:
                continue
            n_genes += 1

            chrom = entry.get("chrom") or getattr(entry, "chrom", None)
            start = entry.get("start") or getattr(entry, "start", None)
            end = entry.get("end") or getattr(entry, "end", None)
            gene_len = end - start

            # Build per-nucleotide labels for this gene
            labels = build_splice_labels(
                gene_id=gene_id,
                gene_start=start,
                gene_sequence_length=gene_len,
                splice_sites_df=splice_sites_df,
            )

            # Window the labels (store in memory — labels are tiny)
            pos = 0
            while pos + window_size <= gene_len:
                win_lbl = labels[pos: pos + window_size]
                for c in range(3):
                    self._class_counts[c] += int((win_lbl == c).sum())
                self._windows.append((chrom, start + pos, win_lbl))
                pos += step_size

        elapsed = time.time() - t0
        n_splice = int(self._class_counts[1] + self._class_counts[2])
        logger.info(
            "  SequenceWindowDataset: %d genes, %d windows, %d splice sites "
            "(%.1f s)",
            n_genes, len(self._windows), n_splice, elapsed,
        )

    def _get_fasta(self) -> Any:
        """Lazy-load FASTA to avoid pickling issues with multiprocessing."""
        if self._fasta is None:
            import pyfaidx
            self._fasta = pyfaidx.Fasta(self._fasta_path)
        return self._fasta

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """Return (dna_string, label_tensor) for a window.

        Args:
            idx: Window index.

        Returns:
            Tuple of:
            - DNA string of length ``window_size`` (uppercase ACGT)
            - Label tensor of shape ``[window_size]`` (long, 0/1/2)
        """
        chrom, win_start, labels = self._windows[idx]
        fasta = self._get_fasta()

        # Resolve chromosome key (handle chr prefix mismatch)
        chrom_key = self._resolve_chrom_key(fasta, chrom)
        dna = str(fasta[chrom_key][win_start: win_start + self._window_size]).upper()

        return dna, torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _resolve_chrom_key(fasta: Any, chrom: str) -> str:
        """Resolve chromosome name to FASTA key (handle chr prefix)."""
        if chrom in fasta:
            return chrom
        # Try adding/removing chr prefix
        if chrom.startswith("chr"):
            bare = chrom[3:]
            if bare in fasta:
                return bare
        else:
            prefixed = f"chr{chrom}"
            if prefixed in fasta:
                return prefixed
        return chrom  # Fall through — will error on access

    def close(self) -> None:
        """Close FASTA file handle."""
        if self._fasta is not None:
            self._fasta.close()
            self._fasta = None

    @property
    def class_counts(self) -> np.ndarray:
        """Per-class sample counts ``[num_classes]``, computed at init."""
        return self._class_counts

    @property
    def window_size(self) -> int:
        """Window size in nucleotides."""
        return self._window_size
