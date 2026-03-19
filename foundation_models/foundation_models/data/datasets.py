"""
PyTorch datasets for streaming windowed embeddings from HDF5 cache.

Two strategies for different scales:

- :class:`HDF5WindowDataset` — per-gene HDF5 files with LRU handle cache.
  Good for single-chromosome runs or when sharding is not worth the setup.
- :class:`ShardedWindowDataset` — pre-packed shard files with all handles
  kept open.  Best for genome-scale training (10-50x faster than per-gene).

Both classes expose ``.class_counts`` (per-class label counts computed at
init by scanning labels only) and ``.close()`` for cleanup.

Typical usage::

    from foundation_models.data import HDF5WindowDataset
    from torch.utils.data import DataLoader

    dataset = HDF5WindowDataset(manifest, train_genes)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for emb, lbl in loader:
        ...
    dataset.close()
"""

import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, List, Protocol, Set, Tuple, runtime_checkable

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


@runtime_checkable
class GeneEntryLike(Protocol):
    """Minimal interface for gene manifest entries."""

    gene_id: str
    n_windows: int
    hdf5_path: str


class HDF5WindowDataset(torch.utils.data.Dataset):
    """Streaming dataset that reads windowed embeddings from per-gene HDF5 files.

    Uses an LRU cache of open file handles to avoid the overhead of
    opening/closing HDF5 files on every ``__getitem__`` call.  Also scans
    labels at init time to compute per-class counts (needed for focal loss
    class weights) without loading embeddings into memory.

    Args:
        manifest: List of gene entries (must have ``gene_id``, ``n_windows``,
            ``hdf5_path`` attributes).
        gene_set: Gene IDs to include.
        max_open_files: Maximum number of simultaneously open HDF5 handles.
    """

    def __init__(
        self,
        manifest: List[GeneEntryLike],
        gene_set: Set[str],
        max_open_files: int = 64,
    ):
        self.index: List[Tuple[str, int]] = []
        self._class_counts = np.zeros(3, dtype=np.int64)
        self._handle_cache: OrderedDict[str, Any] = OrderedDict()
        self._max_open_files = max_open_files

        n_genes = 0
        t0 = time.time()
        for entry in manifest:
            if entry.gene_id not in gene_set:
                continue
            n_genes += 1
            for i in range(entry.n_windows):
                self.index.append((entry.hdf5_path, i))
            # Scan labels for class counts (labels are small: [n_win, W] int8)
            with h5py.File(entry.hdf5_path, "r") as f:
                lbl = f["labels"][:]
                for c in range(3):
                    self._class_counts[c] += int((lbl == c).sum())

        elapsed = time.time() - t0
        logger.info(
            "  HDF5WindowDataset: %d genes, %d windows, scanned labels in %.1f s",
            n_genes, len(self.index), elapsed,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, window_idx = self.index[idx]
        f = self._get_handle(path)
        emb = f["embeddings"][window_idx]
        lbl = f["labels"][window_idx]
        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.long),
        )

    def _get_handle(self, path: str) -> Any:
        """Return an open HDF5 file handle, using LRU cache."""
        if path in self._handle_cache:
            self._handle_cache.move_to_end(path)
            return self._handle_cache[path]
        # Evict oldest if at capacity
        if len(self._handle_cache) >= self._max_open_files:
            _, old_f = self._handle_cache.popitem(last=False)
            old_f.close()
        f = h5py.File(path, "r")
        self._handle_cache[path] = f
        return f

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for f in self._handle_cache.values():
            f.close()
        self._handle_cache.clear()

    @property
    def class_counts(self) -> np.ndarray:
        """Per-class sample counts ``[num_classes]``, computed at init."""
        return self._class_counts


class ShardedWindowDataset(torch.utils.data.Dataset):
    """Fast dataset backed by pre-packed, uncompressed shard files.

    Each shard contains a contiguous ``embeddings [N, W, H]`` and
    ``labels [N, W]`` dataset.  All shards are opened at init and kept
    open (typically < 20 files).

    Also scans labels at init to compute per-class counts for class weights.

    Args:
        shard_paths: Sorted list of shard HDF5 file paths.
    """

    def __init__(self, shard_paths: List[Path]):
        self.shards: List[Any] = []
        self.index: List[Tuple[int, int]] = []  # (shard_idx, win_idx)
        self._class_counts = np.zeros(3, dtype=np.int64)

        for shard_idx, path in enumerate(shard_paths):
            f = h5py.File(path, "r")
            self.shards.append(f)
            n_win = f["embeddings"].shape[0]
            for i in range(n_win):
                self.index.append((shard_idx, i))
            # Scan labels for class counts
            lbl = f["labels"][:]
            for c in range(3):
                self._class_counts[c] += int((lbl == c).sum())

        logger.info(
            "  ShardedWindowDataset: %d shards, %d windows",
            len(self.shards), len(self.index),
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        shard_idx, win_idx = self.index[idx]
        f = self.shards[shard_idx]
        emb = f["embeddings"][win_idx]
        lbl = f["labels"][win_idx]
        return (
            torch.tensor(emb, dtype=torch.float32),
            torch.tensor(lbl, dtype=torch.long),
        )

    def close(self) -> None:
        """Close all shard file handles."""
        for f in self.shards:
            f.close()
        self.shards.clear()

    @property
    def class_counts(self) -> np.ndarray:
        """Per-class sample counts ``[num_classes]``, computed at init."""
        return self._class_counts
