"""
Shard packing for fast training I/O at genome scale.

Converts per-gene HDF5 cache files (potentially gzip-compressed) into
large, uncompressed shard files with shuffled windows.  This is a one-time
cost that enables 10-50x faster training I/O by replacing many small
random reads with sequential reads from a few large files.

Typical usage::

    from foundation_models.data import repack_into_shards

    shard_paths = repack_into_shards(
        manifest, train_genes, output_dir, "train",
    )
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Protocol, Set, Tuple, runtime_checkable

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class GeneEntryLike(Protocol):
    """Minimal interface for gene manifest entries."""

    gene_id: str
    n_windows: int
    hdf5_path: str


# Default memory budget for shard buffers (4 GB)
_DEFAULT_SHARD_MEMORY_BUDGET_BYTES = 4 * 1024**3


def _compute_windows_per_shard(
    window_size: int,
    hidden_dim: int,
    memory_budget: int = _DEFAULT_SHARD_MEMORY_BUDGET_BYTES,
) -> int:
    """Compute max windows per shard that fits within memory budget."""
    bytes_per_window = window_size * hidden_dim * 4  # float32
    windows = max(100, memory_budget // bytes_per_window)
    return windows


def repack_into_shards(
    manifest: List[GeneEntryLike],
    gene_set: Set[str],
    output_dir: Path,
    split_name: str,
    windows_per_shard: int = 0,
    max_shard_memory_gb: float = 4.0,
    seed: int = 42,
) -> List[Path]:
    """Pack per-gene HDF5 windows into large contiguous shard files.

    Reads from existing per-gene cache (compressed or not), shuffles, and
    writes uncompressed shard files for fast sequential training I/O.

    Args:
        manifest: Gene entries from Phase 1.
        gene_set: Gene IDs to include in this split.
        output_dir: Root output directory.
        split_name: ``"train"`` or ``"val"`` — used in shard filenames.
        windows_per_shard: Max windows per shard file.  If 0 (default),
            auto-computed from ``max_shard_memory_gb`` and embedding dims.
        max_shard_memory_gb: Memory budget per shard buffer in GB.
            Used when ``windows_per_shard=0``.  Default 4 GB.
        seed: Random seed for shuffle reproducibility.

    Returns:
        List of shard file paths.
    """
    # Build flat index
    index: List[Tuple[str, int]] = []
    for entry in manifest:
        if entry.gene_id in gene_set:
            for i in range(entry.n_windows):
                index.append((entry.hdf5_path, i))

    if not index:
        return []

    rng = np.random.RandomState(seed)
    rng.shuffle(index)

    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # Probe dimensions from first file
    path0, _ = index[0]
    with h5py.File(path0, "r") as f:
        _, window_size, hidden_dim = f["embeddings"].shape

    # Auto-compute shard size to fit within memory budget
    if windows_per_shard <= 0:
        budget_bytes = int(max_shard_memory_gb * 1024**3)
        windows_per_shard = _compute_windows_per_shard(
            window_size, hidden_dim, budget_bytes,
        )

    shard_mem_gb = windows_per_shard * window_size * hidden_dim * 4 / 1e9
    logger.info(
        "  Shard config: %d windows/shard (%.1f GB buffer), "
        "window=%d, hidden=%d, total=%d windows",
        windows_per_shard, shard_mem_gb,
        window_size, hidden_dim, len(index),
    )

    shard_paths: List[Path] = []
    t0 = time.time()

    for shard_idx in range(0, len(index), windows_per_shard):
        chunk = index[shard_idx : shard_idx + windows_per_shard]
        n_win = len(chunk)

        emb_buf = np.empty((n_win, window_size, hidden_dim), dtype=np.float32)
        lbl_buf = np.empty((n_win, window_size), dtype=np.int8)

        # Read windows (handles gzip transparently)
        handle_cache: Dict[str, Any] = {}
        log_interval = max(1, n_win // 10)
        for i, (path, win_idx) in enumerate(chunk):
            if path not in handle_cache:
                handle_cache[path] = h5py.File(path, "r")
            f = handle_cache[path]
            emb_buf[i] = f["embeddings"][win_idx]
            lbl_buf[i] = f["labels"][win_idx]
            if (i + 1) % log_interval == 0:
                logger.info(
                    "    Reading: %d/%d windows (%.0f s)",
                    i + 1, n_win, time.time() - t0,
                )
        for f in handle_cache.values():
            f.close()

        shard_num = shard_idx // windows_per_shard
        shard_path = shard_dir / f"{split_name}_shard_{shard_num:03d}.h5"
        with h5py.File(shard_path, "w") as f:
            f.create_dataset("embeddings", data=emb_buf)
            f.create_dataset("labels", data=lbl_buf)
        shard_paths.append(shard_path)

        logger.info(
            "  Shard %s_%03d: %d windows (%.1f s)",
            split_name, shard_num, n_win, time.time() - t0,
        )

    logger.info(
        "  %s shards: %d files, %d total windows, %.1f s",
        split_name, len(shard_paths), len(index), time.time() - t0,
    )
    return shard_paths
