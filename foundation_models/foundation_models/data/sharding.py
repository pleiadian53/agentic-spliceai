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


def repack_into_shards(
    manifest: List[GeneEntryLike],
    gene_set: Set[str],
    output_dir: Path,
    split_name: str,
    windows_per_shard: int = 50_000,
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
        windows_per_shard: Max windows per shard file.
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

    shard_paths: List[Path] = []
    t0 = time.time()

    for shard_idx in range(0, len(index), windows_per_shard):
        chunk = index[shard_idx : shard_idx + windows_per_shard]
        n_win = len(chunk)

        emb_buf = np.empty((n_win, window_size, hidden_dim), dtype=np.float32)
        lbl_buf = np.empty((n_win, window_size), dtype=np.int8)

        # Read windows (handles gzip transparently)
        handle_cache: Dict[str, Any] = {}
        for i, (path, win_idx) in enumerate(chunk):
            if path not in handle_cache:
                handle_cache[path] = h5py.File(path, "r")
            f = handle_cache[path]
            emb_buf[i] = f["embeddings"][win_idx]
            lbl_buf[i] = f["labels"][win_idx]
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
