"""
Pack per-gene ``.npz`` cache into per-chromosome HDF5 shards.

Converting 12K+ individual files into ~24 large shards eliminates
per-file open/close overhead in DataLoader workers and enables HDF5
slice reads (only the training window is read, not the full gene).

Typical usage::

    from agentic_spliceai.splice_engine.meta_layer.data.shard_packing import (
        pack_gene_cache_to_shards,
    )

    index_path = pack_gene_cache_to_shards(
        gene_index, gene_annotations, shard_dir=Path("shards/train"),
    )

See ``docs/ml_engineering/data_pipeline/io_bottlenecks_dataloader.md``
for background on why sharding matters.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# DNA nucleotide → uint8 encoding for HDF5 storage.
# One-hot decoding happens at training time in __getitem__.
_NT_TO_UINT8 = {
    "A": 0, "a": 0,
    "C": 1, "c": 1,
    "G": 2, "g": 2,
    "T": 3, "t": 3,
    "N": 4, "n": 4,
}


def _encode_sequence_uint8(seq: str) -> np.ndarray:
    """Encode DNA string to uint8 array (A=0, C=1, G=2, T=3, N=4)."""
    out = np.full(len(seq), 4, dtype=np.uint8)  # default N
    for i, nt in enumerate(seq):
        out[i] = _NT_TO_UINT8.get(nt, 4)
    return out


def pack_gene_cache_to_shards(
    gene_index: List,
    gene_annotations: "polars.DataFrame",
    shard_dir: Path,
    force_rebuild: bool = False,
) -> Path:
    """Pack per-gene .npz cache into per-chromosome HDF5 shards.

    Each chromosome gets one ``.h5`` file containing all genes as
    named HDF5 groups.  A ``shard_index.json`` is written alongside
    the shards mapping gene_id → shard metadata.

    Parameters
    ----------
    gene_index:
        List of ``GeneIndexEntry`` from ``build_gene_cache()``.
    gene_annotations:
        Polars DataFrame with gene_id, chrom columns.
    shard_dir:
        Output directory for shard ``.h5`` files and index.
    force_rebuild:
        If True, rebuild all shards even if they exist.

    Returns
    -------
    Path to ``shard_index.json``.
    """
    import polars as pl

    shard_dir.mkdir(parents=True, exist_ok=True)

    # Build gene_id → chrom lookup
    gene_to_chrom: Dict[str, str] = {}
    for entry in gene_index:
        row = gene_annotations.filter(pl.col("gene_id") == entry.gene_id)
        if row.height == 0:
            row = gene_annotations.filter(pl.col("gene_name") == entry.gene_id)
        if row.height > 0:
            gene_to_chrom[entry.gene_id] = row.row(0, named=True)["chrom"]

    # Group by chromosome
    chrom_to_entries: Dict[str, List] = defaultdict(list)
    for entry in gene_index:
        chrom = gene_to_chrom.get(entry.gene_id)
        if chrom:
            chrom_to_entries[chrom].append(entry)

    # Build shards
    shard_index: Dict[str, dict] = {}
    n_packed = 0

    for chrom, entries in sorted(chrom_to_entries.items()):
        shard_path = shard_dir / f"genes_{chrom}.h5"
        expected_genes = {e.gene_id for e in entries}

        # Skip if shard is complete
        if not force_rebuild and shard_path.exists():
            try:
                with h5py.File(shard_path, "r") as f:
                    existing_genes = set(f.keys())
                if expected_genes <= existing_genes:
                    logger.info(
                        "Shard %s: %d genes already packed, skipping",
                        shard_path.name, len(entries),
                    )
                    # Still need to populate the index
                    for entry in entries:
                        shard_index[entry.gene_id] = {
                            "shard_file": str(shard_path),
                            "chrom": chrom,
                            "length": entry.length,
                            "n_splice_sites": entry.n_splice_sites,
                            "splice_positions": entry.splice_positions.tolist(),
                        }
                    n_packed += len(entries)
                    continue
            except Exception:
                pass  # Corrupted shard, rebuild

        # Build shard from .npz files
        logger.info("Packing shard %s: %d genes", shard_path.name, len(entries))

        # Delete partial shard if it exists
        if shard_path.exists():
            shard_path.unlink()

        with h5py.File(shard_path, "w") as f:
            for entry in entries:
                try:
                    data = np.load(entry.npz_path, allow_pickle=True)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", entry.npz_path, e)
                    continue

                grp = f.create_group(entry.gene_id)
                grp.create_dataset(
                    "base_scores", data=data["base_scores"],
                    dtype="float32",
                )
                grp.create_dataset(
                    "mm_features", data=data["mm_features"],
                    dtype="float32",
                )
                grp.create_dataset(
                    "labels", data=data["labels"],
                    dtype="int64",
                )
                # Encode sequence as uint8 for efficient slice reads
                seq_str = str(data["sequence"])
                grp.create_dataset(
                    "sequence", data=_encode_sequence_uint8(seq_str),
                    dtype="uint8",
                )
                grp.attrs["length"] = int(data["length"])

                shard_index[entry.gene_id] = {
                    "shard_file": str(shard_path),
                    "chrom": chrom,
                    "length": entry.length,
                    "n_splice_sites": entry.n_splice_sites,
                    "splice_positions": entry.splice_positions.tolist(),
                }
                n_packed += 1

    # Write index
    index_path = shard_dir / "shard_index.json"
    with open(index_path, "w") as f:
        json.dump({"genes": shard_index, "n_genes": n_packed}, f)

    logger.info(
        "Shard packing complete: %d genes → %d shards in %s",
        n_packed, len(chrom_to_entries), shard_dir,
    )
    return index_path


def verify_shard_integrity(shard_dir: Path) -> bool:
    """Check shard_index.json and all referenced .h5 files are valid.

    Returns True if all shards exist and contain the expected genes.
    """
    index_path = shard_dir / "shard_index.json"
    if not index_path.exists():
        logger.warning("Missing shard_index.json in %s", shard_dir)
        return False

    with open(index_path) as f:
        data = json.load(f)

    genes = data.get("genes", {})
    if not genes:
        logger.warning("Empty shard index")
        return False

    # Group by shard file and verify
    shard_to_genes: Dict[str, List[str]] = defaultdict(list)
    for gene_id, meta in genes.items():
        shard_to_genes[meta["shard_file"]].append(gene_id)

    for shard_file, expected_genes in shard_to_genes.items():
        shard_path = Path(shard_file)
        if not shard_path.exists():
            logger.warning("Missing shard file: %s", shard_path)
            return False
        try:
            with h5py.File(shard_path, "r") as f:
                existing = set(f.keys())
                missing = set(expected_genes) - existing
                if missing:
                    logger.warning(
                        "Shard %s missing %d genes", shard_path.name, len(missing),
                    )
                    return False
        except Exception as e:
            logger.warning("Cannot read shard %s: %s", shard_path.name, e)
            return False

    logger.info("Shard integrity OK: %d genes, %d shards", len(genes), len(shard_to_genes))
    return True
