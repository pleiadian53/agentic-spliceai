"""
Streaming datasets and shard packing for foundation model training.

Embedding-based datasets (frozen models):

- :class:`HDF5WindowDataset` — reads from per-gene HDF5 files with LRU
  file-handle caching. Works with any cache (gzip or uncompressed).
- :class:`ShardedWindowDataset` — reads from pre-packed shard files for
  maximum I/O throughput at genome scale.
- :func:`repack_into_shards` — packs per-gene HDF5 into shuffled shard files.

Sequence-based datasets (fine-tuning):

- :class:`SequenceWindowDataset` — reads raw DNA from FASTA + labels.
  ~500x smaller than embedding datasets.
"""

from foundation_models.data.datasets import HDF5WindowDataset, ShardedWindowDataset
from foundation_models.data.sequence_dataset import SequenceWindowDataset
from foundation_models.data.sharding import repack_into_shards

__all__ = [
    "HDF5WindowDataset",
    "SequenceWindowDataset",
    "ShardedWindowDataset",
    "repack_into_shards",
]
