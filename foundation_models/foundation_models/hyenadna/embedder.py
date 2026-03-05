"""
HyenaDNA Embedding Extractor

Efficient extraction and caching of HyenaDNA embeddings.
Mirrors Evo2Embedder interface for consistency.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

from foundation_models.hyenadna.config import HyenaDNAConfig
from foundation_models.hyenadna.model import HyenaDNAModel

logger = logging.getLogger(__name__)


class HyenaDNAEmbedder:
    """Extract and cache HyenaDNA embeddings for sequences.

    Handles large sequences by chunking, and caches embeddings to HDF5
    for efficient reuse.

    Example::

        >>> embedder = HyenaDNAEmbedder(model_size="medium-160k")
        >>> embeddings = embedder.encode("ATCG" * 1000)
        >>> print(embeddings.shape)  # [4000, 256]

        >>> gene_embeddings = embedder.encode_batch(
        ...     sequences={"BRCA1": seq1, "TP53": seq2},
        ...     cache_path="embeddings.h5"
        ... )
    """

    def __init__(
        self,
        model_size: str = "medium-160k",
        device: str = "auto",
        config: Optional[HyenaDNAConfig] = None,
    ) -> None:
        if config is None:
            config = HyenaDNAConfig(model_size=model_size, device=device)

        self.config = config
        self.model = HyenaDNAModel(config)

    def encode(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
    ) -> torch.Tensor:
        """Encode a single DNA sequence to embeddings.

        Args:
            sequence: DNA sequence.
            chunk_size: If provided, split sequence into chunks.
            overlap: Overlap between chunks.

        Returns:
            Embeddings of shape ``[seq_len, hidden_dim]``.
        """
        if chunk_size is None:
            chunk_size = self.config.max_length

        if len(sequence) <= chunk_size:
            return self.model.encode(sequence)

        return self._encode_chunked(sequence, chunk_size, overlap)

    def _encode_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """Encode long sequence by chunking."""
        seq_len = len(sequence)
        step = chunk_size - overlap
        all_embeddings = []

        for start in range(0, seq_len, step):
            end = min(start + chunk_size, seq_len)
            chunk = sequence[start:end]

            chunk_emb = self.model.encode(chunk)

            if overlap > 0 and start > 0:
                keep_start = overlap // 2
                chunk_emb = chunk_emb[keep_start:]

            all_embeddings.append(chunk_emb)

            if end >= seq_len:
                break

        return torch.cat(all_embeddings, dim=0)

    def encode_batch(
        self,
        sequences: Dict[str, str],
        cache_path: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Encode multiple sequences and optionally cache to HDF5.

        Args:
            sequences: Dict mapping IDs to sequences.
            cache_path: If provided, cache embeddings to HDF5.
            chunk_size: Chunk size for long sequences.
            show_progress: Show progress bar.

        Returns:
            Dict mapping IDs to embedding tensors.
        """
        embeddings = {}

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                logger.info("Loading cached embeddings from %s", cache_path)
                return self._load_from_cache(cache_path, list(sequences.keys()))

        iterator = tqdm(sequences.items()) if show_progress else sequences.items()

        for seq_id, sequence in iterator:
            if show_progress:
                iterator.set_description(f"Encoding {seq_id}")

            emb = self.encode(sequence, chunk_size=chunk_size)
            embeddings[seq_id] = emb

        if cache_path is not None:
            self._save_to_cache(embeddings, cache_path)

        return embeddings

    def _save_to_cache(
        self, embeddings: Dict[str, torch.Tensor], cache_path: Path
    ) -> None:
        """Save embeddings to HDF5 cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(cache_path, "w") as f:
            f.attrs["model"] = f"hyenadna-{self.config.model_size}"
            f.attrs["hidden_dim"] = self.model.hidden_dim

            for seq_id, emb in embeddings.items():
                f.create_dataset(
                    seq_id,
                    data=emb.cpu().numpy(),
                    compression="gzip",
                    compression_opts=4,
                )

        logger.info("Cached embeddings to %s", cache_path)

    def _load_from_cache(
        self,
        cache_path: Path,
        seq_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load embeddings from HDF5 cache."""
        embeddings = {}

        with h5py.File(cache_path, "r") as f:
            ids_to_load = seq_ids if seq_ids is not None else list(f.keys())

            for seq_id in ids_to_load:
                if seq_id in f:
                    emb = torch.from_numpy(f[seq_id][:])
                    embeddings[seq_id] = emb

        return embeddings

    def __repr__(self) -> str:
        return (
            f"HyenaDNAEmbedder(model_size={self.config.model_size}, "
            f"device={self.model.device})"
        )
