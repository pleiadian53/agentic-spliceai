"""
Evo2 Embedding Extractor

Efficient extraction and caching of Evo2 embeddings for downstream tasks.
"""

import torch
import h5py
import numpy as np
from typing import Optional, Union, List, Dict
from pathlib import Path
from tqdm import tqdm

from foundation_models.evo2.config import Evo2Config
from foundation_models.evo2.model import Evo2Model


class Evo2Embedder:
    """
    Extract and cache Evo2 embeddings for sequences.
    
    Handles large sequences by chunking, and caches embeddings to HDF5
    for efficient reuse.
    
    Example:
        >>> embedder = Evo2Embedder(model_size="7b", quantize=True)
        >>> embeddings = embedder.encode("ATCG" * 1000)
        >>> print(embeddings.shape)  # [4000, 2560]
        
        >>> # Extract and cache for multiple genes
        >>> gene_embeddings = embedder.encode_batch(
        ...     sequences={"BRCA1": seq1, "TP53": seq2},
        ...     cache_path="embeddings.h5"
        ... )
    """
    
    def __init__(
        self,
        model_size: str = "7b",
        quantize: bool = False,
        device: str = "auto",
        config: Optional[Evo2Config] = None,
    ):
        """
        Initialize embedder.
        
        Args:
            model_size: "7b" or "40b"
            quantize: Whether to quantize model
            device: Device to use
            config: Optional Evo2Config (overrides other args)
        """
        if config is None:
            config = Evo2Config(
                model_size=model_size,
                quantize=quantize,
                device=device,
            )
        
        self.config = config
        self.model = Evo2Model(config)
    
    def encode(
        self,
        sequence: str,
        chunk_size: Optional[int] = None,
        overlap: int = 0,
    ) -> torch.Tensor:
        """
        Encode a single DNA sequence to embeddings.
        
        Args:
            sequence: DNA sequence
            chunk_size: If provided, split sequence into chunks
            overlap: Overlap between chunks (for context)
        
        Returns:
            embeddings: [seq_len, hidden_dim]
        """
        seq_len = len(sequence)
        
        # Use chunking if sequence is long or chunk_size specified
        if chunk_size is None:
            chunk_size = self.config.max_length
        
        if seq_len <= chunk_size:
            # Small sequence - process directly
            return self.model.encode(sequence)
        else:
            # Large sequence - process in chunks
            return self._encode_chunked(sequence, chunk_size, overlap)
    
    def _encode_chunked(
        self,
        sequence: str,
        chunk_size: int,
        overlap: int,
    ) -> torch.Tensor:
        """
        Encode long sequence by chunking.
        
        Args:
            sequence: DNA sequence
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        
        Returns:
            embeddings: [seq_len, hidden_dim]
        """
        seq_len = len(sequence)
        step = chunk_size - overlap
        
        all_embeddings = []
        positions = []
        
        # Process chunks
        for start in range(0, seq_len, step):
            end = min(start + chunk_size, seq_len)
            chunk = sequence[start:end]
            
            # Get embeddings for chunk
            chunk_emb = self.model.encode(chunk)  # [chunk_len, hidden_dim]
            
            # Handle overlap - keep only non-overlapping part
            if overlap > 0 and start > 0:
                # Skip overlapping positions from previous chunk
                keep_start = overlap // 2
                chunk_emb = chunk_emb[keep_start:]
                chunk_pos = list(range(start + keep_start, end))
            else:
                chunk_pos = list(range(start, end))
            
            all_embeddings.append(chunk_emb)
            positions.extend(chunk_pos)
            
            if end >= seq_len:
                break
        
        # Concatenate all chunks
        embeddings = torch.cat(all_embeddings, dim=0)
        
        return embeddings
    
    def encode_batch(
        self,
        sequences: Dict[str, str],
        cache_path: Optional[Path] = None,
        chunk_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple sequences and optionally cache to HDF5.
        
        Args:
            sequences: Dict mapping IDs to sequences
            cache_path: If provided, cache embeddings to HDF5
            chunk_size: Chunk size for long sequences
            show_progress: Show progress bar
        
        Returns:
            embeddings: Dict mapping IDs to embeddings
        """
        embeddings = {}
        
        # Check if cache exists
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f"Loading cached embeddings from {cache_path}")
                return self._load_from_cache(cache_path, sequences.keys())
        
        # Extract embeddings
        iterator = tqdm(sequences.items()) if show_progress else sequences.items()
        
        for seq_id, sequence in iterator:
            if show_progress:
                iterator.set_description(f"Encoding {seq_id}")
            
            emb = self.encode(sequence, chunk_size=chunk_size)
            embeddings[seq_id] = emb
        
        # Cache if requested
        if cache_path is not None:
            self._save_to_cache(embeddings, cache_path)
        
        return embeddings
    
    def _save_to_cache(self, embeddings: Dict[str, torch.Tensor], cache_path: Path):
        """Save embeddings to HDF5 cache."""
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(cache_path, 'w') as f:
            # Save metadata
            f.attrs['model_size'] = self.config.model_size
            f.attrs['hidden_dim'] = self.model.hidden_dim
            
            # Save embeddings
            for seq_id, emb in embeddings.items():
                f.create_dataset(
                    seq_id,
                    data=emb.cpu().numpy(),
                    compression="gzip",
                    compression_opts=4,
                )
        
        print(f"✓ Cached embeddings to {cache_path}")
    
    def _load_from_cache(
        self,
        cache_path: Path,
        seq_ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Load embeddings from HDF5 cache."""
        embeddings = {}
        
        with h5py.File(cache_path, 'r') as f:
            # Load all or specified IDs
            ids_to_load = seq_ids if seq_ids is not None else list(f.keys())
            
            for seq_id in ids_to_load:
                if seq_id in f:
                    emb = torch.from_numpy(f[seq_id][:])
                    embeddings[seq_id] = emb
        
        return embeddings
    
    def __repr__(self) -> str:
        return (
            f"Evo2Embedder(model_size={self.config.model_size}, "
            f"device={self.model.device})"
        )
