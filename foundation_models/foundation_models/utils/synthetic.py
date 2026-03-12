"""
Synthetic data utilities for local pipeline testing.

Generates random embeddings and labels that match the real data shapes,
enabling end-to-end pipeline validation without loading Evo2 (~14 GB).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_embeddings(
    n_genes: int = 3,
    seq_lengths: Optional[List[int]] = None,
    hidden_dim: int = 4096,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Generate random embeddings and exon labels for testing.

    Creates synthetic gene data with realistic exon/intron block structure
    (exons: 100-500 bp, introns: 500-5000 bp).

    Args:
        n_genes: Number of synthetic genes to generate.
        seq_lengths: Per-gene sequence lengths. If None, random 5K-50K bp.
        hidden_dim: Embedding dimension (4096 for Evo2 7B, 8192 for 40B).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (embeddings_dict, labels_dict), both keyed by gene ID.
        - embeddings_dict values: [seq_len, hidden_dim] float32 arrays
        - labels_dict values: [seq_len] uint8 arrays (1=exon, 0=intron)
    """
    rng = np.random.default_rng(seed)

    if seq_lengths is None:
        seq_lengths = rng.integers(5000, 50000, size=n_genes).tolist()
    elif len(seq_lengths) != n_genes:
        raise ValueError(
            f"seq_lengths length ({len(seq_lengths)}) != n_genes ({n_genes})"
        )

    embeddings: Dict[str, np.ndarray] = {}
    labels: Dict[str, np.ndarray] = {}

    for i, seq_len in enumerate(seq_lengths):
        gene_id = f"SYNTH_GENE_{i:04d}"

        # Random embeddings (scaled like real embeddings)
        emb = rng.standard_normal((seq_len, hidden_dim)).astype(np.float32) * 0.1
        embeddings[gene_id] = emb

        # Synthetic exon labels: alternating exon/intron blocks
        lbl = np.zeros(seq_len, dtype=np.uint8)
        pos = 0
        is_exon = False  # Start with intron
        while pos < seq_len:
            if is_exon:
                block_len = int(rng.integers(100, 500))
            else:
                block_len = int(rng.integers(500, 5000))
            end = min(pos + block_len, seq_len)
            if is_exon:
                lbl[pos:end] = 1
            pos = end
            is_exon = not is_exon
        labels[gene_id] = lbl

    logger.info(
        "Generated %d synthetic genes (hidden_dim=%d, total positions=%s)",
        n_genes, hidden_dim, sum(seq_lengths),
    )
    return embeddings, labels


def save_synthetic_embeddings(
    output_path: str,
    n_genes: int = 3,
    seq_lengths: Optional[List[int]] = None,
    hidden_dim: int = 4096,
    seed: int = 42,
) -> Tuple[Path, Dict[str, np.ndarray]]:
    """Generate synthetic embeddings and save to HDF5 + labels to NPZ.

    The HDF5 format matches Evo2Embedder._save_to_cache() so existing
    loading code works unchanged.

    Args:
        output_path: Path for the HDF5 file (e.g., "synth.h5").
        n_genes: Number of synthetic genes.
        seq_lengths: Per-gene sequence lengths (None = random 5K-50K).
        hidden_dim: Embedding dimension.
        seed: Random seed.

    Returns:
        Tuple of (hdf5_path, labels_dict).
        Labels are also saved to a companion .labels.npz file.
    """
    embeddings, labels = generate_synthetic_embeddings(
        n_genes=n_genes, seq_lengths=seq_lengths,
        hidden_dim=hidden_dim, seed=seed,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embeddings to HDF5 (same format as Evo2Embedder)
    with h5py.File(output_path, "w") as f:
        f.attrs["model_size"] = "synthetic"
        f.attrs["hidden_dim"] = hidden_dim
        for gene_id, emb in embeddings.items():
            f.create_dataset(
                gene_id, data=emb, compression="gzip", compression_opts=4,
            )

    # Save labels to companion NPZ
    labels_path = output_path.with_suffix(".labels.npz")
    np.savez_compressed(labels_path, **labels)

    logger.info("Saved embeddings to %s", output_path)
    logger.info("Saved labels to %s", labels_path)
    return output_path, labels
