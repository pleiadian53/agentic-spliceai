"""
Streaming Foundation Model Scalar Feature Extraction

Computes per-position scalar features (PCA + norm + gradient) from foundation
model embeddings **without saving intermediate embeddings to disk**. This is
the ephemeral/streaming counterpart to storing 4-64 GB of raw embedding
parquets — total pod storage is ~65 MB.

Strand-aware extraction: for causal models (Evo2, HyenaDNA) each gene is
encoded on both the forward strand and the reverse complement. The RC
embeddings are flipped and averaged with the forward embeddings so that
every position sees both upstream and downstream context. Bidirectional
models (SpliceBERT) already see full context from a single pass.

Two-phase pipeline:
  Phase 1: Fit IncrementalPCA on training chromosomes (streaming, one gene at a time)
  Phase 2: Compute 8 scalar features per sampled position, write per-chromosome parquets

Requires:
  - GPU pod with foundation_models package installed
  - Existing feature parquets from 06_multimodal_genome_workflow.py (for sampled positions)
  - Reference FASTA and gene annotations (auto-resolved via registry)

Output:
  {output_dir}/
      pca_artifacts.npz             # PCA components + mean (~200 KB)
      fm_scalars_chr1.parquet       # 8 scalar columns per sampled position
      fm_scalars_chr2.parquet
      ...
      extraction_summary.json       # Timing, counts, metadata

Usage:
    # Full genome (on GPU pod)
    python 07_streaming_fm_scalars.py \\
        --foundation-model evo2 --model-size 7b \\
        --feature-dir output/features/openspliceai/ \\
        --chromosomes all \\
        -o /runpod-volume/output/fm_scalars/evo2_7b/

    # Single chromosome test
    python 07_streaming_fm_scalars.py \\
        --foundation-model splicebert \\
        --feature-dir output/features/openspliceai/ \\
        --chromosomes 22 \\
        -o /tmp/fm_scalars_test/

    # Phase 2 only (reuse existing PCA artifacts)
    python 07_streaming_fm_scalars.py \\
        --foundation-model evo2 --model-size 7b \\
        --feature-dir output/features/openspliceai/ \\
        --chromosomes all \\
        --pca-artifacts /runpod-volume/output/fm_scalars/evo2_7b/pca_artifacts.npz \\
        -o /runpod-volume/output/fm_scalars/evo2_7b/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Canonical chromosomes
CANONICAL_CHROMS = [str(c) for c in range(1, 23)] + ["X", "Y"]

# SpliceAI chromosome split (Jaganathan et al., 2019)
# Test: chr1, 3, 5, 7, 9 — Training: everything else
SPLICEAI_TEST_CHROMS = {"1", "3", "5", "7", "9"}

# Default PCA components
DEFAULT_PCA_COMPONENTS = 6


# ---------------------------------------------------------------------------
# Phase 1: Streaming IncrementalPCA fit
# ---------------------------------------------------------------------------

def fit_pca_streaming(
    model: "BaseEmbeddingModel",
    feature_dir: Path,
    train_chroms: List[str],
    n_components: int,
    fasta_path: Path,
    gtf_path: Path,
    max_context: int,
    max_positions_per_gene: int = 500,
    max_gene_length: int = 2_000_000,
) -> dict:
    """Fit IncrementalPCA on training chromosome embeddings (streaming).

    Processes one gene at a time: extract full-gene embedding on GPU,
    subsample positions for PCA, call partial_fit(), discard embedding.

    Parameters
    ----------
    model : BaseEmbeddingModel
        Loaded foundation model.
    feature_dir : Path
        Directory with feature parquets (for sampled position lookup).
    train_chroms : list of str
        Training chromosome names (bare, e.g., '2', '10').
    n_components : int
        Number of PCA components.
    fasta_path : Path
        Reference FASTA path.
    gtf_path : Path
        Gene annotation GTF path.
    max_positions_per_gene : int
        Max positions to feed into partial_fit per gene (memory control).
    max_gene_length : int
        Skip genes longer than this (avoid OOM).

    Returns
    -------
    dict
        PCA artifacts: components, mean, explained_variance_ratio.
    """
    import torch
    from sklearn.decomposition import IncrementalPCA
    import pyfaidx

    from agentic_spliceai.splice_engine.base_layer.data.preparation import load_gene_annotations

    ipca = IncrementalPCA(n_components=n_components)
    hidden_dim = model.metadata().hidden_dim

    fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True)
    total_genes = 0
    total_positions = 0

    for chrom in train_chroms:
        chrom_chr = chrom if chrom.startswith("chr") else f"chr{chrom}"

        # Load sampled positions for this chromosome
        sampled_positions = _load_sampled_positions(feature_dir, chrom_chr)
        if sampled_positions is None:
            logger.warning("No feature parquet for %s. Skipping.", chrom_chr)
            continue

        # Load gene annotations
        print(f"[PCA] {chrom_chr}: loading gene annotations...", flush=True)
        gene_data = load_gene_annotations(gtf_path=gtf_path, chromosomes=[chrom_chr])
        if gene_data.height == 0:
            continue

        # Group sampled positions by gene_id
        pos_by_gene = sampled_positions.group_by("gene_id").agg(
            pl.col("position").sort()
        )
        gene_positions = {
            row["gene_id"]: np.array(row["position"], dtype=np.int64)
            for row in pos_by_gene.iter_rows(named=True)
        }

        n_genes = gene_data.height
        n_genes_with_positions = len(gene_positions)
        print(
            f"[PCA] {chrom_chr}: {n_genes} genes, {n_genes_with_positions} "
            f"with sampled positions — starting extraction", flush=True
        )

        chrom_t0 = time.monotonic()
        last_log_time = chrom_t0
        for gene_idx, row in enumerate(gene_data.iter_rows(named=True), 1):
            gene_id = row["gene_id"]
            gene_start = int(row["start"])
            gene_end = int(row["end"])
            gene_len = gene_end - gene_start

            if gene_len > max_gene_length:
                logger.debug("Skipping %s (length %d > %d)", gene_id, gene_len, max_gene_length)
                continue

            positions = gene_positions.get(gene_id)
            if positions is None or len(positions) == 0:
                continue

            # Extract embeddings at sampled positions only (memory-efficient)
            fasta_chrom = chrom_chr if chrom_chr in fasta else chrom
            if fasta_chrom not in fasta:
                continue

            gene_seq = str(fasta[fasta_chrom][gene_start:gene_end]).upper()
            if len(gene_seq) == 0:
                continue

            local_indices = positions - gene_start
            valid_mask = (local_indices >= 0) & (local_indices < len(gene_seq))
            local_indices = local_indices[valid_mask]

            if len(local_indices) == 0:
                continue

            gene_t0 = time.monotonic()
            try:
                pos_embeddings = _extract_gene_embeddings(
                    model, gene_seq, max_context, local_indices,
                )
            except Exception as e:
                logger.warning("Failed to extract %s: %s", gene_id, e)
                continue
            gene_elapsed = time.monotonic() - gene_t0

            # Subsample for PCA if too many positions
            if len(pos_embeddings) > max_positions_per_gene:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(pos_embeddings), max_positions_per_gene, replace=False)
                pos_embeddings = pos_embeddings[idx]

            # IncrementalPCA requires batch_size >= n_components
            if len(pos_embeddings) >= n_components:
                ipca.partial_fit(pos_embeddings.astype(np.float64))

            total_genes += 1
            total_positions += len(pos_embeddings)

            # Log every processed gene for the first 3, then every 60s
            now = time.monotonic()
            if total_genes <= 3 or (now - last_log_time) >= 60:
                elapsed = now - chrom_t0
                rate = total_genes / elapsed if elapsed > 0 else 0
                print(
                    f"  [PCA] {chrom_chr}: gene {gene_idx}/{n_genes} "
                    f"({gene_id}, {gene_len}bp, {gene_elapsed:.1f}s) | "
                    f"{total_genes} genes extracted, {total_positions} "
                    f"positions, {rate * 60:.1f} genes/min",
                    flush=True,
                )
                last_log_time = now

        logger.info(
            "[PCA] %s complete: %d genes, %d cumulative positions",
            chrom_chr, total_genes, total_positions,
        )

    fasta.close()

    if total_positions < n_components:
        logger.error(
            "Not enough positions for PCA (%d < %d components). "
            "Check feature_dir and train chromosomes.",
            total_positions, n_components,
        )
        sys.exit(1)

    logger.info(
        "PCA fit complete: %d genes, %d positions, %d components, "
        "explained_variance_ratio=%.4f (top-%d)",
        total_genes, total_positions, n_components,
        sum(ipca.explained_variance_ratio_), n_components,
    )

    return {
        "components": ipca.components_,  # (k, hidden_dim)
        "mean": ipca.mean_,  # (hidden_dim,)
        "explained_variance_ratio": ipca.explained_variance_ratio_,  # (k,)
    }


# ---------------------------------------------------------------------------
# Phase 2: Streaming scalar feature extraction
# ---------------------------------------------------------------------------

def extract_scalars_streaming(
    model: "BaseEmbeddingModel",
    feature_dir: Path,
    chromosomes: List[str],
    pca_artifacts: dict,
    output_dir: Path,
    fasta_path: Path,
    gtf_path: Path,
    max_context: int,
    max_gene_length: int = 2_000_000,
) -> dict:
    """Compute scalar features for all sampled positions (streaming).

    For each gene: extract full-gene embedding → compute PCA, norm,
    gradient at sampled positions → discard embedding → write per-chromosome
    parquet.

    Returns
    -------
    dict
        Summary statistics (counts, timing per chromosome).
    """
    import torch
    import pyfaidx

    from agentic_spliceai.splice_engine.base_layer.data.preparation import load_gene_annotations

    components = pca_artifacts["components"]  # (k, hidden_dim)
    pca_mean = pca_artifacts["mean"]  # (hidden_dim,)
    n_pca = components.shape[0]

    fasta = pyfaidx.Fasta(str(fasta_path), as_raw=True)
    summary = {"chromosomes": {}, "total_positions": 0, "total_genes": 0}

    for chrom in chromosomes:
        chrom_chr = chrom if chrom.startswith("chr") else f"chr{chrom}"

        # Skip if output already exists (resume support)
        out_path = output_dir / f"fm_scalars_{chrom_chr}.parquet"
        if out_path.exists():
            logger.info("[Extract] %s: already exists, skipping. Delete to recompute.", chrom_chr)
            existing = pl.read_parquet(out_path)
            summary["chromosomes"][chrom_chr] = {
                "genes": 0, "positions": existing.height,
                "time_seconds": 0, "status": "skipped",
            }
            summary["total_positions"] += existing.height
            continue

        t0 = time.monotonic()

        sampled_positions = _load_sampled_positions(feature_dir, chrom_chr)
        if sampled_positions is None:
            logger.warning("No feature parquet for %s. Skipping.", chrom_chr)
            continue

        print(f"[Extract] {chrom_chr}: loading gene annotations...", flush=True)
        gene_data = load_gene_annotations(gtf_path=gtf_path, chromosomes=[chrom_chr])
        if gene_data.height == 0:
            continue

        # Group sampled positions by gene_id
        pos_by_gene = sampled_positions.group_by("gene_id").agg(
            pl.col("position").sort()
        )
        gene_positions = {
            row["gene_id"]: np.array(row["position"], dtype=np.int64)
            for row in pos_by_gene.iter_rows(named=True)
        }

        # Output buffers
        out_chrom: list[str] = []
        out_position: list[int] = []
        out_pca: list[list[float]] = [[] for _ in range(n_pca)]
        out_norm: list[float] = []
        out_gradient: list[float] = []

        n_genes = gene_data.height
        n_genes_with_positions = len(gene_positions)
        chrom_gene_count = 0
        print(
            f"[Extract] {chrom_chr}: {n_genes} genes, {n_genes_with_positions} "
            f"with sampled positions — starting extraction", flush=True
        )

        last_log_time = t0
        for gene_idx, row in enumerate(gene_data.iter_rows(named=True), 1):
            gene_id = row["gene_id"]
            gene_start = int(row["start"])
            gene_end = int(row["end"])
            gene_len = gene_end - gene_start

            if gene_len > max_gene_length:
                continue

            positions = gene_positions.get(gene_id)
            if positions is None or len(positions) == 0:
                continue

            # Extract full-gene embedding
            fasta_chrom = chrom_chr if chrom_chr in fasta else chrom
            if fasta_chrom not in fasta:
                continue

            gene_seq = str(fasta[fasta_chrom][gene_start:gene_end]).upper()
            if len(gene_seq) == 0:
                continue

            local_indices = positions - gene_start
            valid_mask = (local_indices >= 0) & (local_indices < len(gene_seq))
            valid_positions = positions[valid_mask]
            local_indices = local_indices[valid_mask]

            if len(local_indices) == 0:
                continue

            # Expand indices to include ±1 neighbors for gradient computation
            neighbor_indices = np.unique(np.clip(
                np.concatenate([local_indices - 1, local_indices, local_indices + 1]),
                0, len(gene_seq) - 1,
            ))

            gene_t0 = time.monotonic()
            try:
                all_emb = _extract_gene_embeddings(
                    model, gene_seq, max_context, neighbor_indices,
                )
            except Exception as e:
                logger.warning("Failed to extract %s: %s", gene_id, e)
                continue
            gene_elapsed = time.monotonic() - gene_t0

            # Build lookup: gene-local index → row in all_emb
            idx_to_row = {int(idx): row for row, idx in enumerate(neighbor_indices)}

            # Compute features for each sampled position
            for pos, local_idx in zip(valid_positions, local_indices):
                row = idx_to_row.get(int(local_idx))
                if row is None:
                    continue
                emb = all_emb[row].astype(np.float64)

                # PCA projection
                centered = emb - pca_mean
                pca_vals = centered @ components.T  # (k,)

                # Embedding norm
                norm = float(np.linalg.norm(emb))

                # Local gradient (using true genomic neighbors)
                grad = float("nan")
                prev_row = idx_to_row.get(int(local_idx) - 1)
                next_row = idx_to_row.get(int(local_idx) + 1)
                if prev_row is not None and next_row is not None:
                    prev_emb = all_emb[prev_row].astype(np.float64)
                    next_emb = all_emb[next_row].astype(np.float64)
                    neighbor_mean = (prev_emb + next_emb) / 2.0
                    grad = float(np.linalg.norm(emb - neighbor_mean))

                out_chrom.append(chrom_chr)
                out_position.append(int(pos))
                for k in range(n_pca):
                    out_pca[k].append(float(pca_vals[k]))
                out_norm.append(norm)
                out_gradient.append(grad)

            del all_emb

            chrom_gene_count += 1

            now = time.monotonic()
            if chrom_gene_count <= 3 or (now - last_log_time) >= 60:
                elapsed = now - t0
                rate = chrom_gene_count / elapsed if elapsed > 0 else 0
                print(
                    f"  [Extract] {chrom_chr}: gene {gene_idx}/{n_genes} "
                    f"({gene_id}, {gene_len}bp, {gene_elapsed:.1f}s) | "
                    f"{chrom_gene_count} genes extracted, {len(out_position)} "
                    f"positions, {rate * 60:.1f} genes/min",
                    flush=True,
                )
                last_log_time = now

        # Write per-chromosome parquet
        if len(out_position) > 0:
            data = {
                "chrom": out_chrom,
                "position": out_position,
            }
            for k in range(n_pca):
                data[f"fm_pca_{k + 1}"] = out_pca[k]
            data["fm_embedding_norm"] = out_norm
            data["fm_local_gradient"] = out_gradient

            out_df = pl.DataFrame(data)
            out_path = output_dir / f"fm_scalars_{chrom_chr}.parquet"
            out_df.write_parquet(out_path)

            elapsed = time.monotonic() - t0
            logger.info(
                "[Extract] %s complete: %d genes, %d positions, %.1fs → %s",
                chrom_chr, chrom_gene_count, len(out_position), elapsed, out_path,
            )

            summary["chromosomes"][chrom_chr] = {
                "genes": chrom_gene_count,
                "positions": len(out_position),
                "time_seconds": round(elapsed, 1),
            }
            summary["total_positions"] += len(out_position)
            summary["total_genes"] += chrom_gene_count

    fasta.close()
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_sampled_positions(
    feature_dir: Path, chrom: str
) -> Optional[pl.DataFrame]:
    """Load sampled positions from existing feature parquets.

    Tries multiple naming conventions:
    - ``analysis_sequences_{chrom}.parquet`` (06_multimodal_genome_workflow output)
    - ``features_{chrom}.parquet`` (alternative naming)

    Returns DataFrame with columns: chrom, position, gene_id.
    """
    bare = chrom.replace("chr", "")
    candidates = [
        feature_dir / f"analysis_sequences_{chrom}.parquet",
        feature_dir / f"analysis_sequences_chr{bare}.parquet",
        feature_dir / f"features_{chrom}.parquet",
        feature_dir / f"features_chr{bare}.parquet",
    ]
    for path in candidates:
        if path.exists():
            df = pl.read_parquet(path, columns=["chrom", "position", "gene_id"])
            return df

    return None


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    complement = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(complement)[::-1]


def _encode_chunk(model: "BaseEmbeddingModel", sequence: str) -> np.ndarray:
    """Encode a single chunk and return numpy array (chunk_len, hidden_dim)."""
    import torch

    with torch.no_grad():
        emb = model.encode(sequence)
    if hasattr(emb, "cpu"):
        emb = emb.detach().cpu().float().numpy()
    return emb


def _extract_at_positions_single_strand(
    model: "BaseEmbeddingModel",
    sequence: str,
    max_context: int,
    local_indices: np.ndarray,
) -> np.ndarray:
    """Extract embeddings at specific positions only (memory-efficient).

    Processes the sequence in chunks, but instead of stitching the full
    gene embedding (gene_len × hidden_dim), only keeps embeddings at the
    requested positions. Peak RAM: one chunk (~512 MB) instead of full
    gene (potentially 8+ GB).

    Parameters
    ----------
    model : BaseEmbeddingModel
        Foundation model.
    sequence : str
        Gene DNA sequence.
    max_context : int
        Model's max context window.
    local_indices : np.ndarray
        0-based positions within the gene to extract (sorted).

    Returns
    -------
    np.ndarray
        Shape (len(local_indices), hidden_dim), float32.
    """
    import torch
    from foundation_models.utils.chunking import chunk_sequence

    seq_len = len(sequence)
    hidden_dim = model.metadata().hidden_dim
    n_pos = len(local_indices)
    result = np.zeros((n_pos, hidden_dim), dtype=np.float32)

    if seq_len <= max_context:
        emb = _encode_chunk(model, sequence)
        valid = local_indices[local_indices < emb.shape[0]]
        result[: len(valid)] = emb[valid]
        del emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    overlap = max(max_context // 4, 128)
    chunks = chunk_sequence(sequence, chunk_size=max_context, overlap=overlap)

    # For each chunk, determine which sampled positions fall in its "keep" range
    # and extract only those embeddings.
    for chunk in chunks:
        emb = _encode_chunk(model, chunk.sequence)
        chunk_len = len(chunk.sequence)
        if emb.shape[0] < chunk_len:
            padded = np.zeros((chunk_len, hidden_dim), dtype=np.float32)
            padded[: emb.shape[0]] = emb
            emb = padded
        elif emb.shape[0] > chunk_len:
            emb = emb[:chunk_len]

        # Global range this chunk is authoritative for (after overlap trimming)
        global_keep_start = chunk.global_start + chunk.keep_start
        global_keep_end = chunk.global_start + chunk.keep_end

        # Find sampled positions in this chunk's keep range
        mask = (local_indices >= global_keep_start) & (local_indices < global_keep_end)
        if not np.any(mask):
            del emb
            continue

        # Map global positions to chunk-local offsets
        pos_in_chunk = local_indices[mask] - chunk.global_start
        result[mask] = emb[pos_in_chunk]
        del emb

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def _extract_gene_embeddings(
    model: "BaseEmbeddingModel",
    gene_sequence: str,
    max_context: int,
    local_indices: np.ndarray,
) -> np.ndarray:
    """Extract embeddings at sampled positions with strand-aware logic.

    Memory-efficient: processes chunks sequentially, never holds the full
    gene embedding. Peak RAM is one chunk (~512 MB) instead of gene_len ×
    hidden_dim (potentially 8+ GB).

    For **causal** models (Evo2, HyenaDNA): dual-strand extraction.
    Position i's forward embedding only sees left context (0..i).
    We also encode the reverse complement, flip position mapping, and
    average — giving each position both upstream and downstream context.

    For **bidirectional** models (SpliceBERT): single-strand is
    sufficient since every position already sees full context.

    Parameters
    ----------
    model : BaseEmbeddingModel
        Foundation model.
    gene_sequence : str
        Gene DNA sequence.
    max_context : int
        Model's max context window.
    local_indices : np.ndarray
        0-based positions within the gene to extract (sorted, int64).

    Returns
    -------
    np.ndarray
        Shape (len(local_indices), hidden_dim), float32.
    """
    import torch

    model_type = model.metadata().model_type
    gene_len = len(gene_sequence)

    # Forward strand — extract at sampled positions only
    emb_fwd = _extract_at_positions_single_strand(
        model, gene_sequence, max_context, local_indices,
    )

    if model_type != "causal":
        return emb_fwd

    # Reverse complement strand (causal models only)
    # RC position j maps to forward position (gene_len - 1 - j).
    # So forward position i maps to RC position (gene_len - 1 - i).
    rc_seq = _reverse_complement(gene_sequence)
    rc_indices = (gene_len - 1 - local_indices).astype(np.int64)
    # Sort for sequential chunk access, remember original order
    rc_sort_order = np.argsort(rc_indices)
    rc_indices_sorted = rc_indices[rc_sort_order]

    emb_rc_sorted = _extract_at_positions_single_strand(
        model, rc_seq, max_context, rc_indices_sorted,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unsort RC embeddings back to original position order
    emb_rc = np.empty_like(emb_rc_sorted)
    emb_rc[rc_sort_order] = emb_rc_sorted
    del emb_rc_sorted

    # Average forward and reverse-complement in-place
    emb_fwd += emb_rc
    del emb_rc
    emb_fwd *= 0.5
    return emb_fwd


def _resolve_data_paths(base_model: str = "openspliceai") -> Tuple[Path, Path]:
    """Resolve FASTA and GTF paths via the core resource registry."""
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources(base_model)
    return Path(resources.get_fasta_path()), Path(resources.get_gtf_path())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Streaming foundation model scalar feature extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--foundation-model", default="evo2",
        help="Foundation model name (evo2, splicebert, hyenadna). Default: evo2",
    )
    p.add_argument(
        "--model-size", default="7b",
        help="Model size variant (e.g., 7b, 40b for Evo2). Default: 7b",
    )
    p.add_argument(
        "--feature-dir", type=Path, required=True,
        help="Directory with existing feature parquets (for sampled positions)",
    )
    p.add_argument(
        "--chromosomes", default="22",
        help="Comma-separated chromosomes or 'all'. Default: 22",
    )
    p.add_argument(
        "-o", "--output-dir", type=Path, required=True,
        help="Output directory for scalar parquets and PCA artifacts",
    )
    p.add_argument(
        "--pca-artifacts", type=Path, default=None,
        help="Path to existing pca_artifacts.npz (skip Phase 1)",
    )
    p.add_argument(
        "--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS,
        help=f"Number of PCA components. Default: {DEFAULT_PCA_COMPONENTS}",
    )
    p.add_argument(
        "--split", default="spliceai",
        choices=["spliceai"],
        help="Chromosome split for PCA training. Default: spliceai",
    )
    p.add_argument(
        "--max-gene-length", type=int, default=2_000_000,
        help="Skip genes longer than this (avoid OOM). Default: 200000",
    )
    p.add_argument(
        "--base-model", default="openspliceai",
        help="Base model for FASTA/GTF resolution. Default: openspliceai",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve chromosomes
    if args.chromosomes.lower() == "all":
        chromosomes = CANONICAL_CHROMS
    else:
        chromosomes = [c.strip().replace("chr", "") for c in args.chromosomes.split(",")]

    # Train/test split
    train_chroms = [c for c in chromosomes if c not in SPLICEAI_TEST_CHROMS]
    test_chroms = [c for c in chromosomes if c in SPLICEAI_TEST_CHROMS]
    all_chroms = chromosomes

    logger.info("=" * 70)
    logger.info("Streaming FM Scalar Feature Extraction")
    logger.info("=" * 70)
    logger.info("  Foundation model: %s (size: %s)", args.foundation_model, args.model_size)
    logger.info("  Feature dir:      %s", args.feature_dir)
    logger.info("  Output dir:       %s", args.output_dir)
    logger.info("  Chromosomes:      %s", ", ".join(all_chroms))
    logger.info("  Train chroms:     %s", ", ".join(train_chroms) or "(none in selection)")
    logger.info("  PCA components:   %d", args.pca_components)
    logger.info("  PCA artifacts:    %s", args.pca_artifacts or "(will fit in Phase 1)")

    # Validate feature dir
    if not args.feature_dir.exists():
        logger.error("Feature directory does not exist: %s", args.feature_dir)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve FASTA/GTF paths
    fasta_path, gtf_path = _resolve_data_paths(args.base_model)
    logger.info("  FASTA:            %s", fasta_path)
    logger.info("  GTF:              %s", gtf_path)

    # Load foundation model
    logger.info("Loading foundation model...")
    from foundation_models.base import load_embedding_model

    model_kwargs = {}
    if args.foundation_model == "evo2":
        model_kwargs["model_size"] = args.model_size

    model = load_embedding_model(args.foundation_model, **model_kwargs)
    hidden_dim = model.metadata().hidden_dim
    max_context = model.metadata().max_context
    model_type = model.metadata().model_type
    logger.info(
        "  Model loaded: hidden_dim=%d, max_context=%d, model_type=%s, "
        "dual_strand=%s",
        hidden_dim, max_context, model_type, model_type == "causal",
    )

    # ── Phase 1: Fit PCA ──────────────────────────────────────────────
    if args.pca_artifacts is not None and args.pca_artifacts.exists():
        logger.info("Phase 1/2: Loading existing PCA artifacts from %s", args.pca_artifacts)
        data = np.load(args.pca_artifacts)
        pca_artifacts = {
            "components": data["components"],
            "mean": data["mean"],
            "explained_variance_ratio": data["explained_variance_ratio"],
        }
        logger.info(
            "  PCA: %d components, hidden_dim=%d, explained_variance=%.4f",
            pca_artifacts["components"].shape[0],
            pca_artifacts["components"].shape[1],
            sum(pca_artifacts["explained_variance_ratio"]),
        )
    else:
        if not train_chroms:
            logger.error(
                "No training chromosomes in selection. Cannot fit PCA. "
                "Either include training chromosomes or provide --pca-artifacts."
            )
            sys.exit(1)

        logger.info("Phase 1/2: Fitting PCA on training chromosomes (streaming)...")
        t0 = time.monotonic()
        pca_artifacts = fit_pca_streaming(
            model=model,
            feature_dir=args.feature_dir,
            train_chroms=train_chroms,
            n_components=args.pca_components,
            fasta_path=fasta_path,
            gtf_path=gtf_path,
            max_context=max_context,
            max_gene_length=args.max_gene_length,
        )
        elapsed = time.monotonic() - t0
        logger.info("Phase 1 complete: %.1f hours", elapsed / 3600)

        # Save PCA artifacts
        pca_path = args.output_dir / "pca_artifacts.npz"
        np.savez(
            pca_path,
            components=pca_artifacts["components"],
            mean=pca_artifacts["mean"],
            explained_variance_ratio=pca_artifacts["explained_variance_ratio"],
        )
        logger.info("PCA artifacts saved to %s", pca_path)

    # ── Phase 2: Extract scalars ──────────────────────────────────────
    logger.info("Phase 2/2: Extracting scalar features (streaming)...")
    t0 = time.monotonic()
    summary = extract_scalars_streaming(
        model=model,
        feature_dir=args.feature_dir,
        chromosomes=all_chroms,
        pca_artifacts=pca_artifacts,
        output_dir=args.output_dir,
        fasta_path=fasta_path,
        gtf_path=gtf_path,
        max_context=max_context,
        max_gene_length=args.max_gene_length,
    )
    elapsed = time.monotonic() - t0
    logger.info("Phase 2 complete: %.1f hours", elapsed / 3600)

    # Save summary
    summary["foundation_model"] = args.foundation_model
    summary["model_size"] = args.model_size
    summary["hidden_dim"] = hidden_dim
    summary["pca_components"] = args.pca_components
    summary["explained_variance_ratio"] = pca_artifacts["explained_variance_ratio"].tolist()
    summary["phase2_time_hours"] = round(elapsed / 3600, 2)

    summary_path = args.output_dir / "extraction_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("Done. Output: %s", args.output_dir)
    logger.info("  Positions: %d across %d genes", summary["total_positions"], summary["total_genes"])
    logger.info("  PCA explained variance: %.4f", sum(pca_artifacts["explained_variance_ratio"]))
    logger.info("  Summary: %s", summary_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
