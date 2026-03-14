#!/usr/bin/env python3
"""
Sparse Exon Classifier — Multi-Model Foundation Model Embeddings

Implements exon/intron classification using frozen embeddings from genomic
foundation models.  Originally based on the Evo2 paper methodology
(Brixi et al. 2025/2026, Section 4.3.9), now generalized to support
multiple embedding models:

  - **Evo2** (7B/40B) — causal DNA LM, dual-strand extraction (Apache 2.0)
  - **SpliceBERT** (19.4M) — bidirectional BERT, splice-specific (AGPL-3.0)
  - **HyenaDNA** (1.6M-300M) — causal Hyena operator (Apache 2.0)

Pipeline:
  1. Sample N random positions from within gene bodies
  2. Extract context sequences (causal: dual-strand; bidirectional: centered)
  3. Extract embeddings from the foundation model
  4. Train a lightweight MLP classifier (weighted BCE)
  5. Evaluate AUROC/AUPRC on held-out chromosomes

Extraction strategies:
  - **Causal** (Evo2, HyenaDNA): last-position from forward + reverse strands
    → feature dim = 2 × hidden_dim
  - **Bidirectional** (SpliceBERT): center-position from single window
    → feature dim = 1 × hidden_dim

Usage:
    # Evo2 (Evo2 paper setup: 1500 positions, needs A40+)
    python 05_sparse_exon_classifier.py \\
        --foundation-model evo2 --n-positions 1500 \\
        -o /workspace/output/sparse-evo2/

    # SpliceBERT (runs on any GPU or CPU — 19.4M params)
    python 05_sparse_exon_classifier.py \\
        --foundation-model splicebert --n-positions 1500 \\
        -o /workspace/output/sparse-splicebert/

    # HyenaDNA (medium, runs on any GPU or CPU)
    python 05_sparse_exon_classifier.py \\
        --foundation-model hyenadna --model-size medium-160k \\
        --n-positions 1500 -o /workspace/output/sparse-hyenadna/

    # Mock mode (no GPU needed — validates pipeline)
    python 05_sparse_exon_classifier.py \\
        --foundation-model splicebert --mock --n-positions 50 \\
        -o /tmp/sparse-test/

    # Pre-extracted embeddings (skip model loading)
    python 05_sparse_exon_classifier.py \\
        --embeddings-file /path/to/sparse_embeddings.npz \\
        -o /workspace/output/retrain/

References:
    - Brixi et al., Nature (2026), Section 4.3.9
    - Chen & Zheng 2024, Briefings in Bioinformatics 25(3) (SpliceBERT)
    - Nguyen et al. 2023, NeurIPS 2023 (HyenaDNA)
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Step 1: Position sampling
# -----------------------------------------------------------------------

def sample_positions(
    n_positions: int,
    data_dir: Path,
    seed: int = 42,
    filter_chromosomes: str = "canonical",
    exon_fraction: float = 0.0,
) -> list[dict]:
    """Sample random positions from within gene bodies.

    Uses a two-pass approach to avoid materializing all ~1B positions:
      1. Count total positions per gene (just lengths)
      2. Sample N global indices, then resolve each to (gene, offset)

    When exon_fraction > 0, applies enrichment by oversampling then balancing:
      - Samples 3× more positions than requested uniformly
      - Keeps all exonic positions + subsamples intronic to achieve target ratio
      - If not enough exonic positions, keeps all and logs a warning

    Args:
        n_positions: Number of random positions to sample.
        data_dir: Path to data directory with splice site annotations.
        seed: Random seed for reproducibility.
        filter_chromosomes: Chromosome filtering mode.
            "canonical" (default) — keep only chr1-22, chrX, chrY, chrM;
                excludes GRCh38 patch scaffolds (_fix, _alt, _random).
            "none" — keep all chromosomes from the GTF.
        exon_fraction: Target fraction of exonic positions (0.0 = no enrichment,
            0.15 = ~15% exonic). When > 0, oversamples then balances.

    Returns list of dicts with keys:
        gene_id, gene_name, chrom, genomic_pos (1-based), is_exon, strand
    """
    import pandas as pd

    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        filter_to_canonical_chromosomes,
        load_gene_annotations,
        prepare_splice_site_annotations,
    )
    from agentic_spliceai.splice_engine.resources import get_genomic_registry
    from foundation_models.utils.chunking import build_exon_labels

    # Load annotations only — skip FASTA sequence extraction.
    # Sequences are extracted later via pyfaidx point queries in
    # extract_context_sequences(), so loading all 19K gene sequences
    # here is pure overhead (~minutes of wasted I/O).
    logger.info("Loading gene annotations (metadata only, skipping FASTA)...")
    registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
    gtf_path = str(registry.get_gtf_path(validate=True))
    gene_data = load_gene_annotations(gtf_path=gtf_path)
    logger.info("Loaded %d genes", len(gene_data))

    # Filter chromosomes: exclude patch scaffolds (_fix, _alt, _random)
    # that may not be in the reference FASTA.
    chrom_col = "chrom" if "chrom" in gene_data.columns else "seqname"
    if filter_chromosomes == "canonical":
        n_before = len(gene_data)
        gene_data = filter_to_canonical_chromosomes(
            gene_data, build="GRCh38", chrom_column=chrom_col,
        )
        n_filtered = n_before - len(gene_data)
        if n_filtered:
            logger.info(
                "Filtered %d genes on non-canonical chromosomes (kept %d)",
                n_filtered, len(gene_data),
            )

    # Ensure splice site annotations exist
    splice_parquet = data_dir / "splice_sites_enhanced.parquet"
    splice_tsv = data_dir / "splice_sites_enhanced.tsv"
    if splice_parquet.exists():
        splice_sites_df = pd.read_parquet(splice_parquet)
    elif splice_tsv.exists():
        splice_sites_df = pd.read_csv(splice_tsv, sep="\t")
    else:
        logger.info("Generating splice site annotations...")
        splice_sites_df = prepare_splice_site_annotations(
            gtf_path=gtf_path,
            output_dir=str(data_dir),
        )

    # Pass 1: Build gene metadata (gene_id, name, chrom, start, end, strand, length)
    # Only store per-gene info — NOT per-nucleotide
    logger.info("Pass 1: Computing gene lengths...")
    gene_meta = []
    total_positions = 0

    for row in gene_data.iter_rows(named=True):
        gene_start = int(row["start"])  # 1-based
        gene_end = int(row["end"])
        seq_len = gene_end - gene_start + 1
        chrom = row[chrom_col]
        gene_meta.append({
            "gene_id": row["gene_id"],
            "gene_name": row["gene_name"],
            "chrom": chrom,
            "start": gene_start,
            "end": gene_end,
            "strand": row["strand"],
            "length": seq_len,
            "cum_start": total_positions,  # global offset
        })
        total_positions += seq_len

    logger.info("Total genome positions across %d genes: %s", len(gene_meta), f"{total_positions:,}")

    # Pass 2: Sample global indices, then resolve to (gene, relative_offset)
    # When exon enrichment is active, oversample 3× then balance afterward.
    rng = random.Random(seed)
    oversample_factor = 3 if exon_fraction > 0 else 1
    n_raw = min(n_positions * oversample_factor, total_positions)
    n_to_sample = n_raw
    if n_to_sample < n_positions:
        logger.warning(
            "Requested %d positions but pool has only %d — using all",
            n_positions, total_positions,
        )
    sampled_global = sorted(rng.sample(range(total_positions), n_to_sample))

    # Resolve global indices to gene + offset using cumulative starts
    # Build cumulative length array for binary search
    cum_lengths = np.array([g["cum_start"] + g["length"] for g in gene_meta])

    logger.info("Pass 2: Resolving %d sampled positions to genes...", len(sampled_global))
    # Group sampled indices by gene for efficient exon label computation
    gene_indices: dict[int, list[int]] = {}  # gene_idx -> [relative_offsets]
    global_to_gene: dict[int, tuple[int, int]] = {}  # global_idx -> (gene_idx, rel_offset)

    for global_idx in sampled_global:
        gene_idx = int(np.searchsorted(cum_lengths, global_idx, side="right"))
        rel_offset = global_idx - gene_meta[gene_idx]["cum_start"]
        gene_indices.setdefault(gene_idx, []).append(rel_offset)
        global_to_gene[global_idx] = (gene_idx, rel_offset)

    # Build exon labels only for genes that have sampled positions
    logger.info("Computing exon labels for %d genes with sampled positions...", len(gene_indices))
    gene_labels: dict[int, np.ndarray] = {}

    for i, gene_idx in enumerate(gene_indices):
        g = gene_meta[gene_idx]
        labels = build_exon_labels(
            gene_id=g["gene_id"],
            gene_start=g["start"] - 1,  # build_exon_labels expects 0-based
            gene_sequence_length=g["length"],
            splice_sites_df=splice_sites_df,
        )
        gene_labels[gene_idx] = labels
        if (i + 1) % 500 == 0:
            logger.info("  Exon labels: %d/%d genes", i + 1, len(gene_indices))

    # Assemble final position list
    sampled = []
    for global_idx in sampled_global:
        gene_idx, rel_offset = global_to_gene[global_idx]
        g = gene_meta[gene_idx]
        sampled.append({
            "gene_id": g["gene_id"],
            "gene_name": g["gene_name"],
            "chrom": g["chrom"],
            "genomic_pos": g["start"] + rel_offset,  # 1-based
            "is_exon": int(gene_labels[gene_idx][rel_offset]),
            "strand": g["strand"],
        })

    # Exon enrichment: balance the dataset by keeping all exonic positions
    # and subsampling intronic positions to achieve the target exon fraction.
    if exon_fraction > 0:
        exonic = [p for p in sampled if p["is_exon"]]
        intronic = [p for p in sampled if not p["is_exon"]]
        n_exonic = len(exonic)

        if n_exonic == 0:
            logger.warning("No exonic positions found — skipping enrichment")
        else:
            # Target: n_exonic / n_total = exon_fraction
            # → n_intronic = n_exonic * (1 - exon_fraction) / exon_fraction
            n_intronic_target = int(n_exonic * (1 - exon_fraction) / exon_fraction)
            n_intronic_target = min(n_intronic_target, len(intronic))

            # Also cap total at n_positions
            n_total_target = min(n_exonic + n_intronic_target, n_positions)
            n_intronic_target = n_total_target - n_exonic

            rng.shuffle(intronic)
            sampled = exonic + intronic[:n_intronic_target]
            # Re-shuffle so exonic/intronic aren't grouped
            rng.shuffle(sampled)

            actual_frac = n_exonic / len(sampled) if sampled else 0
            logger.info(
                "Exon enrichment: kept %d exonic + %d intronic = %d total "
                "(%.1f%% exonic, target was %.1f%%)",
                n_exonic, n_intronic_target, len(sampled),
                100 * actual_frac, 100 * exon_fraction,
            )

    n_exon = sum(p["is_exon"] for p in sampled)
    logger.info(
        "Final dataset: %d positions — %d exonic (%.1f%%), %d intronic (%.1f%%)",
        len(sampled), n_exon, 100 * n_exon / len(sampled),
        len(sampled) - n_exon, 100 * (len(sampled) - n_exon) / len(sampled),
    )

    return sampled


# -----------------------------------------------------------------------
# Step 2: Sequence extraction (forward + reverse complement)
# -----------------------------------------------------------------------

def extract_context_sequences(
    positions: list[dict],
    fasta_path: str,
    context_size: int = 8192,
    model_type: str = "causal",
) -> list[tuple[str, ...]]:
    """Extract context sequences for each position.

    Strategy depends on model architecture:

    **Causal** (Evo2, HyenaDNA): dual-strand, target at 3' end.
      - Forward:  genome[P - context_size + 1 : P]  (upstream context)
      - Reverse:  revcomp(genome[P : P + context_size - 1])  (downstream)
      - Returns 2-tuples ``(seq_forward, seq_reverse)``.

    **Bidirectional** (SpliceBERT, DNABERT-2): single centered window.
      - Centered: genome[P - context_size//2 : P + context_size//2]
      - Returns 1-tuples ``(seq_centered,)``.

    Args:
        positions: List of position dicts with ``chrom``, ``genomic_pos``.
        fasta_path: Path to reference FASTA.
        context_size: Context window size in nucleotides.
        model_type: ``"causal"`` or ``"bidirectional"``.

    Returns:
        List of tuples — 2-tuples for causal, 1-tuples for bidirectional.
    """
    from pyfaidx import Fasta

    from agentic_spliceai.splice_engine.base_layer.data.sequence_extraction import (
        reverse_complement,
    )

    logger.info("Loading FASTA index...")
    fasta = Fasta(fasta_path)

    sequences: list[tuple[str, ...]] = []
    n_padded = 0

    for i, pos in enumerate(positions):
        chrom = pos["chrom"]
        gpos = pos["genomic_pos"]  # 1-based

        # Resolve chromosome name in FASTA
        if chrom in fasta:
            chrom_key = chrom
        elif chrom.startswith("chr") and chrom[3:] in fasta:
            chrom_key = chrom[3:]
        elif f"chr{chrom}" in fasta:
            chrom_key = f"chr{chrom}"
        else:
            logger.warning("Chromosome %s not in FASTA, skipping position %d", chrom, gpos)
            if model_type == "causal":
                sequences.append(("N" * context_size, "N" * context_size))
            else:
                sequences.append(("N" * context_size,))
            continue

        chrom_len = len(fasta[chrom_key])

        if model_type == "bidirectional":
            # Centered window: target position in the middle
            half = context_size // 2
            ctr_start = gpos - 1 - half  # 0-based
            ctr_end = ctr_start + context_size

            if ctr_start < 0:
                pad_len = -ctr_start
                seq = "N" * pad_len + str(fasta[chrom_key][0:ctr_end])
                n_padded += 1
            elif ctr_end > chrom_len:
                seq = str(fasta[chrom_key][ctr_start:chrom_len])
                pad_len = ctr_end - chrom_len
                seq += "N" * pad_len
                n_padded += 1
            else:
                seq = str(fasta[chrom_key][ctr_start:ctr_end])

            seq = seq.upper()
            assert len(seq) == context_size, (
                f"centered len={len(seq)} != {context_size}"
            )
            sequences.append((seq,))
        else:
            # Causal: dual-strand extraction (original strategy)
            # Forward strand: upstream context, target at 3' end
            fwd_start = gpos - context_size  # 0-based
            fwd_end = gpos

            if fwd_start < 0:
                pad_len = -fwd_start
                seq_fwd = "N" * pad_len + str(fasta[chrom_key][0:fwd_end])
                n_padded += 1
            else:
                seq_fwd = str(fasta[chrom_key][fwd_start:fwd_end])

            # Reverse strand: downstream context, target at 3' end after revcomp
            rev_start = gpos - 1
            rev_end = gpos - 1 + context_size

            if rev_end > chrom_len:
                seq_downstream = str(fasta[chrom_key][rev_start:chrom_len])
                pad_len = rev_end - chrom_len
                seq_downstream += "N" * pad_len
                n_padded += 1
            else:
                seq_downstream = str(fasta[chrom_key][rev_start:rev_end])

            seq_rev = reverse_complement(seq_downstream)

            seq_fwd = seq_fwd.upper()
            seq_rev = seq_rev.upper()
            assert len(seq_fwd) == context_size, (
                f"fwd len={len(seq_fwd)} != {context_size}"
            )
            assert len(seq_rev) == context_size, (
                f"rev len={len(seq_rev)} != {context_size}"
            )
            sequences.append((seq_fwd, seq_rev))

        if (i + 1) % 500 == 0:
            logger.info("  Extracted %d/%d sequences", i + 1, len(positions))

    if n_padded:
        logger.info("  %d positions required N-padding (near chromosome boundaries)", n_padded)

    fasta.close()
    return sequences


# -----------------------------------------------------------------------
# Step 3: Embedding extraction (final position from each strand)
# -----------------------------------------------------------------------

def extract_sparse_embeddings(
    sequences: list[tuple[str, ...]],
    model: "BaseEmbeddingModel",
    layer: str | None = None,
) -> np.ndarray:
    """Extract point embeddings using any foundation model.

    Strategy depends on model architecture (from ``model.metadata()``):

    **Causal** — dual-strand, last-position:
      Each sequence tuple is ``(forward, reverse)``.  Takes ``emb[-1, :]``
      from each strand and concatenates → ``[N, 2 × hidden_dim]``.

    **Bidirectional** — single centered window, center-position:
      Each sequence tuple is ``(centered,)``.  Takes ``emb[center_idx, :]``
      → ``[N, hidden_dim]``.

    Args:
        sequences: List of sequence tuples (from ``extract_context_sequences``).
        model: Foundation embedding model satisfying ``BaseEmbeddingModel``.
        layer: Optional internal layer (only used if model supports it).

    Returns:
        Array of shape ``[N, feature_dim]``.
    """
    import torch

    meta = model.metadata()
    hidden_dim = meta.hidden_dim
    feature_dim = meta.feature_dim_for_classifier
    is_causal = meta.model_type == "causal"

    n_strands = 2 if is_causal else 1
    logger.info(
        "Extracting embeddings: %d positions × %d strand(s) "
        "(model=%s, hidden_dim=%d, feature_dim=%d)",
        len(sequences), n_strands, meta.name, hidden_dim, feature_dim,
    )

    embeddings = np.zeros((len(sequences), feature_dim), dtype=np.float32)
    t_extract = time.time()

    for i, seq_tuple in enumerate(sequences):
        t_pos = time.time()

        with torch.no_grad():
            if is_causal:
                # Dual-strand: take last-position embedding from each
                seq_fwd, seq_rev = seq_tuple
                emb_fwd = model.encode(seq_fwd, layer=layer)
                emb_fwd_last = emb_fwd[-1, :].cpu().float().numpy()

                emb_rev = model.encode(seq_rev, layer=layer)
                emb_rev_last = emb_rev[-1, :].cpu().float().numpy()

                embeddings[i, :hidden_dim] = emb_fwd_last
                embeddings[i, hidden_dim:] = emb_rev_last

                del emb_fwd, emb_rev
            else:
                # Bidirectional: take center-position embedding
                (seq_centered,) = seq_tuple
                emb = model.encode(seq_centered, layer=layer)
                center_idx = emb.shape[0] // 2
                emb_center = emb[center_idx, :].cpu().float().numpy()

                embeddings[i, :] = emb_center

                del emb

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed_pos = time.time() - t_pos
        elapsed_total = time.time() - t_extract

        if i == 0:
            logger.info("  [1/%d] first embedding: %.1fs", len(sequences), elapsed_pos)
        if (i + 1) % 50 == 0 or (i + 1) == len(sequences):
            rate = (i + 1) / elapsed_total
            eta = (len(sequences) - i - 1) / rate if rate > 0 else 0
            logger.info(
                "  [%d/%d] embeddings extracted (%.1fs/pos, ETA %.0fm)",
                i + 1, len(sequences), elapsed_pos, eta / 60,
            )

    logger.info("Embeddings shape: %s (%.1f MB)", embeddings.shape,
                embeddings.nbytes / (1024**2))
    return embeddings


# -----------------------------------------------------------------------
# Step 4: Train and evaluate
# -----------------------------------------------------------------------

def train_and_evaluate(
    embeddings: np.ndarray,
    positions: list[dict],
    output_dir: Path,
    split_preset: str = "spliceai",
    val_fraction: float = 0.1,
    seed: int = 42,
    # Paper hyperparams
    hidden_dim: int = 1024,
    lr: float = 5e-5,
    batch_size: int = 16,
    weight_decay: float = 2e-4,
    epochs: int = 100,
    patience: int = 20,
    monitor_metric: str = "auprc",
) -> dict:
    """Train ExonClassifier and evaluate on held-out chromosomes."""
    import torch
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from foundation_models.evo2 import ExonClassifier

    input_dim = embeddings.shape[1]
    labels = np.array([p["is_exon"] for p in positions], dtype=np.float32)
    chroms = [p["chrom"].replace("chr", "") for p in positions]

    # Chromosome split
    if split_preset == "spliceai":
        test_chroms = {"1", "3", "5", "7", "9"}
    else:
        test_chroms = set()

    train_mask = np.array([c not in test_chroms for c in chroms])
    test_mask = ~train_mask

    # Val split from training data
    rng = random.Random(seed)
    train_indices = np.where(train_mask)[0].tolist()
    rng.shuffle(train_indices)
    n_val = max(1, int(len(train_indices) * val_fraction))
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]
    test_indices = np.where(test_mask)[0].tolist()

    logger.info(
        "Split: %d train, %d val, %d test",
        len(train_indices), len(val_indices), len(test_indices),
    )

    # Prepare tensors — each position is a single vector, not a sequence
    # ExonClassifier expects [batch, seq_len, input_dim] but for point-query
    # we use seq_len=1 and squeeze after
    train_emb = torch.tensor(embeddings[train_indices], dtype=torch.float32).unsqueeze(1)
    train_lbl = torch.tensor(labels[train_indices], dtype=torch.float32).unsqueeze(1)
    val_emb = torch.tensor(embeddings[val_indices], dtype=torch.float32).unsqueeze(1)
    val_lbl = torch.tensor(labels[val_indices], dtype=torch.float32).unsqueeze(1)

    # Resolve device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Build classifier (paper: single hidden layer, 1024 units)
    classifier = ExonClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        architecture="mlp",
        dropout=0.1,
    )

    print()
    print("=" * 60)
    print(f"Training ExonClassifier (Evo2 sparse exon approach)")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Positions:    {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")
    print(f"  Exon ratio:   {labels.mean():.1%}")
    print(f"  Device:       {device}")
    print(f"  Hyperparams:  lr={lr}, batch={batch_size}, wd={weight_decay}")
    print("=" * 60)
    print()

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    history = classifier.fit(
        train_embeddings=train_emb,
        train_labels=train_lbl,
        val_embeddings=val_emb,
        val_labels=val_lbl,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        verbose=True,
        checkpoint_dir=str(model_dir),
        patience=patience,
        lr_schedule=True,
        monitor_metric=monitor_metric,
    )

    # Save diagnostic plots
    save_training_plots(history, output_dir)

    # Evaluate on test set
    test_metrics = {}
    if test_indices:
        test_emb = torch.tensor(embeddings[test_indices], dtype=torch.float32).unsqueeze(1)
        test_lbl_np = labels[test_indices]

        probs = classifier.predict(test_emb).flatten()
        preds = (probs >= 0.5).astype(int)

        try:
            auroc = float(roc_auc_score(test_lbl_np, probs))
        except ValueError:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(test_lbl_np, probs))
        except ValueError:
            auprc = float("nan")

        test_metrics = {
            "auroc": round(auroc, 4),
            "auprc": round(auprc, 4),
            "accuracy": round(float(accuracy_score(test_lbl_np, preds)), 4),
            "f1": round(float(f1_score(test_lbl_np, preds, zero_division=0)), 4),
            "precision": round(float(precision_score(test_lbl_np, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(test_lbl_np, preds, zero_division=0)), 4),
            "n_test": len(test_indices),
            "n_test_exon": int(test_lbl_np.sum()),
        }

        print()
        print("Test Metrics (held-out chromosomes):")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.4f}")
            else:
                print(f"  {k:12s}: {v}")
        print()
        print(f"  Paper reference (Evo2): AUROC 0.82-0.99 across species")
        print()

    # Save results
    results = {
        "training": {
            "best_epoch": history["best_epoch"] + 1,
            "monitor_metric": history.get("best_metric_name", monitor_metric),
            "best_val_metric": round(history["best_val_metric"], 4),
            "best_val_auroc": round(history["best_val_auroc"], 4),
            "stopped_early": history["stopped_early"],
        },
        "config": {
            "n_positions": len(positions),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "split_preset": split_preset,
        },
    }
    if test_metrics:
        results["test"] = test_metrics

    metrics_path = model_dir / "eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save gene split info
    split_info = {
        "train_chroms": sorted(set(chroms[i] for i in train_indices)),
        "val_chroms": sorted(set(chroms[i] for i in val_indices)),
        "test_chroms": sorted(test_chroms) if test_chroms else [],
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_test": len(test_indices),
    }
    with open(model_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    return results


# -----------------------------------------------------------------------
# Step 5: Diagnostic plots
# -----------------------------------------------------------------------

def save_training_plots(history: dict, output_dir: Path) -> list[Path]:
    """Save training diagnostic plots as PNG files.

    Generates two figures:
      1. Loss curves (train + val) — diagnoses overfitting
      2. Metric curves (AUROC + AUPRC) — shows actual classification quality

    Args:
        history: Training history dict from classifier.fit().
        output_dir: Directory to save plots into.

    Returns:
        List of saved plot file paths.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (works on headless pods)
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping diagnostic plots")
        return []

    saved = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = history.get("best_epoch", 0) + 1  # 1-based for display
    monitor = history.get("best_metric_name", "auprc").upper()

    # --- Figure 1: Loss curves ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], "b-", label="Train loss", linewidth=1.5)
    if history["val_loss"]:
        ax.plot(epochs, history["val_loss"], "r-", label="Val loss", linewidth=1.5)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (weighted BCE)")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    loss_path = plot_dir / "loss_curves.png"
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(loss_path)
    logger.info("Saved loss curves: %s", loss_path)

    # --- Figure 2: Metric curves (AUROC + AUPRC) ---
    if history["val_auroc"] and history["val_auprc"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # AUROC
        ax1.plot(epochs, history["val_auroc"], "g-", linewidth=1.5)
        ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("AUROC")
        ax1.set_title("Validation AUROC (misleading with class imbalance)")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)

        # AUPRC — the real metric
        ax2.plot(epochs, history["val_auprc"], "m-", linewidth=1.5)
        ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                     label=f"Best {monitor} (epoch {best_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUPRC")
        ax2.set_title("Validation AUPRC (primary metric for imbalanced data)")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle("Classification Metrics", fontsize=13)
        fig.tight_layout()
        metrics_path = plot_dir / "metric_curves.png"
        fig.savefig(metrics_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(metrics_path)
        logger.info("Saved metric curves: %s", metrics_path)

    # --- Figure 3: Combined overview (single plot for quick glance) ---
    if history["val_auprc"]:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color_loss = "#2c3e50"
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color=color_loss)
        l1, = ax1.plot(epochs, history["train_loss"], "-", color=color_loss,
                        alpha=0.6, label="Train loss")
        l2 = None
        if history["val_loss"]:
            l2, = ax1.plot(epochs, history["val_loss"], "--", color=color_loss,
                           alpha=0.6, label="Val loss")
        ax1.tick_params(axis="y", labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_auprc = "#8e44ad"
        color_auroc = "#27ae60"
        ax2.set_ylabel("Metric", color=color_auprc)
        l3, = ax2.plot(epochs, history["val_auprc"], "-", color=color_auprc,
                        linewidth=2, label="Val AUPRC")
        l4, = ax2.plot(epochs, history["val_auroc"], "-", color=color_auroc,
                        linewidth=1.5, alpha=0.7, label="Val AUROC")
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(axis="y", labelcolor=color_auprc)

        ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)

        lines = [l1] + ([l2] if l2 else []) + [l3, l4]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")

        fig.suptitle(
            f"Training Overview (best {monitor} at epoch {best_epoch})",
            fontsize=13,
        )
        fig.tight_layout()
        overview_path = plot_dir / "training_overview.png"
        fig.savefig(overview_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(overview_path)
        logger.info("Saved training overview: %s", overview_path)

    return saved


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    from foundation_models.base import (
        get_model_metadata,
        list_available_models,
        load_embedding_model,
    )

    available_models = list_available_models()

    parser = argparse.ArgumentParser(
        description="Sparse exon classifier using foundation model embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Foundation model selection
    parser.add_argument("--foundation-model", type=str, default="evo2",
                        choices=available_models,
                        help=f"Foundation model for embeddings "
                             f"(default: evo2; available: {', '.join(available_models)})")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Model-specific variant (e.g. '7b' for Evo2, "
                             "'medium-160k' for HyenaDNA; default: model-dependent)")
    parser.add_argument("--layer", type=str, default=None,
                        help="Internal layer for embeddings (only Evo2; "
                             "default: blocks.26 for Evo2, ignored for others)")

    # Sampling
    parser.add_argument("--n-positions", type=int, default=5000,
                        help="Number of random positions to sample (default: 5000; "
                             "paper used 1500 but more positions improve metric stability)")
    parser.add_argument("--context-size", type=int, default=None,
                        help="Context window size in bp (default: model-dependent; "
                             "clamped to model's max_context if too large)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--split-preset", type=str, default="spliceai",
                        choices=["spliceai", "none"],
                        help="Chromosome split (default: spliceai)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for position sampling and splits")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: data/mane/GRCh38/)")
    parser.add_argument("--fasta", type=str, default=None,
                        help="FASTA file path (default: auto-detect in data dir)")
    parser.add_argument("--filter-chromosomes", type=str, default="canonical",
                        choices=["canonical", "none"],
                        help="Chromosome filter: 'canonical' excludes patch scaffolds "
                             "(_fix, _alt, _random); 'none' keeps all (default: canonical)")
    parser.add_argument("--exon-fraction", type=float, default=0.15,
                        help="Target fraction of exonic positions via enriched sampling "
                             "(default: 0.15 = ~15%% exonic; 0.0 = no enrichment, "
                             "natural rate ~2.3%%)")

    # Classifier hyperparams (configurable for experimentation)
    parser.add_argument("--hidden-dim", type=int, default=1024,
                        help="Classifier hidden dim (default: 1024, Evo2 paper value)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5, Evo2 paper value)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16, Evo2 paper value)")
    parser.add_argument("--weight-decay", type=float, default=2e-4,
                        help="Weight decay (default: 2e-4, Evo2 paper value)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--monitor-metric", type=str, default="auprc",
                        choices=["auprc", "auroc"],
                        help="Metric for early stopping (default: auprc, better for "
                             "class-imbalanced data; auroc is deceptive with rare positives)")

    # Skip embedding extraction (use pre-extracted)
    parser.add_argument("--embeddings-file", type=str, default=None,
                        help="Load pre-extracted embeddings from .npz (skip extraction)")

    # Mock mode: random embeddings for local testing (no GPU needed)
    parser.add_argument("--mock", action="store_true",
                        help="Use random embeddings (for local smoke testing; "
                             "validates sampling + training pipeline without GPU)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else Path("data/mane/GRCh38/")
    t0 = time.time()

    mock_mode = args.mock

    # Build model-specific kwargs for metadata / loading
    model_kwargs: dict = {}
    if args.foundation_model == "evo2":
        model_kwargs["model_size"] = args.model_size or "7b"
        if args.layer:
            model_kwargs["embedding_layer"] = args.layer
    elif args.foundation_model == "hyenadna":
        model_kwargs["model_size"] = args.model_size or "medium-160k"
    elif args.foundation_model == "splicebert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size

    # Get metadata (lightweight — no model loading)
    meta = get_model_metadata(args.foundation_model, **model_kwargs)

    # Resolve context size: user request → model max → default
    if args.context_size is not None:
        context_size = args.context_size
    elif args.foundation_model == "evo2":
        context_size = 8192  # Paper's default for Evo2
    else:
        context_size = min(meta.max_context, 8192)

    if context_size > meta.max_context:
        logger.warning(
            "Requested context_size=%d exceeds %s max_context=%d — clamping",
            context_size, meta.name, meta.max_context,
        )
        context_size = meta.max_context

    # Resolve layer for Evo2
    layer = args.layer
    if args.foundation_model == "evo2" and layer is None:
        layer = "blocks.26"  # Paper's best layer
    elif not meta.supports_layer_selection and layer is not None:
        logger.warning(
            "%s does not support --layer (ignoring '%s')", meta.name, layer,
        )
        layer = None

    feature_dim = meta.feature_dim_for_classifier

    print()
    print("=" * 70)
    print("Sparse Exon Classifier (Foundation Model Embeddings)")
    print("=" * 70)
    print(f"  Model:        {'MOCK (random)' if mock_mode else meta.name}")
    print(f"  Type:         {meta.model_type}")
    print(f"  Hidden dim:   {meta.hidden_dim}")
    print(f"  Feature dim:  {feature_dim}"
          f" ({'2×hidden (dual-strand)' if meta.model_type == 'causal' else '1×hidden (centered)'})")
    print(f"  Positions:    {args.n_positions}")
    print(f"  Exon enrich:  {args.exon_fraction:.0%}" if args.exon_fraction > 0
          else "  Exon enrich:  off (natural rate ~2.3%)")
    print(f"  Context:      {context_size} bp (max: {meta.max_context})")
    if layer:
        print(f"  Layer:        {layer}")
    print(f"  Monitor:      {args.monitor_metric.upper()}")
    print(f"  Split:        {args.split_preset}")
    print(f"  Seed:         {args.seed}")
    print("=" * 70)
    print()

    # Check for pre-extracted embeddings
    if args.embeddings_file:
        logger.info("Loading pre-extracted embeddings from %s", args.embeddings_file)
        data = np.load(args.embeddings_file, allow_pickle=True)
        embeddings = data["embeddings"]
        positions = data["positions"].tolist()
        logger.info("Loaded %d embeddings (dim=%d)", len(embeddings), embeddings.shape[1])
    else:
        # Step 1: Sample positions (always runs — validates annotation pipeline)
        logger.info("Step 1/3: Sampling positions...")
        positions = sample_positions(
            n_positions=args.n_positions,
            data_dir=data_dir,
            seed=args.seed,
            filter_chromosomes=args.filter_chromosomes,
            exon_fraction=args.exon_fraction,
        )

        if mock_mode:
            # Mock mode: random embeddings matching selected model's feature dim
            logger.info(
                "Step 2-3/3: MOCK — generating random embeddings "
                "(%d × %d, simulating %s)",
                len(positions), feature_dim, meta.name,
            )
            rng = np.random.RandomState(args.seed)
            embeddings = rng.randn(len(positions), feature_dim).astype(np.float32)
            logger.info(
                "Mock embeddings: %s (%.1f MB). AUROC will be ~0.5.",
                embeddings.shape, embeddings.nbytes / (1024**2),
            )
        else:
            # Resolve FASTA path
            if args.fasta:
                fasta_path = args.fasta
            else:
                fasta_candidates = [
                    data_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa",
                    data_dir / "GRCh38.primary_assembly.genome.fa",
                    data_dir / "hg38.fa",
                ]
                fasta_path = None
                for candidate in fasta_candidates:
                    if candidate.exists():
                        fasta_path = str(candidate)
                        break
                if fasta_path is None:
                    fa_files = list(data_dir.glob("*.fa"))
                    if fa_files:
                        fasta_path = str(fa_files[0])
                    else:
                        logger.error("No FASTA file found in %s. Use --fasta to specify.", data_dir)
                        sys.exit(1)
                logger.info("Using FASTA: %s", fasta_path)

            # Step 2: Extract context sequences (strategy depends on model type)
            logger.info("Step 2/3: Extracting context sequences (%s strategy)...",
                        meta.model_type)
            sequences = extract_context_sequences(
                positions=positions,
                fasta_path=fasta_path,
                context_size=context_size,
                model_type=meta.model_type,
            )

            # Step 3: Load model and extract embeddings
            logger.info("Step 3/3: Loading %s and extracting embeddings...", meta.name)
            model = load_embedding_model(args.foundation_model, **model_kwargs)
            embeddings = extract_sparse_embeddings(
                sequences=sequences,
                model=model,
                layer=layer,
            )

        # Save embeddings for reuse (include model name for provenance)
        emb_path = output_dir / "sparse_embeddings.npz"
        np.savez_compressed(
            emb_path,
            embeddings=embeddings,
            positions=np.array(positions),
            foundation_model=np.array(meta.name),
        )
        logger.info("Saved embeddings: %s (%.1f MB)", emb_path,
                     emb_path.stat().st_size / (1024**2))

    # Step 4: Train and evaluate
    logger.info("Training classifier...")
    results = train_and_evaluate(
        embeddings=embeddings,
        positions=positions,
        output_dir=output_dir,
        split_preset=args.split_preset,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        monitor_metric=args.monitor_metric,
    )

    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("Complete")
    print("=" * 70)
    print(f"  Model:        {meta.name}")
    print(f"  Output:       {output_dir}")
    print(f"  Embeddings:   {output_dir / 'sparse_embeddings.npz'}")
    print(f"  Classifier:   {output_dir / 'model' / 'best_model.pt'}")
    print(f"  Metrics:      {output_dir / 'model' / 'eval_metrics.json'}")
    if "test" in results:
        print(f"  Test AUPRC:   {results['test']['auprc']}")
        print(f"  Test AUROC:   {results['test']['auroc']}")
    print(f"  Plots:        {output_dir / 'plots' / '*.png'}")
    print(f"  Time:         {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
