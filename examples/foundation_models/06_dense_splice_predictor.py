"""
Dense Per-Nucleotide Splice Site Predictor

Trains a lightweight prediction head on frozen foundation model embeddings
to produce per-nucleotide splice site scores: P(donor), P(acceptor),
P(neither) — matching SpliceAI/OpenSpliceAI's output format.

Supports multiple embedding models:
  - **Evo2** (7B/40B) — causal DNA LM (Apache 2.0)
  - **SpliceBERT** (19.4M) — bidirectional BERT (AGPL-3.0)
  - **HyenaDNA** (1.6M-300M) — causal Hyena operator (Apache 2.0)
  - **DNABERT-2** (117M) — BPE-tokenized BERT (custom license)

Pipeline:
  1. Generate or load per-gene embeddings (windowed)
  2. Build 3-class splice labels (0=none, 1=acceptor, 2=donor)
  3. Train SpliceClassifier with focal loss on windowed data
  4. Evaluate on held-out genes

Usage:
    # Mock mode (no GPU — validates entire pipeline)
    python 06_dense_splice_predictor.py \\
        --foundation-model splicebert --mock --n-genes 5 \\
        -o /tmp/dense-test/

    # Real data — specific genes (local M1 Mac)
    python 06_dense_splice_predictor.py \\
        --foundation-model splicebert --genes TP53 GAPDH HBB \\
        -o /tmp/dense-real/

    # Real data — default gene set on GPU
    python 06_dense_splice_predictor.py \\
        --foundation-model splicebert --n-genes 10 \\
        -o /workspace/output/dense-splicebert/

    # Evo2 on A40+
    python 06_dense_splice_predictor.py \\
        --foundation-model evo2 --n-genes 10 \\
        -o /workspace/output/dense-evo2/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_window_config(
    model_name: str,
    max_context: int,
) -> Tuple[int, int]:
    """Return (window_size, step_size) based on model capabilities.

    Smaller models get smaller windows to match their embedding context.
    """
    if max_context <= 1024:
        # SpliceBERT: 512 bp windows
        return 512, 256
    elif max_context <= 8192:
        return 4096, 2048
    else:
        # Evo2, HyenaDNA: 8192 bp windows
        return 8192, 4096


# Default genes for local testing — short-to-medium genes on distinct
# chromosomes so gene-level splits produce meaningful train/val/test sets.
DEFAULT_GENES = [
    "TP53",    # chr17, ~20 kb, 11 exons
    "GAPDH",   # chr12, ~4.4 kb, 9 exons
    "ACTB",    # chr7, ~3.5 kb, 6 exons
    "SOD1",    # chr21, ~11.5 kb, 5 exons
    "FOS",     # chr14, ~3.4 kb, 4 exons
    "MYC",     # chr8, ~4.3 kb, 3 exons
    "KRAS",    # chr12, ~46 kb, 6 exons
    "HBB",     # chr11, ~1.6 kb, 3 exons
    "ALDOB",   # chr9, ~14.5 kb, 9 exons
    "PKD1",    # chr16, ~47 kb, 46 exons
]


def _resolve_chrom_key(fasta: "pyfaidx.Fasta", chrom: str) -> Optional[str]:
    """Resolve chromosome name against FASTA index (handles chr prefix)."""
    if chrom in fasta:
        return chrom
    bare = chrom.lstrip("chr") if chrom.startswith("chr") else chrom
    if bare in fasta:
        return bare
    prefixed = f"chr{bare}"
    if prefixed in fasta:
        return prefixed
    return None


def _default_overlap(max_context: int) -> int:
    """Sensible overlap for chunked embedding extraction."""
    return min(128, max_context // 8)


def _bpe_to_nucleotide_embeddings(
    model: "BaseEmbeddingModel",
    sequence: str,
) -> torch.Tensor:
    """Upsample BPE token embeddings to per-nucleotide via replication.

    Each BPE token's embedding is replicated to all nucleotides it covers,
    using the tokenizer's offset_mapping when available.

    Returns:
        Tensor of shape ``[len(sequence), hidden_dim]``.
    """
    # Get token-level embeddings (special tokens already stripped)
    token_emb = model.encode(sequence)  # [num_tokens, hidden_dim]
    hidden_dim = token_emb.shape[-1]

    # Try offset_mapping from tokenizer
    try:
        inputs = model.tokenizer(
            sequence, return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = inputs["offset_mapping"][0].tolist()  # [(start, end), ...]
        # Strip special tokens ([CLS] at 0, [SEP] at end)
        offsets = offsets[1:-1]
    except Exception:
        # Fallback: evenly distribute tokens across nucleotides
        n_tokens = token_emb.shape[0]
        seq_len = len(sequence)
        nuc_emb = torch.zeros(seq_len, hidden_dim, dtype=token_emb.dtype,
                              device=token_emb.device)
        for i in range(n_tokens):
            start = int(i * seq_len / n_tokens)
            end = int((i + 1) * seq_len / n_tokens)
            nuc_emb[start:end] = token_emb[i]
        return nuc_emb

    # Replicate each token embedding to its nucleotide span
    seq_len = len(sequence)
    nuc_emb = torch.zeros(seq_len, hidden_dim, dtype=token_emb.dtype,
                          device=token_emb.device)

    n_mapped = min(len(offsets), token_emb.shape[0])
    for i in range(n_mapped):
        start, end = offsets[i]
        if start < seq_len and end > start:
            end = min(end, seq_len)
            nuc_emb[start:end] = token_emb[i]

    return nuc_emb


def extract_dense_gene_embeddings(
    model: "BaseEmbeddingModel",
    gene_sequence: str,
    max_context: int,
    tokenization: str = "character",
    overlap: Optional[int] = None,
) -> np.ndarray:
    """Extract per-nucleotide embeddings for an entire gene.

    Chunks the gene if it exceeds *max_context*, encodes each chunk,
    and stitches the results back to ``[gene_len, hidden_dim]``.

    For BPE-tokenized models (DNABERT-2), token embeddings are upsampled
    to nucleotide resolution before stitching.

    Args:
        model: Loaded foundation model with ``encode()`` method.
        gene_sequence: Full gene DNA sequence.
        max_context: Model's maximum input length in nucleotides.
        tokenization: ``"character"`` or ``"bpe"``.
        overlap: Overlap between chunks (default: model-dependent).

    Returns:
        ``np.ndarray`` of shape ``[gene_len, hidden_dim]``, float32.
    """
    from foundation_models.utils.chunking import chunk_sequence, stitch_embeddings

    gene_len = len(gene_sequence)
    hidden_dim = model.metadata().hidden_dim
    ovlp = overlap if overlap is not None else _default_overlap(max_context)

    if gene_len <= max_context:
        # Single pass — no chunking needed
        with torch.no_grad():
            if tokenization == "bpe":
                emb = _bpe_to_nucleotide_embeddings(model, gene_sequence)
            else:
                emb = model.encode(gene_sequence)
        if hasattr(emb, "cpu"):
            emb = emb.detach().cpu().float().numpy()
        # Pad or trim to exact gene_len (encode may differ by +-1)
        if emb.shape[0] >= gene_len:
            return emb[:gene_len]
        padded = np.zeros((gene_len, hidden_dim), dtype=np.float32)
        padded[:emb.shape[0]] = emb
        return padded

    # Chunked extraction
    chunks = chunk_sequence(gene_sequence, chunk_size=max_context, overlap=ovlp)
    chunk_embeddings = []

    for chunk in chunks:
        with torch.no_grad():
            if tokenization == "bpe":
                emb = _bpe_to_nucleotide_embeddings(model, chunk.sequence)
            else:
                emb = model.encode(chunk.sequence)
        if hasattr(emb, "cpu"):
            emb = emb.detach().cpu().float().numpy()
        # Ensure chunk embedding matches chunk sequence length
        chunk_len = len(chunk.sequence)
        if emb.shape[0] < chunk_len:
            padded = np.zeros((chunk_len, hidden_dim), dtype=np.float32)
            padded[:emb.shape[0]] = emb
            emb = padded
        elif emb.shape[0] > chunk_len:
            emb = emb[:chunk_len]
        chunk_embeddings.append(emb)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stitch_embeddings(chunks, chunk_embeddings, gene_len, hidden_dim)


# -----------------------------------------------------------------------
# Step 1: Generate windowed data (mock or real)
# -----------------------------------------------------------------------

def prepare_mock_data(
    n_genes: int,
    hidden_dim: int,
    window_size: int,
    step_size: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate synthetic windowed embeddings + splice labels.

    Returns:
        (embeddings, labels, gene_ids) where:
        - embeddings: [N_windows, window_size, hidden_dim]
        - labels: [N_windows, window_size] (0/1/2)
        - gene_ids: list of gene IDs per window
    """
    from foundation_models.utils.synthetic import generate_synthetic_splice_data

    emb_dict, lbl_dict = generate_synthetic_splice_data(
        n_genes=n_genes, hidden_dim=hidden_dim, seed=seed,
    )

    # Window each gene
    all_emb = []
    all_lbl = []
    all_genes = []

    for gene_id in sorted(emb_dict.keys()):
        emb = emb_dict[gene_id]   # [seq_len, hidden_dim]
        lbl = lbl_dict[gene_id]   # [seq_len]
        seq_len = len(lbl)

        pos = 0
        while pos + window_size <= seq_len:
            all_emb.append(emb[pos:pos + window_size])
            all_lbl.append(lbl[pos:pos + window_size])
            all_genes.append(gene_id)
            pos += step_size

    embeddings = np.stack(all_emb, axis=0)  # [N, W, H]
    labels = np.stack(all_lbl, axis=0)      # [N, W]

    n_splice = (labels > 0).sum()
    n_total = labels.size
    logger.info(
        "Mock data: %d windows (size=%d, step=%d) from %d genes",
        len(embeddings), window_size, step_size, n_genes,
    )
    logger.info(
        "  Splice sites: %d / %d positions (%.4f%%)",
        n_splice, n_total, n_splice / n_total * 100 if n_total else 0,
    )

    return embeddings, labels, all_genes


def prepare_real_data(
    genes: List[str],
    model_name: str,
    model_kwargs: dict,
    hidden_dim: int,
    max_context: int,
    tokenization: str,
    window_size: int,
    step_size: int,
    build: str = "GRCh38_MANE",
    data_dir: Optional[Path] = None,
    max_gene_length: int = 200_000,
    overlap: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract real embeddings and splice labels for a set of genes.

    Pipeline per gene:
      1. Look up gene coordinates from GTF
      2. Extract gene sequence from FASTA (pyfaidx)
      3. Build 3-class splice labels from annotation
      4. Extract dense per-nucleotide embeddings (chunked if needed)
      5. Window embeddings + labels for training

    Returns:
        Same format as :func:`prepare_mock_data`:
        ``(embeddings, labels, gene_ids)`` where shapes are
        ``[N_windows, window_size, hidden_dim]``,
        ``[N_windows, window_size]``, and ``List[str]``.
    """
    import pandas as pd
    import polars as pl
    from pyfaidx import Fasta

    from foundation_models.base import load_embedding_model
    from foundation_models.utils.chunking import build_splice_labels

    # --- Resolve data paths ---
    release = "1.3" if "MANE" in build.upper() else None
    try:
        from agentic_spliceai.splice_engine.resources import get_genomic_registry
        registry = get_genomic_registry(build=build, release=release)
        gtf_path = str(registry.get_gtf_path(validate=True))
        fasta_path = str(registry.get_fasta_path(validate=True))
        if data_dir is None:
            data_dir = Path(fasta_path).parent
    except Exception as exc:
        logger.error("Cannot resolve data paths for build=%s: %s", build, exc)
        logger.error(
            "Ensure data is prepared. See: "
            "python examples/data_preparation/04_generate_ground_truth.py "
            "--output data/mane/GRCh38/"
        )
        sys.exit(1)

    # --- Load gene annotations ---
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        load_gene_annotations,
    )
    gene_data = load_gene_annotations(gtf_path=gtf_path, genes=genes, verbosity=0)
    if gene_data.height == 0:
        logger.error("No matching genes found in GTF for: %s", genes)
        sys.exit(1)

    # Filter out genes exceeding max_gene_length
    gene_data = gene_data.with_columns(
        (pl.col("end") - pl.col("start")).alias("gene_length"),
    )
    too_long = gene_data.filter(pl.col("gene_length") > max_gene_length)
    if too_long.height > 0:
        for row in too_long.iter_rows(named=True):
            logger.info(
                "  Skipping %s (%d bp > max %d)",
                row["gene_name"], row["gene_length"], max_gene_length,
            )
    gene_data = gene_data.filter(pl.col("gene_length") <= max_gene_length)

    if gene_data.height == 0:
        logger.error("All genes exceeded --max-gene-length %d", max_gene_length)
        sys.exit(1)

    logger.info("Processing %d genes (build=%s)", gene_data.height, build)

    # --- Load splice site annotations ---
    splice_sites_df = None
    for fname in ("splice_sites_enhanced.parquet", "splice_sites_enhanced.tsv"):
        path = data_dir / fname
        if path.exists():
            if fname.endswith(".parquet"):
                splice_sites_df = pd.read_parquet(path)
            else:
                splice_sites_df = pd.read_csv(path, sep="\t")
            logger.info("Loaded splice sites from %s (%d rows)", path, len(splice_sites_df))
            break

    if splice_sites_df is None:
        logger.error(
            "Splice site annotations not found in %s. "
            "Generate with:\n  python examples/data_preparation/"
            "04_generate_ground_truth.py --output %s",
            data_dir, data_dir,
        )
        sys.exit(1)

    # --- Open FASTA ---
    fasta = Fasta(fasta_path)
    logger.info("Using FASTA: %s", fasta_path)

    # --- Load foundation model ---
    logger.info("Loading foundation model '%s'...", model_name)
    model = load_embedding_model(model_name, **model_kwargs)
    meta = model.metadata()
    logger.info("Model loaded: %s (hidden_dim=%d)", meta.name, meta.hidden_dim)

    # --- Per-gene extraction ---
    all_emb: List[np.ndarray] = []
    all_lbl: List[np.ndarray] = []
    all_genes: List[str] = []
    total_splice_sites = 0

    for row in gene_data.iter_rows(named=True):
        gene_id = row["gene_id"]
        gene_name = row["gene_name"]
        chrom = row.get("chrom") or row.get("seqname", "")
        start = int(row["start"])
        end = int(row["end"])
        gene_len = end - start

        # Resolve chromosome in FASTA
        chrom_key = _resolve_chrom_key(fasta, str(chrom))
        if chrom_key is None:
            logger.warning("  %s: chromosome %s not in FASTA, skipping", gene_name, chrom)
            continue

        # Extract sequence (GTF is 1-based, pyfaidx is 0-based)
        gene_seq = str(fasta[chrom_key][start:end]).upper()
        if len(gene_seq) < window_size:
            logger.info("  %s: %d bp < window_size %d, skipping", gene_name, len(gene_seq), window_size)
            continue

        # Build splice labels
        labels = build_splice_labels(
            gene_id=gene_id,
            gene_start=start,
            gene_sequence_length=len(gene_seq),
            splice_sites_df=splice_sites_df,
        )
        n_splice = int((labels > 0).sum())
        total_splice_sites += n_splice

        # Extract dense embeddings
        gene_emb = extract_dense_gene_embeddings(
            model=model,
            gene_sequence=gene_seq,
            max_context=max_context,
            tokenization=tokenization,
            overlap=overlap,
        )

        # Window
        n_windows = 0
        pos = 0
        while pos + window_size <= len(gene_seq):
            all_emb.append(gene_emb[pos:pos + window_size])
            all_lbl.append(labels[pos:pos + window_size])
            all_genes.append(gene_name)
            pos += step_size
            n_windows += 1

        logger.info(
            "  %s (%s): %d bp, %d splice sites, %d windows",
            gene_name, gene_id, gene_len, n_splice, n_windows,
        )

        # Free per-gene memory
        del gene_emb, labels, gene_seq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fasta.close()

    if not all_emb:
        logger.error("No windows extracted. Use more/larger genes or smaller --window-size.")
        sys.exit(1)

    embeddings = np.stack(all_emb, axis=0)
    labels_arr = np.stack(all_lbl, axis=0)

    n_splice = int((labels_arr > 0).sum())
    n_total = labels_arr.size
    logger.info(
        "Real data: %d windows (size=%d, step=%d) from %d genes",
        len(embeddings), window_size, step_size, gene_data.height,
    )
    logger.info(
        "  Splice sites: %d / %d positions (%.4f%%)",
        n_splice, n_total, n_splice / n_total * 100 if n_total else 0,
    )

    return embeddings, labels_arr, all_genes


# -----------------------------------------------------------------------
# Step 2: Train / Val / Test split
# -----------------------------------------------------------------------

def split_by_gene(
    embeddings: np.ndarray,
    labels: np.ndarray,
    gene_ids: List[str],
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Split windowed data by gene (no data leakage).

    Returns dict with keys 'train', 'val', 'test', each containing
    (embeddings, labels) arrays.
    """
    rng = np.random.RandomState(seed)
    unique_genes = sorted(set(gene_ids))
    rng.shuffle(unique_genes)

    n_test = max(1, int(len(unique_genes) * test_fraction))
    n_val = max(1, int(len(unique_genes) * val_fraction))

    test_genes = set(unique_genes[:n_test])
    val_genes = set(unique_genes[n_test:n_test + n_val])
    train_genes = set(unique_genes[n_test + n_val:])

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    gene_arr = np.array(gene_ids)

    for name, gene_set in [
        ("train", train_genes), ("val", val_genes), ("test", test_genes),
    ]:
        mask = np.isin(gene_arr, list(gene_set))
        splits[name] = (embeddings[mask], labels[mask])
        n_splice = (labels[mask] > 0).sum()
        logger.info(
            "  %s: %d windows (%d genes), %d splice sites",
            name, mask.sum(), len(gene_set), n_splice,
        )

    return splits


# -----------------------------------------------------------------------
# Step 3: Train and evaluate
# -----------------------------------------------------------------------

def train_and_evaluate(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    input_dim: int,
    output_dir: Path,
    architecture: str = "dilated_cnn",
    hidden_dim: int = 128,
    lr: float = 1e-3,
    batch_size: int = 16,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 20,
    focal_gamma: float = 2.0,
) -> Dict:
    """Train SpliceClassifier and evaluate on test set.

    Returns:
        Results dict with training history and test metrics.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    from foundation_models.classifiers.splice_classifier import SpliceClassifier

    device = get_device()

    train_emb, train_lbl = splits["train"]
    val_emb, val_lbl = splits["val"]
    test_emb, test_lbl = splits["test"]

    # Convert to tensors
    train_emb_t = torch.tensor(train_emb, dtype=torch.float32)
    train_lbl_t = torch.tensor(train_lbl, dtype=torch.long)
    val_emb_t = torch.tensor(val_emb, dtype=torch.float32)
    val_lbl_t = torch.tensor(val_lbl, dtype=torch.long)

    # Create classifier
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    classifier = SpliceClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        architecture=architecture,
    )

    n_params = sum(p.numel() for p in classifier.parameters())

    print()
    print("=" * 60)
    print("Training SpliceClassifier (dense splice site prediction)")
    print("=" * 60)
    print(f"  Architecture: {architecture}")
    print(f"  Input dim:    {input_dim}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Parameters:   {n_params:,}")
    print(f"  Windows:      {len(train_emb)} train, {len(val_emb)} val, {len(test_emb)} test")
    print(f"  Device:       {device}")
    print(f"  Focal gamma:  {focal_gamma}")
    print(f"  Hyperparams:  lr={lr}, batch={batch_size}, wd={weight_decay}")
    print("=" * 60)
    print()

    # Train
    history = classifier.fit(
        train_emb_t, train_lbl_t,
        val_emb_t, val_lbl_t,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        checkpoint_dir=str(model_dir),
        patience=patience,
        focal_gamma=focal_gamma,
    )

    # Evaluate on test set
    classifier.to(device)
    test_emb_t = torch.tensor(test_emb, dtype=torch.float32).to(device)

    preds = classifier.predict(test_emb_t)

    # Per-class metrics (acceptor + donor)
    test_lbl_flat = test_lbl.flatten()
    results: Dict = {"history": history}

    for cls_name, cls_idx in [("acceptor", 1), ("donor", 2)]:
        y_true = (test_lbl_flat == cls_idx).astype(np.int32)
        y_score = preds[f"{cls_name}_prob"].flatten()

        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            auroc = roc_auc_score(y_true, y_score)
            auprc = average_precision_score(y_true, y_score)
        else:
            auroc = float("nan")
            auprc = float("nan")

        results[cls_name] = {
            "auroc": auroc,
            "auprc": auprc,
            "n_positive": int(y_true.sum()),
            "n_total": int(len(y_true)),
        }

    # Average across splice classes
    aurocs = [
        results[c]["auroc"] for c in ("acceptor", "donor")
        if not np.isnan(results[c]["auroc"])
    ]
    auprcs = [
        results[c]["auprc"] for c in ("acceptor", "donor")
        if not np.isnan(results[c]["auprc"])
    ]
    results["mean_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    results["mean_auprc"] = float(np.mean(auprcs)) if auprcs else float("nan")

    # Save results
    results_path = model_dir / "test_results.json"
    serializable = {
        k: v for k, v in results.items()
        if k != "history"
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print()
    print("Test Metrics (held-out genes):")
    for cls_name in ("acceptor", "donor"):
        m = results[cls_name]
        print(f"  {cls_name:10s}: AUROC={m['auroc']:.4f}, "
              f"AUPRC={m['auprc']:.4f} "
              f"({m['n_positive']} sites / {m['n_total']} positions)")
    print(f"  {'mean':10s}: AUROC={results['mean_auroc']:.4f}, "
          f"AUPRC={results['mean_auprc']:.4f}")
    print()

    return results


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    from foundation_models.base import (
        get_model_metadata,
        list_available_models,
    )

    available_models = list_available_models()

    parser = argparse.ArgumentParser(
        description="Dense per-nucleotide splice site predictor using "
                    "foundation model embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model selection
    parser.add_argument("--foundation-model", type=str, default="splicebert",
                        choices=available_models,
                        help=f"Foundation model (default: splicebert; "
                             f"available: {', '.join(available_models)})")
    parser.add_argument("--model-size", type=str, default=None,
                        help="Model-specific variant")

    # Data
    parser.add_argument("--n-genes", type=int, default=10,
                        help="Number of genes (mock mode: synthetic; "
                             "real mode: max genes to process)")
    parser.add_argument("--window-size", type=int, default=None,
                        help="Window size in bp (default: model-dependent)")
    parser.add_argument("--step-size", type=int, default=None,
                        help="Step size in bp (default: window_size // 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output directory")

    # Classifier architecture
    parser.add_argument("--architecture", type=str, default="dilated_cnn",
                        choices=["dilated_cnn", "mlp", "linear"],
                        help="Prediction head architecture (default: dilated_cnn)")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Prediction head hidden dim (default: 128)")

    # Training hyperparams
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Max epochs (default: 100)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (default: 20)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal loss gamma (default: 2.0)")

    # Data mode
    parser.add_argument("--mock", action="store_true",
                        help="Use synthetic data (no GPU or FASTA needed)")
    parser.add_argument("--genes", nargs="+", type=str, default=None,
                        help="Gene symbols for real-data mode "
                             "(default: built-in small set)")
    parser.add_argument("--build", type=str, default="GRCh38_MANE",
                        help="Genomic build for real data (default: GRCh38_MANE)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory with splice_sites_enhanced.tsv "
                             "(default: auto-detect from registry)")
    parser.add_argument("--max-gene-length", type=int, default=200_000,
                        help="Skip genes longer than this (default: 200000)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Get model metadata (lightweight — no model loading)
    model_kwargs: dict = {}
    if args.foundation_model == "evo2":
        model_kwargs["model_size"] = args.model_size or "7b"
    elif args.foundation_model == "hyenadna":
        model_kwargs["model_size"] = args.model_size or "medium-160k"
    elif args.foundation_model == "splicebert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size
    elif args.foundation_model == "dnabert":
        if args.model_size:
            model_kwargs["model_variant"] = args.model_size

    meta = get_model_metadata(args.foundation_model, **model_kwargs)
    hidden_dim = meta.hidden_dim

    # Resolve window config
    if args.window_size is not None:
        window_size = args.window_size
        step_size = args.step_size or window_size // 2
    else:
        window_size, step_size = get_default_window_config(
            meta.name, meta.max_context,
        )
        if args.step_size is not None:
            step_size = args.step_size

    print()
    print("=" * 70)
    print("Dense Splice Site Predictor (Foundation Model Embeddings)")
    print("=" * 70)
    print(f"  Model:        {'MOCK (synthetic)' if args.mock else meta.name}")
    print(f"  Type:         {meta.model_type}")
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Window:       {window_size} bp (step: {step_size})")
    if args.mock:
        print(f"  Data:         MOCK (synthetic, {args.n_genes} genes)")
    elif args.genes:
        print(f"  Data:         Real ({', '.join(args.genes)})")
    else:
        n = min(args.n_genes, len(DEFAULT_GENES))
        print(f"  Data:         Real (default {n} genes)")
    print(f"  Focal gamma:  {args.focal_gamma}")
    print(f"  Seed:         {args.seed}")
    print("=" * 70)
    print()

    # Step 1: Prepare data
    if args.mock:
        logger.info("Step 1/2: Generating synthetic windowed data...")
        embeddings, labels, gene_ids = prepare_mock_data(
            n_genes=args.n_genes,
            hidden_dim=hidden_dim,
            window_size=window_size,
            step_size=step_size,
            seed=args.seed,
        )
    else:
        gene_list = args.genes if args.genes else DEFAULT_GENES[:args.n_genes]
        logger.info(
            "Step 1/2: Extracting real embeddings for %d genes: %s",
            len(gene_list), ", ".join(gene_list),
        )
        embeddings, labels, gene_ids = prepare_real_data(
            genes=gene_list,
            model_name=args.foundation_model,
            model_kwargs=model_kwargs,
            hidden_dim=hidden_dim,
            max_context=meta.max_context,
            tokenization=meta.tokenization,
            window_size=window_size,
            step_size=step_size,
            build=args.build,
            data_dir=Path(args.data_dir) if args.data_dir else None,
            max_gene_length=args.max_gene_length,
            overlap=None,
        )

    # Step 2: Split and train
    logger.info("Splitting by gene...")
    splits = split_by_gene(
        embeddings, labels, gene_ids, seed=args.seed,
    )

    logger.info("Step 2/2: Training classifier...")
    results = train_and_evaluate(
        splits=splits,
        input_dim=hidden_dim,
        output_dir=output_dir,
        architecture=args.architecture,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
    )

    elapsed = time.time() - t0

    print("=" * 70)
    print("Complete")
    print("=" * 70)
    print(f"  Model:        {meta.name}")
    print(f"  Architecture: {args.architecture}")
    print(f"  Output:       {output_dir}")
    print(f"  Checkpoint:   {output_dir / 'model' / 'best_model.pt'}")
    print(f"  Mean AUROC:   {results['mean_auroc']:.4f}")
    print(f"  Mean AUPRC:   {results['mean_auprc']:.4f}")
    print(f"  Time:         {elapsed / 60:.1f} min")
    print()


if __name__ == "__main__":
    main()
