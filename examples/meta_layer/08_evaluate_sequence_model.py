#!/usr/bin/env python
"""Evaluate a trained M*-S meta-splice model against base model scores.

Compares the meta-layer's refined [L, 3] predictions against the raw
base model scores on held-out test genes, answering: "Does the
meta-layer actually improve over the base model?"

Metrics reported:
- Per-class and macro PR-AUC
- Accuracy, top-1, top-2
- Per-class precision, recall, F1
- FN/FP counts and reduction vs base model
- SpliceAI paper top-k accuracy (k = 0.5, 1, 2, 4 × n_true)
- Confusion matrices for both models

Usage:
    # Evaluate M1-S on test chromosomes (chr1,3,5,7,9)
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --cache-dir output/meta_layer/gene_cache/val

    # Evaluate on specific chromosomes
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --test-chroms chr1 chr3 \\
        --cache-dir /path/to/gene_cache

    # M2a evaluation: Ensembl-only sites
    python 08_evaluate_sequence_model.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --annotation-source ensembl \\
        --filter-mane-sites
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    class_names: List[str] = ["donor", "acceptor", "neither"],
) -> dict:
    """Compute classification metrics from [N, 3] probs and [N] labels."""
    from sklearn.metrics import (
        accuracy_score, average_precision_score, confusion_matrix,
        f1_score, precision_score, recall_score,
    )

    n_classes = len(class_names)
    preds = probs.argmax(axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "top_1_accuracy": float(accuracy_score(labels, preds)),
    }

    # Top-2 accuracy
    top2 = np.argsort(probs, axis=1)[:, -2:]
    metrics["top_2_accuracy"] = float(
        np.any(top2 == labels.reshape(-1, 1), axis=1).mean()
    )

    # Per-class PR-AUC
    pr_aucs = {}
    for i, name in enumerate(class_names):
        binary = (labels == i).astype(int)
        if binary.sum() > 0 and binary.sum() < len(binary):
            pr_aucs[name] = float(average_precision_score(binary, probs[:, i]))
    metrics["pr_aucs"] = pr_aucs
    metrics["macro_pr_auc"] = float(np.mean(list(pr_aucs.values()))) if pr_aucs else 0.0

    # Per-class precision, recall, F1
    for i, name in enumerate(class_names):
        binary_true = (labels == i).astype(int)
        binary_pred = (preds == i).astype(int)
        if binary_true.sum() > 0:
            metrics[f"{name}_precision"] = float(precision_score(binary_true, binary_pred, zero_division=0))
            metrics[f"{name}_recall"] = float(recall_score(binary_true, binary_pred, zero_division=0))
            metrics[f"{name}_f1"] = float(f1_score(binary_true, binary_pred, zero_division=0))

    # FN / FP counts for splice sites (donor + acceptor)
    splice_true = labels < 2  # donor=0, acceptor=1
    splice_pred = preds < 2
    metrics["fn_count"] = int((splice_true & ~splice_pred).sum())
    metrics["fp_count"] = int((~splice_true & splice_pred).sum())
    metrics["tp_count"] = int((splice_true & splice_pred).sum())

    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def compute_topk_accuracy_from_arrays(
    gene_probs: Dict[str, np.ndarray],
    gene_labels: Dict[str, np.ndarray],
    k_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0],
) -> Dict[str, Dict[float, float]]:
    """Compute SpliceAI paper top-k accuracy from per-gene arrays.

    For each gene and splice type:
    1. Count true sites: n_true
    2. Rank positions by predicted probability (descending)
    3. Take top k = k_multiplier × n_true positions
    4. Top-k accuracy = (# true sites in top-k) / n_true

    Parameters
    ----------
    gene_probs : dict
        gene_id → [L, 3] probability array
    gene_labels : dict
        gene_id → [L] label array (0=donor, 1=acceptor, 2=neither)
    k_multipliers : list
        Multipliers for n_true

    Returns
    -------
    dict with keys 'donor', 'acceptor', 'overall', each mapping
    k_multiplier → accuracy.
    """
    results = {
        "donor": defaultdict(list),
        "acceptor": defaultdict(list),
    }

    for gene_id in gene_probs:
        probs = gene_probs[gene_id]
        labels = gene_labels[gene_id]

        for splice_idx, splice_name in [(0, "donor"), (1, "acceptor")]:
            true_mask = labels == splice_idx
            n_true = int(true_mask.sum())
            if n_true == 0:
                continue

            scores = probs[:, splice_idx]
            ranked_indices = np.argsort(scores)[::-1]

            for km in k_multipliers:
                k = max(1, int(np.ceil(km * n_true)))
                top_k_set = set(ranked_indices[:k])
                true_positions = set(np.where(true_mask)[0])
                recovered = len(top_k_set & true_positions)
                results[splice_name][km].append(recovered / n_true)

    # Average across genes
    topk = {}
    for splice_name in ["donor", "acceptor"]:
        topk[splice_name] = {}
        for km in k_multipliers:
            vals = results[splice_name][km]
            topk[splice_name][km] = float(np.mean(vals)) if vals else 0.0

    # Overall = average of donor and acceptor
    topk["overall"] = {}
    for km in k_multipliers:
        topk["overall"][km] = np.mean([topk["donor"][km], topk["acceptor"][km]])

    return topk


# ---------------------------------------------------------------------------
# Full-gene sliding window inference
# ---------------------------------------------------------------------------


def infer_full_gene(
    model,
    gene_data: dict,
    window_size: int = 5001,
    context_padding: int = 400,
    device: "torch.device" = None,
) -> np.ndarray:
    """Run sliding window inference on a full gene, return [L, 3] probs."""
    import torch
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _one_hot_encode,
    )

    if device is None:
        device = torch.device("cpu")

    sequence = gene_data["sequence"]
    base_scores = gene_data["base_scores"]  # [L, 3]
    mm_features = gene_data["mm_features"]  # [L, C]
    gene_len = len(sequence)
    W = window_size
    ctx = context_padding

    if gene_len <= W:
        # Small gene: single window with padding
        seq_onehot = _one_hot_encode(sequence)  # [4, L]
        total_len = W + ctx
        padded_seq = np.zeros((4, total_len), dtype=np.float32)
        offset = (total_len - seq_onehot.shape[1]) // 2
        padded_seq[:, offset:offset + seq_onehot.shape[1]] = seq_onehot

        padded_base = np.full((W, 3), 1.0 / 3, dtype=np.float32)
        padded_base[:gene_len] = base_scores[:gene_len]

        padded_mm = np.zeros((mm_features.shape[1], W), dtype=np.float32)
        padded_mm[:, :gene_len] = mm_features[:gene_len].T

        with torch.no_grad():
            seq_t = torch.from_numpy(padded_seq).unsqueeze(0).to(device)
            base_t = torch.from_numpy(padded_base).unsqueeze(0).to(device)
            mm_t = torch.from_numpy(padded_mm).unsqueeze(0).to(device)
            probs = model(seq_t, base_t, mm_t)  # [1, W, 3]
            return probs[0, :gene_len].cpu().numpy()

    # Sliding window with overlap
    stride = W // 2  # 50% overlap
    accum = np.zeros((gene_len, 3), dtype=np.float64)
    counts = np.zeros(gene_len, dtype=np.float64)

    starts = list(range(0, gene_len - W + 1, stride))
    if starts[-1] + W < gene_len:
        starts.append(gene_len - W)  # ensure last positions are covered

    for out_start in starts:
        out_end = out_start + W

        # Sequence with context
        seq_start = max(0, out_start - ctx // 2)
        seq_end = min(gene_len, out_end + (ctx - ctx // 2))
        seq_onehot = _one_hot_encode(sequence[seq_start:seq_end])

        total_len = W + ctx
        if seq_onehot.shape[1] < total_len:
            padded = np.zeros((4, total_len), dtype=np.float32)
            off = (total_len - seq_onehot.shape[1]) // 2
            padded[:, off:off + seq_onehot.shape[1]] = seq_onehot
            seq_onehot = padded

        bs = base_scores[out_start:out_end]  # [W, 3]
        mm = mm_features[out_start:out_end].T.copy()  # [C, W]

        with torch.no_grad():
            seq_t = torch.from_numpy(seq_onehot).unsqueeze(0).to(device)
            base_t = torch.from_numpy(bs).unsqueeze(0).to(device)
            mm_t = torch.from_numpy(mm).unsqueeze(0).to(device)
            probs = model(seq_t, base_t, mm_t)  # [1, W, 3]
            window_probs = probs[0].cpu().numpy()

        accum[out_start:out_end] += window_probs
        counts[out_start:out_end] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    return (accum / counts[:, None]).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M*-S meta-splice model vs base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to trained model checkpoint (best.pt)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.pt (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, required=True,
        help="Directory with gene cache (.npz files)",
    )
    parser.add_argument(
        "--test-chroms", nargs="+", default=None,
        help="Evaluate on genes from these chromosomes "
             "(default: SpliceAI test set chr1,3,5,7,9)",
    )
    parser.add_argument(
        "--genes", nargs="+", default=None,
        help="Evaluate on specific genes by name/ID (inference mode). "
             "Overrides --test-chroms.",
    )
    parser.add_argument(
        "--annotation-source", choices=["mane", "ensembl"], default="mane",
        help="Annotation source for gene annotations (default: mane)",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit number of genes for quick testing",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for evaluation results (default: checkpoint dir)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for inference (default: cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    # ── Load model ───────────────────────────────────────────────────
    config_path = args.config or args.checkpoint.parent / "config.pt"
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}")
        return 1

    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceModel, MetaSpliceConfig,
    )

    device = torch.device(args.device)
    torch.serialization.add_safe_globals([MetaSpliceConfig])
    cfg = torch.load(config_path, map_location="cpu", weights_only=True)
    model = MetaSpliceModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.variant}, {n_params:,} params")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")

    # ── Resolve test genes ───────────────────────────────────────────
    from agentic_spliceai.splice_engine.resources import get_model_resources, get_genomic_registry
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import extract_gene_annotations
    from agentic_spliceai.splice_engine.eval.splitting import (
        build_gene_split, gene_chromosomes_from_dataframe,
    )

    resources = get_model_resources("openspliceai")
    ann_src = args.annotation_source
    if ann_src == "ensembl":
        ann_registry = get_genomic_registry(build="GRCh38", release="112")
    else:
        ann_registry = resources.get_registry()

    gtf_path = str(ann_registry.get_gtf_path())
    gene_annotations = extract_gene_annotations(gtf_path, verbosity=0)

    # ── Resolve evaluation gene set ────────────────────────────────
    # Three modes:
    # 1. --genes BRCA1 TP53: evaluate specific genes (inference mode)
    # 2. --test-chroms chr21 chr22: evaluate all genes on these chromosomes
    # 3. Default: SpliceAI test split (chr1,3,5,7,9)
    gene_chroms = gene_chromosomes_from_dataframe(gene_annotations)

    if args.genes:
        # Mode 1: user-specified genes
        test_genes = args.genes
        test_chroms = sorted(set(
            gene_chroms.get(g, "unknown") for g in test_genes if g in gene_chroms
        ))
        print(f"  Eval genes: {len(test_genes)} (user-specified)")
    elif args.test_chroms:
        # Mode 2: user-specified chromosomes
        test_chroms = [c if c.startswith("chr") else f"chr{c}" for c in args.test_chroms]
        test_genes = sorted(g for g, c in gene_chroms.items() if c in test_chroms)
        print(f"  Eval genes: {len(test_genes)} on {test_chroms}")
    else:
        # Mode 3: SpliceAI test split (default)
        gene_split = build_gene_split(gene_chroms, preset="spliceai", val_fraction=0.0)
        test_genes = sorted(gene_split.test_genes)
        test_chroms = sorted(set(
            gene_chroms.get(g, "unknown") for g in test_genes if g in gene_chroms
        ))
        print(f"  Eval genes: {len(test_genes)} (SpliceAI test: {test_chroms})")

    if args.max_genes:
        test_genes = test_genes[:args.max_genes]
        print(f"  Limited to {len(test_genes)} genes")

    # ── Load gene cache ──────────────────────────────────────────────
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz, _one_hot_encode,
    )

    cache_dir = args.cache_dir
    min_length = 5001 + 400  # window_size + context_padding

    gene_data_map: Dict[str, dict] = {}
    skipped = 0
    for gene_id in test_genes:
        npz_path = cache_dir / f"{gene_id}.npz"
        if not npz_path.exists():
            skipped += 1
            continue
        data = _load_gene_npz(npz_path)
        if len(data["sequence"]) < min_length:
            skipped += 1
            continue
        gene_data_map[gene_id] = data

    print(f"  Loaded {len(gene_data_map)} genes from cache ({skipped} skipped)")

    if not gene_data_map:
        print("ERROR: No genes loaded. Check --cache-dir path.")
        return 1

    # ── Run evaluation ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Evaluating: Meta model vs Base model")
    print(f"{'='*70}\n")

    meta_gene_probs: Dict[str, np.ndarray] = {}
    base_gene_probs: Dict[str, np.ndarray] = {}
    gene_labels_map: Dict[str, np.ndarray] = {}

    all_meta_probs = []
    all_base_probs = []
    all_labels = []

    t0 = time.time()
    for i, (gene_id, data) in enumerate(gene_data_map.items()):
        # Meta model: sliding window inference
        meta_probs = infer_full_gene(model, data, device=device)
        base_probs = data["base_scores"]  # [L, 3]
        labels = data["labels"]  # [L]

        # Store per-gene for top-k
        meta_gene_probs[gene_id] = meta_probs
        base_gene_probs[gene_id] = base_probs
        gene_labels_map[gene_id] = labels

        # Collect all positions
        all_meta_probs.append(meta_probs)
        all_base_probs.append(base_probs)
        all_labels.append(labels)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(gene_data_map)} genes...")

    elapsed = time.time() - t0
    print(f"  Inference complete: {len(gene_data_map)} genes in {elapsed:.1f}s")

    # Concatenate all positions
    all_meta = np.concatenate(all_meta_probs)
    all_base = np.concatenate(all_base_probs)
    all_lab = np.concatenate(all_labels)
    print(f"  Total positions: {len(all_lab):,}")

    # ── Compute metrics ──────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("Classification Metrics")
    print(f"{'─'*70}\n")

    meta_metrics = compute_classification_metrics(all_meta, all_lab)
    base_metrics = compute_classification_metrics(all_base, all_lab)

    # Display comparison table
    print(f"{'Metric':<30} {'Base Model':>15} {'Meta Model':>15} {'Delta':>12}")
    print(f"{'─'*72}")

    for key in ["accuracy", "top_1_accuracy", "top_2_accuracy", "macro_pr_auc"]:
        b, m = base_metrics[key], meta_metrics[key]
        d = m - b
        print(f"{key:<30} {b:>15.4f} {m:>15.4f} {d:>+12.4f}")

    print()
    for cls in ["donor", "acceptor", "neither"]:
        b_auc = base_metrics["pr_aucs"].get(cls, 0)
        m_auc = meta_metrics["pr_aucs"].get(cls, 0)
        d = m_auc - b_auc
        print(f"PR-AUC ({cls}){'':<17} {b_auc:>15.4f} {m_auc:>15.4f} {d:>+12.4f}")

    print()
    for cls in ["donor", "acceptor"]:
        for met in ["precision", "recall", "f1"]:
            key = f"{cls}_{met}"
            b, m = base_metrics.get(key, 0), meta_metrics.get(key, 0)
            d = m - b
            print(f"{key:<30} {b:>15.4f} {m:>15.4f} {d:>+12.4f}")

    # FN/FP comparison
    print(f"\n{'─'*70}")
    print("Error Analysis (splice sites: donor + acceptor)")
    print(f"{'─'*70}\n")

    b_fn, m_fn = base_metrics["fn_count"], meta_metrics["fn_count"]
    b_fp, m_fp = base_metrics["fp_count"], meta_metrics["fp_count"]
    fn_reduction = (b_fn - m_fn) / max(b_fn, 1) * 100
    fp_reduction = (b_fp - m_fp) / max(b_fp, 1) * 100

    print(f"{'':30} {'Base':>15} {'Meta':>15} {'Reduction':>12}")
    print(f"{'False Negatives':<30} {b_fn:>15,} {m_fn:>15,} {fn_reduction:>+11.1f}%")
    print(f"{'False Positives':<30} {b_fp:>15,} {m_fp:>15,} {fp_reduction:>+11.1f}%")
    print(f"{'True Positives':<30} {base_metrics['tp_count']:>15,} {meta_metrics['tp_count']:>15,}")

    # ── Top-K Accuracy (SpliceAI paper) ──────────────────────────────
    print(f"\n{'─'*70}")
    print("Top-K Accuracy (SpliceAI paper, per-gene ranking)")
    print(f"{'─'*70}\n")

    k_mults = [0.5, 1.0, 2.0, 4.0]
    meta_topk = compute_topk_accuracy_from_arrays(meta_gene_probs, gene_labels_map, k_mults)
    base_topk = compute_topk_accuracy_from_arrays(base_gene_probs, gene_labels_map, k_mults)

    print(f"{'k multiplier':<15}", end="")
    for km in k_mults:
        print(f"  {'k=' + str(km) + 'x':>10}", end="")
    print()
    print("─" * 60)

    for splice in ["donor", "acceptor", "overall"]:
        print(f"\n  {splice.upper()}")
        for label, topk in [("  Base", base_topk), ("  Meta", meta_topk)]:
            print(f"  {label:<12}", end="")
            for km in k_mults:
                print(f"  {topk[splice][km]:>10.4f}", end="")
            print()
        # Delta
        print(f"  {'  Delta':<12}", end="")
        for km in k_mults:
            d = meta_topk[splice][km] - base_topk[splice][km]
            print(f"  {d:>+10.4f}", end="")
        print()

    # ── Save results ─────────────────────────────────────────────────
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": cfg.variant,
        "checkpoint": str(args.checkpoint),
        "annotation_source": ann_src,
        "n_genes": len(gene_data_map),
        "n_positions": int(len(all_lab)),
        "test_chromosomes": test_chroms,
        "meta_model": meta_metrics,
        "base_model": base_metrics,
        "meta_topk": {
            splice: {str(k): v for k, v in vals.items()}
            for splice, vals in meta_topk.items()
        },
        "base_topk": {
            splice: {str(k): v for k, v in vals.items()}
            for splice, vals in base_topk.items()
        },
        "fn_reduction_pct": fn_reduction,
        "fp_reduction_pct": fp_reduction,
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"Summary: Meta model {'improves' if meta_metrics['macro_pr_auc'] > base_metrics['macro_pr_auc'] else 'does not improve'} over base model")
    print(f"  Base PR-AUC: {base_metrics['macro_pr_auc']:.4f}")
    print(f"  Meta PR-AUC: {meta_metrics['macro_pr_auc']:.4f}")
    print(f"  FN reduction: {fn_reduction:+.1f}%")
    print(f"  FP reduction: {fp_reduction:+.1f}%")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
