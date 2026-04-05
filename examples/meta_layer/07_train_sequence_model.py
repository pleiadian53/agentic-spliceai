#!/usr/bin/env python
"""Train the sequence-level multimodal meta-splice model (M*-S).

Two-phase pipeline:
  Phase 1: Build gene cache (base scores + dense features + labels)
  Phase 2: Train MetaSpliceModel on windowed samples

Usage:
    # M1-S: canonical classification, all 9 modalities
    python 07_train_sequence_model.py --mode m1

    # M3-S: novel site prediction, junction excluded
    python 07_train_sequence_model.py --mode m3

    # Quick test on 2 chromosomes
    python 07_train_sequence_model.py --mode m1 --max-genes 100 --epochs 5

    # Resume from checkpoint
    python 07_train_sequence_model.py --mode m1 --checkpoint output/meta_layer/m1s/best.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Focal loss for extreme class imbalance
# ---------------------------------------------------------------------------


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) for per-position classification.

    Reduces the contribution of easy (well-classified) examples so that
    the model focuses on hard splice sites buried in vast "neither" regions.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, L, C] raw logits (before softmax)
        targets : [B, L] integer class labels
        """
        B, L, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        targets_flat = targets.reshape(-1)

        ce = F.cross_entropy(logits_flat, targets_flat, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce)
        focal = ((1 - p_t) ** self.gamma) * ce

        if self.reduction == "mean":
            return focal.mean()
        return focal.sum()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    accumulation_steps: int = 4,
    max_grad_norm: float = 1.0,
) -> float:
    """Train for one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        seq = batch["sequence"].to(device)
        base = batch["base_scores"].to(device)
        mm = batch["mm_features"].to(device)
        labels = batch["labels"].to(device)

        logits = model(seq, base, mm)  # [B, L, C] raw logits (model.training=True)
        loss = criterion(logits, labels) / accumulation_steps
        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        n_batches += 1

        if n_batches % 1000 == 0:
            avg = total_loss / n_batches
            print(f"    batch {n_batches}/{len(loader)} loss={avg:.4f}", flush=True)

    # Final step if batches don't divide evenly
    if n_batches % accumulation_steps != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int = 3,
) -> dict:
    """Evaluate on validation set, return metrics dict."""
    from sklearn.metrics import average_precision_score

    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    n_batches = 0

    for batch in loader:
        seq = batch["sequence"].to(device)
        base = batch["base_scores"].to(device)
        mm = batch["mm_features"].to(device)
        labels = batch["labels"].to(device)

        probs = model(seq, base, mm)  # [B, L, C] softmax probs (model.eval=True)
        logits = torch.log(probs.clamp(min=1e-8))
        loss = criterion(logits.contiguous(), labels)
        total_loss += loss.item()
        n_batches += 1

        # Collect predictions
        all_probs.append(probs.cpu().numpy().reshape(-1, num_classes))
        all_labels.append(labels.cpu().numpy().reshape(-1))

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)

    # Per-class PR-AUC
    label_names = {0: "donor", 1: "acceptor", 2: "neither"} if num_classes == 3 else {0: "positive", 1: "negative"}
    pr_aucs = {}
    for i in range(min(num_classes, 3)):
        binary = (labels == i).astype(int)
        if binary.sum() > 0 and binary.sum() < len(binary):
            pr_aucs[label_names.get(i, str(i))] = float(average_precision_score(binary, probs[:, i]))

    # Accuracy
    preds = probs.argmax(axis=1)
    accuracy = float((preds == labels).mean())

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": accuracy,
        "pr_aucs": pr_aucs,
        "macro_pr_auc": float(np.mean(list(pr_aucs.values()))) if pr_aucs else 0.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train sequence-level meta-splice model (M*-S)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["m1", "m2", "m3"], default="m1",
        help="Model variant: "
             "m1 (canonical, MANE, all modalities), "
             "m2 (alternative sites, Ensembl, all modalities), "
             "m3 (novel sites, no junction features). "
             "m2 defaults to --annotation-source=ensembl.",
    )
    parser.add_argument(
        "--annotation-source", choices=["mane", "ensembl"], default=None,
        help="Annotation source for gene definitions and splice sites. "
             "Default: mane for m1/m3, ensembl for m2. "
             "Override to mix (e.g., m1 on ensembl for M2a evaluation).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--activation", choices=["relu", "gelu", "selu"], default="gelu",
        help="Activation function (default: relu). See meta_layer/docs/architecture_search.md",
    )
    parser.add_argument("--window-size", type=int, default=5001)
    parser.add_argument("--samples-per-epoch", type=int, default=50_000)
    parser.add_argument("--max-genes", type=int, default=None,
                        help="Limit genes for quick testing")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory for disk-backed gene cache (.npz per gene). "
             "Default: <output-dir>/gene_cache/",
    )
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--base-scores-dir", type=Path, default=None,
        help="Override base model predictions directory. By default, uses "
             "OpenSpliceAI MANE precomputed predictions. Set this to use "
             "Ensembl precomputed predictions for Option A M2a evaluation.",
    )
    parser.add_argument(
        "--bigwig-cache", type=Path, default=None,
        help="Local directory with cached conservation bigWig files "
             "(avoids slow remote streaming). E.g., /runpod-volume/bigwig_cache",
    )
    parser.add_argument(
        "--use-shards", action="store_true",
        help="Pack .npz gene cache into per-chromosome HDF5 shards for "
             "faster training I/O (HDF5 slice reads instead of full .npz loads).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--device", default="cpu",
        help="Device: cpu (default, recommended), cuda, mps, or auto. "
             "Note: MPS has autograd bugs with BatchNorm1d backward in "
             "torch 2.5.x. CPU is fast enough for this model (~370K params).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ── Device ───────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Imports ──────────────────────────────────────────────────────
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceModel, MetaSpliceConfig,
    )
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        SequenceLevelDataset, ShardedSequenceLevelDataset, build_gene_cache,
    )
    from agentic_spliceai.splice_engine.meta_layer.data.shard_packing import (
        pack_gene_cache_to_shards, verify_shard_integrity,
    )
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
        DenseFeatureExtractor, DenseFeatureConfig, M3_EXCLUDE,
    )
    from agentic_spliceai.splice_engine.eval.splitting import (
        build_gene_split, gene_chromosomes_from_dataframe,
    )
    from agentic_spliceai.splice_engine.resources import (
        get_model_resources, get_genomic_registry,
    )

    # ── Resolve paths ────────────────────────────────────────────────
    # Base model resources (OpenSpliceAI) — always used for base scores
    resources = get_model_resources("openspliceai")

    # Annotation source: m2 defaults to ensembl, others to mane
    ann_src = args.annotation_source or ("ensembl" if args.mode == "m2" else "mane")
    if ann_src == "ensembl":
        ann_registry = get_genomic_registry(build="GRCh38", release="112")
        print(f"  Annotation source: Ensembl (GRCh38 release 112)")
    else:
        ann_registry = resources.get_registry()
        print(f"  Annotation source: MANE")

    # FASTA is shared across annotation sources (same genome build)
    fasta_path = str(resources.get_fasta_path())
    # Splice sites and GTF come from the annotation source
    splice_sites_path = Path(ann_registry.stash) / "splice_sites_enhanced.tsv"
    # Base scores: explicit override > Ensembl precomputed > MANE precomputed.
    # For Ensembl-only regions without precomputed scores, the model falls
    # back to a uniform 1/3 prior (Option B for M2a evaluation).
    if args.base_scores_dir:
        base_scores_dir = args.base_scores_dir
    else:
        base_scores_dir = resources.get_registry().get_base_model_eval_dir(
            "openspliceai"
        ) / "precomputed"

    print(f"  FASTA:        {fasta_path}")
    print(f"  Splice sites: {splice_sites_path}")
    print(f"  Base scores:  {base_scores_dir}")

    # ── Load gene annotations + splice sites ─────────────────────────
    import pandas as pd
    import polars as pl
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        prepare_splice_site_annotations,
    )

    splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")
    log.info("Splice sites: %d rows", len(splice_sites_df))

    # Load gene annotations for coordinates
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    gtf_path = str(ann_registry.get_gtf_path())
    gene_annotations = extract_gene_annotations(gtf_path, verbosity=0)
    log.info("Gene annotations: %d genes (%s)", gene_annotations.height, ann_src)

    # ── Gene split ───────────────────────────────────────────────────
    gene_chroms = gene_chromosomes_from_dataframe(gene_annotations)
    gene_split = build_gene_split(gene_chroms, preset="spliceai", val_fraction=0.1)
    print(gene_split.summary())

    train_genes = sorted(gene_split.train_genes)
    val_genes = sorted(gene_split.val_genes)

    if args.max_genes:
        train_genes = train_genes[:args.max_genes]
        val_genes = val_genes[:max(10, args.max_genes // 10)]
        print(f"  Limited to {len(train_genes)} train, {len(val_genes)} val genes")

    # ── Build gene caches ────────────────────────────────────────────
    exclude_channels = M3_EXCLUDE if args.mode == "m3" else set()
    feat_config = DenseFeatureConfig(
        build="GRCh38",
        exclude_channels=exclude_channels,
        bigwig_cache_dir=args.bigwig_cache,
    )
    extractor = DenseFeatureExtractor(feat_config)

    # Resolve cache directory (disk-backed to avoid OOM)
    output_dir = args.output_dir or Path(f"output/meta_layer/{args.mode}s")
    cache_dir = args.cache_dir or output_dir / "gene_cache"

    print(f"\n  Building training gene cache ({len(train_genes)} genes)...")
    print(f"    Cache dir: {cache_dir}")
    t0 = time.time()
    train_index = build_gene_cache(
        train_genes, splice_sites_df, fasta_path,
        base_scores_dir, extractor, gene_annotations,
        cache_dir=cache_dir / "train",
    )
    print(f"    Cached {len(train_index)} genes in {time.time() - t0:.1f}s")

    print(f"  Building validation gene cache ({len(val_genes)} genes)...")
    val_index = build_gene_cache(
        val_genes, splice_sites_df, fasta_path,
        base_scores_dir, extractor, gene_annotations,
        cache_dir=cache_dir / "val",
    )
    print(f"    Cached {len(val_index)} genes")

    extractor.close()

    # ── Datasets + loaders ───────────────────────────────────────────
    mm_channels = extractor.num_channels

    if args.use_shards:
        # Pack .npz → per-chromosome HDF5 shards for faster I/O
        print("  Packing training shards...")
        train_shard_dir = cache_dir / "train_shards"
        train_index_path = pack_gene_cache_to_shards(
            train_index, gene_annotations, train_shard_dir,
        )
        print("  Packing validation shards...")
        val_shard_dir = cache_dir / "val_shards"
        val_index_path = pack_gene_cache_to_shards(
            val_index, gene_annotations, val_shard_dir,
        )
        train_ds = ShardedSequenceLevelDataset(
            train_index_path,
            window_size=args.window_size,
            context_padding=400,
            samples_per_epoch=args.samples_per_epoch,
        )
        val_ds = ShardedSequenceLevelDataset(
            val_index_path,
            window_size=args.window_size,
            context_padding=400,
            samples_per_epoch=min(5000, args.samples_per_epoch // 10),
            splice_bias=0.5,
        )
    else:
        train_ds = SequenceLevelDataset(
            train_index,
            window_size=args.window_size,
            context_padding=400,
            samples_per_epoch=args.samples_per_epoch,
        )
        val_ds = SequenceLevelDataset(
            val_index,
            window_size=args.window_size,
            context_padding=400,
            samples_per_epoch=min(5000, args.samples_per_epoch // 10),
            splice_bias=0.5,
        )

    # DataLoader tuning for disk-backed .npz gene cache.
    # See docs/ml_engineering/data_pipeline/io_bottlenecks_dataloader.md
    # - num_workers > 0: parallelize I/O with GPU compute
    # - spawn context: Polars imported in parent → fork() would deadlock
    # - pin_memory: async DMA CPU→GPU transfers
    # - persistent_workers: avoid respawn overhead between epochs
    _cuda = device.type == "cuda"
    _nw = 4 if _cuda else 0
    _mp_ctx = "spawn" if _nw > 0 else None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=_nw, persistent_workers=(_nw > 0),
        pin_memory=_cuda, prefetch_factor=2 if _nw > 0 else None,
        multiprocessing_context=_mp_ctx,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(2, _nw), persistent_workers=(_nw > 0),
        pin_memory=_cuda, prefetch_factor=2 if _nw > 0 else None,
        multiprocessing_context=_mp_ctx,
    )

    # ── Model ────────────────────────────────────────────────────────
    if args.mode in ("m1", "m2"):
        variant = "M1-S" if args.mode == "m1" else "M2-S"
        cfg = MetaSpliceConfig(
            variant=variant, hidden_dim=args.hidden_dim,
            mm_channels=mm_channels, num_classes=3,
            activation=args.activation,
        )
    else:
        cfg = MetaSpliceConfig(
            variant="M3-S", hidden_dim=args.hidden_dim,
            mm_channels=mm_channels, num_classes=2,
            activation=args.activation,
        )

    model = MetaSpliceModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n  Model: {cfg.variant}, {n_params:,} params, H={cfg.hidden_dim}")
    print(f"  mm_channels={mm_channels}, merge_base_scores={cfg.merge_base_scores}")
    print(f"  Receptive field: {model.receptive_field} bp")

    if args.checkpoint and args.checkpoint.exists():
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"  Loaded checkpoint: {args.checkpoint}")

    # ── Loss + optimizer ─────────────────────────────────────────────
    # Class weights: inverse frequency (~0.3% donor, ~0.3% acceptor, ~99.4% neither)
    if cfg.num_classes == 3:
        weights = torch.tensor([166.0, 166.0, 1.0], device=device)
    else:
        weights = torch.tensor([1.0, 1.0], device=device)

    criterion = FocalLoss(alpha=weights, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Output directory ─────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Training {cfg.variant} — {args.epochs} epochs")
    print(f"  Batch: {args.batch_size}, Accum: {args.accumulation_steps}, "
          f"Effective: {args.batch_size * args.accumulation_steps}")
    print(f"  LR: {args.lr}, Patience: {args.patience}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    best_metric = 0.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t_start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            accumulation_steps=args.accumulation_steps,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, cfg.num_classes)
        scheduler.step()

        elapsed = time.time() - t_start
        lr = optimizer.param_groups[0]["lr"]
        macro = val_metrics["macro_pr_auc"]

        print(
            f"  Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_pr_auc={macro:.4f} | "
            f"lr={lr:.2e} | "
            f"{elapsed:.0f}s"
        )

        # Save best
        if macro > best_metric:
            best_metric = macro
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best.pt")
            with open(output_dir / "best_metrics.json", "w") as f:
                json.dump({"epoch": epoch, **val_metrics}, f, indent=2)
            print(f"    -> New best PR-AUC: {macro:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"    Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Save final ───────────────────────────────────────────────────
    torch.save(model.state_dict(), output_dir / "final.pt")
    torch.save(cfg, output_dir / "config.pt")

    print(f"\n{'='*70}")
    print(f"Training complete. Best val PR-AUC: {best_metric:.4f}")
    print(f"  Best model:  {output_dir / 'best.pt'}")
    print(f"  Final model: {output_dir / 'final.pt'}")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
