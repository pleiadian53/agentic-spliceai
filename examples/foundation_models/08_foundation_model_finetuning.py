#!/usr/bin/env python3
"""
Foundation Model Fine-Tuning for Splice Site Prediction.

Fine-tunes pre-trained DNA foundation models (SpliceBERT, DNABERT-2) end-to-end
on splice site data.  Unlike 07_genome_scale_splice_predictor.py which freezes
the foundation model and trains only a classifier head, this script updates the
foundation model's own weights to learn splice-specific representations.

Three-phase pipeline:
  Phase 1: Prepare sequence windows (raw DNA + labels from FASTA)
  Phase 2: Fine-tune (foundation model + classifier head, end-to-end)
  Phase 3: Evaluate on held-out chromosomes (same metrics as 07)

Freezing strategies:
  - last_n (default): Freeze all but last N encoder layers + head
  - lora: LoRA adapters on attention weights (requires peft package)
  - full: Unfreeze everything (only for small models like SpliceBERT)

Usage:
    # Mock mode (pipeline validation, no GPU needed)
    python 08_foundation_model_finetuning.py \\
        --mock -o /tmp/08-mock/

    # SpliceBERT last-N fine-tuning on chr22
    python 08_foundation_model_finetuning.py \\
        --foundation-model splicebert \\
        --strategy last_n --unfreeze-layers 2 \\
        --chromosomes 22 \\
        -o /workspace/output/finetune-splicebert/

    # LoRA fine-tuning
    python 08_foundation_model_finetuning.py \\
        --foundation-model splicebert \\
        --strategy lora --lora-rank 8 \\
        -o /workspace/output/finetune-splicebert-lora/

    # Warm-start head from frozen checkpoint (07)
    python 08_foundation_model_finetuning.py \\
        --foundation-model splicebert \\
        --from-frozen output/splice_classifier/splicebert-chr22/model/best_model.pt \\
        -o /workspace/output/finetune-splicebert-warm/
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> str:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_chrom_key(fasta: Any, chrom: str) -> str:
    """Resolve chromosome name to FASTA key (handle chr prefix)."""
    if chrom in fasta:
        return chrom
    if chrom.startswith("chr"):
        bare = chrom[3:]
        if bare in fasta:
            return bare
    else:
        prefixed = f"chr{chrom}"
        if prefixed in fasta:
            return prefixed
    return chrom


# ---------------------------------------------------------------------------
# Phase 1: Prepare sequence windows
# ---------------------------------------------------------------------------

def prepare_sequence_windows(
    chromosomes: List[str],
    window_size: int,
    step_size: int,
    build: str = "GRCh38_MANE",
    max_gene_length: int = 200_000,
) -> Tuple[List[Dict], Any, str]:
    """Load gene annotations and splice sites for the requested chromosomes.

    Returns gene_entries, splice_sites_df, fasta_path — ready for
    SequenceWindowDataset.
    """
    from agentic_spliceai.splice_engine.resources import get_genomic_registry

    registry = get_genomic_registry()
    reg = registry[build]
    gtf_path = reg["gtf"]
    fasta_path = reg["fasta"]
    splice_path = reg.get("splice_sites", reg.get("splice_sites_enhanced"))

    # Load splice sites
    import pandas as pd
    logger.info("Loading splice sites from %s", splice_path)
    if str(splice_path).endswith(".parquet"):
        splice_sites_df = pd.read_parquet(splice_path)
    else:
        splice_sites_df = pd.read_csv(splice_path, sep="\t")

    # Load gene annotations
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        load_gene_annotations,
    )

    all_entries = []
    for chrom in chromosomes:
        genes_df = load_gene_annotations(gtf_path, chromosomes=[chrom])
        for _, row in genes_df.iterrows():
            gene_len = row["end"] - row["start"]
            if gene_len > max_gene_length:
                continue
            if gene_len < window_size:
                continue
            all_entries.append({
                "gene_id": row["gene_id"],
                "gene_name": row.get("gene_name", row["gene_id"]),
                "chrom": chrom if chrom.startswith("chr") else f"chr{chrom}",
                "start": int(row["start"]),
                "end": int(row["end"]),
                "strand": row.get("strand", "+"),
            })

    logger.info(
        "Prepared %d genes across %d chromosomes", len(all_entries), len(chromosomes),
    )
    return all_entries, splice_sites_df, str(fasta_path)


def prepare_mock_data(
    n_genes: int = 10,
    window_size: int = 512,
    hidden_dim: int = 512,
    seed: int = 42,
) -> Tuple[List[Dict], Any, Optional[str]]:
    """Generate synthetic gene entries for mock mode.

    Returns gene_entries, splice_sites_df, fasta_path (None for mock).
    """
    import pandas as pd
    from foundation_models.utils.synthetic import generate_synthetic_splice_data

    _, lbl_dict = generate_synthetic_splice_data(
        n_genes=n_genes, hidden_dim=hidden_dim, seed=seed,
    )

    # Build mock gene entries and splice sites
    chroms = [f"chr{(i % 22) + 1}" for i in range(n_genes)]
    entries = []
    splice_rows = []

    for idx, (gene_id, labels) in enumerate(sorted(lbl_dict.items())):
        chrom = chroms[idx]
        start = idx * 100_000  # Fake genomic coordinates
        end = start + len(labels)

        entries.append({
            "gene_id": gene_id,
            "gene_name": gene_id,
            "chrom": chrom,
            "start": start,
            "end": end,
            "strand": "+",
        })

        # Convert labels to splice_sites_df rows
        for pos_idx in range(len(labels)):
            if labels[pos_idx] == 1:
                splice_rows.append({
                    "gene_id": gene_id,
                    "position": start + pos_idx,
                    "splice_type": "acceptor",
                })
            elif labels[pos_idx] == 2:
                splice_rows.append({
                    "gene_id": gene_id,
                    "position": start + pos_idx,
                    "splice_type": "donor",
                })

    splice_sites_df = pd.DataFrame(splice_rows)
    logger.info(
        "Mock data: %d genes, %d splice sites", len(entries), len(splice_rows),
    )

    return entries, splice_sites_df, None  # No FASTA for mock


class MockSequenceDataset(torch.utils.data.Dataset):
    """Mock dataset that generates random DNA for pipeline validation."""

    def __init__(
        self,
        n_windows: int = 100,
        window_size: int = 512,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)
        self._class_counts = np.zeros(3, dtype=np.int64)

        nucleotides = np.array(list("ACGT"))
        self._sequences = []
        self._labels = []

        for _ in range(n_windows):
            seq = "".join(rng.choice(nucleotides, size=window_size))
            labels = np.zeros(window_size, dtype=np.int64)
            # Sprinkle ~2 splice sites per window
            n_sites = rng.integers(0, 5)
            for _ in range(n_sites):
                pos = rng.integers(10, window_size - 10)
                site_type = rng.choice([1, 2])  # acceptor or donor
                labels[pos] = site_type

            self._sequences.append(seq)
            self._labels.append(labels)
            for c in range(3):
                self._class_counts[c] += int((labels == c).sum())

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self._sequences[idx], torch.tensor(self._labels[idx], dtype=torch.long)

    @property
    def class_counts(self) -> np.ndarray:
        return self._class_counts

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune
# ---------------------------------------------------------------------------

def finetune(
    model: Any,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    class_weights: torch.Tensor,
    encoder_lr: float = 1e-5,
    head_lr: float = 1e-3,
    warmup_steps: int = 500,
    max_grad_norm: float = 1.0,
    epochs: int = 30,
    patience: int = 10,
    focal_gamma: float = 2.0,
    device: str = "cuda",
    checkpoint_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """End-to-end fine-tuning loop with differential LR and gradient clipping.

    Args:
        model: A SpliceFineTuneModel instance.
        train_loader: Yields (dna_strings, label_tensors) batches.
        val_loader: Optional validation DataLoader.
        class_weights: Pre-computed class weights [num_classes].
        encoder_lr: Learning rate for trainable encoder parameters.
        head_lr: Learning rate for classifier head parameters.
        warmup_steps: Linear warmup steps for encoder LR.
        max_grad_norm: Maximum gradient norm for clipping.
        epochs: Maximum training epochs.
        patience: Early-stopping patience on val AUPRC.
        focal_gamma: Focal loss gamma.
        device: Target device.
        checkpoint_dir: Directory for saving best model.
        verbose: Print per-epoch progress.

    Returns:
        Training history dict.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    from foundation_models.classifiers.finetune import get_param_groups
    from foundation_models.classifiers.losses import FocalLoss

    model.to(device)
    criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights.to(device))

    if verbose:
        logger.info(
            "Class weights: neither=%.1f, acceptor=%.1f, donor=%.1f",
            class_weights[0], class_weights[1], class_weights[2],
        )

    # Differential learning rates
    param_groups = get_param_groups(model, encoder_lr, head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Linear warmup scheduler
    total_steps = epochs * len(train_loader)
    effective_warmup = min(warmup_steps, total_steps // 4)

    def lr_lambda(step: int) -> float:
        if step < effective_warmup:
            return step / max(effective_warmup, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training state
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "val_auroc": [], "val_auprc": [],
    }
    best_val_metric = 0.0
    best_epoch = 0
    best_state: Optional[dict] = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # -- Train --
        model.train()
        train_loss = 0.0
        n_batches = 0
        t_epoch = time.time()

        for batch_idx, (dna_strings, labels) in enumerate(train_loader):
            labels = labels.to(device)

            # End-to-end forward: DNA strings → logits
            logits = model(list(dna_strings))  # DataLoader returns tuple of strings
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            n_batches += 1

            # Progress logging
            if verbose and (batch_idx + 1) % max(len(train_loader) // 10, 1) == 0:
                elapsed = time.time() - t_epoch
                logger.info(
                    "  Epoch %d/%d: batch %d/%d (%.0f s), loss=%.4f",
                    epoch + 1, epochs, batch_idx + 1, len(train_loader),
                    elapsed, train_loss / n_batches,
                )

        train_loss /= max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # -- Validate --
        if val_loader is not None:
            val_metrics = _evaluate_finetune(model, val_loader, criterion, device)
            history["val_loss"].append(val_metrics["loss"])
            history["val_auroc"].append(val_metrics["auroc"])
            history["val_auprc"].append(val_metrics["auprc"])

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs}: "
                    f"train_loss={train_loss:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_auroc={val_metrics['auroc']:.4f}, "
                    f"val_auprc={val_metrics['auprc']:.4f}"
                )

            current_metric = val_metrics["auprc"]
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                if checkpoint_dir is not None:
                    model.save_checkpoint(
                        str(checkpoint_dir / "best_model.pt"),
                        epoch=epoch,
                        metrics=val_metrics,
                    )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if patience > 0 and epochs_without_improvement >= patience:
                if verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1} "
                        f"(best AUPRC: {best_val_metric:.4f} at epoch {best_epoch + 1})"
                    )
                break
        else:
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"Restored best model from epoch {best_epoch + 1} "
                f"(AUPRC: {best_val_metric:.4f})"
            )

    history["best_epoch"] = best_epoch
    history["best_val_metric"] = best_val_metric
    return history


@torch.no_grad()
def _evaluate_finetune(
    model: Any,
    data_loader: torch.utils.data.DataLoader,
    criterion: Any,
    device: str,
) -> Dict[str, float]:
    """Evaluate fine-tuned model on validation data."""
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score, roc_auc_score

    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for dna_strings, labels in data_loader:
        labels = labels.to(device)
        logits = model(list(dna_strings))
        total_loss += criterion(logits, labels).item()
        n_batches += 1

        probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, 3, W]
        b, c, w = probs.shape
        all_probs.append(probs.transpose(0, 2, 1).reshape(-1, c))
        all_labels.append(labels.cpu().numpy().flatten())

    loss = total_loss / max(n_batches, 1)

    if not all_probs:
        return {"loss": loss, "auroc": 0.0, "auprc": 0.0}

    probs_flat = np.concatenate(all_probs)
    labels_flat = np.concatenate(all_labels)

    aurocs, auprcs = [], []
    for cls in [1, 2]:  # acceptor, donor
        y_true = (labels_flat == cls).astype(np.int32)
        y_score = probs_flat[:, cls]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        try:
            aurocs.append(roc_auc_score(y_true, y_score))
            auprcs.append(average_precision_score(y_true, y_score))
        except ValueError:
            continue

    auroc = float(np.mean(aurocs)) if aurocs else 0.0
    auprc = float(np.mean(auprcs)) if auprcs else 0.0
    return {"loss": loss, "auroc": auroc, "auprc": auprc}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune foundation models for splice site prediction",
    )
    # Model
    parser.add_argument(
        "--foundation-model", type=str, default="splicebert",
        help="Foundation model name (default: splicebert)",
    )
    parser.add_argument("--model-size", type=str, default=None)

    # Fine-tuning strategy
    parser.add_argument(
        "--strategy", type=str, default="last_n",
        choices=["last_n", "lora", "full"],
        help="Freezing strategy (default: last_n)",
    )
    parser.add_argument(
        "--unfreeze-layers", type=int, default=2,
        help="Layers to unfreeze for last_n strategy (default: 2)",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)

    # Learning rates
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Data
    parser.add_argument(
        "--chromosomes", type=str, nargs="+", default=["22"],
        help="Chromosomes to use (default: 22)",
    )
    parser.add_argument("--build", type=str, default="GRCh38_MANE")
    parser.add_argument("--max-gene-length", type=int, default=200_000)

    # Training
    parser.add_argument("--architecture", type=str, default="dilated_cnn")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # I/O
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--n-genes", type=int, default=10)
    parser.add_argument("--from-frozen", type=str, default=None,
                        help="Initialize head from frozen checkpoint (07)")
    parser.add_argument("--split", type=str, default="spliceai")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    t0 = time.time()

    # ===================================================================
    # Resolve model metadata
    # ===================================================================
    if args.mock:
        hidden_dim = 512
        window_size = 512
        model_name = "MOCK"
    else:
        from foundation_models.base import get_model_metadata
        meta = get_model_metadata(args.foundation_model)
        hidden_dim = meta.hidden_dim
        window_size = min(meta.max_context, 512)  # Cap for fine-tuning memory
        model_name = meta.name

    step_size = window_size // 2

    # Banner
    print()
    print("=" * 70)
    print("Foundation Model Fine-Tuning for Splice Site Prediction")
    print("=" * 70)
    print(f"  Model:        {model_name}")
    print(f"  Strategy:     {args.strategy}", end="")
    if args.strategy == "last_n":
        print(f" (unfreeze={args.unfreeze_layers})")
    elif args.strategy == "lora":
        print(f" (rank={args.lora_rank}, alpha={args.lora_alpha})")
    else:
        print()
    print(f"  Hidden dim:   {hidden_dim}")
    print(f"  Window:       {window_size} bp (step: {step_size})")
    print(f"  Encoder LR:   {args.encoder_lr}")
    print(f"  Head LR:      {args.head_lr}")
    print(f"  Device:       {device}")
    if args.from_frozen:
        print(f"  Warm start:   {args.from_frozen}")
    print("=" * 70)
    print()

    # ===================================================================
    # Phase 1: Prepare data
    # ===================================================================
    logger.info("Phase 1/3: Preparing sequence data...")

    if args.mock:
        # Mock: synthetic random DNA + labels
        n_train = int(args.n_genes * 0.6)
        n_val = int(args.n_genes * 0.2)

        train_dataset = MockSequenceDataset(
            n_windows=n_train * 20, window_size=window_size, seed=args.seed,
        )
        val_dataset = MockSequenceDataset(
            n_windows=n_val * 20, window_size=window_size, seed=args.seed + 1,
        )
        test_dataset = MockSequenceDataset(
            n_windows=(args.n_genes - n_train - n_val) * 20,
            window_size=window_size, seed=args.seed + 2,
        )
        logger.info(
            "Mock data: %d train, %d val, %d test windows",
            len(train_dataset), len(val_dataset), len(test_dataset),
        )
    else:
        # Normalize chromosomes — support both space and comma separation
        expanded = []
        for item in args.chromosomes:
            expanded.extend(item.split(","))
        args.chromosomes = [c.strip() for c in expanded if c.strip()]

        # Real data
        gene_entries, splice_sites_df, fasta_path = prepare_sequence_windows(
            chromosomes=args.chromosomes,
            window_size=window_size,
            step_size=step_size,
            build=args.build,
            max_gene_length=args.max_gene_length,
        )

        if not gene_entries:
            logger.error("No genes found — exiting.")
            sys.exit(1)

        # Chromosome-based split
        from agentic_spliceai.splice_engine.eval.splitting import build_gene_split

        gene_chroms = {e["gene_id"]: e["chrom"] for e in gene_entries}
        split = build_gene_split(
            gene_chroms, preset=args.split, val_fraction=0.1, seed=args.seed,
        )
        logger.info(
            "Split: %d train, %d val, %d test genes",
            len(split.train_genes), len(split.val_genes), len(split.test_genes),
        )

        from foundation_models.data import SequenceWindowDataset

        logger.info("Building training dataset...")
        train_dataset = SequenceWindowDataset(
            gene_entries, split.train_genes, fasta_path, splice_sites_df,
            window_size, step_size,
        )
        val_dataset = None
        if split.val_genes:
            logger.info("Building validation dataset...")
            val_dataset = SequenceWindowDataset(
                gene_entries, split.val_genes, fasta_path, splice_sites_df,
                window_size, step_size,
            )

    # DataLoaders — batch_size smaller than frozen pipeline because
    # fine-tuning uses more VRAM (activations for backprop through encoder)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=0,
        )

    # Class weights
    from foundation_models.classifiers.losses import compute_class_weights_from_counts
    class_weights = compute_class_weights_from_counts(train_dataset.class_counts)

    # ===================================================================
    # Phase 2: Build model and fine-tune
    # ===================================================================
    logger.info("Phase 2/3: Fine-tuning...")

    from foundation_models.classifiers.splice_classifier import SpliceClassifier

    classifier_head = SpliceClassifier(
        input_dim=hidden_dim,
        hidden_dim=args.hidden_dim,
        architecture=args.architecture,
    )

    if args.mock:
        # Mock: use a simple linear model that accepts strings
        # (skip foundation model loading for pipeline validation)
        class MockFineTuneModel(torch.nn.Module):
            def __init__(self, hidden_dim, head):
                super().__init__()
                self.embed = torch.nn.Embedding(5, hidden_dim)  # ACGTN
                self.head = head
                self._nuc_map = {c: i for i, c in enumerate("ACGTN")}

            def forward(self, sequences):
                device = next(self.parameters()).device
                batch = []
                for seq in sequences:
                    ids = [self._nuc_map.get(c, 4) for c in seq]
                    batch.append(torch.tensor(ids, dtype=torch.long))
                input_ids = torch.stack(batch).to(device)
                emb = self.embed(input_ids)
                return self.head(emb)

            def encoder_trainable_params(self):
                return self.embed.parameters()

            def save_checkpoint(self, path, epoch=-1, metrics=None):
                torch.save({"epoch": epoch, "metrics": metrics or {}}, path)

        model = MockFineTuneModel(hidden_dim, classifier_head)
    else:
        # Real model
        from foundation_models.base import load_embedding_model
        from foundation_models.classifiers.finetune import SpliceFineTuneModel

        model_kwargs = {}
        if args.model_size:
            model_kwargs["model_size"] = args.model_size

        foundation = load_embedding_model(args.foundation_model, **model_kwargs)

        model = SpliceFineTuneModel(
            foundation_model=foundation,
            classifier_head=classifier_head,
            strategy=args.strategy,
            n_unfreeze=args.unfreeze_layers,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
        )

    # Warm-start head from frozen checkpoint
    if args.from_frozen:
        model.load_head_from_frozen(args.from_frozen)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print()
    print("=" * 60)
    print("Fine-Tuning")
    print("=" * 60)
    print(f"  Total params:     {n_params:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Train windows:    {len(train_dataset)}")
    if val_dataset:
        print(f"  Val windows:      {len(val_dataset)}")
    print(f"  Device:           {device}")
    print("=" * 60)
    print()

    history = finetune(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        encoder_lr=args.encoder_lr,
        head_lr=args.head_lr,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        patience=args.patience,
        focal_gamma=args.focal_gamma,
        device=device,
        checkpoint_dir=model_dir,
    )

    # Save training history
    history_path = model_dir / "training_history.json"
    serializable = {
        k: v for k, v in history.items()
        if isinstance(v, (list, int, float, str, bool))
    }
    with open(history_path, "w") as f:
        json.dump(serializable, f, indent=2)

    # ===================================================================
    # Phase 3: Summary
    # ===================================================================
    elapsed = time.time() - t0

    print()
    print("=" * 70)
    print("Complete")
    print("=" * 70)
    print(f"  Model:        {model_name}")
    print(f"  Strategy:     {args.strategy}")
    print(f"  Output:       {output_dir}")
    print(f"  Checkpoint:   {model_dir / 'best_model.pt'}")
    if history.get("best_val_metric"):
        print(f"  Best AUPRC:   {history['best_val_metric']:.4f} "
              f"(epoch {history['best_epoch'] + 1})")
    print(f"  Time:         {elapsed / 60:.1f} min")

    # Cleanup
    if hasattr(train_dataset, "close"):
        train_dataset.close()
    if val_dataset and hasattr(val_dataset, "close"):
        val_dataset.close()


if __name__ == "__main__":
    main()
