"""
Dense per-nucleotide splice site classifier.

Produces 3-class output: P(no_splice), P(acceptor), P(donor) at every
position in a sequence, matching the SpliceAI/OpenSpliceAI output format.

Operates on **frozen** foundation model embeddings — the head is lightweight
(~250K params for dilated_cnn) and trains in minutes even on CPU.

Architecture options:
  - ``dilated_cnn`` (default): Dilated residual Conv1D stack.  Captures
    local splice motif patterns on top of contextual embeddings.
  - ``mlp``: Position-wise MLP (no cross-position interaction).
  - ``linear``: Single linear layer (baseline).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundation_models.classifiers.losses import FocalLoss, compute_class_weights

logger = logging.getLogger(__name__)

# Label convention (matches OpenSpliceAI)
LABEL_NO_SPLICE = 0
LABEL_ACCEPTOR = 1
LABEL_DONOR = 2
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DilatedResidualBlock(nn.Module):
    """Single dilated residual block (inspired by OpenSpliceAI's ResidualUnit).

    BatchNorm → ReLU → DilatedConv → BatchNorm → ReLU → DilatedConv + skip.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 11,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2  # same-length output

        self.block = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size,
                      dilation=dilation, padding=padding),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual connection: x + block(x)."""
        return x + self.block(x)


# ---------------------------------------------------------------------------
# SpliceClassifier
# ---------------------------------------------------------------------------

class SpliceClassifier(nn.Module):
    """Dense per-nucleotide splice site classifier.

    Sits on top of frozen foundation model embeddings and predicts 3-class
    splice site probabilities at every position.

    Args:
        input_dim: Embedding dimension from the foundation model.
        hidden_dim: Number of channels in the prediction head (default: 128).
        architecture: Head architecture — ``"dilated_cnn"`` (default),
            ``"mlp"``, or ``"linear"``.
        num_blocks: Number of dilated residual blocks (dilated_cnn only).
        kernel_size: Convolution kernel size (dilated_cnn only).
        dilations: List of dilation rates per block (dilated_cnn only).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        architecture: Literal["dilated_cnn", "mlp", "linear"] = "dilated_cnn",
        num_blocks: int = 3,
        kernel_size: int = 11,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.architecture = architecture
        self.num_blocks = num_blocks
        self.kernel_size = kernel_size
        self.dilations = dilations or [1, 4, 16]
        self.dropout = dropout

        if architecture == "dilated_cnn":
            layers: list[nn.Module] = [
                # 1×1 projection: hidden_dim → channels
                nn.Conv1d(input_dim, hidden_dim, 1),
            ]
            for i in range(num_blocks):
                d = self.dilations[i % len(self.dilations)]
                layers.append(
                    DilatedResidualBlock(hidden_dim, kernel_size, d, dropout)
                )
            layers.append(nn.Conv1d(hidden_dim, NUM_CLASSES, 1))
            self.head = nn.Sequential(*layers)

        elif architecture == "mlp":
            self.head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, NUM_CLASSES),
            )

        elif architecture == "linear":
            self.head = nn.Linear(input_dim, NUM_CLASSES)

        else:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                f"Choose from: dilated_cnn, mlp, linear"
            )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SpliceClassifier(%s): input_dim=%d, hidden_dim=%d, %d params",
            architecture, input_dim, hidden_dim, n_params,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: ``[batch, seq_len, input_dim]`` or
                ``[seq_len, input_dim]``.

        Returns:
            Logits ``[batch, 3, seq_len]`` (before softmax).
        """
        single = embeddings.dim() == 2
        if single:
            embeddings = embeddings.unsqueeze(0)  # [1, seq_len, input_dim]

        if self.architecture == "dilated_cnn":
            # Conv1d expects [batch, channels, seq_len]
            x = embeddings.transpose(1, 2)  # [batch, input_dim, seq_len]
            logits = self.head(x)  # [batch, 3, seq_len]
        elif self.architecture == "mlp":
            # Per-position MLP: [batch, seq_len, input_dim] → [batch, seq_len, 3]
            logits = self.head(embeddings)
            logits = logits.transpose(1, 2)  # [batch, 3, seq_len]
        else:
            # Linear: same as MLP
            logits = self.head(embeddings)
            logits = logits.transpose(1, 2)

        if single:
            logits = logits.squeeze(0)  # [3, seq_len]

        return logits

    def predict(self, embeddings: torch.Tensor) -> Dict[str, np.ndarray]:
        """Predict per-nucleotide splice site probabilities.

        Args:
            embeddings: ``[seq_len, input_dim]`` or
                ``[batch, seq_len, input_dim]``.

        Returns:
            Dict with keys ``donor_prob``, ``acceptor_prob``, ``neither_prob``,
            each a numpy array of shape ``[seq_len]`` or ``[batch, seq_len]``.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(embeddings)
            probs = F.softmax(logits, dim=-2 if logits.dim() == 2 else 1)

        # probs shape: [3, seq_len] or [batch, 3, seq_len]
        if probs.dim() == 2:
            probs_np = probs.cpu().float().numpy()
            return {
                "neither_prob": probs_np[LABEL_NO_SPLICE],
                "acceptor_prob": probs_np[LABEL_ACCEPTOR],
                "donor_prob": probs_np[LABEL_DONOR],
            }
        else:
            probs_np = probs.cpu().float().numpy()
            return {
                "neither_prob": probs_np[:, LABEL_NO_SPLICE],
                "acceptor_prob": probs_np[:, LABEL_ACCEPTOR],
                "donor_prob": probs_np[:, LABEL_DONOR],
            }

    def fit(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cuda",
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        patience: int = 20,
        lr_schedule: bool = True,
        monitor_metric: Literal["auprc", "auroc"] = "auprc",
        focal_gamma: float = 2.0,
    ) -> Dict:
        """Train splice site classifier.

        Args:
            train_embeddings: ``[N, seq_len, input_dim]``.
            train_labels: ``[N, seq_len]`` — integer labels (0/1/2).
            val_embeddings: Optional validation embeddings.
            val_labels: Optional validation labels.
            epochs: Maximum training epochs.
            batch_size: Batch size.
            lr: Learning rate.
            weight_decay: Weight decay for AdamW.
            device: Training device.
            verbose: Print progress.
            checkpoint_dir: Directory for best model checkpoint.
            patience: Early stopping patience (0 = disabled).
            lr_schedule: Use ReduceLROnPlateau.
            monitor_metric: Metric for early stopping.
            focal_gamma: Focal loss gamma parameter.

        Returns:
            History dict with train_loss, val_loss, val_auroc, val_auprc,
            best_epoch, stopped_early.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score
        from torch.utils.data import TensorDataset, DataLoader

        self.to(device)
        train_embeddings = train_embeddings.to(device)
        train_labels = train_labels.long().to(device)

        if val_embeddings is not None:
            val_embeddings = val_embeddings.to(device)
            val_labels = val_labels.long().to(device)

        # Data loader
        train_dataset = TensorDataset(train_embeddings, train_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
        )

        # Loss: focal loss with class weights
        class_weights = compute_class_weights(
            train_labels.cpu().numpy(), num_classes=NUM_CLASSES,
        ).to(device)
        criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)

        if verbose:
            logger.info(
                "Class weights: neither=%.1f, acceptor=%.1f, donor=%.1f",
                class_weights[0].item(), class_weights[1].item(),
                class_weights[2].item(),
            )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = None
        if lr_schedule and val_embeddings is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )

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
            self.train()
            train_loss = 0.0
            for batch_emb, batch_lbl in train_loader:
                logits = self.forward(batch_emb)  # [batch, 3, seq_len]
                loss = criterion(logits, batch_lbl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # -- Validate --
            if val_embeddings is not None:
                val_metrics = self._evaluate(
                    val_embeddings, val_labels, criterion,
                )
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

                current_metric = val_metrics[monitor_metric]
                if current_metric > best_val_metric:
                    best_val_metric = current_metric
                    best_epoch = epoch
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.state_dict().items()
                    }
                    if checkpoint_dir is not None:
                        self._save_checkpoint(
                            checkpoint_dir, epoch, val_metrics,
                        )
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if scheduler is not None:
                    scheduler.step(current_metric)

                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch + 1} "
                            f"(best {monitor_metric.upper()}: "
                            f"{best_val_metric:.4f} at epoch {best_epoch + 1})"
                        )
                    break
            else:
                if verbose:
                    print(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={train_loss:.4f}"
                    )

        # Restore best model
        stopped_early = (
            epochs_without_improvement >= patience if patience > 0 else False
        )
        if val_embeddings is not None and best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(
                    f"Restored best model from epoch {best_epoch + 1} "
                    f"({monitor_metric.upper()}: {best_val_metric:.4f})"
                )

        history["best_epoch"] = best_epoch
        history["best_val_metric"] = best_val_metric
        history["best_metric_name"] = monitor_metric
        history["stopped_early"] = stopped_early
        return history

    @torch.no_grad()
    def _evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        """Evaluate on a dataset, returning loss + per-class metrics.

        Computes one-vs-rest AUROC and AUPRC averaged over acceptor + donor
        classes only (ignoring the "neither" class which dominates).
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        self.eval()
        logits = self.forward(embeddings)  # [batch, 3, seq_len]
        loss = criterion(logits, labels).item()

        probs = F.softmax(logits, dim=1).cpu().numpy()  # [batch, 3, seq_len]
        labels_np = labels.cpu().numpy()  # [batch, seq_len]

        # Flatten to per-position
        probs_flat = probs.reshape(3, -1).T  # [N_total, 3]
        labels_flat = labels_np.flatten()  # [N_total]

        # One-vs-rest for splice classes (1=acceptor, 2=donor)
        aurocs, auprcs = [], []
        for cls in [LABEL_ACCEPTOR, LABEL_DONOR]:
            y_true_cls = (labels_flat == cls).astype(np.int32)
            y_score_cls = probs_flat[:, cls]

            if y_true_cls.sum() == 0 or y_true_cls.sum() == len(y_true_cls):
                continue  # Skip degenerate cases

            try:
                aurocs.append(roc_auc_score(y_true_cls, y_score_cls))
                auprcs.append(average_precision_score(y_true_cls, y_score_cls))
            except ValueError:
                continue

        auroc = float(np.mean(aurocs)) if aurocs else 0.0
        auprc = float(np.mean(auprcs)) if auprcs else 0.0

        return {"loss": loss, "auroc": auroc, "auprc": auprc}

    def _save_checkpoint(
        self, checkpoint_dir: str, epoch: int, metrics: dict,
    ) -> Path:
        """Save model checkpoint to disk."""
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = ckpt_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "architecture": self.architecture,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_blocks": self.num_blocks,
                "kernel_size": self.kernel_size,
                "dilations": self.dilations,
                "dropout": self.dropout,
                "epoch": epoch,
                "metrics": metrics,
            },
            ckpt_path,
        )

        # Also save metrics as JSON
        metrics_path = ckpt_dir / "eval_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {**metrics, "epoch": epoch, "architecture": self.architecture},
                f, indent=2,
            )

        return ckpt_path

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SpliceClassifier":
        """Load a SpliceClassifier from a checkpoint.

        Args:
            path: Path to ``best_model.pt``.
            device: Device to load onto.

        Returns:
            Restored SpliceClassifier.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)

        model = cls(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            architecture=ckpt["architecture"],
            num_blocks=ckpt.get("num_blocks", 3),
            kernel_size=ckpt.get("kernel_size", 11),
            dilations=ckpt.get("dilations", [1, 4, 16]),
            dropout=ckpt.get("dropout", 0.1),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info(
            "Loaded SpliceClassifier from %s (epoch %d, %s)",
            path, ckpt.get("epoch", -1), ckpt.get("architecture", "?"),
        )
        return model
