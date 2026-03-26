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
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from foundation_models.classifiers.losses import (
    ECELoss,
    FocalLoss,
    compute_class_weights,
    compute_class_weights_from_counts,
)

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
        # Temperature for post-hoc calibration (set by calibrate())
        self.temperature: Optional[nn.Parameter] = None

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
            # Apply temperature scaling if calibrated
            if self.temperature is not None:
                temp = torch.clamp(
                    self.temperature, min=0.05, max=5.0,
                ).to(logits.device)
                if logits.dim() == 2:
                    # [3, seq_len] / [3, 1]
                    logits = logits / temp.unsqueeze(1)
                else:
                    # [batch, 3, seq_len] / [1, 3, 1]
                    logits = logits / temp.unsqueeze(0).unsqueeze(2)
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

    def fit_streaming(
        self,
        train_loader: "torch.utils.data.DataLoader",
        val_loader: Optional["torch.utils.data.DataLoader"] = None,
        class_weights: Optional[torch.Tensor] = None,
        epochs: int = 100,
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
        """Train from DataLoaders — constant-memory streaming.

        Same training logic as :meth:`fit` but reads batches from DataLoaders
        instead of pre-loaded tensors, keeping peak memory proportional to
        batch size rather than dataset size.

        Args:
            train_loader: Yields ``(embeddings, labels)`` batches.
            val_loader: Optional validation DataLoader.
            class_weights: Pre-computed class weights ``[num_classes]``.  Use
                :func:`compute_class_weights_from_counts` to derive these from
                :attr:`HDF5WindowDataset.class_counts`.
            epochs: Maximum training epochs.
            lr: Learning rate.
            weight_decay: AdamW weight decay.
            device: Target device.
            verbose: Print per-epoch progress.
            checkpoint_dir: Directory for saving best model checkpoint.
            patience: Early-stopping patience (0 = disabled).
            lr_schedule: Use ReduceLROnPlateau scheduler.
            monitor_metric: Metric to monitor for early stopping.
            focal_gamma: Focal loss gamma.

        Returns:
            Training history dict (same schema as :meth:`fit`).
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        self.to(device)

        # Loss
        if class_weights is not None:
            class_weights = class_weights.to(device)
        criterion = FocalLoss(gamma=focal_gamma, alpha=class_weights)

        if verbose and class_weights is not None:
            logger.info(
                "Class weights: neither=%.1f, acceptor=%.1f, donor=%.1f",
                class_weights[0].item(),
                class_weights[1].item(),
                class_weights[2].item(),
            )

        # Optimizer + scheduler
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )
        scheduler = None
        if lr_schedule and val_loader is not None:
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

        total_batches = len(train_loader)
        # Log every ~10% of batches, at least every 100 batches
        log_interval = max(1, min(total_batches // 10, 100))

        for epoch in range(epochs):
            # -- Train --
            self.train()
            train_loss = 0.0
            n_batches = 0
            epoch_t0 = time.time()
            for batch_emb, batch_lbl in train_loader:
                batch_emb = batch_emb.to(device)
                batch_lbl = batch_lbl.to(device)
                logits = self.forward(batch_emb)
                loss = criterion(logits, batch_lbl)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1

                if verbose and n_batches % log_interval == 0:
                    elapsed = time.time() - epoch_t0
                    avg_loss = train_loss / n_batches
                    logger.info(
                        "  Epoch %d/%d: batch %d/%d (%.0f s), loss=%.4f",
                        epoch + 1, epochs, n_batches, total_batches,
                        elapsed, avg_loss,
                    )

            train_loss /= max(n_batches, 1)
            history["train_loss"].append(train_loss)

            # -- Validate --
            if val_loader is not None:
                val_metrics = self._evaluate_streaming(
                    val_loader, criterion, device,
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
        if val_loader is not None and best_state is not None:
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
    def _evaluate_streaming(
        self,
        data_loader: "torch.utils.data.DataLoader",
        criterion: nn.Module,
        device: str,
    ) -> Dict[str, float]:
        """Evaluate on a DataLoader, accumulating predictions in CPU memory.

        Only stores flat probability/label arrays (not embeddings), keeping
        memory proportional to ``N_positions * num_classes * 4 bytes``.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        self.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        for batch_emb, batch_lbl in data_loader:
            batch_emb = batch_emb.to(device)
            batch_lbl = batch_lbl.to(device)
            logits = self.forward(batch_emb)
            total_loss += criterion(logits, batch_lbl).item()
            n_batches += 1

            probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, 3, W]
            # Reshape to [B*W, 3]
            b, c, w = probs.shape
            all_probs.append(probs.transpose(0, 2, 1).reshape(-1, c))
            all_labels.append(batch_lbl.cpu().numpy().flatten())

        loss = total_loss / max(n_batches, 1)

        if not all_probs:
            return {"loss": loss, "auroc": 0.0, "auprc": 0.0}

        probs_flat = np.concatenate(all_probs, axis=0)   # [N, 3]
        labels_flat = np.concatenate(all_labels, axis=0)  # [N]

        aurocs, auprcs = [], []
        for cls in [LABEL_ACCEPTOR, LABEL_DONOR]:
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

    def calibrate(
        self,
        val_loader: "torch.utils.data.DataLoader",
        device: str = "cuda",
        max_epochs: int = 2000,
        lr: float = 0.01,
        patience: int = 5,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Post-hoc temperature scaling calibration on validation data.

        Learns a class-wise temperature vector that divides logits before
        softmax, improving probability calibration without changing model
        weights.  Adapted from OpenSpliceAI
        ``calibrate/temperature_scaling.py``.

        Args:
            val_loader: Validation DataLoader yielding ``(emb, lbl)`` batches.
            device: Target device for optimization.
            max_epochs: Maximum optimization epochs for temperature.
            lr: Adam learning rate.
            patience: Early-stopping patience on NLL.
            verbose: Log before/after metrics.

        Returns:
            Dict with ``nll_before``, ``nll_after``, ``ece_before``,
            ``ece_after``, and ``temperature`` (list of 3 floats).
        """
        self.to(device)
        self.eval()

        # --- Collect logits from val set (streaming, CPU accumulate) ---
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []

        with torch.no_grad():
            for batch_emb, batch_lbl in val_loader:
                batch_emb = batch_emb.to(device)
                logits = self.forward(batch_emb)  # [B, 3, W]
                # Flatten to [B*W, 3]
                b, c, w = logits.shape
                flat_logits = logits.permute(0, 2, 1).reshape(-1, c)
                flat_labels = batch_lbl.reshape(-1).long()
                all_logits.append(flat_logits.cpu())
                all_labels.append(flat_labels)

        logits_flat = torch.cat(all_logits).to(device)   # [N, 3]
        labels_flat = torch.cat(all_labels).to(device)   # [N]

        logger.info(
            "  Calibration data: %d positions (%.1f MB logits)",
            len(labels_flat), logits_flat.nelement() * 4 / 1e6,
        )

        # --- Initialize temperature ---
        self.temperature = nn.Parameter(
            torch.ones(NUM_CLASSES, device=device),
        )

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss(n_bins=15).to(device)

        # Before calibration
        with torch.no_grad():
            nll_before = nll_criterion(logits_flat, labels_flat).item()
            ece_before = ece_criterion(logits_flat, labels_flat).item()

        if verbose:
            logger.info(
                "  Before calibration: NLL=%.4f, ECE=%.4f",
                nll_before, ece_before,
            )

        # --- Optimize temperature ---
        optimizer = torch.optim.Adam([self.temperature], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=2,
        )

        best_loss = float("inf")
        best_temp = self.temperature.data.clone()
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            temp = torch.clamp(self.temperature, min=0.05, max=5.0)
            scaled = logits_flat / temp.unsqueeze(0)  # [N, 3] / [1, 3]
            loss = nll_criterion(scaled, labels_flat)
            loss.backward()
            optimizer.step()
            # Clamp after step
            self.temperature.data.clamp_(min=0.05, max=5.0)

            current_loss = loss.item()
            scheduler.step(current_loss)

            if best_loss - current_loss > 1e-6:
                best_loss = current_loss
                best_temp = self.temperature.data.clone()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        self.temperature = nn.Parameter(best_temp)

        # After calibration
        with torch.no_grad():
            temp = torch.clamp(self.temperature, min=0.05, max=5.0)
            scaled = logits_flat / temp.unsqueeze(0)
            nll_after = nll_criterion(scaled, labels_flat).item()
            ece_after = ece_criterion(scaled, labels_flat).item()

        temp_vals = self.temperature.data.cpu().tolist()
        if verbose:
            logger.info(
                "  After calibration:  NLL=%.4f, ECE=%.4f  (%d epochs)",
                nll_after, ece_after, epoch + 1,
            )
            logger.info(
                "  Temperature: neither=%.3f, acceptor=%.3f, donor=%.3f",
                temp_vals[0], temp_vals[1], temp_vals[2],
            )

        return {
            "nll_before": nll_before,
            "nll_after": nll_after,
            "ece_before": ece_before,
            "ece_after": ece_after,
            "temperature": temp_vals,
        }

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
        ckpt_data = {
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
        }
        if self.temperature is not None:
            ckpt_data["temperature"] = self.temperature.data.cpu()
        torch.save(ckpt_data, ckpt_path)

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
        state_dict = ckpt["model_state_dict"]
        # Temperature is handled separately below — remove from state_dict
        # to avoid "unexpected key" errors on fresh (uncalibrated) models.
        state_dict.pop("temperature", None)
        model.load_state_dict(state_dict)
        if "temperature" in ckpt:
            model.temperature = nn.Parameter(ckpt["temperature"].to(device))
        model.to(device)
        model.eval()

        logger.info(
            "Loaded SpliceClassifier from %s (epoch %d, %s, calibrated=%s)",
            path, ckpt.get("epoch", -1), ckpt.get("architecture", "?"),
            "temperature" in ckpt,
        )
        return model

    @classmethod
    def load_model(
        cls,
        path: str | Path,
        device: str = "cpu",
    ) -> "SpliceClassifier":
        """Load a trained (and optionally calibrated) SpliceClassifier.

        Alias for :meth:`load_checkpoint` — use when loading a finalized
        model for inference rather than resuming training.

        Args:
            path: Path to ``best_model.pt`` or directory containing it.
            device: Device to load onto.

        Returns:
            Ready-to-use SpliceClassifier in eval mode.
        """
        p = Path(path)
        if p.is_dir():
            p = p / "best_model.pt"
        return cls.load_checkpoint(p, device=device)
