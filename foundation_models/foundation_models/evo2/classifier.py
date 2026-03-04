"""
Exon Classifier using Evo2 Embeddings

Lightweight classifier trained on Evo2 embeddings for single-nucleotide
resolution exon/intron classification.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ExonClassifier(nn.Module):
    """
    Exon/intron classifier using Evo2 embeddings.
    
    Architecture options:
    - "linear": Simple linear layer (baseline)
    - "mlp": Multi-layer perceptron (2-3 layers)
    - "cnn": 1D CNN for local context
    - "lstm": Bidirectional LSTM for long-range dependencies
    
    Example:
        >>> classifier = ExonClassifier(
        ...     input_dim=2560,  # Evo2 7B hidden dim
        ...     architecture="mlp"
        ... )
        >>> 
        >>> # Train
        >>> classifier.fit(train_embeddings, train_labels, epochs=50)
        >>> 
        >>> # Predict
        >>> probs = classifier.predict(test_embeddings)
    """
    
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 256,
        num_layers: int = 2,
        architecture: Literal["linear", "mlp", "cnn", "lstm"] = "mlp",
        dropout: float = 0.1,
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Dimension of input embeddings (Evo2 hidden dim)
            hidden_dim: Hidden dimension for classifier
            num_layers: Number of hidden layers (for mlp, lstm)
            architecture: Classifier architecture
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.architecture = architecture
        self.dropout = dropout
        
        # Build architecture
        if architecture == "linear":
            self.classifier = nn.Linear(input_dim, 1)
        
        elif architecture == "mlp":
            layers = []
            dims = [input_dim] + [hidden_dim] * num_layers + [1]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:  # No activation after last layer
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            self.classifier = nn.Sequential(*layers)
        
        elif architecture == "cnn":
            self.classifier = nn.Sequential(
                # First conv layer
                nn.Conv1d(input_dim, hidden_dim, kernel_size=9, padding=4),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                # Second conv layer
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                # Output layer
                nn.Conv1d(hidden_dim, 1, kernel_size=1),
            )
        
        elif architecture == "lstm":
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim // 2,
                num_layers=num_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
            self.classifier = nn.Linear(hidden_dim, 1)
        
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: [batch, seq_len, input_dim] or [seq_len, input_dim]
        
        Returns:
            logits: [batch, seq_len] or [seq_len]
        """
        # Handle single sequence input
        if embeddings.ndim == 2:
            embeddings = embeddings.unsqueeze(0)  # Add batch dim
            single_input = True
        else:
            single_input = False
        
        # Forward pass based on architecture
        if self.architecture == "cnn":
            # Conv expects [batch, channels, seq_len]
            x = embeddings.transpose(1, 2)
            logits = self.classifier(x).transpose(1, 2).squeeze(-1)
        
        elif self.architecture == "lstm":
            x, _ = self.lstm(embeddings)
            logits = self.classifier(x).squeeze(-1)
        
        else:  # linear or mlp
            logits = self.classifier(embeddings).squeeze(-1)
        
        # Remove batch dim if single input
        if single_input:
            logits = logits.squeeze(0)
        
        return logits
    
    def predict(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Predict exon probabilities.
        
        Args:
            embeddings: [batch, seq_len, input_dim] or [seq_len, input_dim]
        
        Returns:
            probs: [batch, seq_len] or [seq_len] - probability of being exon
        """
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            logits = self(embeddings.to(device))
            probs = torch.sigmoid(logits)

        return probs.cpu().numpy()
    
    def fit(
        self,
        train_embeddings: torch.Tensor,
        train_labels: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        val_labels: Optional[torch.Tensor] = None,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        device: str = "cuda",
        verbose: bool = True,
        checkpoint_dir: Optional[str] = None,
        patience: int = 10,
        lr_schedule: bool = True,
    ) -> Dict:
        """
        Train classifier.

        Args:
            train_embeddings: [N, seq_len, input_dim]
            train_labels: [N, seq_len] - binary labels (0=intron, 1=exon)
            val_embeddings: Optional validation embeddings
            val_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            device: Device to train on
            verbose: Print training progress
            checkpoint_dir: Directory to save best model checkpoint (None = no disk save)
            patience: Early stopping patience in epochs (0 = disabled)
            lr_schedule: Use ReduceLROnPlateau on val AUROC

        Returns:
            Training history dict with keys: train_loss, val_loss, val_auroc,
            val_auprc, best_epoch, best_val_auroc, stopped_early.
        """
        from torch.utils.data import TensorDataset, DataLoader

        # Move to device
        self.to(device)
        train_embeddings = train_embeddings.to(device)
        train_labels = train_labels.to(device)

        if val_embeddings is not None:
            val_embeddings = val_embeddings.to(device)
            val_labels = val_labels.to(device)

        # Create data loader
        train_dataset = TensorDataset(train_embeddings, train_labels)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Loss (handle class imbalance)
        n_pos = (train_labels == 1).sum().float()
        n_neg = (train_labels == 0).sum().float()
        pos_weight = n_neg / n_pos.clamp(min=1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # LR scheduler (only with validation data)
        scheduler = None
        if lr_schedule and val_embeddings is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
            )

        # Training state
        history: Dict[str, List] = {
            "train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [],
        }
        best_val_auroc = 0.0
        best_epoch = 0
        best_state: Optional[dict] = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Train
            self.train()
            train_loss = 0.0

            for batch_emb, batch_labels in train_loader:
                logits = self(batch_emb)
                loss = criterion(logits, batch_labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)

            # Validate
            if val_embeddings is not None:
                val_metrics = self._evaluate(
                    val_embeddings, val_labels, criterion
                )
                history["val_loss"].append(val_metrics["loss"])
                history["val_auroc"].append(val_metrics["auroc"])
                history["val_auprc"].append(val_metrics["auprc"])

                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_auroc={val_metrics['auroc']:.4f}, "
                        f"val_auprc={val_metrics['auprc']:.4f}"
                    )

                # Track best model
                if val_metrics["auroc"] > best_val_auroc:
                    best_val_auroc = val_metrics["auroc"]
                    best_epoch = epoch
                    best_state = {
                        k: v.cpu().clone() for k, v in self.state_dict().items()
                    }
                    if checkpoint_dir is not None:
                        self._save_checkpoint(checkpoint_dir, epoch, val_metrics)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                # LR scheduling
                if scheduler is not None:
                    scheduler.step(val_metrics["auroc"])

                # Early stopping
                if patience > 0 and epochs_without_improvement >= patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch+1} "
                            f"(best AUROC: {best_val_auroc:.4f} at epoch {best_epoch+1})"
                        )
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")

        # Restore best model weights
        stopped_early = epochs_without_improvement >= patience if patience > 0 else False
        if val_embeddings is not None and best_state is not None:
            self.load_state_dict(best_state)
            if verbose:
                print(
                    f"Restored best model from epoch {best_epoch+1} "
                    f"(AUROC: {best_val_auroc:.4f})"
                )

        history["best_epoch"] = best_epoch
        history["best_val_auroc"] = best_val_auroc
        history["stopped_early"] = stopped_early
        return history
    
    def _save_checkpoint(
        self, checkpoint_dir: str, epoch: int, metrics: dict
    ) -> Path:
        """Save model checkpoint to disk.

        Args:
            checkpoint_dir: Directory to save checkpoint into.
            epoch: Current epoch number.
            metrics: Validation metrics dict.

        Returns:
            Path to the saved checkpoint file.
        """
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "architecture": self.architecture,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "epoch": epoch,
            "metrics": metrics,
        }
        path = ckpt_dir / "best_model.pt"
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s (epoch %d)", path, epoch + 1)
        return path

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str, device: str = "cpu"
    ) -> "ExonClassifier":
        """Load a classifier from a saved checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to load model onto.

        Returns:
            ExonClassifier with restored weights.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False,
        )
        model = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            num_layers=checkpoint["num_layers"],
            architecture=checkpoint["architecture"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        logger.info(
            "Loaded checkpoint from %s (epoch %d, AUROC %.4f)",
            checkpoint_path,
            checkpoint["epoch"] + 1,
            checkpoint.get("metrics", {}).get("auroc", 0.0),
        )
        return model

    def _evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> dict:
        """Evaluate on validation set."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        
        self.eval()
        
        with torch.no_grad():
            logits = self(embeddings)
            loss = criterion(logits, labels.float()).item()
            probs = torch.sigmoid(logits).cpu().numpy()
        
        labels_np = labels.cpu().numpy()
        
        # Flatten for metrics
        probs_flat = probs.flatten()
        labels_flat = labels_np.flatten()
        
        metrics = {
            'loss': loss,
            'auroc': roc_auc_score(labels_flat, probs_flat),
            'auprc': average_precision_score(labels_flat, probs_flat),
        }
        
        return metrics
    
    def __repr__(self) -> str:
        return (
            f"ExonClassifier(architecture={self.architecture}, "
            f"input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )
