"""
Exon Classifier using Evo2 Embeddings

Lightweight classifier trained on Evo2 embeddings for single-nucleotide
resolution exon/intron classification.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional
import numpy as np


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
        
        with torch.no_grad():
            logits = self(embeddings)
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
    ):
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
        """
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.metrics import roc_auc_score, average_precision_score
        
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
        pos_weight = (train_labels == 0).sum() / (train_labels == 1).sum()
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Training loop
        best_val_auroc = 0.0
        
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
            
            # Validate
            if val_embeddings is not None:
                val_metrics = self._evaluate(
                    val_embeddings, val_labels, criterion
                )
                
                if verbose:
                    print(
                        f"Epoch {epoch+1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={val_metrics['loss']:.4f}, "
                        f"val_auroc={val_metrics['auroc']:.4f}, "
                        f"val_auprc={val_metrics['auprc']:.4f}"
                    )
                
                # Track best model
                if val_metrics['auroc'] > best_val_auroc:
                    best_val_auroc = val_metrics['auroc']
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}")
    
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
