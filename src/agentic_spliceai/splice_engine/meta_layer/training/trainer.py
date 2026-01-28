"""
Training loop for the meta-layer model.

Supports:
- Mixed precision training
- Gradient accumulation
- Early stopping
- Learning rate scheduling
- Comprehensive logging
- Checkpoint saving/loading
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

from agentic_spliceai.splice_engine.meta_layer.models import MetaSpliceModel
from agentic_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig
from .evaluator import Evaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Basic training
    epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Scheduler
    scheduler: str = 'cosine'  # 'cosine', 'onecycle', 'none'
    warmup_epochs: int = 5
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Optimization
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Mixed precision
    use_amp: bool = False  # Disable for M1 compatibility
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = 'val_pr_auc'
    early_stopping_mode: str = 'max'
    
    # Checkpointing
    save_every: int = 5
    checkpoint_dir: Optional[str] = None
    
    # Logging
    log_every: int = 100
    eval_every: int = 1  # Evaluate every N epochs
    
    # Device
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    
    def get_device(self) -> torch.device:
        """Get the appropriate device."""
        if self.device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device)


@dataclass
class TrainingResult:
    """Results from training."""
    best_epoch: int
    best_metrics: Dict[str, float]
    training_history: List[Dict[str, float]]
    validation_history: List[Dict[str, float]]
    final_model_path: Optional[str] = None
    total_time_seconds: float = 0


class Trainer:
    """
    Trainer for meta-layer model.
    
    Examples
    --------
    >>> trainer = Trainer(model, train_loader, val_loader, config)
    >>> result = trainer.train()
    >>> print(f"Best PR-AUC: {result.best_metrics['pr_auc']:.4f}")
    """
    
    def __init__(
        self,
        model: MetaSpliceModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.get_device()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Class weights for imbalanced data
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup evaluator
        self.evaluator = Evaluator()
        
        # Mixed precision scaler
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Tracking
        self.best_metric = float('-inf') if config.early_stopping_mode == 'max' else float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.epochs
        
        if self.config.scheduler == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_epochs / self.config.epochs
            )
        else:
            return None
    
    def train(self) -> TrainingResult:
        """
        Run full training loop.
        
        Returns
        -------
        TrainingResult
            Training results including history and best model.
        """
        training_history = []
        validation_history = []
        start_time = time.time()
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(1, self.config.epochs + 1):
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            training_history.append(train_metrics)
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_metrics = self._validate_epoch(epoch)
                validation_history.append(val_metrics)
                
                # Check for improvement
                current_metric = val_metrics.get(
                    self.config.early_stopping_metric, 
                    val_metrics.get('val_loss', 0)
                )
                
                improved = self._check_improvement(current_metric)
                
                if improved:
                    self.best_epoch = epoch
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    
                    # Save best model
                    if self.config.checkpoint_dir:
                        self._save_checkpoint(epoch, is_best=True)
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Regular checkpoint
            if self.config.checkpoint_dir and epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)
        
        total_time = time.time() - start_time
        
        # Load best model
        best_model_path = None
        if self.config.checkpoint_dir:
            best_model_path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            if best_model_path.exists():
                self._load_checkpoint(best_model_path)
        
        # Final validation
        final_metrics = self._validate_epoch(-1, final=True)
        
        return TrainingResult(
            best_epoch=self.best_epoch,
            best_metrics=final_metrics,
            training_history=training_history,
            validation_history=validation_history,
            final_model_path=str(best_model_path) if best_model_path else None,
            total_time_seconds=total_time
        )
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            sequence = batch['sequence'].to(self.device)
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            weights = batch.get('weight', None)
            if weights is not None:
                weights = weights.to(self.device)
            
            # Forward pass
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = self.model(sequence, features)
                    loss = self._compute_loss(logits, labels, weights)
            else:
                logits = self.model(sequence, features)
                loss = self._compute_loss(logits, labels, weights)
            
            # Scale for accumulation
            loss = loss / self.config.accumulation_steps
            
            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler is not None:
                    self.scheduler.step()
            
            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1
            
            # Logging
            if batch_idx % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.debug(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} LR: {lr:.6f}"
                )
        
        avg_loss = total_loss / num_batches
        lr = self.optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch}: train_loss={avg_loss:.4f}, lr={lr:.6f}")
        
        return {
            'epoch': epoch,
            'train_loss': avg_loss,
            'learning_rate': lr
        }
    
    def _validate_epoch(
        self, 
        epoch: int, 
        final: bool = False
    ) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                sequence = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.model(sequence, features)
                loss = self._compute_loss(logits, labels)
                
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())
                total_loss += loss.item()
                num_batches += 1
        
        # Concatenate
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_probs = torch.cat(all_probs)
        
        # Evaluate
        result = self.evaluator.evaluate(
            predictions=all_preds.numpy(),
            labels=all_labels.numpy(),
            probabilities=all_probs.numpy(),
            compute_detailed=final
        )
        
        avg_loss = total_loss / num_batches
        
        metrics = {
            'epoch': epoch,
            'val_loss': avg_loss,
            'val_accuracy': result.accuracy,
            'val_pr_auc': result.pr_auc_macro,
            'val_roc_auc': result.roc_auc_macro,
            'val_average_precision': result.average_precision_macro
        }
        
        # Add per-class metrics
        for i, name in enumerate(['donor', 'acceptor', 'neither']):
            if result.per_class_pr_auc:
                metrics[f'val_pr_auc_{name}'] = result.per_class_pr_auc[i]
        
        if final:
            metrics['confusion_matrix'] = result.confusion_matrix.tolist()
        
        logger.info(
            f"Epoch {epoch}: val_loss={avg_loss:.4f}, "
            f"val_pr_auc={result.pr_auc_macro:.4f}, "
            f"val_acc={result.accuracy:.4f}"
        )
        
        return metrics
    
    def _compute_loss(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss with optional class weights and label smoothing."""
        if self.config.label_smoothing > 0:
            loss = F.cross_entropy(
                logits, 
                labels,
                weight=self.class_weights,
                label_smoothing=self.config.label_smoothing
            )
        else:
            loss = F.cross_entropy(
                logits, 
                labels,
                weight=self.class_weights
            )
        
        # Apply sample weights
        if sample_weights is not None:
            loss = (loss * sample_weights).mean()
        
        return loss
    
    def _check_improvement(self, current_metric: float) -> bool:
        """Check if current metric is an improvement."""
        if self.config.early_stopping_mode == 'max':
            return current_metric > self.best_metric
        else:
            return current_metric < self.best_metric
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint
        if is_best:
            path = checkpoint_dir / 'best_model.pt'
        else:
            path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path: Union[str, Path]):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint: {path}")


def train_meta_model(
    config: MetaLayerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    training_config: Optional[TrainingConfig] = None,
    model: Optional[MetaSpliceModel] = None
) -> TrainingResult:
    """
    Convenience function to train a meta-layer model.
    
    Parameters
    ----------
    config : MetaLayerConfig
        Meta-layer configuration.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    training_config : TrainingConfig, optional
        Training configuration.
    model : MetaSpliceModel, optional
        Pre-configured model. If None, creates from config.
    
    Returns
    -------
    TrainingResult
        Training results.
    """
    if training_config is None:
        training_config = TrainingConfig()
    
    # Create model if not provided
    if model is None:
        # Get number of features from dataset
        sample = next(iter(train_loader))
        num_features = sample['features'].shape[-1]
        
        model = MetaSpliceModel(
            sequence_encoder=config.sequence_encoder,
            num_score_features=num_features,
            hidden_dim=config.hidden_dim,
            dropout=training_config.dropout
        )
    
    # Get class weights
    class_weights = None
    if hasattr(train_loader.dataset, 'dataset'):
        # Handle Subset from random_split
        base_dataset = train_loader.dataset.dataset
        if hasattr(base_dataset, 'get_class_weights'):
            class_weights = base_dataset.get_class_weights()
    elif hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights()
    
    # Create trainer and train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        class_weights=class_weights
    )
    
    return trainer.train()












