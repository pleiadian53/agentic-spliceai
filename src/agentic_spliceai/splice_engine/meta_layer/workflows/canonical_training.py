"""
Canonical Splice Site Classification Training Workflow.

This workflow trains the meta-layer on canonical splice sites from base layer
artifacts, then evaluates its ability to detect variant-induced splice changes
using SpliceVarDB as an external validation set.

Key Principle:
  SpliceVarDB is NOT used for training, only for evaluation. This validates
  whether the meta-layer generalizes to variant effect detection without
  directly training on variant labels.

Training Data: Base layer artifacts (analysis_sequences, splice_errors)
  - Canonical splice sites (donor/acceptor) from gene annotations
  - Non-splice sites (neither) for negative examples

Evaluation Data:
  1. Held-out canonical sites (test chromosomes) - No degradation check
  2. SpliceVarDB variants - Variant effect detection ability

This approach is useful for:
  - Recalibrating base model predictions
  - Understanding meta-layer generalization
  - Baseline comparison for other approaches

See also:
  - validated_delta_training.py: Uses SpliceVarDB for training targets (BEST)
  - hyenadna_training.py: GPU training with HyenaDNA encoder

Ported from: meta_spliceai/splice_engine/meta_layer/workflows/phase1_training.py
Renamed for clarity: "Canonical" describes training data source, not research phase.
"""

import logging
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import polars as pl

from agentic_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig
from agentic_spliceai.splice_engine.meta_layer.core.artifact_loader import ArtifactLoader
from agentic_spliceai.splice_engine.meta_layer.data.dataset import (
    MetaLayerDataset, 
    create_dataloaders, 
    prepare_training_data
)
from agentic_spliceai.splice_engine.meta_layer.data.splicevardb_loader import load_splicevardb
from agentic_spliceai.splice_engine.meta_layer.models import MetaSpliceModel
from agentic_spliceai.splice_engine.meta_layer.training.trainer import (
    Trainer, 
    TrainingConfig, 
    TrainingResult
)
from agentic_spliceai.splice_engine.meta_layer.training.variant_evaluator import (
    VariantEffectEvaluator, 
    VariantEvaluationResult
)

logger = logging.getLogger(__name__)


@dataclass
class CanonicalTrainingConfig:
    """
    Configuration for canonical splice site classification training.
    
    Supports both base models:
    - openspliceai: Uses GRCh38/MANE, hg38 coordinates from SpliceVarDB
    - spliceai: Uses GRCh37/Ensembl, hg19 coordinates from SpliceVarDB
    """
    
    # Base model (determines genome build automatically)
    base_model: str = 'openspliceai'
    genome_build: Optional[str] = None  # Auto-derived from base_model if None
    
    # Data
    train_chromosomes: Optional[List[str]] = None  # None = all except test
    test_chromosomes: List[str] = field(default_factory=lambda: ['21', '22'])
    max_train_samples: Optional[int] = None
    balance_classes: bool = True
    
    # Model
    sequence_encoder: str = 'cnn'  # 'cnn' for M1 Mac, 'hyenadna' for GPU
    hidden_dim: int = 256
    fusion_type: str = 'concat'
    dropout: float = 0.1
    
    # Training
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    early_stopping_patience: int = 10
    
    # Output
    output_dir: str = './canonical_training_output'
    experiment_name: str = 'canonical_classification'
    
    # Evaluation
    eval_max_variants: Optional[int] = 1000  # Limit for faster evaluation
    
    def __post_init__(self):
        """Auto-derive genome_build from base_model if not specified."""
        if self.genome_build is None:
            build_map = {
                'openspliceai': 'GRCh38',
                'spliceai': 'GRCh37',
            }
            self.genome_build = build_map.get(self.base_model.lower(), 'GRCh38')
    
    def to_meta_layer_config(self) -> MetaLayerConfig:
        """Convert to MetaLayerConfig."""
        return MetaLayerConfig(
            base_model=self.base_model,
            sequence_encoder=self.sequence_encoder,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            balance_classes=self.balance_classes
        )
    
    def to_training_config(self) -> TrainingConfig:
        """Convert to TrainingConfig."""
        return TrainingConfig(
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            early_stopping_patience=self.early_stopping_patience,
            checkpoint_dir=str(Path(self.output_dir) / 'checkpoints'),
            device='auto'
        )


@dataclass
class CanonicalTrainingResult:
    """Results from canonical training and evaluation."""
    
    # Training results
    training_result: TrainingResult
    
    # Canonical site evaluation (held-out test set)
    canonical_test_metrics: Dict[str, float]
    
    # Variant effect evaluation (SpliceVarDB)
    variant_evaluation: Optional[VariantEvaluationResult]
    
    # Meta
    config: CanonicalTrainingConfig
    start_time: str
    end_time: str
    total_duration_seconds: float
    
    def save(self, path: Path):
        """Save results to JSON."""
        result_dict = {
            'config': asdict(self.config),
            'training': {
                'best_epoch': self.training_result.best_epoch,
                'best_metrics': self.training_result.best_metrics,
                'total_time_seconds': self.training_result.total_time_seconds,
            },
            'canonical_test': self.canonical_test_metrics,
            'variant_evaluation': {
                'total_variants': self.variant_evaluation.total_variants,
                'base_accuracy': self.variant_evaluation.base_accuracy,
                'meta_accuracy': self.variant_evaluation.meta_accuracy,
                'base_detection_rate': self.variant_evaluation.base_detection_rate,
                'meta_detection_rate': self.variant_evaluation.meta_detection_rate,
                'improvement_count': self.variant_evaluation.improvement_count,
            } if self.variant_evaluation else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_duration_seconds': self.total_duration_seconds,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved results to {path}")


class CanonicalTrainingWorkflow:
    """
    Canonical Splice Site Classification Training Workflow.
    
    Steps:
    1. Load canonical splice site data from base layer artifacts
    2. Split into train/val/test by chromosome
    3. Train multimodal meta-layer
    4. Evaluate on held-out canonical sites
    5. Evaluate on SpliceVarDB variants (external validation)
    
    Examples
    --------
    >>> config = CanonicalTrainingConfig(
    ...     base_model='openspliceai',
    ...     epochs=30,
    ...     output_dir='./experiments/canonical'
    ... )
    >>> workflow = CanonicalTrainingWorkflow(config)
    >>> result = workflow.run()
    >>> print(f"Test PR-AUC: {result.canonical_test_metrics['pr_auc_macro']:.4f}")
    >>> if result.variant_evaluation:
    ...     print(result.variant_evaluation.summary())
    """
    
    def __init__(self, config: CanonicalTrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("=" * 60)
        logger.info("CANONICAL SPLICE SITE CLASSIFICATION TRAINING")
        logger.info("=" * 60)
        logger.info(f"Base model: {config.base_model}")
        logger.info(f"Genome build: {config.genome_build}")
        logger.info(f"Sequence encoder: {config.sequence_encoder}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _setup_logging(self):
        """Setup file logging."""
        log_file = self.output_dir / 'canonical_training.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logging.getLogger().addHandler(file_handler)
    
    def run(self) -> CanonicalTrainingResult:
        """
        Run the complete training workflow.
        
        Returns
        -------
        CanonicalTrainingResult
            Complete training and evaluation results.
        """
        import time
        start_time = datetime.now()
        start_timestamp = time.time()
        
        # Step 1: Prepare data
        logger.info("\n📊 STEP 1: Preparing training data...")
        train_loader, val_loader, test_loader = self._prepare_data()
        
        # Step 2: Create model
        logger.info("\n🧠 STEP 2: Creating model...")
        model = self._create_model(train_loader)
        
        # Step 3: Train
        logger.info("\n🏋️ STEP 3: Training model...")
        training_result = self._train(model, train_loader, val_loader)
        
        # Step 4: Evaluate on canonical test set
        logger.info("\n📈 STEP 4: Evaluating on canonical test set...")
        canonical_metrics = self._evaluate_canonical(model, test_loader)
        
        # Step 5: Evaluate on SpliceVarDB
        logger.info("\n🧬 STEP 5: Evaluating on SpliceVarDB variants...")
        variant_result = self._evaluate_variants(model)
        
        end_time = datetime.now()
        total_duration = time.time() - start_timestamp
        
        # Create result
        result = CanonicalTrainingResult(
            training_result=training_result,
            canonical_test_metrics=canonical_metrics,
            variant_evaluation=variant_result,
            config=self.config,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=total_duration
        )
        
        # Save results
        result.save(self.output_dir / 'canonical_training_results.json')
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _prepare_data(self):
        """Prepare training, validation, and test data loaders."""
        meta_config = self.config.to_meta_layer_config()
        
        # Get all chromosomes except test
        all_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
        train_val_chroms = [c for c in all_chroms if c not in self.config.test_chromosomes]
        
        # Load training data
        logger.info(f"Loading data from base layer artifacts...")
        logger.info(f"  Train/val chromosomes: {train_val_chroms}")
        logger.info(f"  Test chromosomes: {self.config.test_chromosomes}")
        
        # Prepare training data (train + val)
        train_val_df = prepare_training_data(
            config=meta_config,
            chromosomes=train_val_chroms,
            max_samples=self.config.max_train_samples,
            balance_classes=self.config.balance_classes
        )
        
        # Prepare test data
        test_df = prepare_training_data(
            config=meta_config,
            chromosomes=self.config.test_chromosomes,
            max_samples=None,
            balance_classes=False  # Don't balance test set
        )
        
        logger.info(f"  Train/val samples: {len(train_val_df)}")
        logger.info(f"  Test samples: {len(test_df)}")
        
        # Create datasets
        train_val_dataset = MetaLayerDataset(train_val_df)
        test_dataset = MetaLayerDataset(test_df)
        
        # Split train/val
        from torch.utils.data import DataLoader, random_split
        
        n_train = int(0.9 * len(train_val_dataset))
        n_val = len(train_val_dataset) - n_train
        
        train_dataset, val_dataset = random_split(
            train_val_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"  Train batches: {len(train_loader)}")
        logger.info(f"  Val batches: {len(val_loader)}")
        logger.info(f"  Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _create_model(self, train_loader) -> MetaSpliceModel:
        """Create the meta-layer model."""
        sample = next(iter(train_loader))
        num_features = sample['features'].shape[-1]
        
        model = MetaSpliceModel(
            sequence_encoder=self.config.sequence_encoder,
            num_score_features=num_features,
            hidden_dim=self.config.hidden_dim,
            num_classes=3,
            dropout=self.config.dropout,
            fusion_type=self.config.fusion_type
        )
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model parameters: {n_params:,}")
        
        return model
    
    def _train(
        self,
        model: MetaSpliceModel,
        train_loader,
        val_loader
    ) -> TrainingResult:
        """Train the model."""
        training_config = self.config.to_training_config()
        
        # Get class weights
        class_weights = None
        if hasattr(train_loader.dataset, 'dataset'):
            base_dataset = train_loader.dataset.dataset
            if hasattr(base_dataset, 'get_class_weights'):
                class_weights = base_dataset.get_class_weights()
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            class_weights=class_weights
        )
        
        result = trainer.train()
        
        logger.info(f"  Best epoch: {result.best_epoch}")
        logger.info(f"  Best val PR-AUC: {result.best_metrics.get('val_pr_auc', 0):.4f}")
        
        return result
    
    def _evaluate_canonical(
        self,
        model: MetaSpliceModel,
        test_loader
    ) -> Dict[str, float]:
        """Evaluate on held-out canonical splice sites."""
        from agentic_spliceai.splice_engine.meta_layer.training.evaluator import Evaluator
        import torch.nn.functional as F
        
        device = next(model.parameters()).device
        model.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in test_loader:
                sequence = batch['sequence'].to(device)
                features = batch['features'].to(device)
                labels = batch['label']
                
                logits = model(sequence, features)
                probs = F.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
                all_probs.append(probs.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        
        evaluator = Evaluator()
        result = evaluator.evaluate(
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs,
            compute_detailed=True
        )
        
        metrics = {
            'accuracy': result.accuracy,
            'pr_auc_macro': result.pr_auc_macro,
            'roc_auc_macro': result.roc_auc_macro,
            'average_precision_macro': result.average_precision_macro,
        }
        
        for i, name in enumerate(['donor', 'acceptor', 'neither']):
            if result.per_class_pr_auc:
                metrics[f'pr_auc_{name}'] = result.per_class_pr_auc[i]
        
        logger.info(f"  Test accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Test PR-AUC: {metrics['pr_auc_macro']:.4f}")
        
        return metrics
    
    def _evaluate_variants(
        self,
        model: MetaSpliceModel
    ) -> Optional[VariantEvaluationResult]:
        """Evaluate on SpliceVarDB variants."""
        try:
            meta_config = self.config.to_meta_layer_config()
            
            loader = load_splicevardb(genome_build=self.config.genome_build)
            
            _, test_variants = loader.get_train_test_split(
                test_chromosomes=self.config.test_chromosomes
            )
            
            if self.config.eval_max_variants and len(test_variants) > self.config.eval_max_variants:
                import random
                random.seed(42)
                random.shuffle(test_variants)
                test_variants = test_variants[:self.config.eval_max_variants]
            
            logger.info(f"  Evaluating {len(test_variants)} variants...")
            
            evaluator = VariantEffectEvaluator(model, meta_config)
            result = evaluator.evaluate(test_variants)
            
            logger.info(f"  Meta-layer accuracy: {result.meta_accuracy:.1%}")
            logger.info(f"  Detection rate (splice-altering): {result.meta_detection_rate:.1%}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Variant evaluation failed: {e}")
            return None
    
    def _print_summary(self, result: CanonicalTrainingResult):
        """Print final summary."""
        logger.info("\n" + "=" * 60)
        logger.info("CANONICAL TRAINING COMPLETE")
        logger.info("=" * 60)
        
        logger.info(f"\n📊 Training Summary:")
        logger.info(f"  Best epoch: {result.training_result.best_epoch}")
        logger.info(f"  Training time: {result.training_result.total_time_seconds:.1f}s")
        
        logger.info(f"\n📈 Canonical Test Performance:")
        logger.info(f"  Accuracy: {result.canonical_test_metrics.get('accuracy', 0):.4f}")
        logger.info(f"  PR-AUC: {result.canonical_test_metrics.get('pr_auc_macro', 0):.4f}")
        
        if result.variant_evaluation:
            logger.info(f"\n🧬 Variant Effect Detection:")
            logger.info(result.variant_evaluation.summary())
        
        logger.info(f"\n📁 Results saved to: {self.output_dir}")


def run_canonical_training(
    base_model: str = 'openspliceai',
    epochs: int = 30,
    output_dir: str = './canonical_training_output',
    sequence_encoder: str = 'cnn',
    max_train_samples: Optional[int] = None,
    eval_max_variants: int = 1000
) -> CanonicalTrainingResult:
    """
    Convenience function to run canonical training.
    
    Parameters
    ----------
    base_model : str
        Base model to use ('openspliceai' or 'spliceai').
    epochs : int
        Number of training epochs.
    output_dir : str
        Output directory for results.
    sequence_encoder : str
        Sequence encoder type ('cnn' or 'hyenadna').
    max_train_samples : int, optional
        Maximum training samples (for quick testing).
    eval_max_variants : int
        Maximum variants for SpliceVarDB evaluation.
    
    Returns
    -------
    CanonicalTrainingResult
        Training and evaluation results.
    
    Examples
    --------
    >>> result = run_canonical_training(epochs=10, max_train_samples=10000)
    >>> print(f"Test PR-AUC: {result.canonical_test_metrics['pr_auc_macro']:.4f}")
    """
    config = CanonicalTrainingConfig(
        base_model=base_model,
        epochs=epochs,
        output_dir=output_dir,
        sequence_encoder=sequence_encoder,
        max_train_samples=max_train_samples,
        eval_max_variants=eval_max_variants
    )
    
    workflow = CanonicalTrainingWorkflow(config)
    return workflow.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Canonical Splice Site Classification Training'
    )
    parser.add_argument('--base-model', default='openspliceai', help='Base model')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--output-dir', default='./canonical_training_output', 
                        help='Output directory')
    parser.add_argument('--sequence-encoder', default='cnn', help='Sequence encoder')
    parser.add_argument('--max-train-samples', type=int, help='Max training samples')
    parser.add_argument('--eval-max-variants', type=int, default=1000, 
                        help='Max eval variants')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    result = run_canonical_training(
        base_model=args.base_model,
        epochs=args.epochs,
        output_dir=args.output_dir,
        sequence_encoder=args.sequence_encoder,
        max_train_samples=args.max_train_samples,
        eval_max_variants=args.eval_max_variants
    )
    
    print("\n✅ Canonical training complete!")
    print(f"Results saved to: {args.output_dir}")












