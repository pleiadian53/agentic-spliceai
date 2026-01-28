"""
Inference pipeline for meta-layer model.

Provides:
- Batch prediction on DNA sequences
- Delta score computation (meta vs base)
- Splice site calling and filtering
- Confidence scoring
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from agentic_spliceai.splice_engine.meta_layer.models import MetaSpliceModel
from agentic_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig
from agentic_spliceai.splice_engine.meta_layer.core.feature_schema import LABEL_DECODING

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of splice site prediction."""
    # Per-position predictions
    positions: np.ndarray  # Genomic positions
    donor_scores: np.ndarray  # Donor probability
    acceptor_scores: np.ndarray  # Acceptor probability
    neither_scores: np.ndarray  # Neither probability
    
    # Predicted class
    predicted_class: np.ndarray  # 0=donor, 1=acceptor, 2=neither
    confidence: np.ndarray  # Max probability
    
    # Delta from base model (optional)
    donor_delta: Optional[np.ndarray] = None
    acceptor_delta: Optional[np.ndarray] = None
    
    # Metadata
    chromosome: Optional[str] = None
    strand: Optional[str] = None
    
    def to_dataframe(self) -> pl.DataFrame:
        """Convert to Polars DataFrame."""
        data = {
            'position': self.positions,
            'donor_score': self.donor_scores,
            'acceptor_score': self.acceptor_scores,
            'neither_score': self.neither_scores,
            'predicted_class': self.predicted_class,
            'predicted_label': [LABEL_DECODING[c] for c in self.predicted_class],
            'confidence': self.confidence,
        }
        
        if self.donor_delta is not None:
            data['donor_delta'] = self.donor_delta
        if self.acceptor_delta is not None:
            data['acceptor_delta'] = self.acceptor_delta
        if self.chromosome is not None:
            data['chromosome'] = [self.chromosome] * len(self.positions)
        if self.strand is not None:
            data['strand'] = [self.strand] * len(self.positions)
        
        return pl.DataFrame(data)


class MetaLayerPredictor:
    """
    Predictor for splice site classification using the meta-layer model.
    
    Examples
    --------
    >>> predictor = MetaLayerPredictor.from_checkpoint('models/best_model.pt')
    >>> result = predictor.predict(sequences, features)
    >>> 
    >>> # Get high-confidence splice sites
    >>> splice_sites = predictor.call_splice_sites(result, threshold=0.8)
    """
    
    def __init__(
        self,
        model: MetaSpliceModel,
        config: MetaLayerConfig,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"MetaLayerPredictor initialized on {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[MetaLayerConfig] = None
    ) -> 'MetaLayerPredictor':
        """
        Load predictor from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : str or Path
            Path to model checkpoint.
        config : MetaLayerConfig, optional
            Configuration. If None, loaded from checkpoint.
        
        Returns
        -------
        MetaLayerPredictor
            Loaded predictor.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get config
        if config is None:
            if 'config' in checkpoint:
                config = checkpoint['config']
            else:
                config = MetaLayerConfig()
                logger.warning("No config in checkpoint, using defaults")
        
        # Create model
        state_dict = checkpoint['model_state_dict']
        
        # Infer num_features from score encoder
        if 'score_encoder.encoder.0.weight' in state_dict:
            num_features = state_dict['score_encoder.encoder.0.weight'].shape[1]
        else:
            num_features = 50  # Default
            logger.warning(f"Could not infer num_features, using {num_features}")
        
        model = MetaSpliceModel(
            sequence_encoder=config.sequence_encoder,
            num_score_features=num_features,
            hidden_dim=config.hidden_dim
        )
        
        model.load_state_dict(state_dict)
        
        return cls(model, config)
    
    @torch.no_grad()
    def predict(
        self,
        sequences: torch.Tensor,
        features: torch.Tensor,
        batch_size: int = 256,
        return_delta: bool = False,
        base_predictions: Optional[torch.Tensor] = None
    ) -> PredictionResult:
        """
        Predict splice site probabilities.
        
        Parameters
        ----------
        sequences : torch.Tensor
            DNA sequences [N, 4, L] (one-hot) or [N, L] (tokens).
        features : torch.Tensor
            Score features [N, num_features].
        batch_size : int
            Batch size for prediction.
        return_delta : bool
            Whether to compute delta from base predictions.
        base_predictions : torch.Tensor, optional
            Base model predictions [N, 3] for delta computation.
        
        Returns
        -------
        PredictionResult
            Prediction results.
        """
        n_samples = len(sequences)
        all_probs = []
        
        for i in range(0, n_samples, batch_size):
            batch_seq = sequences[i:i+batch_size].to(self.device)
            batch_feat = features[i:i+batch_size].to(self.device)
            
            logits = self.model(batch_seq, batch_feat)
            probs = F.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu())
        
        probs = torch.cat(all_probs, dim=0).numpy()
        
        # Extract scores
        donor_scores = probs[:, 0]
        acceptor_scores = probs[:, 1]
        neither_scores = probs[:, 2]
        
        # Get predictions and confidence
        predicted_class = probs.argmax(axis=1)
        confidence = probs.max(axis=1)
        
        # Positions (placeholder)
        positions = np.arange(n_samples)
        
        # Delta computation
        donor_delta = None
        acceptor_delta = None
        
        if return_delta and base_predictions is not None:
            base_probs = base_predictions.numpy() if torch.is_tensor(base_predictions) else base_predictions
            donor_delta = donor_scores - base_probs[:, 0]
            acceptor_delta = acceptor_scores - base_probs[:, 1]
        
        return PredictionResult(
            positions=positions,
            donor_scores=donor_scores,
            acceptor_scores=acceptor_scores,
            neither_scores=neither_scores,
            predicted_class=predicted_class,
            confidence=confidence,
            donor_delta=donor_delta,
            acceptor_delta=acceptor_delta
        )
    
    def predict_dataframe(
        self,
        df: pl.DataFrame,
        sequence_col: str = 'sequence',
        batch_size: int = 256
    ) -> pl.DataFrame:
        """
        Predict from DataFrame with sequences and features.
        
        Parameters
        ----------
        df : pl.DataFrame
            Input DataFrame with sequence and feature columns.
        sequence_col : str
            Name of sequence column.
        batch_size : int
            Batch size for prediction.
        
        Returns
        -------
        pl.DataFrame
            DataFrame with predictions added.
        """
        from agentic_spliceai.splice_engine.meta_layer.data import MetaLayerDataset
        
        # Create dataset
        dataset = MetaLayerDataset(df, max_seq_length=self.config.max_seq_length)
        
        # Collect all data
        sequences = []
        features = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            sequences.append(sample['sequence'])
            features.append(sample['features'])
        
        sequences = torch.stack(sequences)
        features = torch.stack(features)
        
        # Predict
        result = self.predict(sequences, features, batch_size=batch_size)
        
        # Add to DataFrame
        df = df.with_columns([
            pl.Series('meta_donor_score', result.donor_scores),
            pl.Series('meta_acceptor_score', result.acceptor_scores),
            pl.Series('meta_neither_score', result.neither_scores),
            pl.Series('meta_predicted_class', result.predicted_class),
            pl.Series('meta_predicted_label', [LABEL_DECODING[c] for c in result.predicted_class]),
            pl.Series('meta_confidence', result.confidence),
        ])
        
        return df
    
    def call_splice_sites(
        self,
        result: PredictionResult,
        donor_threshold: float = 0.5,
        acceptor_threshold: float = 0.5,
        min_confidence: float = 0.0
    ) -> Dict[str, np.ndarray]:
        """
        Call splice sites from prediction results.
        
        Parameters
        ----------
        result : PredictionResult
            Prediction results.
        donor_threshold : float
            Minimum donor probability to call a donor site.
        acceptor_threshold : float
            Minimum acceptor probability to call an acceptor site.
        min_confidence : float
            Minimum confidence to include in results.
        
        Returns
        -------
        dict
            Dictionary with 'donors' and 'acceptors' arrays of positions.
        """
        # Apply thresholds
        donor_mask = (
            (result.donor_scores >= donor_threshold) & 
            (result.confidence >= min_confidence)
        )
        acceptor_mask = (
            (result.acceptor_scores >= acceptor_threshold) & 
            (result.confidence >= min_confidence)
        )
        
        return {
            'donors': result.positions[donor_mask],
            'acceptors': result.positions[acceptor_mask],
            'donor_scores': result.donor_scores[donor_mask],
            'acceptor_scores': result.acceptor_scores[acceptor_mask],
        }


class DeltaScorer:
    """
    Computes and analyzes delta scores between meta and base models.
    
    Delta scores help identify where the meta-layer corrects base model errors.
    """
    
    def __init__(self, predictor: MetaLayerPredictor):
        self.predictor = predictor
    
    def compute_deltas(
        self,
        sequences: torch.Tensor,
        features: torch.Tensor,
        base_predictions: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute delta scores.
        
        Parameters
        ----------
        sequences : torch.Tensor
            DNA sequences.
        features : torch.Tensor
            Score features.
        base_predictions : np.ndarray
            Base model predictions [N, 3].
        
        Returns
        -------
        dict
            Delta scores and analysis.
        """
        # Get meta predictions
        result = self.predictor.predict(
            sequences, 
            features, 
            return_delta=True,
            base_predictions=base_predictions
        )
        
        # Compute various delta metrics
        donor_delta = result.donor_scores - base_predictions[:, 0]
        acceptor_delta = result.acceptor_scores - base_predictions[:, 1]
        
        return {
            'donor_delta': donor_delta,
            'acceptor_delta': acceptor_delta,
            'donor_delta_abs': np.abs(donor_delta),
            'acceptor_delta_abs': np.abs(acceptor_delta),
            'max_delta': np.maximum(np.abs(donor_delta), np.abs(acceptor_delta)),
            'meta_confidence': result.confidence,
            'base_confidence': base_predictions.max(axis=1),
            'confidence_change': result.confidence - base_predictions.max(axis=1),
        }
    
    def identify_corrections(
        self,
        deltas: Dict[str, np.ndarray],
        labels: np.ndarray,
        base_predictions: np.ndarray,
        delta_threshold: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Identify positions where meta-layer corrects base model errors.
        
        Parameters
        ----------
        deltas : dict
            Delta scores from compute_deltas.
        labels : np.ndarray
            True labels.
        base_predictions : np.ndarray
            Base model predictions.
        delta_threshold : float
            Minimum delta to consider a correction.
        
        Returns
        -------
        dict
            Correction analysis.
        """
        base_preds = base_predictions.argmax(axis=1)
        
        # Find base model errors
        base_errors = base_preds != labels
        
        # Find significant changes
        significant_delta = deltas['max_delta'] >= delta_threshold
        
        # Corrections: base was wrong, meta changes significantly
        correction_candidates = base_errors & significant_delta
        
        return {
            'base_error_count': base_errors.sum(),
            'correction_candidate_count': correction_candidates.sum(),
            'correction_candidate_positions': np.where(correction_candidates)[0],
            'donor_correction_magnitude': deltas['donor_delta'][correction_candidates].mean() if correction_candidates.any() else 0,
            'acceptor_correction_magnitude': deltas['acceptor_delta'][correction_candidates].mean() if correction_candidates.any() else 0,
        }












