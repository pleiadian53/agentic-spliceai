"""
Base model predictor for delta score computation.

Wraps OpenSpliceAI/SpliceAI models to provide a simple interface for:
1. Single sequence prediction
2. Delta score computation (ref vs alt)

Supports both:
- OpenSpliceAI (GRCh38/MANE)
- SpliceAI (GRCh37/Ensembl)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from base model prediction."""
    
    # Raw probabilities [L, 3] where L = sequence length
    probabilities: np.ndarray  # [L, 3] - none, acceptor, donor
    
    # Per-position scores
    acceptor_scores: np.ndarray  # [L]
    donor_scores: np.ndarray  # [L]
    neither_scores: np.ndarray  # [L]
    
    # Sequence info
    sequence_length: int
    
    def get_center_scores(self) -> np.ndarray:
        """Get scores at the center position (for single-position predictions)."""
        center = self.sequence_length // 2
        return np.array([
            self.donor_scores[center],
            self.acceptor_scores[center],
            self.neither_scores[center]
        ])


@dataclass
class DeltaScoreResult:
    """Delta scores between reference and variant sequences."""
    
    # Position-level delta scores [L, 3]
    delta_probabilities: np.ndarray
    
    # Center position delta (for variant at center)
    delta_donor: float
    delta_acceptor: float
    delta_neither: float
    
    # Max delta scores (for splice site gain/loss detection)
    max_donor_gain: float  # max positive delta_donor
    max_donor_loss: float  # max negative delta_donor (as positive value)
    max_acceptor_gain: float
    max_acceptor_loss: float
    
    # Positions of max deltas
    pos_max_donor_gain: int
    pos_max_donor_loss: int
    pos_max_acceptor_gain: int
    pos_max_acceptor_loss: int


class BaseModelPredictor:
    """
    Predictor for OpenSpliceAI/SpliceAI base models.
    
    Provides a simple interface for getting splice predictions and delta scores.
    
    Parameters
    ----------
    model_path : Path or str
        Path to model weights file (.pt for PyTorch, .h5 for Keras).
    flanking_size : int
        Flanking sequence size (80, 400, 2000, or 10000).
    model_type : str
        'pytorch' or 'keras'.
    device : str, optional
        Device for inference ('cuda', 'mps', 'cpu', or 'auto').
    
    Examples
    --------
    >>> predictor = BaseModelPredictor(model_path='models/openspliceai.pt', flanking_size=400)
    >>> 
    >>> # Single sequence prediction
    >>> result = predictor.predict('ACGT...')
    >>> print(result.donor_scores.shape)
    >>> 
    >>> # Delta score computation
    >>> delta = predictor.compute_delta('ACGT...ref...', 'ACGT...alt...')
    >>> print(f"Donor gain: {delta.max_donor_gain:.4f}")
    """
    
    def __init__(
        self,
        model_path: Union[Path, str],
        flanking_size: int = 400,
        model_type: str = 'pytorch',
        device: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.flanking_size = flanking_size
        self.model_type = model_type
        
        # Set device
        if device is None or device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load models
        self.models = self._load_models()
        
        # Compute context length
        self.context_length = self._compute_context_length()
        
        logger.info(f"BaseModelPredictor initialized:")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Flanking size: {self.flanking_size}")
        logger.info(f"  Context length: {self.context_length}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Models loaded: {len(self.models)}")
    
    def _load_models(self) -> List:
        """
        Load model(s) using the existing model loaders.
        """
        try:
            from agentic_spliceai.splice_engine.base_layer.models.loader import load_base_model_ensemble
            
            context = self.flanking_size
            models, metadata = load_base_model_ensemble(
                base_model='openspliceai' if self.model_type == 'pytorch' else 'spliceai',
                context=context,
                device=str(self.device) if self.model_type == 'pytorch' else None,
                verbosity=0
            )
            self._metadata = metadata
            logger.info(f"Loaded {metadata['num_models']} {metadata['base_model']} models")
            return models
        except ImportError as e:
            logger.warning(f"Could not load base model: {e}")
            return []
    
    def _compute_context_length(self) -> int:
        """Compute context length based on flanking size."""
        context_mapping = {
            80: 80,
            400: 400,
            2000: 2000,
            10000: 10000
        }
        return context_mapping.get(self.flanking_size, self.flanking_size)
    
    def predict(self, sequence: str) -> PredictionResult:
        """Get splice predictions for a sequence."""
        x = self._one_hot_encode(sequence)
        x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 2:
            x = x.T.unsqueeze(0)  # [1, 4, L]
        
        x = x.to(self.device)
        
        with torch.no_grad():
            if self.model_type == 'pytorch':
                predictions = torch.mean(
                    torch.stack([m(x).detach().cpu() for m in self.models]),
                    dim=0
                )
                predictions = predictions.permute(0, 2, 1)
            else:
                predictions = np.mean([m.predict(x.cpu().numpy()) for m in self.models], axis=0)
                predictions = torch.tensor(predictions)
        
        probs = predictions.squeeze(0).numpy()
        
        if probs.min() < 0 or probs.max() > 1:
            probs = F.softmax(torch.tensor(probs), dim=-1).numpy()
        
        return PredictionResult(
            probabilities=probs,
            neither_scores=probs[:, 0],
            acceptor_scores=probs[:, 1],
            donor_scores=probs[:, 2],
            sequence_length=len(sequence)
        )
    
    def compute_delta(
        self,
        ref_sequence: str,
        alt_sequence: str,
        variant_position: Optional[int] = None
    ) -> DeltaScoreResult:
        """Compute delta scores between reference and variant sequences."""
        ref_result = self.predict(ref_sequence)
        alt_result = self.predict(alt_sequence)
        
        output_len = len(ref_result.probabilities)
        context_offset = self.context_length // 2
        
        ref_probs = ref_result.probabilities
        alt_probs = alt_result.probabilities
        
        min_len = min(len(ref_probs), len(alt_probs))
        ref_probs = ref_probs[:min_len]
        alt_probs = alt_probs[:min_len]
        
        delta = alt_probs - ref_probs
        
        if variant_position is None:
            variant_position = len(ref_sequence) // 2
        
        output_position = variant_position - context_offset
        output_position = max(0, min(output_position, min_len - 1))
        
        delta_neither = float(delta[output_position, 0])
        delta_acceptor = float(delta[output_position, 1])
        delta_donor = float(delta[output_position, 2])
        
        delta_donor_vec = delta[:, 2]
        delta_acceptor_vec = delta[:, 1]
        
        pos_max_donor_gain = int(np.argmax(delta_donor_vec))
        pos_max_donor_loss = int(np.argmin(delta_donor_vec))
        max_donor_gain = max(0, float(delta_donor_vec[pos_max_donor_gain]))
        max_donor_loss = max(0, float(-delta_donor_vec[pos_max_donor_loss]))
        
        pos_max_acceptor_gain = int(np.argmax(delta_acceptor_vec))
        pos_max_acceptor_loss = int(np.argmin(delta_acceptor_vec))
        max_acceptor_gain = max(0, float(delta_acceptor_vec[pos_max_acceptor_gain]))
        max_acceptor_loss = max(0, float(-delta_acceptor_vec[pos_max_acceptor_loss]))
        
        return DeltaScoreResult(
            delta_probabilities=delta,
            delta_donor=delta_donor,
            delta_acceptor=delta_acceptor,
            delta_neither=delta_neither,
            max_donor_gain=max_donor_gain,
            max_donor_loss=max_donor_loss,
            max_acceptor_gain=max_acceptor_gain,
            max_acceptor_loss=max_acceptor_loss,
            pos_max_donor_gain=pos_max_donor_gain,
            pos_max_donor_loss=pos_max_donor_loss,
            pos_max_acceptor_gain=pos_max_acceptor_gain,
            pos_max_acceptor_loss=pos_max_acceptor_loss
        )
    
    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode DNA sequence."""
        mapping = np.array([
            [0, 0, 0, 0],  # N
            [1, 0, 0, 0],  # A
            [0, 1, 0, 0],  # C
            [0, 0, 1, 0],  # G
            [0, 0, 0, 1],  # T
        ])
        
        seq = sequence.upper()
        seq = seq.replace('A', '\x01').replace('C', '\x02')
        seq = seq.replace('G', '\x03').replace('T', '\x04').replace('N', '\x00')
        
        indices = np.frombuffer(seq.encode('latin-1'), dtype=np.int8) % 5
        
        return mapping[indices]


def get_base_model_predictor(
    base_model: str,
    flanking_size: int = 10000,
    device: Optional[str] = None
) -> BaseModelPredictor:
    """
    Get a base model predictor for the specified model.
    
    Parameters
    ----------
    base_model : str
        Base model name ('openspliceai' or 'spliceai').
    flanking_size : int
        Flanking sequence size (default 10000 for best accuracy).
    device : str, optional
        Device for inference.
    
    Returns
    -------
    BaseModelPredictor
        Configured predictor instance.
    """
    base_model_lower = base_model.lower()
    if base_model_lower == 'openspliceai':
        model_type = 'pytorch'
    elif base_model_lower == 'spliceai':
        model_type = 'keras'
    else:
        raise ValueError(
            f"Unknown base model: {base_model}. "
            f"Supported: ['openspliceai', 'spliceai']"
        )
    
    model_path = Path(f"data/models/{base_model_lower}")
    
    return BaseModelPredictor(
        model_path=model_path,
        flanking_size=flanking_size,
        model_type=model_type,
        device=device
    )












