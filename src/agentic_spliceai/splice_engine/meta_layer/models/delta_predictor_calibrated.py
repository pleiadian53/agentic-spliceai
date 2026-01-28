"""
Calibrated Delta Predictors with Multiple Improvement Options.

This module implements 4 calibration strategies to improve the
Gated CNN delta predictor (which achieved r=0.36 but poor calibration).

Options:
1. Output Scaling - Simple multiplicative scaling
2. Temperature Scaling - Learnable temperature parameter
3. Quantile Regression - Predict quantiles instead of mean
4. Classification + Regression Hybrid - Multi-task learning

Usage:
    from agentic_spliceai.splice_engine.meta_layer.models.delta_predictor_calibrated import (
        ScaledDeltaPredictor,
        TemperatureScaledPredictor,
        QuantileDeltaPredictor,
        HybridDeltaPredictor
    )

Ported from: meta_spliceai/splice_engine/meta_layer/models/delta_predictor_calibrated.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np

from .hyenadna_delta_predictor import SimpleCNNDeltaPredictor


class ScaledDeltaPredictor(nn.Module):
    """
    Option 1: Output Scaling.
    
    Applies a fixed or learnable scale factor to predictions
    to match the target distribution.
    
    Parameters
    ----------
    base_model : nn.Module
        Pre-trained delta predictor
    scale_factor : float
        Initial scale factor (default: 2.5, estimated from training stats)
    learnable : bool
        Whether to make scale factor learnable
    """
    
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        scale_factor: float = 2.5,
        learnable: bool = True,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Base model
        if base_model is None:
            self.base_model = SimpleCNNDeltaPredictor(hidden_dim=hidden_dim)
        else:
            self.base_model = base_model
        
        # Scale factor
        if learnable:
            self.scale = nn.Parameter(torch.tensor(scale_factor))
        else:
            self.register_buffer('scale', torch.tensor(scale_factor))
        
        self.learnable = learnable
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Forward with scaling."""
        delta = self.base_model(ref_seq, alt_seq)
        return delta * self.scale
    
    def get_scale(self) -> float:
        """Get current scale factor."""
        return float(self.scale.item())


class TemperatureScaledPredictor(nn.Module):
    """
    Option 2: Temperature Scaling.
    
    Learns a temperature parameter that adjusts prediction confidence.
    Unlike simple scaling, temperature affects the distribution shape.
    
    Parameters
    ----------
    base_model : nn.Module
        Pre-trained delta predictor
    initial_temperature : float
        Initial temperature value
    """
    
    def __init__(
        self,
        base_model: Optional[nn.Module] = None,
        initial_temperature: float = 0.5,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        if base_model is None:
            self.base_model = SimpleCNNDeltaPredictor(hidden_dim=hidden_dim)
        else:
            self.base_model = base_model
        
        # Temperature (constrained to be positive)
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(initial_temperature))
        )
    
    @property
    def temperature(self) -> torch.Tensor:
        """Get temperature (always positive)."""
        return torch.exp(self.log_temperature)
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Forward with temperature scaling."""
        delta = self.base_model(ref_seq, alt_seq)
        return delta / self.temperature


class QuantileDeltaPredictor(nn.Module):
    """
    Option 3: Quantile Regression.
    
    Predicts multiple quantiles (e.g., 10%, 50%, 90%) instead of mean.
    This provides uncertainty estimates and allows focusing on high deltas.
    
    Parameters
    ----------
    quantiles : list of float
        Quantiles to predict (e.g., [0.1, 0.5, 0.9])
    hidden_dim : int
        Hidden dimension for base model
    """
    
    def __init__(
        self,
        quantiles: Optional[list] = None,
        hidden_dim: int = 128,
        n_layers: int = 6
    ):
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        
        # Shared encoder
        self.encoder = SimpleCNNDeltaPredictor(hidden_dim=hidden_dim, n_layers=n_layers)
        
        # Modify output head to predict multiple quantiles
        # Each quantile predicts [L, 2] deltas
        self.quantile_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2) for _ in quantiles
        ])
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor,
        return_all_quantiles: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns median (0.5 quantile) by default, or all quantiles if requested.
        """
        # Get encoder features
        # Need to access intermediate features, so we modify the flow
        ref_feat = self.encoder.encode(ref_seq)  # [B, L, H]
        alt_feat = self.encoder.encode(alt_seq)
        diff = alt_feat - ref_feat
        
        # Predict each quantile
        quantile_preds = []
        for head in self.quantile_heads:
            q_pred = head(diff)  # [B, L, 2]
            quantile_preds.append(q_pred)
        
        # Stack: [B, L, 2, n_quantiles]
        all_quantiles = torch.stack(quantile_preds, dim=-1)
        
        if return_all_quantiles:
            return all_quantiles
        else:
            # Return median (middle quantile)
            median_idx = len(self.quantiles) // 2
            return quantile_preds[median_idx]
    
    def quantile_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute pinball loss for all quantiles.
        
        Parameters
        ----------
        predictions : torch.Tensor
            All quantile predictions [B, L, 2, n_quantiles]
        targets : torch.Tensor
            Target deltas [B, L, 2]
        weights : torch.Tensor, optional
            Sample weights [B]
        """
        total_loss = 0
        
        for i, tau in enumerate(self.quantiles):
            q_pred = predictions[..., i]  # [B, L, 2]
            error = targets - q_pred
            
            # Pinball loss: asymmetric penalty
            loss = torch.where(
                error >= 0,
                tau * error,
                (tau - 1) * error
            )
            
            loss = loss.mean(dim=[1, 2])  # [B]
            
            if weights is not None:
                loss = loss * weights
            
            total_loss = total_loss + loss.mean()
        
        return total_loss / len(self.quantiles)


class HybridDeltaPredictor(nn.Module):
    """
    Option 4: Classification + Regression Hybrid.
    
    Multi-task model that:
    1. Classifies: "Is this variant splice-altering?" (binary)
    2. Regresses: "What are the delta scores?" (continuous)
    
    The classification task provides additional supervision signal
    and the regression benefits from the learned representations.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    n_layers : int
        Number of encoder layers
    classification_weight : float
        Weight for classification loss (vs regression)
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 6,
        classification_weight: float = 0.5
    ):
        super().__init__()
        
        self.classification_weight = classification_weight
        
        # Shared encoder
        self.encoder = SimpleCNNDeltaPredictor(hidden_dim=hidden_dim, n_layers=n_layers)
        
        # Classification head (global pooling → binary)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global pooling
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Binary: splice-altering?
        )
        
        # Regression head (per-position deltas)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns
        -------
        Tuple
            (delta_pred [B, L, 2], class_logit [B, 1])
        """
        # Encode
        ref_feat = self.encoder.encode(ref_seq)  # [B, L, H]
        alt_feat = self.encoder.encode(alt_seq)
        diff = alt_feat - ref_feat  # [B, L, H]
        
        # Classification (on pooled features)
        # Need [B, H, L] for pooling
        diff_t = diff.permute(0, 2, 1)
        class_logit = self.classifier(diff_t)  # [B, 1]
        
        # Regression (per-position)
        delta_pred = self.regressor(diff)  # [B, L, 2]
        
        return delta_pred, class_logit
    
    def compute_loss(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor,
        delta_target: torch.Tensor,
        is_splice_altering: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Parameters
        ----------
        delta_target : torch.Tensor
            Target deltas [B, L, 2]
        is_splice_altering : torch.Tensor
            Binary labels [B] (1 = splice-altering, 0 = not)
        sample_weights : torch.Tensor, optional
            Sample weights [B]
        """
        delta_pred, class_logit = self.forward(ref_seq, alt_seq)
        
        # Regression loss (MSE)
        reg_loss = F.mse_loss(delta_pred, delta_target, reduction='none')
        reg_loss = reg_loss.mean(dim=[1, 2])  # [B]
        
        if sample_weights is not None:
            reg_loss = reg_loss * sample_weights
        reg_loss = reg_loss.mean()
        
        # Classification loss (BCE)
        class_loss = F.binary_cross_entropy_with_logits(
            class_logit.squeeze(-1),
            is_splice_altering.float(),
            reduction='mean'
        )
        
        # Combined
        total_loss = (
            (1 - self.classification_weight) * reg_loss +
            self.classification_weight * class_loss
        )
        
        return {
            'total': total_loss,
            'regression': reg_loss,
            'classification': class_loss
        }
    
    def predict_delta(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Get just delta predictions (for inference)."""
        delta_pred, _ = self.forward(ref_seq, alt_seq)
        return delta_pred
    
    def predict_splice_altering(
        self,
        ref_seq: torch.Tensor,
        alt_seq: torch.Tensor
    ) -> torch.Tensor:
        """Get splice-altering probability."""
        _, class_logit = self.forward(ref_seq, alt_seq)
        return torch.sigmoid(class_logit)


def create_calibrated_predictor(
    option: str = 'scaled',
    base_model: Optional[nn.Module] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function for calibrated predictors.
    
    Parameters
    ----------
    option : str
        One of: 'scaled', 'temperature', 'quantile', 'hybrid'
    base_model : nn.Module, optional
        Pre-trained base model (for scaled/temperature options)
    **kwargs
        Additional arguments for the specific option
    
    Returns
    -------
    nn.Module
        Calibrated delta predictor
    """
    if option == 'scaled':
        return ScaledDeltaPredictor(base_model=base_model, **kwargs)
    elif option == 'temperature':
        return TemperatureScaledPredictor(base_model=base_model, **kwargs)
    elif option == 'quantile':
        return QuantileDeltaPredictor(**kwargs)
    elif option == 'hybrid':
        return HybridDeltaPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown option: {option}. Use: scaled, temperature, quantile, hybrid")









