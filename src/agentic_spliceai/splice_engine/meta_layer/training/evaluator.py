"""
Evaluation metrics for meta-layer model.

Computes comprehensive metrics including:
- PR-AUC (primary metric)
- ROC-AUC
- Average Precision (AP)
- Top-k accuracy
- Per-class metrics
- Calibration metrics
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    # Primary metrics
    accuracy: float = 0.0
    pr_auc_macro: float = 0.0
    roc_auc_macro: float = 0.0
    average_precision_macro: float = 0.0
    
    # Per-class metrics
    per_class_pr_auc: Optional[List[float]] = None
    per_class_roc_auc: Optional[List[float]] = None
    per_class_average_precision: Optional[List[float]] = None
    per_class_f1: Optional[List[float]] = None
    
    # Top-k accuracy
    top_1_accuracy: float = 0.0
    top_2_accuracy: float = 0.0
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    
    # Calibration
    expected_calibration_error: float = 0.0
    
    # FP/FN analysis
    false_positive_rate: Optional[Dict[str, float]] = None
    false_negative_rate: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'accuracy': self.accuracy,
            'pr_auc_macro': self.pr_auc_macro,
            'roc_auc_macro': self.roc_auc_macro,
            'average_precision_macro': self.average_precision_macro,
            'top_1_accuracy': self.top_1_accuracy,
            'top_2_accuracy': self.top_2_accuracy,
            'expected_calibration_error': self.expected_calibration_error,
        }
        
        # Add per-class metrics
        for i, name in enumerate(['donor', 'acceptor', 'neither']):
            if self.per_class_pr_auc:
                result[f'pr_auc_{name}'] = self.per_class_pr_auc[i]
            if self.per_class_roc_auc:
                result[f'roc_auc_{name}'] = self.per_class_roc_auc[i]
            if self.per_class_average_precision:
                result[f'ap_{name}'] = self.per_class_average_precision[i]
            if self.per_class_f1:
                result[f'f1_{name}'] = self.per_class_f1[i]
        
        return result


class Evaluator:
    """
    Comprehensive evaluator for meta-layer model.
    
    Examples
    --------
    >>> evaluator = Evaluator()
    >>> result = evaluator.evaluate(predictions, labels, probabilities)
    >>> print(f"PR-AUC: {result.pr_auc_macro:.4f}")
    """
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = ['donor', 'acceptor', 'neither']
    
    def evaluate(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
        compute_detailed: bool = True
    ) -> EvaluationResult:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        predictions : np.ndarray
            Predicted class indices [N].
        labels : np.ndarray
            True class indices [N].
        probabilities : np.ndarray
            Class probabilities [N, num_classes].
        compute_detailed : bool
            Whether to compute detailed metrics (slower).
        
        Returns
        -------
        EvaluationResult
            Comprehensive evaluation results.
        """
        result = EvaluationResult()
        
        # Basic accuracy
        result.accuracy = (predictions == labels).mean()
        result.top_1_accuracy = result.accuracy
        
        # Top-2 accuracy
        result.top_2_accuracy = self._top_k_accuracy(probabilities, labels, k=2)
        
        # Per-class metrics
        result.per_class_pr_auc = []
        result.per_class_roc_auc = []
        result.per_class_average_precision = []
        result.per_class_f1 = []
        
        for i in range(self.num_classes):
            binary_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]
            binary_preds = (predictions == i).astype(int)
            
            # PR-AUC
            pr_auc = self._compute_pr_auc(binary_labels, class_probs)
            result.per_class_pr_auc.append(pr_auc)
            
            # ROC-AUC
            roc_auc = self._compute_roc_auc(binary_labels, class_probs)
            result.per_class_roc_auc.append(roc_auc)
            
            # Average Precision
            ap = self._compute_average_precision(binary_labels, class_probs)
            result.per_class_average_precision.append(ap)
            
            # F1
            f1 = self._compute_f1(binary_labels, binary_preds)
            result.per_class_f1.append(f1)
        
        # Macro averages
        result.pr_auc_macro = np.mean(result.per_class_pr_auc)
        result.roc_auc_macro = np.mean(result.per_class_roc_auc)
        result.average_precision_macro = np.mean(result.per_class_average_precision)
        
        if compute_detailed:
            # Confusion matrix
            result.confusion_matrix = self._compute_confusion_matrix(labels, predictions)
            
            # Calibration error
            result.expected_calibration_error = self._compute_ece(
                probabilities, labels, n_bins=10
            )
            
            # FP/FN rates
            result.false_positive_rate, result.false_negative_rate = \
                self._compute_fp_fn_rates(labels, predictions)
        
        return result
    
    def _top_k_accuracy(
        self, 
        probabilities: np.ndarray, 
        labels: np.ndarray, 
        k: int
    ) -> float:
        """Compute top-k accuracy."""
        top_k_preds = np.argsort(probabilities, axis=1)[:, -k:]
        correct = np.any(top_k_preds == labels.reshape(-1, 1), axis=1)
        return correct.mean()
    
    def _compute_pr_auc(
        self, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> float:
        """Compute PR-AUC for binary classification."""
        try:
            from sklearn.metrics import precision_recall_curve, auc
            
            # Handle edge case: all same label
            if len(np.unique(labels)) < 2:
                return 0.5
            
            precision, recall, _ = precision_recall_curve(labels, scores)
            return auc(recall, precision)
        except ImportError:
            return self._compute_pr_auc_manual(labels, scores)
    
    def _compute_pr_auc_manual(
        self, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> float:
        """Manual PR-AUC computation without sklearn."""
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        labels_sorted = labels[sorted_indices]
        
        # Compute precision and recall at each threshold
        tp_cumsum = np.cumsum(labels_sorted)
        fp_cumsum = np.cumsum(1 - labels_sorted)
        
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recall = tp_cumsum / (labels_sorted.sum() + 1e-10)
        
        # Compute AUC using trapezoidal rule
        auc_value = np.trapz(precision, recall)
        
        return abs(auc_value)
    
    def _compute_roc_auc(
        self, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> float:
        """Compute ROC-AUC for binary classification."""
        try:
            from sklearn.metrics import roc_auc_score
            
            if len(np.unique(labels)) < 2:
                return 0.5
            
            return roc_auc_score(labels, scores)
        except ImportError:
            return self._compute_roc_auc_manual(labels, scores)
    
    def _compute_roc_auc_manual(
        self, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> float:
        """Manual ROC-AUC computation."""
        n_pos = labels.sum()
        n_neg = len(labels) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        sorted_indices = np.argsort(scores)[::-1]
        labels_sorted = labels[sorted_indices]
        
        # Compute TPR and FPR
        tp = np.cumsum(labels_sorted)
        fp = np.cumsum(1 - labels_sorted)
        
        tpr = tp / n_pos
        fpr = fp / n_neg
        
        # AUC
        auc_value = np.trapz(tpr, fpr)
        
        return abs(auc_value)
    
    def _compute_average_precision(
        self, 
        labels: np.ndarray, 
        scores: np.ndarray
    ) -> float:
        """Compute Average Precision."""
        try:
            from sklearn.metrics import average_precision_score
            
            if len(np.unique(labels)) < 2:
                return labels.mean()  # Return base rate
            
            return average_precision_score(labels, scores)
        except ImportError:
            # Fallback: approximate AP
            return self._compute_pr_auc_manual(labels, scores)
    
    def _compute_f1(
        self, 
        labels: np.ndarray, 
        predictions: np.ndarray
    ) -> float:
        """Compute F1 score."""
        tp = np.sum((labels == 1) & (predictions == 1))
        fp = np.sum((labels == 0) & (predictions == 1))
        fn = np.sum((labels == 1) & (predictions == 0))
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f1
    
    def _compute_confusion_matrix(
        self, 
        labels: np.ndarray, 
        predictions: np.ndarray
    ) -> np.ndarray:
        """Compute confusion matrix."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        
        for true_label in range(self.num_classes):
            for pred_label in range(self.num_classes):
                cm[true_label, pred_label] = np.sum(
                    (labels == true_label) & (predictions == pred_label)
                )
        
        return cm
    
    def _compute_ece(
        self, 
        probabilities: np.ndarray, 
        labels: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        Measures how well predicted probabilities match observed frequencies.
        """
        # Get max probability and prediction
        max_probs = probabilities.max(axis=1)
        predictions = probabilities.argmax(axis=1)
        
        # Bin by confidence
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            mask = (max_probs > lower) & (max_probs <= upper)
            bin_size = mask.sum()
            
            if bin_size > 0:
                bin_accuracy = (predictions[mask] == labels[mask]).mean()
                bin_confidence = max_probs[mask].mean()
                ece += (bin_size / len(labels)) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _compute_fp_fn_rates(
        self, 
        labels: np.ndarray, 
        predictions: np.ndarray
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute FP and FN rates per class."""
        fp_rates = {}
        fn_rates = {}
        
        for i, name in enumerate(self.class_names):
            binary_labels = (labels == i)
            binary_preds = (predictions == i)
            
            # False positives: predicted positive but actually negative
            fp = np.sum((~binary_labels) & binary_preds)
            n_neg = np.sum(~binary_labels)
            fp_rates[name] = fp / (n_neg + 1e-10)
            
            # False negatives: predicted negative but actually positive
            fn = np.sum(binary_labels & (~binary_preds))
            n_pos = np.sum(binary_labels)
            fn_rates[name] = fn / (n_pos + 1e-10)
        
        return fp_rates, fn_rates
    
    def compare_models(
        self,
        base_result: EvaluationResult,
        meta_result: EvaluationResult
    ) -> Dict[str, float]:
        """
        Compare base model and meta-layer performance.
        
        Returns improvement metrics.
        """
        improvements = {
            'pr_auc_improvement': meta_result.pr_auc_macro - base_result.pr_auc_macro,
            'roc_auc_improvement': meta_result.roc_auc_macro - base_result.roc_auc_macro,
            'accuracy_improvement': meta_result.accuracy - base_result.accuracy,
            'ece_improvement': base_result.expected_calibration_error - meta_result.expected_calibration_error,
        }
        
        # Per-class improvements
        for i, name in enumerate(self.class_names):
            if base_result.per_class_pr_auc and meta_result.per_class_pr_auc:
                improvements[f'pr_auc_{name}_improvement'] = \
                    meta_result.per_class_pr_auc[i] - base_result.per_class_pr_auc[i]
        
        return improvements












