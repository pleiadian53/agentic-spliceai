"""
Variant-effect evaluator for Phase 1 (Approach A).

Evaluates whether the meta-layer improves variant effect detection
by comparing delta scores (variant vs reference) between:
1. Base model only
2. Base model + meta-layer

Uses SpliceVarDB as ground truth for whether variants affect splicing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from tqdm import tqdm

from agentic_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig
from agentic_spliceai.splice_engine.meta_layer.data.splicevardb_loader import (
    SpliceVarDBLoader, VariantRecord, load_splicevardb
)
from agentic_spliceai.splice_engine.meta_layer.models import MetaSpliceModel

logger = logging.getLogger(__name__)


@dataclass
class DeltaResult:
    """Delta scores for a single variant."""
    
    variant_id: int
    chrom: str
    position: int
    classification: str
    
    # Base model deltas
    base_delta_donor: float
    base_delta_acceptor: float
    base_delta_neither: float
    
    # Meta-layer deltas
    meta_delta_donor: float
    meta_delta_acceptor: float
    meta_delta_neither: float
    
    # Maximum absolute deltas
    base_max_delta: float = field(init=False)
    meta_max_delta: float = field(init=False)
    
    def __post_init__(self):
        self.base_max_delta = max(
            abs(self.base_delta_donor),
            abs(self.base_delta_acceptor)
        )
        self.meta_max_delta = max(
            abs(self.meta_delta_donor),
            abs(self.meta_delta_acceptor)
        )
    
    @property
    def is_splice_altering(self) -> bool:
        return self.classification == "Splice-altering"
    
    @property
    def base_detects_effect(self) -> bool:
        """Whether base model detects a significant effect."""
        return self.base_max_delta > 0.1
    
    @property
    def meta_detects_effect(self) -> bool:
        """Whether meta-layer detects a significant effect."""
        return self.meta_max_delta > 0.1
    
    @property
    def meta_improves(self) -> bool:
        """Whether meta-layer detects an effect that base model missed."""
        return self.meta_detects_effect and not self.base_detects_effect


@dataclass
class VariantEvaluationResult:
    """Aggregated evaluation results."""
    
    total_variants: int
    
    # Accuracy metrics
    base_accuracy: float  # Fraction where base delta matches classification
    meta_accuracy: float  # Fraction where meta delta matches classification
    
    # Detection rates for splice-altering variants
    base_detection_rate: float
    meta_detection_rate: float
    
    # False positive rates for non-splice-altering variants
    base_false_positive_rate: float
    meta_false_positive_rate: float
    
    # Improvement metrics
    improvement_count: int  # Cases where meta detected but base didn't
    degradation_count: int  # Cases where base detected but meta didn't
    
    # Per-classification metrics
    per_classification: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Delta statistics
    mean_base_delta_splice_altering: float = 0.0
    mean_meta_delta_splice_altering: float = 0.0
    mean_base_delta_non_splice_altering: float = 0.0
    mean_meta_delta_non_splice_altering: float = 0.0
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "VARIANT EFFECT EVALUATION RESULTS",
            "=" * 60,
            f"Total variants evaluated: {self.total_variants}",
            "",
            "DETECTION ACCURACY:",
            f"  Base model:  {self.base_accuracy:.1%}",
            f"  Meta-layer:  {self.meta_accuracy:.1%}",
            f"  Improvement: {self.meta_accuracy - self.base_accuracy:+.1%}",
            "",
            "SPLICE-ALTERING DETECTION:",
            f"  Base model:  {self.base_detection_rate:.1%}",
            f"  Meta-layer:  {self.meta_detection_rate:.1%}",
            "",
            "FALSE POSITIVE RATE (non-splice-altering):",
            f"  Base model:  {self.base_false_positive_rate:.1%}",
            f"  Meta-layer:  {self.meta_false_positive_rate:.1%}",
            "",
            "IMPROVEMENT ANALYSIS:",
            f"  Meta improved: {self.improvement_count} variants",
            f"  Meta degraded: {self.degradation_count} variants",
            f"  Net improvement: {self.improvement_count - self.degradation_count}",
            "",
            "MEAN DELTA MAGNITUDES:",
            f"  Splice-altering (base):      {self.mean_base_delta_splice_altering:.3f}",
            f"  Splice-altering (meta):      {self.mean_meta_delta_splice_altering:.3f}",
            f"  Non-splice-altering (base):  {self.mean_base_delta_non_splice_altering:.3f}",
            f"  Non-splice-altering (meta):  {self.mean_meta_delta_non_splice_altering:.3f}",
            "=" * 60,
        ]
        return "\n".join(lines)


class VariantEffectEvaluator:
    """
    Evaluates meta-layer's ability to detect variant effects.
    
    Approach A evaluation:
    1. For each SpliceVarDB variant, generate ref and alt sequences
    2. Run base model on both → compute base_delta
    3. Run meta-layer on both → compute meta_delta
    4. Compare deltas with SpliceVarDB classification
    
    Supports both:
    - OpenSpliceAI (GRCh38/MANE) - uses hg38 coordinates from SpliceVarDB
    - SpliceAI (GRCh37/Ensembl) - uses hg19 coordinates from SpliceVarDB
    
    Parameters
    ----------
    meta_model : MetaSpliceModel
        Trained meta-layer model.
    config : MetaLayerConfig
        Configuration with base model and genome info.
    device : str, optional
        Device for inference.
    genome_build : str, optional
        Override genome build ('GRCh37' or 'GRCh38'). If None, inferred from base model.
    
    Examples
    --------
    >>> evaluator = VariantEffectEvaluator(meta_model, config)
    >>> results = evaluator.evaluate(test_variants)
    >>> print(results.summary())
    """
    
    def __init__(
        self,
        meta_model: MetaSpliceModel,
        config: MetaLayerConfig,
        device: Optional[str] = None,
        detection_threshold: float = 0.1,
        genome_build: Optional[str] = None
    ):
        self.meta_model = meta_model
        self.config = config
        self.detection_threshold = detection_threshold
        
        # Determine genome build from config or override
        if genome_build:
            self.genome_build = genome_build
        else:
            # Infer from base model
            if config.base_model.lower() == 'spliceai':
                self.genome_build = 'GRCh37'
            else:
                self.genome_build = 'GRCh38'
        
        logger.info(f"Using genome build: {self.genome_build} for base model: {config.base_model}")
        
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        self.meta_model.to(self.device)
        self.meta_model.eval()
        
        # Load FASTA reference (lazy)
        self._fasta = None
        self._registry = None
        
        # Load base model predictor (lazy)
        self._base_predictor = None
        
        logger.info(f"VariantEffectEvaluator initialized on {self.device}")
    
    @property
    def base_predictor(self):
        """Lazy load base model predictor."""
        if self._base_predictor is None:
            try:
                from agentic_spliceai.splice_engine.meta_layer.inference.base_model_predictor import (
                    get_base_model_predictor
                )
                self._base_predictor = get_base_model_predictor(
                    base_model=self.config.base_model,
                    flanking_size=400,  # Default flanking size
                    device=str(self.device)
                )
                logger.info(f"Loaded base model predictor for {self.config.base_model}")
            except Exception as e:
                logger.warning(f"Could not load base model predictor: {e}")
                self._base_predictor = None
        return self._base_predictor
    
    @property
    def fasta(self):
        """
        Lazy load FASTA reference using pyfaidx.
        
        Uses the correct genome build based on base model:
        - OpenSpliceAI → GRCh38
        - SpliceAI → GRCh37
        """
        if self._fasta is None:
            try:
                from pyfaidx import Fasta
                # Try common paths based on build
                from pathlib import Path
                
                if self.genome_build == 'GRCh38':
                    common_paths = [
                        Path.home() / "reference/GRCh38/GRCh38.primary_assembly.genome.fa",
                        Path("/data/reference/GRCh38.fa"),
                        Path.home() / "reference/hg38/hg38.fa",
                    ]
                else:  # GRCh37
                    common_paths = [
                        Path.home() / "reference/GRCh37/GRCh37.primary_assembly.genome.fa",
                        Path("/data/reference/GRCh37.fa"),
                        Path.home() / "reference/hg19/hg19.fa",
                    ]
                
                for path in common_paths:
                    if path.exists():
                        self._fasta = Fasta(str(path), sequence_always_upper=True, rebuild=False)
                        logger.info(f"Loaded FASTA for {self.genome_build} from: {path}")
                        break
                else:
                    raise FileNotFoundError(
                        f"Could not find FASTA reference file for {self.genome_build}. "
                        f"Please place FASTA in one of: {common_paths}"
                    )
            except Exception as e:
                logger.warning(f"Could not load FASTA: {e}")
                self._fasta = None
        return self._fasta
    
    def evaluate(
        self,
        variants: List[VariantRecord],
        window_size: int = 501,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> VariantEvaluationResult:
        """
        Evaluate meta-layer on a set of variants.
        
        Parameters
        ----------
        variants : list of VariantRecord
            Variants to evaluate.
        window_size : int
            Sequence window size (should match training).
        batch_size : int
            Batch size for inference.
        show_progress : bool
            Whether to show progress bar.
        
        Returns
        -------
        VariantEvaluationResult
            Evaluation results.
        """
        delta_results = []
        
        iterator = tqdm(variants, desc="Evaluating variants") if show_progress else variants
        
        for variant in iterator:
            try:
                result = self._evaluate_single_variant(variant, window_size)
                if result is not None:
                    delta_results.append(result)
            except Exception as e:
                logger.debug(f"Failed to evaluate variant {variant.variant_id}: {e}")
                continue
        
        if not delta_results:
            raise ValueError("No variants could be evaluated")
        
        return self._aggregate_results(delta_results)
    
    def _evaluate_single_variant(
        self,
        variant: VariantRecord,
        window_size: int
    ) -> Optional[DeltaResult]:
        """
        Evaluate a single variant.
        """
        half_window = window_size // 2
        
        try:
            # Normalize chromosome name to match FASTA
            chrom = self._normalise_chrom(variant.chrom)
            
            # Fetch reference sequence (pyfaidx uses 0-based, half-open intervals)
            start = variant.position - half_window - 1  # Convert to 0-based
            end = variant.position + half_window
            
            ref_seq = str(self.fasta[chrom][start:end].seq)
            
            if len(ref_seq) != window_size:
                logger.debug(f"Sequence length mismatch for {variant.variant_id}")
                return None
            
            # Verify reference allele matches
            ref_in_seq = ref_seq[half_window:half_window + len(variant.ref_allele)]
            if ref_in_seq.upper() != variant.ref_allele.upper():
                logger.debug(f"Reference mismatch for {variant.variant_id}")
                return None
            
            # Create variant sequence
            alt_seq = (
                ref_seq[:half_window] +
                variant.alt_allele +
                ref_seq[half_window + len(variant.ref_allele):]
            )
            
            # Handle length differences for indels
            if len(alt_seq) < window_size:
                alt_seq = alt_seq + 'N' * (window_size - len(alt_seq))
            elif len(alt_seq) > window_size:
                excess = len(alt_seq) - window_size
                alt_seq = alt_seq[excess // 2 : -(excess - excess // 2)] if excess > 1 else alt_seq[:-1]
            
        except Exception as e:
            logger.debug(f"Sequence extraction failed for {variant.variant_id}: {e}")
            return None
        
        # Encode sequences
        ref_encoded = self._encode_sequence(ref_seq)
        alt_encoded = self._encode_sequence(alt_seq)
        
        # Get feature dimension from model
        try:
            num_features = self.meta_model.score_encoder.encoder[0].in_features
        except AttributeError:
            num_features = 43  # Default
        
        # Use zero features (placeholder)
        dummy_features = torch.zeros(1, num_features)
        
        with torch.no_grad():
            ref_encoded = ref_encoded.to(self.device)
            alt_encoded = alt_encoded.to(self.device)
            dummy_features = dummy_features.to(self.device)
            
            # Meta-layer predictions
            ref_logits = self.meta_model(ref_encoded, dummy_features)
            alt_logits = self.meta_model(alt_encoded, dummy_features)
            
            ref_probs = F.softmax(ref_logits, dim=-1)
            alt_probs = F.softmax(alt_logits, dim=-1)
            
            meta_delta = (alt_probs - ref_probs).squeeze().cpu().numpy()
        
        # Compute base model delta
        base_delta = self._compute_base_model_delta(ref_seq, alt_seq)
        
        return DeltaResult(
            variant_id=variant.variant_id,
            chrom=variant.chrom,
            position=variant.position,
            classification=variant.classification,
            base_delta_donor=base_delta[0],
            base_delta_acceptor=base_delta[1],
            base_delta_neither=base_delta[2],
            meta_delta_donor=meta_delta[0],
            meta_delta_acceptor=meta_delta[1],
            meta_delta_neither=meta_delta[2]
        )
    
    def _normalise_chrom(self, chrom: str) -> str:
        """Normalize chromosome name to match FASTA."""
        if self.fasta is None:
            return chrom
            
        fasta_chroms = list(self.fasta.keys())
        if not fasta_chroms:
            return chrom
        
        target = fasta_chroms[0]
        has_chr_prefix = target.startswith('chr')
        
        if has_chr_prefix and not chrom.startswith('chr'):
            return 'chr' + chrom
        elif not has_chr_prefix and chrom.startswith('chr'):
            return chrom[3:]
        
        return chrom
    
    def _compute_base_model_delta(
        self,
        ref_seq: str,
        alt_seq: str
    ) -> np.ndarray:
        """Compute base model delta scores."""
        if self.base_predictor is None:
            return np.zeros(3)
        
        try:
            delta_result = self.base_predictor.compute_delta(ref_seq, alt_seq)
            return np.array([
                delta_result.delta_donor,
                delta_result.delta_acceptor,
                delta_result.delta_neither
            ])
        except Exception as e:
            logger.debug(f"Base model delta computation failed: {e}")
            return np.zeros(3)
    
    def _encode_sequence(self, sequence: str) -> torch.Tensor:
        """One-hot encode sequence for CNN."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        sequence = sequence.upper()
        
        encoded = np.zeros((4, len(sequence)), dtype=np.float32)
        for i, base in enumerate(sequence):
            idx = mapping.get(base, 0)
            encoded[idx, i] = 1.0
        
        return torch.from_numpy(encoded).unsqueeze(0)
    
    def _aggregate_results(
        self,
        delta_results: List[DeltaResult]
    ) -> VariantEvaluationResult:
        """Aggregate individual variant results."""
        total = len(delta_results)
        
        # Separate by classification
        splice_altering = [r for r in delta_results if r.is_splice_altering]
        non_splice_altering = [r for r in delta_results if r.classification == "Non-splice-altering"]
        low_frequency = [r for r in delta_results if r.classification == "Low-frequency"]
        
        # Detection rates for splice-altering
        base_detected_sa = sum(1 for r in splice_altering if r.base_detects_effect)
        meta_detected_sa = sum(1 for r in splice_altering if r.meta_detects_effect)
        
        base_detection_rate = base_detected_sa / len(splice_altering) if splice_altering else 0
        meta_detection_rate = meta_detected_sa / len(splice_altering) if splice_altering else 0
        
        # False positive rates for non-splice-altering
        base_fp = sum(1 for r in non_splice_altering if r.base_detects_effect)
        meta_fp = sum(1 for r in non_splice_altering if r.meta_detects_effect)
        
        base_fp_rate = base_fp / len(non_splice_altering) if non_splice_altering else 0
        meta_fp_rate = meta_fp / len(non_splice_altering) if non_splice_altering else 0
        
        # Accuracy
        base_correct = 0
        meta_correct = 0
        
        for r in splice_altering:
            if r.base_detects_effect:
                base_correct += 1
            if r.meta_detects_effect:
                meta_correct += 1
        
        for r in non_splice_altering:
            if not r.base_detects_effect:
                base_correct += 1
            if not r.meta_detects_effect:
                meta_correct += 1
        
        classified_total = len(splice_altering) + len(non_splice_altering)
        base_accuracy = base_correct / classified_total if classified_total else 0
        meta_accuracy = meta_correct / classified_total if classified_total else 0
        
        # Improvement/degradation
        improvement = sum(1 for r in delta_results if r.meta_improves)
        degradation = sum(1 for r in delta_results if r.base_detects_effect and not r.meta_detects_effect)
        
        # Mean deltas
        mean_base_delta_sa = np.mean([r.base_max_delta for r in splice_altering]) if splice_altering else 0
        mean_meta_delta_sa = np.mean([r.meta_max_delta for r in splice_altering]) if splice_altering else 0
        mean_base_delta_nsa = np.mean([r.base_max_delta for r in non_splice_altering]) if non_splice_altering else 0
        mean_meta_delta_nsa = np.mean([r.meta_max_delta for r in non_splice_altering]) if non_splice_altering else 0
        
        return VariantEvaluationResult(
            total_variants=total,
            base_accuracy=base_accuracy,
            meta_accuracy=meta_accuracy,
            base_detection_rate=base_detection_rate,
            meta_detection_rate=meta_detection_rate,
            base_false_positive_rate=base_fp_rate,
            meta_false_positive_rate=meta_fp_rate,
            improvement_count=improvement,
            degradation_count=degradation,
            mean_base_delta_splice_altering=mean_base_delta_sa,
            mean_meta_delta_splice_altering=mean_meta_delta_sa,
            mean_base_delta_non_splice_altering=mean_base_delta_nsa,
            mean_meta_delta_non_splice_altering=mean_meta_delta_nsa,
            per_classification={
                'Splice-altering': {
                    'count': len(splice_altering),
                    'base_detection_rate': base_detection_rate,
                    'meta_detection_rate': meta_detection_rate,
                },
                'Non-splice-altering': {
                    'count': len(non_splice_altering),
                    'base_fp_rate': base_fp_rate,
                    'meta_fp_rate': meta_fp_rate,
                },
                'Low-frequency': {
                    'count': len(low_frequency),
                }
            }
        )


def evaluate_variant_effects(
    meta_model: MetaSpliceModel,
    config: MetaLayerConfig,
    splicevardb_path: Optional[Path] = None,
    test_chromosomes: List[str] = ['21', '22'],
    max_variants: Optional[int] = None,
    genome_build: Optional[str] = None
) -> VariantEvaluationResult:
    """
    Convenience function to evaluate variant effects.
    
    Parameters
    ----------
    meta_model : MetaSpliceModel
        Trained meta-layer model.
    config : MetaLayerConfig
        Configuration.
    splicevardb_path : Path, optional
        Path to SpliceVarDB data.
    test_chromosomes : list of str
        Chromosomes to evaluate on.
    max_variants : int, optional
        Maximum number of variants to evaluate.
    genome_build : str, optional
        Genome build ('GRCh37' or 'GRCh38'). If None, inferred from base model.
    
    Returns
    -------
    VariantEvaluationResult
        Evaluation results.
    """
    # Determine genome build
    if genome_build is None:
        if config.base_model.lower() == 'spliceai':
            genome_build = 'GRCh37'
        else:
            genome_build = 'GRCh38'
    
    logger.info(f"Using genome build: {genome_build} (base model: {config.base_model})")
    
    # Load SpliceVarDB with correct build
    loader = load_splicevardb(
        genome_build=genome_build,
        data_path=splicevardb_path
    )
    
    # Get test variants
    _, test_variants = loader.get_train_test_split(
        test_chromosomes=test_chromosomes
    )
    
    if max_variants and len(test_variants) > max_variants:
        import random
        random.shuffle(test_variants)
        test_variants = test_variants[:max_variants]
    
    logger.info(f"Evaluating on {len(test_variants)} variants from chromosomes {test_chromosomes}")
    
    # Evaluate with correct genome build
    evaluator = VariantEffectEvaluator(meta_model, config, genome_build=genome_build)
    return evaluator.evaluate(test_variants)









