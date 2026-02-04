"""Base model runner for splice site predictions.

Provides utilities for running base models (SpliceAI, OpenSpliceAI)
and comparing their performance.

Ported from: meta_spliceai/system/base_model_runner.py
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field

import polars as pl

from .config import BaseModelConfig


@dataclass
class BaseModelResult:
    """Result from running a base model.
    
    Attributes
    ----------
    model_name : str
        Base model name
    success : bool
        Whether execution succeeded
    runtime_seconds : float
        Execution time in seconds
    positions : pl.DataFrame
        Analyzed positions with predictions
    nucleotide_scores : Optional[pl.DataFrame]
        Nucleotide-level scores (if enabled)
    gene_manifest : Optional[pl.DataFrame]
        Gene processing manifest
    processed_genes : Set[str]
        Genes successfully processed
    missing_genes : Set[str]
        Genes not processed
    metrics : Dict[str, float]
        Performance metrics
    paths : Dict[str, str]
        Artifact paths
    error : Optional[str]
        Error message (if failed)
    """
    model_name: str
    success: bool
    runtime_seconds: float
    positions: pl.DataFrame
    nucleotide_scores: Optional[pl.DataFrame] = None
    gene_manifest: Optional[pl.DataFrame] = None
    processed_genes: Set[str] = field(default_factory=set)
    missing_genes: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result from comparing multiple base models.
    
    Attributes
    ----------
    test_name : str
        Test identifier
    models : Dict[str, BaseModelResult]
        Results for each model
    comparison_metrics : Dict[str, Any]
        Comparison metrics
    output_dir : Path
        Output directory
    """
    test_name: str
    models: Dict[str, BaseModelResult]
    comparison_metrics: Dict[str, Any]
    output_dir: Path


class BaseModelRunner:
    """Runner for base model predictions and comparisons.
    
    This class provides utilities for:
    - Running multiple base models with consistent configuration
    - Comparing performance metrics
    - Handling missing genes gracefully
    - Generating comprehensive reports
    
    Examples
    --------
    >>> from agentic_spliceai.splice_engine.base_layer.models import BaseModelRunner
    >>> runner = BaseModelRunner()
    >>> result = runner.run_single_model(
    ...     model_name='spliceai',
    ...     target_genes=['BRCA1', 'TP53'],
    ...     test_name='validation_test'
    ... )
    >>> print(f"Analyzed {result.positions.height} positions")
    """
    
    def __init__(self):
        """Initialize base model runner."""
        pass
    
    def run_single_model(
        self,
        model_name: str,
        target_genes: List[Optional[str]],
        test_name: str,
        mode: str = 'test',
        coverage: str = 'gene_subset',
        threshold: float = 0.5,
        consensus_window: int = 2,
        error_window: int = 500,
        use_auto_position_adjustments: bool = True,
        save_nucleotide_scores: bool = False,
        no_tn_sampling: bool = True,
        verbosity: int = 1
    ) -> BaseModelResult:
        """Run a single base model.
        
        Parameters
        ----------
        model_name : str
            Base model name ('spliceai', 'openspliceai')
        target_genes : List[Optional[str]]
            Target gene IDs
        test_name : str
            Test identifier
        mode : str, default='test'
            Execution mode
        coverage : str, default='gene_subset'
            Coverage mode
        threshold : float, default=0.5
            Splice site score threshold
        consensus_window : int, default=2
            Window for consensus calling
        error_window : int, default=500
            Window for error analysis
        use_auto_position_adjustments : bool, default=True
            Auto-detect position offsets
        save_nucleotide_scores : bool, default=False
            Save nucleotide-level scores
        no_tn_sampling : bool, default=True
            Disable true negative sampling
        verbosity : int, default=1
            Output verbosity
        
        Returns
        -------
        BaseModelResult
            Model execution results
        """
        if verbosity >= 1:
            print("=" * 80)
            print(f"Running {model_name.upper()}")
            print("=" * 80)
        
        start_time = time.time()
        
        try:
            # Import here to avoid circular imports
            from .config import BaseModelConfig
            
            # Create configuration
            config = BaseModelConfig(
                base_model=model_name,
                mode=mode,
                coverage=coverage,
                test_name=f"{test_name}_{model_name}",
                threshold=threshold,
                consensus_window=consensus_window,
                error_window=error_window,
                use_auto_position_adjustments=use_auto_position_adjustments,
                save_nucleotide_scores=save_nucleotide_scores,
                verbosity=verbosity
            )
            
            # ==== Phase 1: Wire up existing prediction code ====
            
            if verbosity >= 1:
                print(f"\n{'='*60}")
                print(f"Running base model predictions: {model_name}")
                print(f"{'='*60}")
            
            # 1. Prepare gene data (load annotations + extract sequences)
            gene_df = self._prepare_gene_data(config, target_genes=target_genes, verbosity=verbosity)
            
            if gene_df is None or len(gene_df) == 0:
                runtime = time.time() - start_time
                return BaseModelResult(
                    model_name=model_name,
                    success=False,
                    runtime_seconds=runtime,
                    positions=pl.DataFrame(),
                    processed_genes=set(),
                    missing_genes=set(target_genes) if target_genes else set(),
                    error="No genes found in annotations"
                )
            
            # 2. Load models
            from ..prediction.core import load_spliceai_models
            
            # Determine build from model name
            build = 'GRCh38' if model_name.lower() == 'openspliceai' else 'GRCh37'
            
            models = load_spliceai_models(
                model_type=model_name,
                build=build,
                verbosity=verbosity
            )
            
            if not models:
                runtime = time.time() - start_time
                return BaseModelResult(
                    model_name=model_name,
                    success=False,
                    runtime_seconds=runtime,
                    positions=pl.DataFrame(),
                    processed_genes=set(),
                    missing_genes=set(target_genes) if target_genes else set(),
                    error=f"Failed to load {model_name} models"
                )
            
            # 3. Run predictions
            from ..prediction.core import predict_splice_sites_for_genes
            
            if verbosity >= 1:
                print(f"\nPredicting on {len(gene_df)} genes...")
            
            predictions_dict = predict_splice_sites_for_genes(
                gene_df=gene_df,
                models=models,
                context=10000,  # SpliceAI-10k context
                output_format='dict',
                verbosity=verbosity
            )
            
            # 4. Convert predictions to DataFrame
            positions_df = self._predictions_to_dataframe(predictions_dict, config)
            
            runtime = time.time() - start_time
            
            if verbosity >= 1:
                print(f"\n{'='*60}")
                print(f"Prediction complete!")
                print(f"  Processed genes: {len(predictions_dict)}")
                print(f"  Total positions: {len(positions_df) if len(positions_df) > 0 else 0}")
                print(f"  Runtime: {runtime:.2f}s")
                print(f"{'='*60}\n")
            
            # 5. Return result
            processed_genes = set(predictions_dict.keys())
            missing_genes = set(target_genes) - processed_genes if target_genes else set()
            
            return BaseModelResult(
                model_name=model_name,
                success=True,
                runtime_seconds=runtime,
                positions=positions_df,
                processed_genes=processed_genes,
                missing_genes=missing_genes,
                error=None
            )
        
        except Exception as e:
            runtime = time.time() - start_time
            
            if verbosity >= 1:
                print(f"❌ {model_name.upper()} error: {e}")
            
            return BaseModelResult(
                model_name=model_name,
                success=False,
                runtime_seconds=runtime,
                positions=pl.DataFrame(),
                processed_genes=set(),
                missing_genes=set(g for g in target_genes if g is not None),
                error=str(e)
            )
    
    def calculate_metrics(self, positions_df: pl.DataFrame) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Parameters
        ----------
        positions_df : pl.DataFrame
            Positions dataframe with pred_type column
        
        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        if positions_df.height == 0:
            return {
                'positions': 0, 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'donor_ap': 0.0, 'donor_pr_auc': 0.0,
                'acceptor_ap': 0.0, 'acceptor_pr_auc': 0.0,
                'macro_ap': 0.0, 'macro_pr_auc': 0.0,
            }
        
        if 'pred_type' not in positions_df.columns:
            return {
                'positions': positions_df.height,
                'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0,
                'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0,
                'donor_ap': 0.0, 'donor_pr_auc': 0.0,
                'acceptor_ap': 0.0, 'acceptor_pr_auc': 0.0,
                'macro_ap': 0.0, 'macro_pr_auc': 0.0,
            }
        
        tp = positions_df.filter(pl.col('pred_type') == 'TP').height
        tn = positions_df.filter(pl.col('pred_type') == 'TN').height
        fp = positions_df.filter(pl.col('pred_type') == 'FP').height
        fn = positions_df.filter(pl.col('pred_type') == 'FN').height
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

        pr_metrics: Dict[str, float] = {
            'donor_ap': 0.0, 'donor_pr_auc': 0.0,
            'acceptor_ap': 0.0, 'acceptor_pr_auc': 0.0,
            'macro_ap': 0.0, 'macro_pr_auc': 0.0,
        }
        try:
            from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import compute_pr_metrics

            pr_metrics = compute_pr_metrics(positions_df)
        except Exception:
            # Best-effort; keep thresholded metrics even if sklearn isn't available.
            pr_metrics = pr_metrics

        return {
            'positions': positions_df.height,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            **pr_metrics,
        }
    
    # ==== Phase 1 Helper Methods ====
    
    def _prepare_gene_data(
        self,
        config: BaseModelConfig,
        target_genes: Optional[List[str]] = None,
        verbosity: int = 1
    ) -> Optional[pl.DataFrame]:
        """Load gene annotations and extract sequences."""
        from ...resources import get_genomic_registry
        from ..data.sequence_extraction import extract_gene_sequences
        
        # Get registry for path resolution (use the build from config)
        build = config.genomic_build  # This is a property, not a method
        # For OpenSpliceAI, use MANE annotations with correct release
        if config.base_model.lower() == 'openspliceai':
            registry = get_genomic_registry(build='GRCh38_MANE', release='1.3')
        else:
            registry = get_genomic_registry(build='GRCh37', release='87')
        gtf_path = registry.get_gtf_path(validate=True)
        fasta_path = registry.get_fasta_path(validate=True)
        
        if verbosity >= 1:
            print(f"\n📂 Loading genomic resources:")
            print(f"  GTF: {gtf_path}")
            print(f"  FASTA: {fasta_path}")
        
        # Load gene annotations
        genes_df = self._load_gene_annotations(
            gtf_path=gtf_path,
            target_genes=target_genes,
            chromosomes=config.chromosomes,
            verbosity=verbosity
        )
        
        if genes_df is None or len(genes_df) == 0:
            if verbosity >= 1:
                print("❌ No genes found in annotations")
            return None
        
        if verbosity >= 1:
            print(f"✓ Loaded {len(genes_df)} gene annotations")
        
        # Extract sequences
        if verbosity >= 1:
            print(f"\n🧬 Extracting sequences from FASTA...")
        
        genes_df = extract_gene_sequences(
            gene_df=genes_df,
            fasta_file=str(fasta_path),
            verbosity=verbosity
        )
        
        if verbosity >= 1:
            n_with_seq = len(genes_df.filter(pl.col('sequence').is_not_null()))
            print(f"✓ Extracted {n_with_seq}/{len(genes_df)} sequences")
        
        # Filter to genes with sequences
        genes_df = genes_df.filter(pl.col('sequence').is_not_null())
        return genes_df
    
    def _load_gene_annotations(
        self,
        gtf_path: Path,
        target_genes: Optional[List[str]] = None,
        chromosomes: Optional[List[str]] = None,
        verbosity: int = 1
    ) -> Optional[pl.DataFrame]:
        """Load gene annotations from GTF file."""
        from ..data.genomic_extraction import extract_gene_annotations
        
        try:
            genes_df = extract_gene_annotations(
                gtf_file=str(gtf_path),
                chromosomes=chromosomes,
                verbosity=verbosity
            )
            
            # Filter to target genes if specified
            if target_genes:
                genes_df = genes_df.filter(
                    (pl.col('gene_id').is_in(target_genes)) |
                    (pl.col('gene_name').is_in(target_genes))
                )
            
            return genes_df
        except Exception as e:
            if verbosity >= 1:
                print(f"❌ Error loading gene annotations: {e}")
            return None
    
    def _predictions_to_dataframe(
        self,
        predictions_dict: Dict[str, Dict],
        config: BaseModelConfig
    ) -> pl.DataFrame:
        """Convert predictions dictionary to DataFrame."""
        if not predictions_dict:
            return pl.DataFrame()
        
        # Convert to list of records
        records = []
        for gene_id, pred in predictions_dict.items():
            gene_name = pred.get('gene_name', gene_id)
            seqname = pred.get('seqname', '')
            strand = pred.get('strand', '+')
            gene_start = pred.get('gene_start', 0)
            gene_end = pred.get('gene_end', 0)
            
            positions = pred.get('positions', [])
            donor_probs = pred.get('donor_prob', [])
            acceptor_probs = pred.get('acceptor_prob', [])
            neither_probs = pred.get('neither_prob', [])
            
            for i, pos in enumerate(positions):
                records.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'seqname': seqname,
                    'position': pos,
                    'strand': strand,
                    'gene_start': gene_start,
                    'gene_end': gene_end,
                    'donor_prob': donor_probs[i] if i < len(donor_probs) else 0.0,
                    'acceptor_prob': acceptor_probs[i] if i < len(acceptor_probs) else 0.0,
                    'neither_prob': neither_probs[i] if i < len(neither_probs) else 0.0,
                })
        
        return pl.DataFrame(records)
    
    def compare_models(
        self,
        models: List[str],
        gene_symbols: List[str],
        gene_ids_by_model: Dict[str, List[Optional[str]]],
        test_name: str,
        verbosity: int = 1,
        **kwargs
    ) -> ComparisonResult:
        """Compare multiple base models.
        
        Parameters
        ----------
        models : List[str]
            List of model names to compare
        gene_symbols : List[str]
            Gene symbols being tested
        gene_ids_by_model : Dict[str, List[Optional[str]]]
            Gene IDs for each model (model_name → gene_ids)
        test_name : str
            Test identifier
        verbosity : int, default=1
            Output verbosity
        **kwargs
            Additional parameters passed to run_single_model
        
        Returns
        -------
        ComparisonResult
            Comparison results
        """
        model_results = {}
        
        for model_name in models:
            target_genes = gene_ids_by_model.get(model_name, [])
            
            result = self.run_single_model(
                model_name=model_name,
                target_genes=target_genes,
                test_name=test_name,
                verbosity=verbosity,
                **kwargs
            )
            
            model_results[model_name] = result
            
            if verbosity >= 1:
                print()
        
        # Calculate comparison metrics
        comparison_metrics = {
            'total_genes': len(gene_symbols),
            'models': {
                name: {
                    'success': result.success,
                    'genes_processed': len(result.processed_genes),
                    'genes_missing': len(result.missing_genes),
                    'runtime_seconds': result.runtime_seconds,
                    **result.metrics
                }
                for name, result in model_results.items()
            }
        }
        
        # Create output directory
        output_dir = Path(f"results/{test_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return ComparisonResult(
            test_name=test_name,
            models=model_results,
            comparison_metrics=comparison_metrics,
            output_dir=output_dir
        )
    
    def print_comparison_summary(
        self,
        result: ComparisonResult,
        verbosity: int = 1
    ) -> None:
        """Print comparison summary.
        
        Parameters
        ----------
        result : ComparisonResult
            Comparison results
        verbosity : int, default=1
            Output verbosity
        """
        print("=" * 80)
        print("PERFORMANCE COMPARISON")
        print("=" * 80)
        print()
        
        model_names = list(result.models.keys())
        col_width = 25
        
        print(f"{'Metric':<20}", end='')
        for name in model_names:
            print(f"{name.upper():<{col_width}}", end='')
        print()
        print("-" * (20 + col_width * len(model_names)))
        
        metrics_to_show = [
            ('Genes Processed', 'genes_processed', ''),
            ('Positions', 'positions', ','),
            ('TP', 'tp', ','),
            ('FP', 'fp', ','),
            ('FN', 'fn', ','),
            ('Precision', 'precision', '.4f'),
            ('Recall', 'recall', '.4f'),
            ('F1 Score', 'f1', '.4f'),
            ('Accuracy', 'accuracy', '.4f'),
            ('Runtime (sec)', 'runtime_seconds', '.1f')
        ]
        
        for label, key, fmt in metrics_to_show:
            print(f"{label:<20}", end='')
            for name in model_names:
                model_result = result.models[name]
                if model_result.success:
                    value = result.comparison_metrics['models'][name].get(key, 0)
                    if fmt:
                        if ',' in fmt:
                            print(f"{value:<{col_width},}", end='')
                        else:
                            print(f"{value:<{col_width}{fmt}}", end='')
                    else:
                        print(f"{value:<{col_width}}", end='')
                else:
                    print(f"{'FAILED':<{col_width}}", end='')
            print()
        
        print("-" * (20 + col_width * len(model_names)))
        print()
