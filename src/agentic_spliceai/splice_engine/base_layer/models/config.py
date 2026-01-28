"""Configuration classes for base model predictions.

Provides configuration dataclasses for SpliceAI and OpenSpliceAI models.

Ported from: meta_spliceai/splice_engine/meta_models/core/data_types.py
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
import os


@dataclass
class BaseModelConfig:
    """Base configuration for splice site prediction models.
    
    This is the core configuration class used by both SpliceAI and OpenSpliceAI.
    It handles path resolution, artifact management, and prediction parameters.
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to use: 'spliceai' or 'openspliceai'
    mode : str, default='test'
        Execution mode: 'production' (immutable) or 'test' (overwritable)
    coverage : str, default='gene_subset'
        Data coverage: 'full_genome', 'chromosome', 'gene_subset'
    test_name : str, optional
        Test identifier for test mode artifacts
    threshold : float, default=0.5
        Splice site score threshold
    consensus_window : int, default=2
        Window for consensus calling
    error_window : int, default=500
        Window for error analysis
    use_auto_position_adjustments : bool, default=True
        Auto-detect position offsets
    save_nucleotide_scores : bool, default=False
        Save nucleotide-level scores (generates large files)
    verbosity : int, default=1
        Output verbosity (0-2)
    """
    
    # Base model selection
    base_model: str = "spliceai"
    
    # Artifact management
    mode: str = "test"
    coverage: str = "gene_subset"
    test_name: Optional[str] = None
    
    # File paths (auto-resolved based on base_model)
    gtf_file: Optional[str] = None
    genome_fasta: Optional[str] = None
    eval_dir: Optional[str] = None
    local_dir: Optional[str] = None
    output_subdir: str = "meta_models"
    
    # File format settings
    format: str = "tsv"
    seq_format: str = "parquet"
    seq_mode: str = "gene"
    seq_type: str = "full"
    separator: str = "\t"
    
    # Prediction parameters
    threshold: float = 0.5
    consensus_window: int = 2
    error_window: int = 500
    
    # Processing settings
    test_mode: bool = False
    chromosomes: Optional[List[str]] = None
    
    # Data preparation switches
    do_extract_annotations: bool = False
    do_extract_splice_sites: bool = False
    do_extract_sequences: bool = False
    do_find_overlapping_genes: bool = False
    use_precomputed_overlapping_genes: bool = False
    
    # Three-probability-score workflow settings
    use_three_probabilities: bool = True
    use_auto_position_adjustments: bool = True
    save_example_sequences: bool = True
    
    # Output control
    save_nucleotide_scores: bool = False
    
    # Verbosity
    verbosity: int = 1
    
    def __post_init__(self):
        """Initialize derived values after dataclass initialization."""
        from agentic_spliceai.splice_engine.resources import get_genomic_registry
        
        base_model_lower = self.base_model.lower()
        
        # Resolve paths based on base_model if not explicitly provided
        if self.gtf_file is None or self.genome_fasta is None or self.eval_dir is None:
            # Get appropriate registry based on base model
            if base_model_lower == 'openspliceai':
                registry = get_genomic_registry(build='GRCh38_MANE', release='1.3')
            else:
                registry = get_genomic_registry(build='GRCh37', release='87')
            
            if self.gtf_file is None:
                gtf_path = registry.get_gtf_path(validate=False)
                self.gtf_file = str(gtf_path) if gtf_path else None
            
            if self.genome_fasta is None:
                fasta_path = registry.get_fasta_path(validate=False)
                self.genome_fasta = str(fasta_path) if fasta_path else None
            
            if self.eval_dir is None:
                self.eval_dir = str(registry.get_base_model_eval_dir(base_model_lower, create=True))
        
        # Derive local_dir from eval_dir if not provided
        if self.local_dir is None and self.eval_dir:
            self.local_dir = os.path.dirname(self.eval_dir)
        
        # Auto-detect mode from coverage
        if self.coverage == "full_genome" and self.mode == "test":
            self.mode = "production"
        
        # Generate test_name if needed
        if self.mode == "test" and self.test_name is None:
            from datetime import datetime
            self.test_name = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_full_eval_dir(self) -> str:
        """Get the full evaluation directory path including the output subdirectory."""
        return os.path.join(self.eval_dir, self.output_subdir)
    
    @property
    def genomic_build(self) -> str:
        """Get the genomic build for this configuration."""
        if self.base_model.lower() == 'openspliceai':
            return 'GRCh38'
        return 'GRCh37'
    
    @property
    def annotation_source(self) -> str:
        """Get the annotation source for this configuration."""
        if self.base_model.lower() == 'openspliceai':
            return 'mane'
        return 'ensembl'


# Aliases for backward compatibility
SpliceAIConfig = BaseModelConfig


@dataclass
class OpenSpliceAIConfig(BaseModelConfig):
    """Configuration specifically for OpenSpliceAI model.
    
    This is a convenience class that sets OpenSpliceAI-specific defaults.
    """
    base_model: str = "openspliceai"


def create_config(
    base_model: str = 'spliceai',
    **kwargs
) -> BaseModelConfig:
    """Factory function to create appropriate configuration.
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to use: 'spliceai' or 'openspliceai'
    **kwargs
        Additional configuration parameters
        
    Returns
    -------
    BaseModelConfig
        Configuration instance for the specified model
    """
    return BaseModelConfig(base_model=base_model, **kwargs)
