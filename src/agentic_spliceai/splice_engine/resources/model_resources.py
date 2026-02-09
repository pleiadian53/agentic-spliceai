"""Model-specific resource manager.

This module provides a unified interface for resolving genomic resources
(GTF, FASTA, annotations) based on the base model being used.

Each base model was trained on a specific genomic build and annotation source,
so predictions and evaluations must use matching resources.

Examples
--------
>>> # Get resources for SpliceAI
>>> resources = get_model_resources('spliceai')
>>> print(resources.build)  # GRCh37
>>> print(resources.annotation_source)  # ensembl
>>> registry = resources.get_registry()
>>> annotations_dir = resources.get_annotations_dir()

>>> # Get resources for OpenSpliceAI
>>> resources = get_model_resources('openspliceai')
>>> print(resources.build)  # GRCh38
>>> print(resources.annotation_source)  # mane
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .registry import Registry, get_genomic_registry
from ..config.genomic_config import load_config


@dataclass
class ModelResources:
    """Resources for a specific base model.
    
    Attributes
    ----------
    model_name : str
        Base model name (e.g., 'spliceai', 'openspliceai')
    build : str
        Genomic build the model was trained on (e.g., 'GRCh37', 'GRCh38')
    annotation_source : str
        Annotation source used for training (e.g., 'ensembl', 'mane')
    release : str, optional
        Specific release version (e.g., '87', '1.3')
    training_annotation : str, optional
        Full training annotation description (e.g., 'GENCODE V24lift37')
    notes : str, optional
        Additional notes about the model
    """
    
    model_name: str
    build: str
    annotation_source: str
    release: Optional[str] = None
    training_annotation: Optional[str] = None
    notes: Optional[str] = None
    
    def get_registry(self) -> Registry:
        """Get the genomic registry for this model's resources.
        
        Returns
        -------
        Registry
            Registry configured for the model's build and annotation source
            
        Examples
        --------
        >>> resources = get_model_resources('spliceai')
        >>> registry = resources.get_registry()
        >>> gtf_path = registry.get_gtf_path()
        """
        # Handle special cases (e.g., MANE needs GRCh38_MANE build key)
        if self.build == 'GRCh38' and self.annotation_source == 'mane':
            return get_genomic_registry(build='GRCh38_MANE', release=self.release or '1.3')
        else:
            return get_genomic_registry(build=self.build, release=self.release)
    
    def get_annotations_dir(self, create: bool = True) -> Path:
        """Get the directory for storing build-specific annotations.
        
        This directory stores splice site annotations and other derived data
        specific to this genomic build. Ensures GRCh37 and GRCh38 annotations
        are kept separate.
        
        Parameters
        ----------
        create : bool, default=True
            Whether to create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Build-specific annotations directory
            Example: data/ensembl/GRCh37/
            
        Examples
        --------
        >>> resources = get_model_resources('spliceai')
        >>> annotations_dir = resources.get_annotations_dir()
        >>> print(annotations_dir)
        .../data/ensembl/GRCh37/
        """
        registry = self.get_registry()
        data_dir = registry.data_dir
        
        if create:
            data_dir.mkdir(parents=True, exist_ok=True)
        
        return data_dir
    
    def get_eval_dir(self, create: bool = True) -> Path:
        """Get the evaluation directory for this model.
        
        This directory stores model-specific evaluation results, predictions,
        and derived datasets used for meta-model training.
        
        Parameters
        ----------
        create : bool, default=True
            Whether to create the directory if it doesn't exist
            
        Returns
        -------
        Path
            Model-specific evaluation directory
            Example: data/ensembl/GRCh37/spliceai_eval/
            
        Examples
        --------
        >>> resources = get_model_resources('openspliceai')
        >>> eval_dir = resources.get_eval_dir()
        >>> print(eval_dir)
        .../data/mane/GRCh38/openspliceai_eval/
        """
        registry = self.get_registry()
        eval_dir = registry.eval_dir
        
        if create:
            eval_dir.mkdir(parents=True, exist_ok=True)
        
        return eval_dir
    
    def get_gtf_path(self) -> Optional[Path]:
        """Get the GTF file path for this model's genomic build.
        
        Returns
        -------
        Path or None
            Path to GTF file, or None if not found
            
        Examples
        --------
        >>> resources = get_model_resources('spliceai')
        >>> gtf_path = resources.get_gtf_path()
        >>> print(gtf_path)
        .../data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf
        """
        registry = self.get_registry()
        return registry.get_gtf_path()
    
    def get_fasta_path(self) -> Optional[Path]:
        """Get the FASTA file path for this model's genomic build.
        
        Returns
        -------
        Path or None
            Path to FASTA file, or None if not found
            
        Examples
        --------
        >>> resources = get_model_resources('openspliceai')
        >>> fasta_path = resources.get_fasta_path()
        >>> print(fasta_path)
        .../data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
        """
        registry = self.get_registry()
        return registry.get_fasta_path()
    
    def __str__(self) -> str:
        """String representation of model resources."""
        return (
            f"ModelResources(model={self.model_name}, "
            f"build={self.build}, "
            f"annotation={self.annotation_source})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation of model resources."""
        return self.__str__()


def get_model_resources(model_name: str) -> ModelResources:
    """Get genomic resources for a specific base model.
    
    This function looks up the model's training configuration and returns
    a ModelResources object that provides access to the correct genomic
    build, annotation source, and resource paths.
    
    Parameters
    ----------
    model_name : str
        Base model name (e.g., 'spliceai', 'openspliceai')
        Case-insensitive.
        
    Returns
    -------
    ModelResources
        Resource manager for the model
        
    Raises
    ------
    ValueError
        If the model is not configured in settings.yaml
        
    Examples
    --------
    >>> # Basic usage
    >>> resources = get_model_resources('spliceai')
    >>> print(resources.build)
    GRCh37
    >>> print(resources.annotation_source)
    ensembl
    
    >>> # Get registry and paths
    >>> registry = resources.get_registry()
    >>> gtf_path = resources.get_gtf_path()
    >>> annotations_dir = resources.get_annotations_dir()
    
    >>> # Use in prediction workflow
    >>> resources = get_model_resources('openspliceai')
    >>> from agentic_spliceai.splice_engine.base_layer.data import prepare_gene_data
    >>> genes_df = prepare_gene_data(
    ...     genes=['TP53'],
    ...     build=resources.build,
    ...     annotation_source=resources.annotation_source
    ... )
    
    See Also
    --------
    ModelResources : The resource manager class
    get_genomic_registry : Low-level registry access
    """
    cfg = load_config()
    model_name_lower = model_name.lower()
    
    if not cfg.base_models or model_name_lower not in cfg.base_models:
        available = list(cfg.base_models.keys()) if cfg.base_models else []
        raise ValueError(
            f"Unknown base model: {model_name}. "
            f"Configured models: {available}\n"
            f"Add model configuration to settings.yaml under 'base_models'"
        )
    
    model_config = cfg.base_models[model_name_lower]
    
    return ModelResources(
        model_name=model_name_lower,
        build=model_config.get('training_build', cfg.build),
        annotation_source=model_config.get('annotation_source', cfg.default_annotation_source),
        release=model_config.get('release'),  # Optional
        training_annotation=model_config.get('training_annotation'),
        notes=model_config.get('notes')
    )


def list_available_models() -> list[str]:
    """List all configured base models.
    
    Returns
    -------
    list of str
        Names of available base models
        
    Examples
    --------
    >>> models = list_available_models()
    >>> print(models)
    ['spliceai', 'openspliceai']
    """
    cfg = load_config()
    return list(cfg.base_models.keys()) if cfg.base_models else []


def get_model_info(model_name: str) -> dict:
    """Get full configuration information for a model.
    
    Parameters
    ----------
    model_name : str
        Base model name
        
    Returns
    -------
    dict
        Full model configuration from settings.yaml
        
    Examples
    --------
    >>> info = get_model_info('spliceai')
    >>> print(info['training_build'])
    GRCh37
    >>> print(info['training_annotation'])
    GENCODE V24lift37
    """
    cfg = load_config()
    model_name_lower = model_name.lower()
    
    if not cfg.base_models or model_name_lower not in cfg.base_models:
        raise ValueError(f"Unknown base model: {model_name}")
    
    return cfg.base_models[model_name_lower].copy()
