"""Registry for resolving genomic resource paths.

Provides a unified interface for locating GTF, FASTA, and derived TSV files
across multiple possible locations.

Ported from: meta_spliceai/system/genomic_resources/registry.py
"""

import os
from pathlib import Path
from typing import Optional

from agentic_spliceai.splice_engine.config.genomic_config import load_config, filename


class Registry:
    """Registry for resolving genomic resource paths.
    
    Directory structure: data / annotation_source / build
    
    Examples:
    - data/ensembl/GRCh37/
    - data/ensembl/GRCh38/
    - data/mane/GRCh38/
    
    Search order for files:
    1. Explicit path from environment variable (SS_GTF_PATH, SS_FASTA_PATH, etc.)
    2. data/<annotation_source>/<BUILD>/ (build-specific directory)
    3. data/<annotation_source>/<BUILD>/spliceai_analysis/ (derived datasets)
    
    Parameters
    ----------
    build : str, optional
        Override build from config (e.g., 'GRCh38', 'GRCh37', 'GRCh38_MANE')
    release : str, optional
        Override release from config (e.g., '112', '106', '1.3')
    """
    
    def __init__(self, build: Optional[str] = None, release: Optional[str] = None):
        self.cfg = load_config()
        if build:
            self.cfg.build = build
        if release:
            self.cfg.release = release
        
        # Get annotation source for this build
        self.annotation_source = self.cfg.get_annotation_source(self.cfg.build)
        
        # Build directory structure using annotation source
        # Structure: data_root / annotation_source / build
        self.top = self.cfg.data_root / self.annotation_source
        
        # For MANE builds, strip suffix from directory name
        build_dir = self.cfg.build.replace("_MANE", "").replace("_GENCODE", "")
        self.stash = self.top / build_dir
        self.legacy = self.top / "spliceai_analysis"  # Legacy location (rarely used)
        
        # Build-specific directories
        self.data_dir = self.stash  # Alias for build-specific directory
        self.eval_dir = self.stash / "spliceai_eval"
        self.analysis_dir = self.stash / "spliceai_analysis"
    
    def resolve(self, kind: str) -> Optional[str]:
        """Resolve path for a given resource kind.
        
        Parameters
        ----------
        kind : str
            Resource kind: 'gtf', 'fasta', 'fasta_index', 'splice_sites',
            'gene_features', 'transcript_features', 'exon_features', 'junctions'
            
        Returns
        -------
        str or None
            Absolute path to the resource if found, None otherwise
        """
        # Check for explicit environment variable override
        env_var = f"SS_{kind.upper()}_PATH"
        if env_var in os.environ:
            path = Path(os.environ[env_var])
            if path.exists():
                return str(path.resolve())
        
        # Determine filename
        if kind == "gtf":
            name = filename("gtf", self.cfg)
        elif kind == "fasta":
            name = filename("fasta", self.cfg)
        elif kind == "fasta_index":
            name = filename("fasta", self.cfg) + ".fai"
        else:
            # Derived datasets - use config if available, otherwise use defaults
            if self.cfg.derived_datasets and kind in self.cfg.derived_datasets:
                name = self.cfg.derived_datasets[kind]
            else:
                # Fallback to defaults if not in config
                mapping = {
                    "splice_sites": "splice_sites_enhanced.tsv",  # Updated to enhanced format
                    "gene_features": "gene_features.tsv",
                    "transcript_features": "transcript_features.tsv",
                    "exon_features": "exon_features.tsv",
                    "junctions": "junctions.tsv"
                }
                if kind not in mapping:
                    raise ValueError(f"Unknown resource kind: {kind}")
                name = mapping[kind]
        
        # Enhanced splice sites fallback
        if kind == "splice_sites" and "enhanced" not in name:
            enhanced_name = "splice_sites_enhanced.tsv"
            search_order = [self.stash, self.top, self.legacy]
            for root in search_order:
                enhanced_path = Path(root) / enhanced_name
                if enhanced_path.exists():
                    return str(enhanced_path.resolve())
        
        # Search in order: stash (build-specific), top (default), legacy
        for root in [self.stash, self.top, self.legacy]:
            p = Path(root) / name
            if p.exists():
                return str(p.resolve())
        
        return None
    
    def get_gtf_path(self, validate=True) -> Optional[Path]:
        """Get the path to the GTF file.
        
        Parameters
        ----------
        validate : bool, default=True
            If True, raises FileNotFoundError if GTF file doesn't exist
            
        Returns
        -------
        Path or None
            Path to GTF file, or None if not found and validate=False
        """
        gtf_path = self.resolve("gtf")
        if gtf_path is None:
            if validate:
                raise FileNotFoundError(
                    f"GTF file not found for build {self.cfg.build}, release {self.cfg.release}"
                )
            return None
        return Path(gtf_path)
    
    def get_fasta_path(self, validate=True) -> Optional[Path]:
        """Get the path to the FASTA file.
        
        Parameters
        ----------
        validate : bool, default=True
            If True, raises FileNotFoundError if FASTA file doesn't exist
            
        Returns
        -------
        Path or None
            Path to FASTA file, or None if not found and validate=False
        """
        fasta_path = self.resolve("fasta")
        if fasta_path is None:
            if validate:
                raise FileNotFoundError(
                    f"FASTA file not found for build {self.cfg.build}, release {self.cfg.release}"
                )
            return None
        return Path(fasta_path)
    
    def list_all(self) -> dict:
        """List all known resource kinds and their resolved paths.
        
        Returns
        -------
        dict
            Mapping from resource kind to resolved path (or None if not found)
        """
        kinds = [
            "gtf",
            "fasta",
            "fasta_index",
            "splice_sites",
            "gene_features",
            "transcript_features",
            "exon_features",
            "junctions",
        ]
        return {kind: self.resolve(kind) for kind in kinds}
    
    def get_annotations_db_path(self, validate: bool = False) -> Optional[Path]:
        """Get path to annotations.db file."""
        for root in [self.stash, self.legacy, self.top]:
            path = Path(root) / "annotations.db"
            if path.exists():
                return path
        
        if validate:
            raise FileNotFoundError(
                f"annotations.db not found for build {self.cfg.build}"
            )
        return None
    
    def get_overlapping_genes_path(self, validate: bool = False) -> Optional[Path]:
        """Get path to overlapping_genes.tsv or overlapping_gene_counts.tsv file."""
        for filename in ["overlapping_genes.tsv", "overlapping_gene_counts.tsv"]:
            for root in [self.stash, self.legacy, self.top]:
                path = Path(root) / filename
                if path.exists():
                    return path
        
        if validate:
            raise FileNotFoundError(
                f"overlapping genes file not found for build {self.cfg.build}"
            )
        return None
    
    def get_chromosome_sequence_path(
        self, 
        chromosome: str, 
        format: str = "parquet",
        validate: bool = False
    ) -> Optional[Path]:
        """Get path to chromosome-specific sequence file."""
        filename = f"gene_sequence_{chromosome}.{format}"
        
        for root in [self.stash, self.legacy, self.top]:
            path = Path(root) / filename
            if path.exists():
                return path
        
        if validate:
            raise FileNotFoundError(
                f"Chromosome {chromosome} sequence file not found for build {self.cfg.build}"
            )
        return None
    
    def get_local_dir(self) -> Path:
        """Get the local directory for intermediate files."""
        return self.stash
    
    def get_eval_dir(self, create: bool = False) -> Path:
        """Get the evaluation directory for SpliceAI outputs."""
        if create:
            self.eval_dir.mkdir(parents=True, exist_ok=True)
        return self.eval_dir
    
    def get_analysis_dir(self, create: bool = False) -> Path:
        """Get the analysis directory for derived datasets."""
        if create:
            self.analysis_dir.mkdir(parents=True, exist_ok=True)
        return self.analysis_dir
    
    def get_base_model_eval_dir(
        self, 
        base_model: str, 
        create: bool = False
    ) -> Path:
        """Get evaluation directory for a specific base model.
        
        Examples:
        - data/ensembl/GRCh37/spliceai_eval/
        - data/mane/GRCh38/openspliceai_eval/
        """
        eval_dir = self.stash / f'{base_model.lower()}_eval'
        if create:
            eval_dir.mkdir(parents=True, exist_ok=True)
        return eval_dir
    
    def get_meta_models_artifact_dir(
        self, 
        base_model: str, 
        create: bool = False
    ) -> Path:
        """Get meta_models artifact directory for a specific base model.
        
        Examples:
        - data/ensembl/GRCh37/spliceai_eval/meta_models/
        - data/mane/GRCh38/openspliceai_eval/meta_models/
        """
        artifact_dir = self.get_base_model_eval_dir(base_model) / 'meta_models'
        if create:
            artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir
    
    def get_model_weights_dir(self, base_model: str) -> Path:
        """Get directory containing model weights for a base model.
        
        Pattern: <data_root>/models/<base_model>/
        
        Examples:
        - data/models/spliceai/
        - data/models/openspliceai/
        """
        return self.cfg.data_root / "models" / base_model.lower()


# Global registry cache
_registry_cache = {}


def get_genomic_registry(build: str = None, release: str = None) -> Registry:
    """Get or create a genomic registry for a specific build.
    
    This function caches registry instances to avoid recreating them.
    
    Parameters
    ----------
    build : str, optional
        Genomic build (e.g., 'GRCh37', 'GRCh38', 'GRCh38_MANE')
    release : str, optional
        Release version
    
    Returns
    -------
    Registry
        Registry instance for the specified build
    """
    key = f"{build or 'default'}/{release or 'default'}"
    
    if key not in _registry_cache:
        _registry_cache[key] = Registry(build=build, release=release)
    
    return _registry_cache[key]
