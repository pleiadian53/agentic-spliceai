"""Configuration management for genomic resources.

Loads configuration from YAML file with environment variable overrides.

Ported from: meta_spliceai/system/genomic_resources/config.py
"""

from dataclasses import dataclass
from pathlib import Path
import os
import yaml


@dataclass
class Config:
    """Configuration for genomic resources."""
    species: str
    build: str
    release: str
    data_root: Path
    builds: dict
    derived_datasets: dict = None  # Filenames for derived datasets
    annotation_sources: dict = None  # Available annotation sources (Ensembl, MANE, GENCODE)
    base_models: dict = None  # Base model specifications
    default_annotation_source: str = "ensembl"  # Default annotation source
    
    def get_annotation_source(self, build: str = None) -> str:
        """Get annotation source for a build.
        
        Parameters
        ----------
        build : str, optional
            Build name. If None, uses current build.
            
        Returns
        -------
        str
            Annotation source name (e.g., 'ensembl', 'mane')
        """
        if build is None:
            build = self.build
        return self.builds[build].get("annotation_source", self.default_annotation_source)
    
    def get_data_dir(self, build: str = None, annotation_source: str = None) -> Path:
        """Get data directory for a build and annotation source.
        
        The directory structure is: data_root / annotation_source / build
        
        Parameters
        ----------
        build : str, optional
            Build name. If None, uses current build.
        annotation_source : str, optional
            Annotation source. If None, inferred from build.
            
        Returns
        -------
        Path
            Data directory path
            
        Examples
        --------
        >>> config.get_data_dir("GRCh37")
        Path("data/ensembl/GRCh37")
        
        >>> config.get_data_dir("GRCh38_MANE")
        Path("data/mane/GRCh38")
        """
        if build is None:
            build = self.build
        if annotation_source is None:
            annotation_source = self.get_annotation_source(build)
        
        # data_root / annotation_source / build
        # For MANE builds, strip the _MANE suffix from directory name
        build_dir = build.replace("_MANE", "").replace("_GENCODE", "")
        return self.data_root / annotation_source / build_dir
    
    def get_base_model_info(self, model_name: str) -> dict:
        """Get base model information.
        
        Parameters
        ----------
        model_name : str
            Base model name (e.g., 'spliceai', 'openspliceai')
            
        Returns
        -------
        dict
            Base model information including training build and annotation
        """
        if self.base_models and model_name in self.base_models:
            return self.base_models[model_name]
        return {}
    
    def get_base_model_build(self, model_name: str) -> str:
        """Get the genomic build for a base model.
        
        Parameters
        ----------
        model_name : str
            Base model name (e.g., 'spliceai', 'openspliceai')
            
        Returns
        -------
        str
            Genomic build (e.g., 'GRCh37', 'GRCh38')
            
        Raises
        ------
        ValueError
            If base model is not configured
            
        Examples
        --------
        >>> config.get_base_model_build('spliceai')
        'GRCh37'
        >>> config.get_base_model_build('openspliceai')
        'GRCh38'
        """
        info = self.get_base_model_info(model_name.lower())
        if not info:
            raise ValueError(
                f"Unknown base model: {model_name}. "
                f"Configured models: {list(self.base_models.keys()) if self.base_models else []}"
            )
        return info.get('training_build', self.build)
    
    def get_base_model_annotation_source(self, model_name: str) -> str:
        """Get the annotation source for a base model.
        
        Parameters
        ----------
        model_name : str
            Base model name (e.g., 'spliceai', 'openspliceai')
            
        Returns
        -------
        str
            Annotation source (e.g., 'ensembl', 'mane')
            
        Examples
        --------
        >>> config.get_base_model_annotation_source('spliceai')
        'ensembl'
        >>> config.get_base_model_annotation_source('openspliceai')
        'mane'
        """
        info = self.get_base_model_info(model_name.lower())
        if not info:
            return self.default_annotation_source
        return info.get('annotation_source', self.default_annotation_source)


def get_project_root() -> Path:
    """Get the project root directory.
    
    Uses markers to find the project root by looking for common project files.
    
    Returns
    -------
    Path
        Absolute path to the project root directory
    """
    return Path(find_project_root(os.path.dirname(__file__)))


def find_project_root(current_path: str = './') -> str:
    """Find the project root directory by looking for common project markers.
    
    Args:
        current_path: The path to start searching from (defaults to current directory)
        
    Returns:
        The absolute path to the project root directory
        
    Notes:
        This function looks for common project markers (.git, setup.py, etc.) to identify
        the project root. If no markers are found, it falls back to a reasonable default.
    """
    # Convert to absolute path if not already
    path = os.path.abspath(current_path)
    
    # Project root markers in order of preference
    # Note: .git and pyproject.toml are prioritized as they are definitive markers
    # 'agentic_spliceai' is removed as it can match src/agentic_spliceai
    root_markers = ['.git', 'pyproject.toml', 'setup.py', 'requirements.txt']
    
    # Start from the directory and move up until we find project markers
    while True:
        # Check if we've reached the filesystem root
        if path == os.path.dirname(path):  # Works on both Unix and Windows
            break
            
        # Check for project markers
        for marker in root_markers:
            if os.path.exists(os.path.join(path, marker)):
                return path
                
        # Move up one directory
        path = os.path.dirname(path)
    
    # If we reach here, we couldn't find a project root
    # For backward compatibility, fall back to 5 directories up from original path
    fallback_path = current_path
    for _ in range(5):
        fallback_path = os.path.dirname(fallback_path)
    
    return fallback_path


def load_config(path: str = None) -> Config:
    """Load configuration from YAML file with environment variable overrides.
    
    Parameters
    ----------
    path : str, optional
        Path to YAML config file. If None, uses default location.
        
    Returns
    -------
    Config
        Configuration object with species, build, release, data_root, and builds.
        
    Environment Variables
    ---------------------
    SS_SPECIES : str
        Override species (default: homo_sapiens)
    SS_BUILD : str
        Override build (default: GRCh38)
    SS_RELEASE : str
        Override release (default: 112)
    SS_DATA_ROOT : str
        Override data root directory (default: data)
    """
    # Get project root
    project_root = get_project_root()
    
    if path is None:
        # Look for config in splice_engine/config/ first, then project root
        config_locations = [
            Path(__file__).parent / "settings.yaml",
            project_root / "configs" / "genomic_resources.yaml",
            project_root / "config" / "genomic_resources.yaml",
        ]
        
        for config_path in config_locations:
            if config_path.exists():
                path = config_path
                break
        
        if path is None:
            raise FileNotFoundError(
                f"Config file not found. Searched:\n" +
                "\n".join(f"  - {p}" for p in config_locations) +
                f"\nPlease create a config file in one of these locations."
            )
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        y = yaml.safe_load(f)
    
    # Resolve data_root to absolute path
    data_root = Path(os.getenv("SS_DATA_ROOT", y["data_root"]))
    if not data_root.is_absolute():
        # Make relative to project root
        data_root = project_root / data_root
    
    return Config(
        species=os.getenv("SS_SPECIES", y["species"]),
        build=os.getenv("SS_BUILD", y["default_build"]),
        release=os.getenv("SS_RELEASE", y["default_release"]),
        data_root=data_root,
        builds=y["builds"],
        derived_datasets=y.get("derived_datasets", {}),
        annotation_sources=y.get("annotation_sources", {}),
        base_models=y.get("base_models", {}),
        default_annotation_source=y.get("default_annotation_source", "ensembl")
    )


def filename(kind: str, cfg: Config) -> str:
    """Get filename for a given resource kind.
    
    Parameters
    ----------
    kind : str
        Resource kind: 'gtf' or 'fasta'
    cfg : Config
        Configuration object
        
    Returns
    -------
    str
        Filename with release substituted
    """
    return cfg.builds[cfg.build][kind].format(release=cfg.release)
