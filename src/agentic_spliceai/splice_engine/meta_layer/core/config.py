"""
Configuration for the meta-layer.

Provides a unified configuration system that handles:
- Base model selection (automatic artifact routing)
- Sequence encoder configuration (HyenaDNA, CNN, etc.)
- Training hyperparameters
- Variant data integration
- Safe path management (read from production, write to isolated dev)

Key Design:
- READ paths: Always from production artifacts (pre-computed, complete)
- WRITE paths: To isolated timestamped dev directories (prevents overwrites)

Ported from: meta_spliceai/splice_engine/meta_layer/core/config.py
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union
import yaml


@dataclass
class MetaLayerConfig:
    """
    Configuration for the meta-learning layer.
    
    This config follows the same design philosophy as the base layer:
    - Single parameter (base_model) routes to correct artifacts
    - Automatic path resolution
    - Sensible defaults
    
    Parameters
    ----------
    base_model : str
        Base model to use ('spliceai', 'openspliceai', etc.)
        Automatically routes to correct artifact directory.
    
    sequence_encoder : str
        DNA sequence encoder type:
        - 'hyenadna': State space model (best, requires GPU)
        - 'dnabert2': Transformer-based (good, moderate GPU)
        - 'cnn': Lightweight CNN (fast, CPU-friendly)
        - 'none': No sequence encoding (score-only baseline)
    
    variant_source : str, optional
        External variant database for training enhancement:
        - 'splicevardb': SpliceVarDB validated variants
        - 'clinvar': ClinVar pathogenic variants
        - None: Use base layer artifacts only
    
    Examples
    --------
    >>> config = MetaLayerConfig(
    ...     base_model='openspliceai',
    ...     sequence_encoder='hyenadna',
    ...     variant_source='splicevardb'
    ... )
    >>> print(config.artifacts_dir)
    data/mane/GRCh38/openspliceai_eval/meta_models
    """
    
    # Base model selection
    base_model: str = 'openspliceai'
    
    # Sequence encoder
    sequence_encoder: str = 'cnn'  # Default to lightweight
    sequence_encoder_config: Dict = field(default_factory=lambda: {
        'output_dim': 256,
        'pretrained': True,
        'freeze': True
    })
    
    # Variant data
    variant_source: Optional[str] = None
    variant_config: Dict = field(default_factory=lambda: {
        'use_soft_labels': True,
        'low_frequency_weight': 0.5
    })
    
    # Model architecture
    hidden_dim: int = 256
    num_attention_heads: int = 8
    dropout: float = 0.1
    
    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    patience: int = 10  # Early stopping
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    
    # Data configuration
    max_samples: Optional[int] = None  # Subset for quick testing
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    balance_classes: bool = True
    
    # Paths (auto-resolved from base_model)
    data_root: Path = field(default_factory=lambda: Path('.'))
    output_dir: Optional[Path] = None
    
    # Evaluation
    top_k_values: List[int] = field(default_factory=lambda: [10, 50, 100, 500])
    threshold: float = 0.5
    
    # Runtime
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    num_workers: int = 4
    seed: int = 42
    verbosity: int = 1
    
    # Development session ID (for isolated write paths)
    dev_session_id: Optional[str] = None  # None = auto-generate timestamp
    
    # Internal cache for genomic resources
    _genomic_config: Optional[object] = field(default=None, repr=False)
    _registry: Optional[object] = field(default=None, repr=False)
    _path_manager: Optional[object] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate and resolve paths after initialization."""
        # Normalize base model name
        self.base_model = self.base_model.lower()
        
        # Convert paths
        self.data_root = Path(self.data_root)
        if self.output_dir:
            self.output_dir = Path(self.output_dir)
        
        # Initialize genomic resources (lazy loading)
        self._init_genomic_resources()
        
        # Initialize path manager for safe read/write separation
        self._init_path_manager()
    
    def _init_genomic_resources(self):
        """Initialize genomic resources configuration and registry."""
        try:
            from agentic_spliceai.splice_engine.config import load_config
            from agentic_spliceai.splice_engine.resources import get_genomic_registry
            
            # Load genomic config
            self._genomic_config = load_config()
            
            # Validate base model is configured
            info = self._genomic_config.get_base_model_info(self.base_model)
            if not info:
                available = list(self._genomic_config.base_models.keys()) if self._genomic_config.base_models else []
                raise ValueError(
                    f"Unknown base model: {self.base_model}. "
                    f"Configured in settings.yaml: {available}"
                )
            
            # Get build for this base model and create registry
            build = self._genomic_config.get_base_model_build(self.base_model)
            # Handle MANE suffix for OpenSpliceAI
            if self.base_model == 'openspliceai':
                build = 'GRCh38_MANE'
            
            self._registry = get_genomic_registry(build=build)
            
        except (ImportError, FileNotFoundError) as e:
            # Allow config to be created even without genomic resources
            # This enables testing and development without full setup
            import logging
            logging.getLogger(__name__).warning(
                f"Could not initialize genomic resources: {e}. "
                "Using fallback path resolution."
            )
            self._genomic_config = None
            self._registry = None
    
    def _init_path_manager(self):
        """Initialize path manager for safe read/write separation."""
        try:
            from .path_manager import MetaLayerPathManager
            
            self._path_manager = MetaLayerPathManager(
                base_model=self.base_model,
                dev_session_id=self.dev_session_id
            )
            
        except Exception as e:
            # Path manager is optional - fall back to legacy paths
            import logging
            logging.getLogger(__name__).warning(
                f"Could not initialize path manager: {e}. "
                "Using legacy path resolution."
            )
            self._path_manager = None
    
    @property
    def artifacts_dir(self) -> Path:
        """Get artifact directory for the selected base model.
        
        Uses the genomic resources Registry for consistent path resolution.
        Path structure: data/<source>/<build>/<base_model>_eval/meta_models/
        """
        if self._registry is not None:
            return self._registry.get_meta_models_artifact_dir(self.base_model)
        # Fallback if registry not available
        return self.data_root / 'artifacts' / self.base_model
    
    @property
    def genome_build(self) -> str:
        """Get genome build for the selected base model.
        
        Uses the genomic resources config for consistent build mapping.
        """
        if self._genomic_config is not None:
            return self._genomic_config.get_base_model_build(self.base_model)
        # Fallback defaults
        return 'GRCh38' if self.base_model == 'openspliceai' else 'GRCh37'
    
    @property
    def annotation_source(self) -> str:
        """Get annotation source for the selected base model.
        
        Uses the genomic resources config for consistent source mapping.
        """
        if self._genomic_config is not None:
            return self._genomic_config.get_base_model_annotation_source(self.base_model)
        # Fallback defaults
        return 'mane' if self.base_model == 'openspliceai' else 'ensembl'
    
    @property
    def coordinate_column(self) -> str:
        """Get coordinate column for SpliceVarDB based on genome build."""
        if self.genome_build == 'GRCh37':
            return 'hg19'
        else:
            return 'hg38'
    
    @property
    def registry(self) -> object:
        """Get the genomic resources Registry for this base model."""
        if self._registry is None:
            self._init_genomic_resources()
        return self._registry
    
    @property
    def path_manager(self) -> object:
        """Get the path manager for safe read/write separation."""
        if self._path_manager is None:
            self._init_path_manager()
        return self._path_manager
    
    @property
    def output_write_dir(self) -> Path:
        """Get isolated output directory for development writes.
        
        This returns a timestamped directory that won't overwrite production:
            .../openspliceai_eval/meta_layer_dev/20251214_155918/
        """
        if self._path_manager:
            return self._path_manager.get_output_write_dir()
        elif self.output_dir:
            return self.output_dir
        else:
            # Fallback: create timestamped dir next to artifacts
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return self.artifacts_dir.parent / 'meta_layer_dev' / timestamp
    
    @property
    def checkpoints_dir(self) -> Path:
        """Get directory for model checkpoints."""
        if self._path_manager:
            return self._path_manager.get_model_checkpoint_dir()
        return self.output_write_dir / 'checkpoints'
    
    @property
    def predictions_dir(self) -> Path:
        """Get directory for prediction outputs."""
        if self._path_manager:
            return self._path_manager.get_predictions_dir()
        return self.output_write_dir / 'predictions'
    
    @property
    def evaluation_dir(self) -> Path:
        """Get directory for evaluation results."""
        if self._path_manager:
            return self._path_manager.get_evaluation_dir()
        return self.output_write_dir / 'evaluation'
    
    def get_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != 'auto':
            return self.device
        
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except ImportError:
            pass
        return 'cpu'
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'MetaLayerConfig':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict (excluding private fields)
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        # Convert Paths to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"MetaLayerConfig(\n"
            f"  base_model='{self.base_model}',\n"
            f"  sequence_encoder='{self.sequence_encoder}',\n"
            f"  variant_source={self.variant_source!r},\n"
            f"  genome_build='{self.genome_build}'\n"
            f")"
        )
