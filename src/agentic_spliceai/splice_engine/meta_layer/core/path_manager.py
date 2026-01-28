"""
Path Manager for Meta-Layer Development

This module provides safe path resolution for the meta-layer:
- READ: Always from production artifact directories (pre-computed, complete)
- WRITE: To isolated development directories with timestamps

This prevents accidental overwriting of production artifacts during development.

Production Artifact Directories (READ-ONLY):
    - data/mane/GRCh38/openspliceai_eval/meta_models/
    - data/ensembl/GRCh37/spliceai_eval/meta_models/

Development Output Directories (WRITE):
    - data/mane/GRCh38/openspliceai_eval/meta_layer_dev/{timestamp}/
    - data/ensembl/GRCh37/spliceai_eval/meta_layer_dev/{timestamp}/

Usage:
    from agentic_spliceai.splice_engine.meta_layer.core.path_manager import MetaLayerPathManager
    
    pm = MetaLayerPathManager(base_model='openspliceai')
    
    # Reading artifacts (from production)
    artifacts_dir = pm.get_artifacts_read_dir()  # → .../meta_models/
    
    # Writing outputs (to isolated dev directory)
    output_dir = pm.get_output_write_dir()  # → .../meta_layer_dev/20251214_143025/

Ported from: meta_spliceai/splice_engine/meta_layer/core/path_manager.py
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MetaLayerPathManager:
    """
    Manages read/write paths for meta-layer development.
    
    Ensures:
    - Production artifacts are read-only
    - Development outputs go to isolated timestamped directories
    - No accidental overwrites of production data
    
    Parameters
    ----------
    base_model : str
        Base model name ('openspliceai' or 'spliceai')
    dev_session_id : str, optional
        Custom session identifier. If None, uses timestamp.
    project_root : Path, optional
        Project root directory. Auto-detected if None.
    
    Examples
    --------
    >>> pm = MetaLayerPathManager(base_model='openspliceai')
    >>> 
    >>> # Read pre-computed artifacts
    >>> artifacts = pm.get_artifacts_read_dir()
    >>> print(f"Reading from: {artifacts}")
    >>> 
    >>> # Write development outputs
    >>> output = pm.get_output_write_dir()
    >>> print(f"Writing to: {output}")
    """
    
    # Production artifact directories (READ-ONLY)
    _PRODUCTION_PATHS = {
        'openspliceai': 'data/mane/GRCh38/openspliceai_eval/meta_models',
        'spliceai': 'data/ensembl/GRCh37/spliceai_eval/meta_models',
    }
    
    # Base paths for evaluation directories
    _EVAL_BASE_PATHS = {
        'openspliceai': 'data/mane/GRCh38/openspliceai_eval',
        'spliceai': 'data/ensembl/GRCh37/spliceai_eval',
    }
    
    def __init__(
        self,
        base_model: str = 'openspliceai',
        dev_session_id: Optional[str] = None,
        project_root: Optional[Path] = None
    ):
        self.base_model = base_model.lower()
        
        if self.base_model not in self._PRODUCTION_PATHS:
            raise ValueError(
                f"Unknown base model: {base_model}. "
                f"Supported: {list(self._PRODUCTION_PATHS.keys())}"
            )
        
        # Auto-detect project root
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)
        
        # Session ID for isolated writes
        if dev_session_id is None:
            self.dev_session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        else:
            self.dev_session_id = dev_session_id
        
        self._output_dir_created = False
    
    def _find_project_root(self) -> Path:
        """Find project root by looking for markers."""
        current = Path(__file__).resolve()
        
        for parent in current.parents:
            # Look for project markers (agentic_spliceai or pyproject.toml)
            if (parent / 'src' / 'agentic_spliceai').exists() or (parent / 'pyproject.toml').exists():
                return parent
        
        # Fallback to cwd
        return Path.cwd()
    
    # =========================================================================
    # READ PATHS (Production Artifacts)
    # =========================================================================
    
    def get_artifacts_read_dir(self) -> Path:
        """
        Get path to production artifact directory (READ-ONLY).
        
        Returns
        -------
        Path
            Path to pre-computed artifacts (analysis_sequences_*.tsv, etc.)
        
        Raises
        ------
        FileNotFoundError
            If production artifacts don't exist
        """
        rel_path = self._PRODUCTION_PATHS[self.base_model]
        artifacts_dir = self.project_root / rel_path
        
        if not artifacts_dir.exists():
            raise FileNotFoundError(
                f"Production artifacts not found at: {artifacts_dir}\n"
                f"Run the base model first: run_base_model --base-model {self.base_model}"
            )
        
        logger.debug(f"Artifacts READ path: {artifacts_dir}")
        return artifacts_dir
    
    def get_analysis_sequence_files(self) -> list:
        """
        Get list of analysis sequence artifact files.
        
        Returns
        -------
        list
            Sorted list of analysis_sequences_*.tsv files
        """
        artifacts_dir = self.get_artifacts_read_dir()
        files = sorted(artifacts_dir.glob('analysis_sequences_*.tsv'))
        
        if not files:
            raise FileNotFoundError(
                f"No analysis_sequences_*.tsv files found in: {artifacts_dir}"
            )
        
        return files
    
    def get_fasta_path(self) -> Path:
        """Get path to reference FASTA file."""
        if self.base_model == 'openspliceai':
            return self.project_root / 'data/mane/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna'
        else:  # spliceai
            return self.project_root / 'data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa'
    
    # =========================================================================
    # WRITE PATHS (Isolated Development)
    # =========================================================================
    
    def get_output_write_dir(self, create: bool = True) -> Path:
        """
        Get path to isolated development output directory (WRITE).
        
        This directory is timestamped to prevent overwrites:
            .../openspliceai_eval/meta_layer_dev/20251214_143025/
        
        Parameters
        ----------
        create : bool, default=True
            Create directory if it doesn't exist
        
        Returns
        -------
        Path
            Isolated output directory for this development session
        """
        eval_base = self.project_root / self._EVAL_BASE_PATHS[self.base_model]
        output_dir = eval_base / 'meta_layer_dev' / self.dev_session_id
        
        if create and not self._output_dir_created:
            output_dir.mkdir(parents=True, exist_ok=True)
            self._output_dir_created = True
            logger.info(f"Created dev output directory: {output_dir}")
        
        return output_dir
    
    def get_model_checkpoint_dir(self) -> Path:
        """Get directory for model checkpoints."""
        return self.get_output_write_dir() / 'checkpoints'
    
    def get_predictions_dir(self) -> Path:
        """Get directory for prediction outputs."""
        return self.get_output_write_dir() / 'predictions'
    
    def get_evaluation_dir(self) -> Path:
        """Get directory for evaluation results."""
        return self.get_output_write_dir() / 'evaluation'
    
    def get_logs_dir(self) -> Path:
        """Get directory for training logs."""
        return self.get_output_write_dir() / 'logs'
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def summary(self) -> Dict[str, Any]:
        """
        Get summary of all paths.
        
        Returns
        -------
        dict
            Summary with all path information
        """
        return {
            'base_model': self.base_model,
            'project_root': str(self.project_root),
            'dev_session_id': self.dev_session_id,
            'read': {
                'artifacts_dir': str(self.get_artifacts_read_dir()),
                'fasta_path': str(self.get_fasta_path()),
            },
            'write': {
                'output_dir': str(self.get_output_write_dir(create=False)),
                'checkpoints': str(self.get_model_checkpoint_dir()),
                'predictions': str(self.get_predictions_dir()),
                'evaluation': str(self.get_evaluation_dir()),
            }
        }
    
    def print_summary(self):
        """Print formatted path summary."""
        summary = self.summary()
        
        print("=" * 70)
        print("MetaLayerPathManager Summary")
        print("=" * 70)
        print(f"Base Model:     {summary['base_model']}")
        print(f"Session ID:     {summary['dev_session_id']}")
        print(f"Project Root:   {summary['project_root']}")
        print()
        print("READ Paths (Production Artifacts - DO NOT MODIFY):")
        print(f"  Artifacts:    {summary['read']['artifacts_dir']}")
        print(f"  FASTA:        {summary['read']['fasta_path']}")
        print()
        print("WRITE Paths (Isolated Development):")
        print(f"  Output Dir:   {summary['write']['output_dir']}")
        print(f"  Checkpoints:  {summary['write']['checkpoints']}")
        print(f"  Predictions:  {summary['write']['predictions']}")
        print(f"  Evaluation:   {summary['write']['evaluation']}")
        print("=" * 70)
    
    def __repr__(self) -> str:
        return (
            f"MetaLayerPathManager("
            f"base_model='{self.base_model}', "
            f"session='{self.dev_session_id}')"
        )


# Convenience function
def get_path_manager(
    base_model: str = 'openspliceai',
    session_id: Optional[str] = None
) -> MetaLayerPathManager:
    """
    Get a configured path manager.
    
    Parameters
    ----------
    base_model : str
        'openspliceai' or 'spliceai'
    session_id : str, optional
        Custom session ID. If None, uses timestamp.
    
    Returns
    -------
    MetaLayerPathManager
        Configured path manager
    """
    return MetaLayerPathManager(
        base_model=base_model,
        dev_session_id=session_id
    )

