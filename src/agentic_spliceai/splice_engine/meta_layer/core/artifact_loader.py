"""
Artifact loader for base layer outputs.

Provides unified access to base layer artifacts regardless of which
base model was used to generate them.

This module integrates with the resources package for consistent
path resolution across the entire Agentic-SpliceAI system.

Path Convention:
    data/<annotation_source>/<build>/<base_model>_eval/meta_models/

Examples:
    - data/ensembl/GRCh37/spliceai_eval/meta_models/
    - data/mane/GRCh38/openspliceai_eval/meta_models/

Ported from: meta_spliceai/splice_engine/meta_layer/core/artifact_loader.py
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import polars as pl
from tqdm import tqdm

from .config import MetaLayerConfig

logger = logging.getLogger(__name__)


class ArtifactLoader:
    """
    Load base layer artifacts for any supported base model.
    
    This class provides a unified interface to load analysis_sequences,
    splice_errors, and other artifacts from the base layer, regardless
    of which base model (SpliceAI, OpenSpliceAI, etc.) generated them.
    
    Parameters
    ----------
    config : MetaLayerConfig
        Configuration specifying base model and paths.
    
    Examples
    --------
    >>> config = MetaLayerConfig(base_model='openspliceai')
    >>> loader = ArtifactLoader(config)
    >>> 
    >>> # Load all analysis sequences
    >>> df = loader.load_analysis_sequences()
    >>> print(f"Loaded {len(df)} positions")
    >>> 
    >>> # Load specific chromosomes
    >>> df_chr21 = loader.load_analysis_sequences(chromosomes=['21'])
    """
    
    def __init__(self, config: MetaLayerConfig):
        """
        Initialize artifact loader with configuration.
        
        Uses the Registry (via config) for path resolution,
        ensuring consistency with the rest of the Agentic-SpliceAI system.
        
        Parameters
        ----------
        config : MetaLayerConfig
            Configuration specifying base model and paths.
            The config uses Registry internally.
        """
        self.config = config
        self.artifacts_dir = config.artifacts_dir
        
        # Access the registry through config for additional path resolution
        self._registry = config.registry
        
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: {self.artifacts_dir}\n"
                f"Have you run the base layer for {config.base_model}?\n"
                f"Expected path: {self.artifacts_dir}\n"
                f"Annotation source: {config.annotation_source}\n"
                f"Genome build: {config.genome_build}"
            )
        
        logger.info(f"ArtifactLoader initialized for {config.base_model}")
        logger.info(f"  Artifacts dir: {self.artifacts_dir}")
        logger.info(f"  Annotation source: {config.annotation_source}")
        logger.info(f"  Genome build: {config.genome_build}")
    
    def load_analysis_sequences(
        self,
        chromosomes: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        verbose: bool = True
    ) -> pl.DataFrame:
        """
        Load analysis_sequences artifacts.
        
        These contain per-position features including:
        - 501nt contextual sequences
        - Base model scores (donor, acceptor, neither)
        - 50+ derived features
        - Labels (splice_type)
        
        Parameters
        ----------
        chromosomes : list of str, optional
            Specific chromosomes to load. If None, loads all.
        columns : list of str, optional
            Specific columns to load. If None, loads all.
        verbose : bool
            Show progress bar.
        
        Returns
        -------
        pl.DataFrame
            Combined analysis sequences from all chunks.
        """
        pattern = "analysis_sequences_*.tsv"
        files = sorted(self.artifacts_dir.glob(pattern))
        
        if not files:
            raise FileNotFoundError(
                f"No analysis_sequences files found in {self.artifacts_dir}"
            )
        
        # Filter by chromosome if specified
        if chromosomes:
            chromosomes_set = set(chromosomes)
            files = [
                f for f in files 
                if self._extract_chrom(f) in chromosomes_set
            ]
        
        if not files:
            raise ValueError(f"No files found for chromosomes: {chromosomes}")
        
        # Load files
        dfs = []
        iterator = tqdm(files, desc="Loading artifacts") if verbose else files
        
        for file_path in iterator:
            df = pl.read_csv(file_path, separator='\t')
            if columns:
                available_cols = [c for c in columns if c in df.columns]
                df = df.select(available_cols)
            dfs.append(df)
        
        # Find common columns across all dataframes
        if dfs:
            common_cols = set(dfs[0].columns)
            for df in dfs[1:]:
                common_cols &= set(df.columns)
            common_cols = sorted(common_cols)  # Ensure consistent order
            
            if len(common_cols) < len(dfs[0].columns):
                logger.warning(
                    f"Schema mismatch detected. Using {len(common_cols)} common columns."
                )
            
            # Select only common columns and ensure consistent types
            # Use the first dataframe's schema as reference
            reference_schema = {c: dfs[0][c].dtype for c in common_cols if c in dfs[0].columns}
            
            aligned_dfs = []
            for df in dfs:
                df_selected = df.select(common_cols)
                # Cast columns to match reference schema
                try:
                    for col, dtype in reference_schema.items():
                        if col in df_selected.columns and df_selected[col].dtype != dtype:
                            df_selected = df_selected.with_columns(
                                pl.col(col).cast(dtype, strict=False)
                            )
                except Exception as e:
                    logger.debug(f"Type casting warning: {e}")
                aligned_dfs.append(df_selected)
            
            dfs = aligned_dfs
        
        # Combine with diagonal (handles any remaining schema differences)
        try:
            combined = pl.concat(dfs, how='diagonal')
        except Exception:
            # Fallback to align
            combined = pl.concat(dfs, how='align')
        
        logger.info(f"Loaded {len(combined)} positions from {len(files)} files")
        
        return combined
    
    def load_splice_errors(
        self,
        chromosomes: Optional[List[str]] = None,
        error_types: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Load splice_errors artifacts (FP/FN classifications).
        
        Parameters
        ----------
        chromosomes : list of str, optional
            Specific chromosomes to load.
        error_types : list of str, optional
            Filter by error type ('FP', 'FN').
        
        Returns
        -------
        pl.DataFrame
            Error positions with classifications.
        """
        pattern = "splice_errors_*.tsv"
        files = sorted(self.artifacts_dir.glob(pattern))
        
        if not files:
            logger.warning("No splice_errors files found")
            return pl.DataFrame()
        
        # Filter by chromosome
        if chromosomes:
            chromosomes_set = set(chromosomes)
            files = [
                f for f in files 
                if self._extract_chrom(f) in chromosomes_set
            ]
        
        dfs = [pl.read_csv(f, separator='\t') for f in files]
        combined = pl.concat(dfs)
        
        # Filter by error type
        if error_types and 'error_type' in combined.columns:
            combined = combined.filter(pl.col('error_type').is_in(error_types))
        
        return combined
    
    def load_gene_manifest(self) -> pl.DataFrame:
        """Load gene processing manifest."""
        manifest_path = self.artifacts_dir / "gene_manifest.tsv"
        
        if manifest_path.exists():
            return pl.read_csv(manifest_path, separator='\t')
        else:
            logger.warning(f"Gene manifest not found: {manifest_path}")
            return pl.DataFrame()
    
    def get_feature_columns(self, df: Optional[pl.DataFrame] = None) -> Dict[str, List[str]]:
        """
        Get categorized feature columns from analysis_sequences.
        
        Returns
        -------
        dict
            Dictionary mapping category to list of column names:
            - 'sequence': The sequence column
            - 'base_scores': donor_score, acceptor_score, neither_score
            - 'context_scores': context_score_m2, m1, p1, p2
            - 'derived_features': All other numeric features
            - 'labels': splice_type, pred_type
            - 'metadata': gene_id, transcript_id, chrom, etc.
        """
        if df is None:
            # Load a sample to get columns
            files = list(self.artifacts_dir.glob("analysis_sequences_*.tsv"))
            if not files:
                raise FileNotFoundError("No analysis_sequences files found")
            df = pl.read_csv(files[0], separator='\t', n_rows=1)
        
        all_columns = df.columns
        
        categories = {
            'sequence': ['sequence'],
            
            'base_scores': [
                'donor_score', 'acceptor_score', 'neither_score'
            ],
            
            'context_scores': [
                'context_score_m2', 'context_score_m1',
                'context_score_p1', 'context_score_p2'
            ],
            
            'labels': ['splice_type', 'pred_type'],
            
            'metadata': [
                'gene_id', 'transcript_id', 'chrom', 'strand',
                'position', 'window_start', 'window_end', 'transcript_count'
            ]
        }
        
        # Derived features = all numeric columns not in other categories
        known_cols = set()
        for cols in categories.values():
            known_cols.update(cols)
        
        derived = []
        for col in all_columns:
            if col not in known_cols:
                # Check if numeric
                if col in df.columns and df[col].dtype in [
                    pl.Float32, pl.Float64, pl.Int32, pl.Int64
                ]:
                    derived.append(col)
        
        categories['derived_features'] = derived
        
        # Filter to only existing columns
        for category, cols in categories.items():
            categories[category] = [c for c in cols if c in all_columns]
        
        return categories
    
    def _extract_chrom(self, file_path: Path) -> str:
        """Extract chromosome from filename like analysis_sequences_1_chunk_1_500.tsv"""
        name = file_path.stem  # analysis_sequences_1_chunk_1_500
        parts = name.split('_')
        
        # Find chromosome (after 'sequences' and before 'chunk')
        for i, part in enumerate(parts):
            if part == 'sequences' and i + 1 < len(parts):
                return parts[i + 1]
        
        return ''
    
    def get_available_chromosomes(self) -> List[str]:
        """Get list of chromosomes with available artifacts."""
        files = list(self.artifacts_dir.glob("analysis_sequences_*.tsv"))
        chromosomes = set()
        
        for f in files:
            chrom = self._extract_chrom(f)
            if chrom:
                chromosomes.add(chrom)
        
        # Sort chromosomes (1-22, X, Y)
        def chrom_key(c):
            try:
                return (0, int(c))
            except ValueError:
                return (1, c)
        
        return sorted(chromosomes, key=chrom_key)
    
    def get_statistics(self) -> Dict:
        """Get summary statistics about available artifacts."""
        files = list(self.artifacts_dir.glob("analysis_sequences_*.tsv"))
        
        total_positions = 0
        file_sizes = []
        
        for f in files:
            # Quick line count (approximate)
            with open(f) as fh:
                lines = sum(1 for _ in fh) - 1  # Subtract header
            total_positions += lines
            file_sizes.append(f.stat().st_size)
        
        return {
            'base_model': self.config.base_model,
            'genome_build': self.config.genome_build,
            'artifacts_dir': str(self.artifacts_dir),
            'num_files': len(files),
            'chromosomes': self.get_available_chromosomes(),
            'total_positions': total_positions,
            'total_size_mb': sum(file_sizes) / (1024 * 1024)
        }

