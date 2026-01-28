"""Data types for base layer predictions.

Provides dataclasses for gene manifests and prediction results.

Ported from: meta_spliceai/splice_engine/meta_models/core/data_types.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime

import pandas as pd
import polars as pl


@dataclass
class GeneManifestEntry:
    """Single entry in the gene manifest tracking gene processing status.
    
    Attributes
    ----------
    gene_id : str
        Gene identifier (e.g., ENSG00000012048)
    gene_name : str
        Gene symbol (e.g., BRCA1)
    requested : bool
        Whether this gene was explicitly requested
    status : str
        Processing status: 'processed', 'not_in_annotation', 'no_sequence', 
        'sequence_too_short', 'prediction_failed'
    reason : Optional[str]
        Explanation if status != 'processed'
    num_positions : int
        Number of positions analyzed (0 if failed)
    num_nucleotides : int
        Total sequence length (0 if failed)
    num_splice_sites : int
        Number of annotated splice sites (0 if failed)
    processing_time_sec : float
        Time taken to process this gene
    base_model : str
        Base model used ('spliceai', 'openspliceai', etc.)
    genomic_build : str
        Genomic build ('GRCh37', 'GRCh38', etc.)
    timestamp : str
        ISO format timestamp of processing
    """
    gene_id: str
    gene_name: str
    requested: bool
    status: str
    reason: Optional[str] = None
    num_positions: int = 0
    num_nucleotides: int = 0
    num_splice_sites: int = 0
    processing_time_sec: float = 0.0
    base_model: str = "spliceai"
    genomic_build: str = "GRCh37"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return {
            'gene_id': self.gene_id,
            'gene_name': self.gene_name,
            'requested': self.requested,
            'status': self.status,
            'reason': self.reason,
            'num_positions': self.num_positions,
            'num_nucleotides': self.num_nucleotides,
            'num_splice_sites': self.num_splice_sites,
            'processing_time_sec': self.processing_time_sec,
            'base_model': self.base_model,
            'genomic_build': self.genomic_build,
            'timestamp': self.timestamp
        }


class GeneManifest:
    """Tracks gene processing status across the workflow.
    
    This class maintains a record of all genes that were requested for processing,
    their processing status, and diagnostic information for failed genes.
    
    Examples
    --------
    >>> manifest = GeneManifest(base_model='spliceai', genomic_build='GRCh37')
    >>> manifest.add_requested_genes(['BRCA1', 'TP53', 'UNKNOWN_GENE'])
    >>> manifest.mark_processed('BRCA1', num_positions=1234, num_nucleotides=5592)
    >>> manifest.mark_failed('UNKNOWN_GENE', status='not_in_annotation')
    >>> df = manifest.to_dataframe()
    """
    
    def __init__(self, base_model: str = 'spliceai', genomic_build: str = 'GRCh37'):
        """Initialize gene manifest.
        
        Parameters
        ----------
        base_model : str
            Base model being used
        genomic_build : str
            Genomic build being used
        """
        self.base_model = base_model
        self.genomic_build = genomic_build
        self.entries: Dict[str, GeneManifestEntry] = {}
        self._requested_genes: Set[str] = set()
    
    def add_requested_genes(self, gene_ids: Union[List[str], Set[str]]):
        """Mark genes as requested for processing."""
        self._requested_genes.update(gene_ids)
        
        for gene_id in gene_ids:
            if gene_id not in self.entries:
                self.entries[gene_id] = GeneManifestEntry(
                    gene_id=gene_id,
                    gene_name=gene_id,
                    requested=True,
                    status='pending',
                    base_model=self.base_model,
                    genomic_build=self.genomic_build
                )
    
    def mark_processed(
        self,
        gene_id: str,
        gene_name: Optional[str] = None,
        num_positions: int = 0,
        num_nucleotides: int = 0,
        num_splice_sites: int = 0,
        processing_time: float = 0.0
    ):
        """Mark a gene as successfully processed."""
        self.entries[gene_id] = GeneManifestEntry(
            gene_id=gene_id,
            gene_name=gene_name or gene_id,
            requested=gene_id in self._requested_genes,
            status='processed',
            reason=None,
            num_positions=num_positions,
            num_nucleotides=num_nucleotides,
            num_splice_sites=num_splice_sites,
            processing_time_sec=processing_time,
            base_model=self.base_model,
            genomic_build=self.genomic_build
        )
    
    def mark_failed(
        self,
        gene_id: str,
        gene_name: Optional[str] = None,
        status: str = 'prediction_failed',
        reason: Optional[str] = None
    ):
        """Mark a gene as failed to process."""
        self.entries[gene_id] = GeneManifestEntry(
            gene_id=gene_id,
            gene_name=gene_name or gene_id,
            requested=gene_id in self._requested_genes,
            status=status,
            reason=reason,
            num_positions=0,
            num_nucleotides=0,
            num_splice_sites=0,
            processing_time_sec=0.0,
            base_model=self.base_model,
            genomic_build=self.genomic_build
        )
    
    def to_dataframe(self, use_polars: bool = True) -> Union[pl.DataFrame, pd.DataFrame]:
        """Convert manifest to DataFrame."""
        data = [entry.to_dict() for entry in self.entries.values()]
        
        if use_polars:
            return pl.DataFrame(data) if data else pl.DataFrame()
        else:
            return pd.DataFrame(data)
    
    def save(self, path: str, use_polars: bool = True):
        """Save manifest to TSV file."""
        df = self.to_dataframe(use_polars=use_polars)
        
        if use_polars:
            df.write_csv(path, separator='\t')
        else:
            df.to_csv(path, sep='\t', index=False)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the manifest."""
        df = self.to_dataframe(use_polars=True)
        
        if df.height == 0:
            return {
                'total_genes': 0,
                'requested_genes': 0,
                'processed_genes': 0,
                'failed_genes': 0,
                'status_counts': {}
            }
        
        status_counts = df.group_by('status').agg(pl.count()).to_dict(as_series=False)
        
        return {
            'total_genes': df.height,
            'requested_genes': df.filter(pl.col('requested')).height,
            'processed_genes': df.filter(pl.col('status') == 'processed').height,
            'failed_genes': df.filter(pl.col('status') != 'processed').height,
            'status_counts': dict(zip(status_counts['status'], status_counts['count'])),
            'total_processing_time_sec': df['processing_time_sec'].sum(),
            'base_model': self.base_model,
            'genomic_build': self.genomic_build
        }


@dataclass
class PredictionResult:
    """Container for prediction results.
    
    Attributes
    ----------
    positions : pl.DataFrame
        All analyzed positions with predictions
    error_analysis : pl.DataFrame
        Positions with errors (FP, FN)
    analysis_sequences : pl.DataFrame
        Sequences around each position
    nucleotide_scores : Optional[pl.DataFrame]
        Per-nucleotide scores (if enabled)
    gene_manifest : GeneManifest
        Gene processing manifest
    paths : Dict[str, str]
        Output file paths
    success : bool
        Whether prediction completed successfully
    """
    positions: pl.DataFrame
    error_analysis: Optional[pl.DataFrame] = None
    analysis_sequences: Optional[pl.DataFrame] = None
    nucleotide_scores: Optional[pl.DataFrame] = None
    gene_manifest: Optional[GeneManifest] = None
    paths: Dict[str, str] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of prediction results."""
        return {
            'success': self.success,
            'total_positions': self.positions.height if self.positions is not None else 0,
            'error_positions': self.error_analysis.height if self.error_analysis is not None else 0,
            'has_sequences': self.analysis_sequences is not None and self.analysis_sequences.height > 0,
            'has_nucleotide_scores': self.nucleotide_scores is not None,
            'gene_summary': self.gene_manifest.get_summary() if self.gene_manifest else None,
            'paths': self.paths
        }
