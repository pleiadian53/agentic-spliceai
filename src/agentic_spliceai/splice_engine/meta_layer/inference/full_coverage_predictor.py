"""
Full Coverage Predictor for Meta-Layer (from pre-computed artifacts).

Produces per-nucleotide predictions [L, 3] matching the base model output format.
Uses pre-computed artifacts which already contain all features.

Key Design:
- Reads from production artifacts (pre-computed features + sequences)
- Batched inference through the meta-layer
- Outputs [L, 3] tensor matching base model format

Usage:
    from agentic_spliceai.splice_engine.meta_layer.inference import FullCoveragePredictor
    
    predictor = FullCoveragePredictor(meta_model, config)
    
    # Predict for a gene
    scores = predictor.predict_gene('ENSG00000000003')  # Returns [L, 3]
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import polars as pl
from tqdm import tqdm

from agentic_spliceai.splice_engine.meta_layer.core.config import MetaLayerConfig
from agentic_spliceai.splice_engine.meta_layer.core.artifact_loader import ArtifactLoader
from agentic_spliceai.splice_engine.meta_layer.data.dataset import MetaLayerDataset
from agentic_spliceai.splice_engine.meta_layer.core.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)


class FullCoveragePredictor:
    """
    Produces per-nucleotide predictions [L, 3] for entire genes.
    
    This predictor uses pre-computed artifacts from the base layer,
    which already contain:
    - 501nt contextual sequences for each position
    - Base model scores (donor, acceptor, neither)
    - Derived features (context scores, -omics features)
    
    For genes NOT in artifacts, use inference.FullCoverageFromScratch instead.
    
    Parameters
    ----------
    meta_model : torch.nn.Module
        Trained meta-layer model
    config : MetaLayerConfig
        Configuration with path resolution
    batch_size : int
        Batch size for inference (default: 256)
    device : str, optional
        Device for inference. Auto-detected if None.
    """
    
    def __init__(
        self,
        meta_model: torch.nn.Module,
        config: MetaLayerConfig,
        batch_size: int = 256,
        device: Optional[str] = None
    ):
        self.meta_model = meta_model
        self.config = config
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            device = config.get_device()
        self.device = torch.device(device)
        
        # Move model to device
        self.meta_model = self.meta_model.to(self.device)
        self.meta_model.eval()
        
        # Load artifacts
        self.loader = ArtifactLoader(config)
        self.schema = FeatureSchema()
        
        # Cache for loaded data
        self._cached_data: Optional[pl.DataFrame] = None
        self._cached_chromosomes: Optional[List[str]] = None
        
        logger.info(f"FullCoveragePredictor initialized on {self.device}")
    
    def load_gene_data(
        self,
        gene_id: str,
        chromosomes: Optional[List[str]] = None
    ) -> Optional[pl.DataFrame]:
        """
        Load pre-computed features for a gene from artifacts.
        
        Parameters
        ----------
        gene_id : str
            Gene ID (e.g., 'ENSG00000000003' or 'UNC13A')
        chromosomes : list of str, optional
            Chromosomes to search. If None, searches all.
        
        Returns
        -------
        pl.DataFrame or None
            Gene data with all positions and features, or None if not found
        """
        # Use cached data if available
        if self._cached_data is None or self._cached_chromosomes != chromosomes:
            logger.info("Loading artifacts...")
            self._cached_data = self.loader.load_analysis_sequences(
                chromosomes=chromosomes,
                verbose=True
            )
            self._cached_chromosomes = chromosomes
        
        # Filter to gene
        gene_data = self._cached_data.filter(
            (pl.col('gene_id') == gene_id) | 
            (pl.col('gene_id').str.contains(gene_id))
        )
        
        if len(gene_data) == 0:
            logger.warning(f"Gene {gene_id} not found in artifacts")
            return None
        
        # Sort by position
        gene_data = gene_data.sort('position')
        
        return gene_data
    
    def predict_gene(
        self,
        gene_id: str,
        chromosomes: Optional[List[str]] = None,
        return_positions: bool = True
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Predict splice site scores for all positions in a gene.
        
        Parameters
        ----------
        gene_id : str
            Gene ID
        chromosomes : list of str, optional
            Chromosomes to search
        return_positions : bool
            Include position coordinates in output
        
        Returns
        -------
        dict or None
            Dictionary containing:
            - 'meta_scores': np.ndarray [L, 3] - Meta-layer predictions
            - 'base_scores': np.ndarray [L, 3] - Base model scores (from artifacts)
            - 'delta': np.ndarray [L, 3] - Difference (meta - base)
            - 'positions': np.ndarray [L] - Genomic positions (if return_positions)
            - 'labels': np.ndarray [L] - Ground truth labels (if available)
        """
        # Load gene data
        gene_data = self.load_gene_data(gene_id, chromosomes)
        if gene_data is None:
            return None
        
        L = len(gene_data)
        logger.info(f"Predicting {L} positions for gene {gene_id}")
        
        # Create dataset from gene data
        dataset = MetaLayerDataset(
            gene_data,
            check_leakage=False  # Inference mode
        )
        
        # Batch inference
        meta_scores = self._batch_inference(dataset)
        
        # Extract base model scores from artifacts
        base_score_cols = ['donor_score', 'acceptor_score', 'neither_score']
        base_scores = gene_data.select(base_score_cols).to_numpy()
        
        # Compute delta
        delta = meta_scores - base_scores
        
        result = {
            'meta_scores': meta_scores,
            'base_scores': base_scores,
            'delta': delta,
            'gene_id': gene_id,
            'num_positions': L
        }
        
        if return_positions:
            result['positions'] = gene_data['position'].to_numpy()
        
        if 'splice_type' in gene_data.columns:
            result['labels'] = gene_data['splice_type'].to_numpy()
        
        return result
    
    def _batch_inference(self, dataset: MetaLayerDataset) -> np.ndarray:
        """Run batched inference through meta-layer."""
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Preserve order!
            num_workers=0
        )
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference", disable=len(dataset) < 1000):
                sequences = batch['sequence'].to(self.device)
                features = batch['features'].to(self.device)
                
                logits = self.meta_model(sequences, features)
                probs = F.softmax(logits, dim=-1)
                
                all_predictions.append(probs.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        
        return predictions
    
    def predict_chromosome(
        self,
        chromosome: str,
        max_genes: Optional[int] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Predict for all genes on a chromosome.
        
        Parameters
        ----------
        chromosome : str
            Chromosome to process
        max_genes : int, optional
            Maximum number of genes (for testing)
        
        Returns
        -------
        dict
            Mapping from gene_id to prediction results
        """
        chrom_data = self.loader.load_analysis_sequences(
            chromosomes=[chromosome],
            verbose=True
        )
        
        gene_ids = chrom_data['gene_id'].unique().to_list()
        
        if max_genes:
            gene_ids = gene_ids[:max_genes]
        
        logger.info(f"Predicting {len(gene_ids)} genes on chromosome {chromosome}")
        
        self._cached_data = chrom_data
        self._cached_chromosomes = [chromosome]
        
        results = {}
        for gene_id in tqdm(gene_ids, desc=f"Chr {chromosome}"):
            result = self.predict_gene(gene_id, chromosomes=[chromosome])
            if result:
                results[gene_id] = result
        
        return results
    
    def compare_with_base_model(
        self,
        gene_id: str,
        threshold: float = 0.5
    ) -> Optional[Dict]:
        """
        Compare meta-layer predictions with base model.
        
        Returns comparison metrics.
        """
        result = self.predict_gene(gene_id)
        if result is None:
            return None
        
        meta_scores = result['meta_scores']
        base_scores = result['base_scores']
        labels = result.get('labels')
        
        meta_preds = np.argmax(meta_scores, axis=1)
        base_preds = np.argmax(base_scores, axis=1)
        
        comparison = {
            'gene_id': gene_id,
            'num_positions': result['num_positions'],
            'meta_donor_mean': float(meta_scores[:, 0].mean()),
            'base_donor_mean': float(base_scores[:, 0].mean()),
            'meta_acceptor_mean': float(meta_scores[:, 1].mean()),
            'base_acceptor_mean': float(base_scores[:, 1].mean()),
            'agreement_rate': float((meta_preds == base_preds).mean()),
            'max_delta': float(np.abs(result['delta']).max()),
            'mean_delta': float(np.abs(result['delta']).mean()),
        }
        
        if labels is not None:
            label_to_idx = {'donor': 0, 'acceptor': 1, '': 2, 'neither': 2}
            true_labels = np.array([label_to_idx.get(l, 2) for l in labels])
            
            comparison['meta_accuracy'] = float((meta_preds == true_labels).mean())
            comparison['base_accuracy'] = float((base_preds == true_labels).mean())
            comparison['accuracy_improvement'] = (
                comparison['meta_accuracy'] - comparison['base_accuracy']
            )
        
        return comparison


def create_full_coverage_predictor(
    meta_model: torch.nn.Module,
    base_model: str = 'openspliceai',
    batch_size: int = 256,
    device: Optional[str] = None
) -> FullCoveragePredictor:
    """
    Convenience function to create a FullCoveragePredictor.
    
    Parameters
    ----------
    meta_model : torch.nn.Module
        Trained meta-layer model
    base_model : str
        Base model name ('openspliceai' or 'spliceai')
    batch_size : int
        Batch size for inference
    device : str, optional
        Device for inference
    
    Returns
    -------
    FullCoveragePredictor
        Configured predictor
    """
    config = MetaLayerConfig(base_model=base_model)
    return FullCoveragePredictor(
        meta_model=meta_model,
        config=config,
        batch_size=batch_size,
        device=device
    )












