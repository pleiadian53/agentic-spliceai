"""
Variant Dataset for Delta Prediction Training.

Prepares (ref_sequence, alt_sequence, delta_target) pairs from SpliceVarDB
for training the delta prediction model.

Key Features:
- Extracts sequences around variants from FASTA
- Computes base model delta scores as targets
- Supports class-based sample weighting
- Handles indels and edge cases

Usage:
    from agentic_spliceai.splice_engine.meta_layer.data.variant_dataset import (
        VariantDeltaDataset,
        prepare_variant_data
    )
    
    dataset = VariantDeltaDataset(
        samples=variant_samples,
        context_size=501
    )
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

Ported from: meta_spliceai/splice_engine/meta_layer/data/variant_dataset.py
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class VariantSample:
    """A single variant sample for delta prediction."""
    variant_id: str
    chrom: str
    position: int
    ref_allele: str
    alt_allele: str
    classification: str
    
    ref_sequence: str
    alt_sequence: str
    
    # Target delta scores (from base model)
    delta_donor: float
    delta_acceptor: float
    
    # Sample weight based on classification
    weight: float = 1.0


class VariantDeltaDataset(Dataset):
    """
    Dataset for variant delta prediction training.
    
    Each sample contains:
    - ref_sequence: Reference sequence around variant (one-hot)
    - alt_sequence: Alternate sequence with variant (one-hot)
    - delta_target: [Δ_donor, Δ_acceptor] from base model
    - weight: Sample weight based on classification
    
    Parameters
    ----------
    samples : list of VariantSample
        Pre-processed variant samples
    context_size : int
        Sequence context size (default: 501)
    
    Examples
    --------
    >>> dataset = VariantDeltaDataset(samples)
    >>> sample = dataset[0]
    >>> print(sample['ref_sequence'].shape)  # [4, 501]
    >>> print(sample['delta_target'].shape)  # [2]
    """
    
    def __init__(
        self,
        samples: List[VariantSample],
        context_size: int = 501
    ):
        self.samples = samples
        self.context_size = context_size
        
        # Class weights
        self.class_weights = {
            'Splice-altering': 2.0,
            'Normal': 1.0,
            'Low-frequency': 0.5,
            'Conflicting': 0.5
        }
        
        logger.info(f"VariantDeltaDataset: {len(samples)} samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # One-hot encode sequences
        ref_encoded = self._one_hot_encode(sample.ref_sequence)
        alt_encoded = self._one_hot_encode(sample.alt_sequence)
        
        # Delta target
        delta_target = np.array([
            sample.delta_donor,
            sample.delta_acceptor
        ], dtype=np.float32)
        
        # Get weight
        weight = self.class_weights.get(sample.classification, 1.0)
        
        return {
            'ref_sequence': torch.tensor(ref_encoded, dtype=torch.float32),
            'alt_sequence': torch.tensor(alt_encoded, dtype=torch.float32),
            'delta_target': torch.tensor(delta_target, dtype=torch.float32),
            'weight': torch.tensor(weight, dtype=torch.float32),
            'classification': sample.classification
        }
    
    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode a DNA sequence."""
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        seq = sequence.upper()
        
        # Ensure correct length
        if len(seq) < self.context_size:
            seq = seq + 'N' * (self.context_size - len(seq))
        elif len(seq) > self.context_size:
            seq = seq[:self.context_size]
        
        indices = np.array([mapping.get(n, 0) for n in seq])
        one_hot = np.zeros((4, len(seq)), dtype=np.float32)
        one_hot[indices, np.arange(len(seq))] = 1.0
        
        return one_hot
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of variant classifications."""
        from collections import Counter
        return Counter(s.classification for s in self.samples)


def prepare_variant_data(
    splicevardb_loader,
    fasta,
    base_models,
    base_device: str,
    chromosomes: Optional[List[str]] = None,
    context_size: int = 501,
    max_variants: Optional[int] = None,
    verbose: bool = True
) -> List[VariantSample]:
    """
    Prepare variant samples from SpliceVarDB.
    
    Parameters
    ----------
    splicevardb_loader : SpliceVarDBLoader
        Loaded SpliceVarDB data
    fasta : pyfaidx.Fasta
        Reference genome FASTA
    base_models : list
        Base model ensemble for computing deltas
    base_device : str
        Device for base model inference
    chromosomes : list of str, optional
        Chromosomes to include. If None, uses all.
    context_size : int
        Context window size
    max_variants : int, optional
        Maximum number of variants to process
    verbose : bool
        Show progress bar
    
    Returns
    -------
    list of VariantSample
        Processed variant samples
    """
    import torch.nn.functional as F
    
    # Get variants
    if chromosomes:
        variants = [
            v for v in splicevardb_loader.variants
            if str(v.chrom).replace('chr', '') in [str(c).replace('chr', '') for c in chromosomes]
        ]
    else:
        variants = splicevardb_loader.variants
    
    if max_variants:
        variants = variants[:max_variants]
    
    logger.info(f"Processing {len(variants)} variants...")
    
    samples = []
    iterator = tqdm(variants, desc="Preparing variants") if verbose else variants
    
    for v in iterator:
        try:
            sample = _process_single_variant(
                v, fasta, base_models, base_device, context_size
            )
            if sample:
                samples.append(sample)
        except Exception as e:
            logger.debug(f"Skipping variant {v.chrom}:{v.position}: {e}")
            continue
    
    logger.info(f"Prepared {len(samples)} variant samples")
    
    return samples


def _process_single_variant(
    variant,
    fasta,
    base_models,
    base_device: str,
    context_size: int
) -> Optional[VariantSample]:
    """Process a single variant into a training sample."""
    import torch
    import torch.nn.functional as F
    
    # Normalize chromosome
    chrom = str(variant.chrom)
    if chrom.startswith('chr'):
        chrom = chrom[3:]
    
    # Check if chromosome exists in FASTA
    fasta_chroms = list(fasta.keys())
    if chrom not in fasta_chroms:
        # Try with 'chr' prefix
        if f'chr{chrom}' in fasta_chroms:
            chrom = f'chr{chrom}'
        else:
            return None
    
    pos = variant.position
    half_ctx = context_size // 2
    
    # Get extended sequence for base model (needs more context)
    base_ctx = 6000  # Extended context for base model
    start = max(0, pos - base_ctx)
    end = pos + base_ctx
    
    try:
        full_seq = str(fasta[chrom][start:end].seq)
    except Exception:
        return None
    
    if len(full_seq) < 11000:
        return None
    
    # Find variant position in sequence
    var_pos_in_seq = pos - start - 1
    
    # Create alternate sequence
    ref_allele = variant.ref_allele
    alt_allele = variant.alt_allele
    
    alt_full_seq = (
        full_seq[:var_pos_in_seq] + 
        alt_allele + 
        full_seq[var_pos_in_seq + len(ref_allele):]
    )
    
    # Ensure same length for comparison
    min_len = min(len(full_seq), len(alt_full_seq))
    ref_full = full_seq[:min_len]
    alt_full = alt_full_seq[:min_len]
    
    if len(ref_full) < 11000:
        return None
    
    # Get base model predictions
    def one_hot(seq):
        mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
        indices = np.array([mapping.get(n.upper(), 0) for n in seq])
        oh = np.zeros((len(seq), 4), dtype=np.float32)
        oh[np.arange(len(seq)), indices] = 1
        return oh
    
    def get_scores(seq, models, device):
        oh = one_hot(seq)
        x = torch.tensor(oh.T, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = [m(x).cpu() for m in models]
            avg = torch.mean(torch.stack(preds), dim=0)
            probs = F.softmax(avg.permute(0, 2, 1), dim=-1)
        return probs[0].numpy()  # [L, 3]
    
    ref_scores = get_scores(ref_full, base_models, base_device)
    alt_scores = get_scores(alt_full, base_models, base_device)
    
    # Find center position in output
    out_len = len(ref_scores)
    out_center = out_len // 2
    
    # Get max delta in ±50bp window
    scan_start = max(0, out_center - 50)
    scan_end = min(out_len, out_center + 51)
    
    delta = alt_scores - ref_scores
    
    # OpenSpliceAI output: [neither, acceptor, donor]
    donor_deltas = delta[scan_start:scan_end, 2]
    acceptor_deltas = delta[scan_start:scan_end, 1]
    
    # Get max absolute delta
    max_donor_delta = donor_deltas[np.abs(donor_deltas).argmax()]
    max_acceptor_delta = acceptor_deltas[np.abs(acceptor_deltas).argmax()]
    
    # Extract context sequences (501nt around variant)
    ctx_start = var_pos_in_seq - half_ctx
    ctx_end = var_pos_in_seq + half_ctx + 1
    
    if ctx_start < 0 or ctx_end > len(full_seq):
        return None
    
    ref_context = full_seq[ctx_start:ctx_end]
    
    # For alt context, adjust for indel
    alt_var_pos = var_pos_in_seq
    alt_ctx_start = alt_var_pos - half_ctx
    alt_ctx_end = alt_var_pos + half_ctx + 1
    
    if alt_ctx_start < 0 or alt_ctx_end > len(alt_full_seq):
        alt_context = ref_context  # Fallback
    else:
        alt_context = alt_full_seq[alt_ctx_start:alt_ctx_end]
    
    # Ensure correct length
    if len(ref_context) != context_size:
        ref_context = ref_context[:context_size].ljust(context_size, 'N')
    if len(alt_context) != context_size:
        alt_context = alt_context[:context_size].ljust(context_size, 'N')
    
    return VariantSample(
        variant_id=f"{variant.chrom}:{variant.position}:{variant.ref_allele}>{variant.alt_allele}",
        chrom=str(variant.chrom),
        position=variant.position,
        ref_allele=ref_allele,
        alt_allele=alt_allele,
        classification=variant.classification,
        ref_sequence=ref_context,
        alt_sequence=alt_context,
        delta_donor=float(max_donor_delta),
        delta_acceptor=float(max_acceptor_delta),
        weight=2.0 if variant.classification == 'Splice-altering' else 1.0
    )


def create_variant_dataloader(
    splicevardb_loader,
    fasta,
    base_models,
    base_device: str,
    batch_size: int = 32,
    chromosomes: Optional[List[str]] = None,
    max_variants: Optional[int] = None,
    shuffle: bool = True
):
    """
    Create a DataLoader for variant delta prediction.
    
    Convenience function that prepares data and creates loader.
    """
    from torch.utils.data import DataLoader
    
    samples = prepare_variant_data(
        splicevardb_loader=splicevardb_loader,
        fasta=fasta,
        base_models=base_models,
        base_device=base_device,
        chromosomes=chromosomes,
        max_variants=max_variants
    )
    
    dataset = VariantDeltaDataset(samples)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )












