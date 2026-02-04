"""
Core prediction functions for splice site prediction.

This module contains the essential prediction logic for running SpliceAI and
OpenSpliceAI models on gene sequences.

Ported from: meta_spliceai/splice_engine/run_spliceai_workflow.py
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
from tqdm import tqdm

from ..data.sequence_extraction import one_hot_encode


# SpliceAI context lengths
SPLICEAI_CONTEXT = 10000  # Default context for SpliceAI-10k
SPLICEAI_BLOCK_SIZE = 5000  # Core block size


def prepare_input_sequence(
    sequence: str,
    context: int = SPLICEAI_CONTEXT
) -> np.ndarray:
    """
    Prepare an input DNA sequence for SpliceAI prediction.
    
    Converts the sequence to one-hot encoded format, adds flanking sequences,
    padding, and splits into overlapping blocks of the required size.
    
    Parameters
    ----------
    sequence : str
        Input DNA sequence (string of A, C, G, T)
    context : int, default=10000
        Context length for SpliceAI models. Defines flanking sequence on both
        sides of each 5,000-nucleotide chunk. Valid: 80, 400, 2000, 10000.
        
    Returns
    -------
    np.ndarray
        Array of overlapping blocks, shape (num_blocks, block_size, 4)
        where block_size = context/2 + 5000 + context/2
        
    Notes
    -----
    The SpliceAI model expects input blocks of size (context + 5000).
    The model outputs predictions for the central 5000 bp, cropping the context.
    """
    # One-hot encode the input sequence
    encoded_seq = np.array(one_hot_encode(sequence), dtype=np.float32)
    
    # Pad sequence to make length a multiple of 5000
    seq_length = len(encoded_seq)
    if seq_length % SPLICEAI_BLOCK_SIZE != 0:
        pad_length = SPLICEAI_BLOCK_SIZE - (seq_length % SPLICEAI_BLOCK_SIZE)
        encoded_seq = np.pad(encoded_seq, ((0, pad_length), (0, 0)), 'constant')
    
    # Add flanking sequences (N padding as zeros)
    flanking = np.zeros((context // 2, 4), dtype=np.float32)
    padded_seq = np.vstack([flanking, encoded_seq, flanking])
    
    # Split into overlapping blocks
    block_size = context // 2 + SPLICEAI_BLOCK_SIZE + context // 2
    num_blocks = (len(padded_seq) - block_size) // SPLICEAI_BLOCK_SIZE + 1
    
    blocks = np.array([
        padded_seq[SPLICEAI_BLOCK_SIZE * i: SPLICEAI_BLOCK_SIZE * i + block_size]
        for i in range(num_blocks)
    ])
    
    return blocks


def predict_with_model(model, x: np.ndarray) -> np.ndarray:
    """
    Universal prediction function for Keras and PyTorch models.
    
    Parameters
    ----------
    model : keras.Model or torch.nn.Module
        The model to use for prediction
    x : np.ndarray
        Input array (batch_size, sequence_length, channels)
        
    Returns
    -------
    np.ndarray
        Predictions (batch, sequence, channels)
    """
    if hasattr(model, 'predict'):
        # Keras model (SpliceAI)
        return model.predict(x, verbose=0)
    else:
        # PyTorch model (OpenSpliceAI)
        import torch
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).float()
            # PyTorch expects (batch, channels, sequence)
            x_tensor = x_tensor.permute(0, 2, 1)
            device = next(model.parameters()).device
            x_tensor = x_tensor.to(device)
            pred = model(x_tensor)
            # Transpose back to (batch, sequence, channels)
            pred = pred.permute(0, 2, 1)
            pred = pred.cpu().numpy()
        return pred


def normalize_strand(strand: str) -> str:
    """Normalize strand representation to '+' or '-'."""
    if strand in ['+', '1', 1, 'plus', 'forward']:
        return '+'
    elif strand in ['-', '-1', -1, 'minus', 'reverse']:
        return '-'
    else:
        return strand


def predict_splice_sites_for_genes(
    gene_df: pl.DataFrame,
    models: List,
    context: int = SPLICEAI_CONTEXT,
    output_format: str = 'dict',
    verbosity: int = 1,
    **kwargs
) -> Union[Dict[str, Dict], pl.DataFrame]:
    """
    Generate splice site predictions for each gene sequence.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        DataFrame with columns: gene_id, gene_name, chrom/seqname, start, end, strand, sequence
        Note: Negative-strand sequences must already be reverse-complemented.
    models : List
        List of loaded SpliceAI/OpenSpliceAI models
    context : int, default=10000
        Context length for SpliceAI
    output_format : str, default='dict'
        Output format: 'dict' for efficient dictionary, 'dataframe' for full DataFrame
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Union[Dict, pl.DataFrame]
        If output_format='dict': Dictionary with gene predictions
        If output_format='dataframe': DataFrame with all positions
        
    Notes
    -----
    Output dictionary structure when output_format='dict':
    {
        gene_id: {
            'seqname': str,
            'gene_name': str,
            'strand': str,
            'gene_start': int,
            'gene_end': int,
            'donor_prob': List[float],
            'acceptor_prob': List[float],
            'neither_prob': List[float],
            'positions': List[int]
        }
    }
    """
    # Dictionary to store merged results by position
    merged_results = defaultdict(lambda: {
        'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []
    })
    n_genes_processed = 0
    
    # Progress bar
    iterator = gene_df.iter_rows(named=True)
    if verbosity >= 1:
        iterator = tqdm(iterator, total=gene_df.height, desc="Processing genes", mininterval=10.0)
    
    for row in iterator:
        gene_id = row['gene_id']
        gene_name = str(row.get('gene_name', '')) or ''
        sequence = row['sequence']
        seqname = row.get('seqname', row.get('chrom', ''))
        strand = normalize_strand(row['strand'])
        seq_len = len(sequence)
        
        if verbosity >= 2:
            print(f"[predict] Processing gene {gene_id} (chr={seqname}, len={seq_len})")
        
        # Check for absolute positions
        has_absolute_positions = 'start' in row and 'end' in row
        gene_start = row.get('start', None)
        gene_end = row.get('end', None)
        
        # Prepare input blocks
        input_blocks = prepare_input_sequence(sequence, context)
        
        if verbosity >= 2:
            print(f"  Generated {len(input_blocks)} blocks")
        
        # Predict for each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension
            
            # Average predictions across all models
            y = np.mean([predict_with_model(model, x) for model in models], axis=0)
            
            # Extract probabilities
            # SpliceAI output channels: [neither, acceptor, donor]
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]
            
            # Calculate block start position
            block_start = block_index * SPLICEAI_BLOCK_SIZE
            
            # Store results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(
                zip(donor_prob, acceptor_prob, neither_prob)
            ):
                if has_absolute_positions:
                    if strand == '+':
                        absolute_position = gene_start + (block_start + i)
                    else:  # strand == '-'
                        absolute_position = gene_end - (block_start + i)
                else:
                    absolute_position = block_start + i + 1
                
                pos_key = (gene_id, absolute_position)
                
                # Append probabilities (will be averaged for overlapping positions)
                merged_results[pos_key]['donor_prob'].append(float(donor_p))
                merged_results[pos_key]['acceptor_prob'].append(float(acceptor_p))
                merged_results[pos_key]['neither_prob'].append(float(neither_p))
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['seqname'] = seqname
                merged_results[pos_key]['gene_name'] = gene_name
                merged_results[pos_key]['absolute_position'] = absolute_position
                merged_results[pos_key]['gene_start'] = gene_start
                merged_results[pos_key]['gene_end'] = gene_end
        
        n_genes_processed += 1
    
    if len(merged_results) == 0:
        if verbosity >= 0:
            print("[warning] No splice site predictions generated")
        return {} if output_format == 'dict' else pl.DataFrame()
    
    # Trim positions outside gene boundaries
    gene_positions = {
        row['gene_id']: (row['start'], row['end'])
        for row in gene_df.iter_rows(named=True)
        if 'start' in row and 'end' in row
    }
    
    trimmed_results = {}
    for (gene_id, position), data in merged_results.items():
        if gene_id in gene_positions:
            gene_start, gene_end = gene_positions[gene_id]
            abs_pos = data['absolute_position']
            if abs_pos is not None and gene_start <= abs_pos <= gene_end:
                trimmed_results[(gene_id, position)] = data
        else:
            trimmed_results[(gene_id, position)] = data
    
    if verbosity >= 1:
        print(f"[predict] Processed {n_genes_processed} genes, {len(trimmed_results)} positions")
    
    # Convert to output format
    if output_format.startswith(('eff', 'dict')):
        return _convert_to_efficient_output(trimmed_results, gene_df)
    else:
        return _convert_to_dataframe(trimmed_results)


def _convert_to_efficient_output(
    merged_results: Dict,
    gene_df: pl.DataFrame
) -> Dict[str, Dict]:
    """Convert merged results to efficient dictionary format."""
    efficient_results = {}
    
    # Group by gene_id
    gene_data = defaultdict(lambda: {
        'positions': [],
        'donor_prob': [],
        'acceptor_prob': [],
        'neither_prob': []
    })
    
    for (gene_id, position), data in merged_results.items():
        # Average overlapping predictions
        avg_donor = np.mean(data['donor_prob'])
        avg_acceptor = np.mean(data['acceptor_prob'])
        avg_neither = np.mean(data['neither_prob'])
        
        gene_data[gene_id]['positions'].append(data['absolute_position'])
        gene_data[gene_id]['donor_prob'].append(avg_donor)
        gene_data[gene_id]['acceptor_prob'].append(avg_acceptor)
        gene_data[gene_id]['neither_prob'].append(avg_neither)
        gene_data[gene_id]['seqname'] = data['seqname']
        gene_data[gene_id]['gene_name'] = data['gene_name']
        gene_data[gene_id]['strand'] = data['strand']
        gene_data[gene_id]['gene_start'] = data.get('gene_start')
        gene_data[gene_id]['gene_end'] = data.get('gene_end')
    
    # Sort positions within each gene
    for gene_id, data in gene_data.items():
        sorted_indices = np.argsort(data['positions'])
        efficient_results[gene_id] = {
            'seqname': data['seqname'],
            'gene_name': data['gene_name'],
            'strand': data['strand'],
            'gene_start': data['gene_start'],
            'gene_end': data['gene_end'],
            'positions': [data['positions'][i] for i in sorted_indices],
            'donor_prob': [data['donor_prob'][i] for i in sorted_indices],
            'acceptor_prob': [data['acceptor_prob'][i] for i in sorted_indices],
            'neither_prob': [data['neither_prob'][i] for i in sorted_indices],
        }
    
    return efficient_results


def _convert_to_dataframe(merged_results: Dict) -> pl.DataFrame:
    """Convert merged results to Polars DataFrame."""
    records = []
    
    for (gene_id, position), data in merged_results.items():
        # Average overlapping predictions
        avg_donor = np.mean(data['donor_prob'])
        avg_acceptor = np.mean(data['acceptor_prob'])
        avg_neither = np.mean(data['neither_prob'])
        
        records.append({
            'gene_id': gene_id,
            'gene_name': data['gene_name'],
            'seqname': data['seqname'],
            'position': position,
            'absolute_position': data['absolute_position'],
            'gene_start': data.get('gene_start'),
            'gene_end': data.get('gene_end'),
            'strand': data['strand'],
            'donor_prob': avg_donor,
            'acceptor_prob': avg_acceptor,
            'neither_prob': avg_neither,
        })
    
    return pl.DataFrame(records).sort(['gene_id', 'position'])


def load_spliceai_models(
    model_dir: Optional[str] = None,
    model_type: str = 'spliceai',
    build: Optional[str] = None,
    verbosity: int = 1
) -> List:
    """
    Load SpliceAI or OpenSpliceAI models.
    
    Parameters
    ----------
    model_dir : str, optional
        Directory containing model files. If None, uses resource registry.
    model_type : str, default='spliceai'
        Model type: 'spliceai' or 'openspliceai'
    build : str, optional
        Genome build (e.g., 'GRCh38', 'GRCh37'). Used to initialize registry if model_dir is None.
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    List
        List of loaded models
        
    Examples
    --------
    >>> # Use resource registry (recommended)
    >>> models = load_spliceai_models(model_type='openspliceai', build='GRCh38')
    >>> 
    >>> # Explicit path
    >>> models = load_spliceai_models(
    ...     model_type='openspliceai',
    ...     model_dir='/path/to/models/openspliceai'
    ... )
    """
    # Resolve model directory using registry if not explicitly provided
    if model_dir is None:
        from ...resources.registry import get_genomic_registry
        
        # Use build if provided, otherwise use config default
        registry = get_genomic_registry(build=build)
        model_dir = str(registry.get_model_weights_dir(model_type))
        
        if verbosity >= 2:
            print(f"[load] Resolved model directory from registry: {model_dir}")
    
    if model_type.lower() == 'openspliceai':
        return _load_openspliceai_models(model_dir, verbosity)
    else:
        return _load_spliceai_models(model_dir, verbosity)


def _load_spliceai_models(
    model_dir: Optional[str] = None,
    verbosity: int = 1
) -> List:
    """Load SpliceAI Keras models."""
    import os
    import glob
    
    try:
        from keras.models import load_model
    except ImportError:
        from tensorflow.keras.models import load_model
    
    # Default model locations
    if model_dir is None:
        possible_dirs = [
            os.path.expanduser("~/.spliceai/models"),
            "/data/models/spliceai",
            "data/models/spliceai",
        ]
        # Try to find from spliceai package
        try:
            from pkg_resources import resource_filename
            pkg_dir = resource_filename('spliceai', 'models')
            possible_dirs.insert(0, pkg_dir)
        except:
            pass
        
        for d in possible_dirs:
            if os.path.exists(d):
                model_dir = d
                break
    
    if model_dir is None or not os.path.exists(model_dir):
        raise FileNotFoundError(f"SpliceAI model directory not found: {model_dir}")
    
    # Load all .h5 models
    model_files = sorted(glob.glob(os.path.join(model_dir, "*.h5")))
    
    if not model_files:
        raise FileNotFoundError(f"No .h5 model files found in {model_dir}")
    
    models = []
    for model_file in model_files:
        if verbosity >= 2:
            print(f"[load] Loading model: {os.path.basename(model_file)}")
        model = load_model(model_file, compile=False)
        models.append(model)
    
    if verbosity >= 1:
        print(f"[load] Loaded {len(models)} SpliceAI models from {model_dir}")
    
    return models


def _load_openspliceai_models(
    model_dir: Optional[str] = None,
    verbosity: int = 1
) -> List:
    """Load OpenSpliceAI PyTorch models."""
    import os
    import glob
    
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for OpenSpliceAI models")
    
    # Default model locations
    if model_dir is None:
        possible_dirs = [
            "data/models/openspliceai",
            os.path.expanduser("~/.openspliceai/models"),
        ]
        for d in possible_dirs:
            if os.path.exists(d):
                model_dir = d
                break
    
    if model_dir is None or not os.path.exists(model_dir):
        raise FileNotFoundError(f"OpenSpliceAI model directory not found: {model_dir}")
    
    # Load .pt or .pth models
    model_files = sorted(
        glob.glob(os.path.join(model_dir, "*.pt")) +
        glob.glob(os.path.join(model_dir, "*.pth"))
    )
    
    if not model_files:
        raise FileNotFoundError(f"No .pt/.pth model files found in {model_dir}")
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    if verbosity >= 1:
        print(f"[load] Loading OpenSpliceAI models on device: {device}")
    
    # Use OpenSpliceAI's own model loader
    # This properly instantiates the model architecture and loads weights
    try:
        # Use local openspliceai module (100% independent from meta-spliceai)
        from agentic_spliceai.openspliceai.predict.predict import load_pytorch_models
        
        if verbosity >= 2:
            print(f"[load] Using load_pytorch_models from agentic_spliceai.openspliceai")
        
    except ImportError as e:
        raise ImportError(
            f"Cannot load OpenSpliceAI models: {e}\n"
            "OpenSpliceAI module not found in agentic-spliceai.\n"
            "The module should be at: agentic_spliceai/openspliceai/predict/predict.py"
        )
    
    # load_pytorch_models(model_path, device, SL, CL)
    # SL = output sequence length (5000)
    # CL = context length (10000 for SpliceAI-10k)
    models = load_pytorch_models(
        model_dir,
        device,
        SL=5000,
        CL=10000
    )
    
    # Extract models list if returned as tuple
    if isinstance(models, tuple):
        models = models[0]
    
    if verbosity >= 1:
        model_count = len(models) if isinstance(models, list) else 1
        print(f"[load] Loaded {model_count} OpenSpliceAI models successfully")
    
    return models
