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
from ..utils.coordinate_adjustment import apply_custom_adjustments


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


def genomic_positions_for_indices(
    *,
    gene_start: int,
    gene_end: int,
    strand: str,
    n: int,
    transcript_offset: int = 0,
) -> np.ndarray:
    """Map per-nucleotide model output indices to absolute genomic coordinates.

    This is the single definition of the index -> coordinate convention used
    across the base layer. Index 0 is the first nucleotide the model saw, which
    for a minus-strand gene is the *highest* genomic coordinate, because gene
    sequences are handed to predictors already reverse-complemented (see the
    contract on :func:`predict_splice_sites_for_genes`).

    Getting this wrong is silent and severe: an ascending range on a
    minus-strand gene places every call in its mirror-image position, so the
    model scores 0.0 at every true site and looks broken when it is correct.

    Parameters
    ----------
    gene_start, gene_end : int
        Gene bounds in the same frame the caller's annotations use.
    strand : str
        '+' or '-' (anything ``normalize_strand`` accepts).
    n : int
        Number of output positions.
    transcript_offset : int, default=0
        Shift applied in *transcript* direction (downstream positive), for
        checkpoints whose training labels sit off the annotated base. Default
        0 = the annotation's own frame. Anchoring on ``gene_end`` for the minus
        strand keeps this correct even when ``n`` is shorter than the gene.

    Returns
    -------
    np.ndarray
        int64 array of length ``n``.

    Examples
    --------
    >>> genomic_positions_for_indices(
    ...     gene_start=100, gene_end=104, strand='+', n=3
    ... ).tolist()
    [100, 101, 102]
    >>> genomic_positions_for_indices(
    ...     gene_start=100, gene_end=104, strand='-', n=3
    ... ).tolist()
    [104, 103, 102]
    """
    idx = np.arange(n, dtype=np.int64)
    if normalize_strand(strand) == '-':
        return gene_end - idx - transcript_offset
    return gene_start + idx + transcript_offset


def predict_splice_sites_for_genes(
    gene_df: pl.DataFrame,
    models: List,
    context: int = SPLICEAI_CONTEXT,
    adjustment_dict: Optional[Dict[str, Dict[str, int]]] = None,
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
    adjustment_dict : Optional[Dict[str, Dict[str, int]]], default=None
        UNUSED - This pipeline's position mapping is correct and does not require
        coordinate adjustments. Parameter retained for API compatibility.
        See the Notes on multi-transcript annotations before reaching for a
        coordinate adjustment to explain a low recall.
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
            'chrom': str,
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
    
    **Recall depends on how many transcripts you score against**

    Recall against an annotation is driven by that annotation's transcript
    multiplicity, not by the model. MANE carries one transcript per gene
    (median 1.0); Ensembl carries a median of 8 for the same genes. Scoring a
    MANE-trained model against the Ensembl union therefore adds minor-isoform
    boundaries it was never trained to emit, and those score 0.0 outright —
    this is a denominator effect, not a threshold or coordinate problem.

    Measured on chr21 (208 genes with stored OpenSpliceAI scores, threshold
    0.5, +/-2 window), the same predictions scored three ways:

        MANE union (the training annotation)   3,561 sites   96.0% recall
        Ensembl, canonical transcript only     3,826 sites   87.6%
        Ensembl, all transcripts               4,765 sites   72.3%

    Closing that last gap WITHOUT retraining the base model is what the meta
    layer exists to do; the base scores become an input feature and multimodal
    evidence recalibrates them. See the M2-S alternative-site evaluation in
    examples/meta_layer/09_evaluate_alternative_sites.py, which scores only
    sites present in the eval annotation and absent from MANE, so shared
    canonical sites cannot inflate the result.

    **A ~40% figure today means something else.** An earlier version of this
    note reported ~40% all-transcript recall from a 2026-02 investigation and
    attributed it to isoform multiplicity. That measurement predates the
    minus-strand annotation fix of 2026-05-25. Re-measured on the same genes,
    the pre-fix annotation gives 59.6% overall — plus strand 72.7%, minus
    strand 48.4% — while the current annotation gives 72.3% with the strands
    balanced (72.7% / 71.8%) and ~786 spurious minus-strand sites gone. Most
    of that deficit was a coordinate bug, not biology. So if you see recall
    near 40% now, suspect coordinates FIRST and validate with the GT/AG
    dinucleotide oracle split by strand: a collapse on one strand only is the
    signature, and a correct extractor scores ~0.98 on both.

    Use the evaluation's transcript filtering and gap analysis to separate the
    two effects:
    
    ```python
    from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
        filter_annotations_by_transcript,
        splice_site_gap_analysis,
    )
    
    # Evaluate against canonical transcript only (true base model performance)
    canonical_annots = filter_annotations_by_transcript(annotations_df, mode='canonical')
    
    # Or run dual evaluation (canonical + all) with gap analysis
    # See: examples/base_layer/03_prediction_with_evaluation.py --gap-analysis
    gap = splice_site_gap_analysis(predictions, annotations_df)
    # gap['recovery_potential'] shows what meta/agentic layers need to address
    ```
    
    **Note on coordinate adjustments**: The MetaSpliceAI codebase requires np.roll()
    adjustments due to its internal position-mapping convention. This pipeline uses 
    absolute genomic positions and does NOT need those adjustments. The automatic
    detection system (coordinate_adjustment.py) is retained for future base models
    that may have different position-mapping conventions.
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
        seqname = row.get('chrom', row.get('seqname', ''))
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

        # Index -> coordinate map for the whole gene, built once per gene from
        # the shared convention rather than re-derived per position. Sized to
        # the padded block span, since the final block runs past the gene end;
        # those positions are trimmed below against the gene bounds.
        n_mapped = len(input_blocks) * SPLICEAI_BLOCK_SIZE
        if has_absolute_positions:
            position_map = genomic_positions_for_indices(
                gene_start=gene_start, gene_end=gene_end, strand=strand, n=n_mapped,
            )
        else:
            # No gene bounds on the row: fall back to 1-based positions relative
            # to the sequence rather than absolute genomic coordinates.
            position_map = np.arange(1, n_mapped + 1, dtype=np.int64)
        
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
            
            # Position mapping is read from the per-gene map built above; see
            # genomic_positions_for_indices for the convention itself.
            # This pipeline does NOT need np.roll() coordinate adjustments.
            # See docstring Notes for details on low recall.

            # Calculate block start position
            block_start = block_index * SPLICEAI_BLOCK_SIZE

            # Store results with adjusted positions
            for i, (donor_p, acceptor_p, neither_p) in enumerate(
                zip(donor_prob, acceptor_prob, neither_prob)
            ):
                absolute_position = int(position_map[block_start + i])

                pos_key = (gene_id, absolute_position)
                
                # Append probabilities (will be averaged for overlapping positions)
                merged_results[pos_key]['donor_prob'].append(float(donor_p))
                merged_results[pos_key]['acceptor_prob'].append(float(acceptor_p))
                merged_results[pos_key]['neither_prob'].append(float(neither_p))
                merged_results[pos_key]['strand'] = strand
                merged_results[pos_key]['chrom'] = seqname
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
        gene_data[gene_id]['chrom'] = data['chrom']
        gene_data[gene_id]['gene_name'] = data['gene_name']
        gene_data[gene_id]['strand'] = data['strand']
        gene_data[gene_id]['gene_start'] = data.get('gene_start')
        gene_data[gene_id]['gene_end'] = data.get('gene_end')

    # Sort positions within each gene
    for gene_id, data in gene_data.items():
        sorted_indices = np.argsort(data['positions'])
        efficient_results[gene_id] = {
            'chrom': data['chrom'],
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
            'chrom': data['chrom'],
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
