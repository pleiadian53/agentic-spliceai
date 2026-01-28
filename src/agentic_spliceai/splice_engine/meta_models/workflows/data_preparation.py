"""
Data preparation utilities for splice site prediction workflows.

This module contains functions for preparing genomic data for splice site prediction,
including annotation extraction, sequence loading, and chromosome optimization.

STANDALONE VERSION - No dependencies on meta-spliceai.
"""

import os
import glob
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Union, Any

from agentic_spliceai.splice_engine.utils.display import print_emphasized, print_with_indent
from agentic_spliceai.splice_engine.utils.dataframe import is_dataframe_empty
from agentic_spliceai.splice_engine.utils.filesystem import read_splice_sites
from agentic_spliceai.splice_engine.resources import standardize_splice_sites_schema

from agentic_spliceai.splice_engine.base_layer.data import (
    extract_gene_annotations as _extract_gene_annotations,
    extract_transcript_annotations,
    extract_exon_annotations,
    extract_splice_sites as _extract_splice_sites,
    extract_all_annotations,
    extract_gene_sequences as _extract_gene_sequences,
    load_chromosome_sequences,
)

from agentic_spliceai.splice_engine.meta_models.utils.chrom_utils import (
    normalize_chromosome_names,
    determine_target_chromosomes
)


def prepare_gene_annotations(
    local_dir: str, 
    gtf_file: str, 
    do_extract: bool = False,
    output_filename: str = "annotations_all_transcripts.tsv",
    use_shared_db: bool = True,
    target_chromosomes: Optional[List[str]] = None,
    separator: str = "\t",
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Prepare gene annotations from GTF file for splice site prediction.
    
    Parameters
    ----------
    local_dir : str
        Directory to store annotation files
    gtf_file : str
        Path to GTF file
    do_extract : bool, default=False
        Whether to extract annotations from GTF file
    output_filename : str, default="annotations_all_transcripts.tsv"
        Name of the output file for annotations
    use_shared_db : bool, default=True
        Whether to use the shared database (ignored in standalone version)
    target_chromosomes : Optional[List[str]], default=None
        List of chromosomes to filter annotations to
    separator : str, default="\t"
        Separator to use for output files
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing paths and dataframes for annotations
    """
    result = {
        'success': False,
        'annotation_file': None,
        'annotation_df': None
    }
    
    output_file = os.path.join(local_dir, output_filename)
    os.makedirs(local_dir, exist_ok=True)
    
    # Extract or load annotations
    if do_extract or not os.path.exists(output_file):
        if verbosity >= 1:
            print_emphasized("[action] Extracting gene annotations from GTF...")
        
        # Use standalone extraction
        annot_df = extract_all_annotations(
            gtf_file,
            output_file=output_file,
            chromosomes=target_chromosomes,
            separator=separator,
            verbosity=verbosity
        )
        
        result['annotation_file'] = output_file
        result['annotation_df'] = annot_df.to_pandas()
        result['success'] = True
        
    elif os.path.exists(output_file):
        if verbosity >= 1:
            print(f"[info] Loading existing annotations file: {output_file}")
        
        # Load existing file
        annot_df = pd.read_csv(output_file, sep=separator, low_memory=False, dtype={'chrom': str})
        
        # Filter by chromosomes if specified
        if target_chromosomes:
            normalized_chromosomes = normalize_chromosome_names(target_chromosomes)
            if verbosity >= 1:
                print(f"[info] Filtering annotations to chromosomes: {normalized_chromosomes}")
            
            original_count = len(annot_df)
            annot_df = annot_df[annot_df['chrom'].isin(normalized_chromosomes)]
            if verbosity >= 1:
                print(f"[info] Filtered from {original_count} to {len(annot_df)} rows")
        
        result['annotation_df'] = annot_df
        result['annotation_file'] = output_file
        result['success'] = True
    
    if verbosity >= 1 and result['annotation_df'] is not None:
        print(f"[info] Loaded {len(result['annotation_df'])} annotation records")
    
    return result


def prepare_splice_site_annotations(
    local_dir: str, 
    gtf_file: str, 
    do_extract: bool = True,
    output_filename: str = "splice_sites_enhanced.tsv",
    gene_annotations_df: Optional[Union[pd.DataFrame, pl.DataFrame]] = None,
    consensus_window: int = 2,
    separator: str = '\t',
    use_shared_db: bool = True,
    target_chromosomes: Optional[List[str]] = None,
    fasta_file: Optional[str] = None,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Extract and prepare splice site annotations.
    
    Parameters
    ----------
    local_dir : str
        Directory to store extracted splice site annotations
    gtf_file : str
        Path to GTF file
    do_extract : bool, default=True
        Whether to extract annotations from GTF file
    output_filename : str, default="splice_sites_enhanced.tsv"
        Name of the output file
    gene_annotations_df : Optional[DataFrame], default=None
        Pre-loaded gene annotations dataframe for filtering
    consensus_window : int, default=2
        Window size for splice site consensus (unused in standalone)
    separator : str, default='\t'
        Separator for annotation files
    use_shared_db : bool, default=True
        Whether to use shared database (ignored in standalone)
    target_chromosomes : Optional[List[str]], default=None
        List of chromosomes to filter splice sites to
    fasta_file : Optional[str], default=None
        Path to FASTA genome file (unused in standalone)
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing path and dataframe for splice site annotations
    """
    result = {
        'success': False,
        'splice_sites_file': None,
        'splice_sites_df': None
    }
    
    if verbosity >= 1:
        print_emphasized("[data] Preparing splice site annotations...")
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Ensure proper extension
    if not output_filename.endswith('.tsv'):
        output_filename = f"{output_filename}.tsv"
    
    splice_sites_file = os.path.join(local_dir, output_filename)
    result['splice_sites_file'] = splice_sites_file
    
    # Get target genes if filtering
    target_genes = None
    if gene_annotations_df is not None:
        if isinstance(gene_annotations_df, pd.DataFrame):
            target_genes = set(gene_annotations_df['gene_id'].unique())
        else:
            target_genes = set(gene_annotations_df['gene_id'].unique().to_list())
        if verbosity >= 1:
            print(f"[info] Will filter to {len(target_genes)} target genes")
    
    # Extract or load splice sites
    if do_extract or not os.path.exists(splice_sites_file):
        if verbosity >= 1:
            print_emphasized("[action] Extracting splice sites from GTF...")
        
        # Use standalone extraction
        ss_df = _extract_splice_sites(
            gtf_file,
            output_file=splice_sites_file,
            chromosomes=target_chromosomes,
            separator=separator,
            verbosity=verbosity
        )
        
    elif os.path.exists(splice_sites_file):
        if verbosity >= 1:
            print(f"[info] Loading existing splice sites: {splice_sites_file}")
        
        ss_df = read_splice_sites(splice_sites_file, separator=separator, dtypes=None)
        
        if ss_df is None or is_dataframe_empty(ss_df):
            if verbosity >= 0:
                print("[error] Failed to load splice site annotations")
            return result
    else:
        if verbosity >= 0:
            print(f"[error] Splice sites file not found: {splice_sites_file}")
        return result
    
    # Apply gene filtering
    if target_genes and ss_df is not None:
        if verbosity >= 1:
            print(f"[action] Filtering to {len(target_genes)} target genes...")
        
        original_count = len(ss_df)
        if isinstance(ss_df, pd.DataFrame):
            ss_df = ss_df[ss_df['gene_id'].isin(target_genes)]
        else:
            ss_df = ss_df.filter(pl.col("gene_id").is_in(target_genes))
        
        if verbosity >= 1:
            print(f"[info] Filtered from {original_count} to {len(ss_df)} splice sites")
    
    # Apply chromosome filtering
    if target_chromosomes and ss_df is not None:
        normalized_chroms = normalize_chromosome_names(target_chromosomes)
        if verbosity >= 1:
            print(f"[info] Filtering to chromosomes: {normalized_chroms}")
        
        original_count = len(ss_df)
        if isinstance(ss_df, pd.DataFrame):
            ss_df = ss_df[ss_df['chrom'].isin(normalized_chroms)]
        else:
            ss_df = ss_df.filter(pl.col("chrom").is_in(normalized_chroms))
        
        if verbosity >= 1:
            print(f"[info] Filtered from {original_count} to {len(ss_df)} by chromosomes")
    
    # Standardize schema
    if ss_df is not None:
        if verbosity >= 1:
            print_with_indent("[schema] Standardizing splice site annotations...", indent=2)
        ss_df = standardize_splice_sites_schema(ss_df, verbose=(verbosity >= 2))
    
    result['splice_sites_df'] = ss_df
    result['success'] = True
    
    if verbosity >= 1 and ss_df is not None:
        print(f"[info] Final splice sites: {len(ss_df)} records")
    
    return result


def prepare_genomic_sequences(
    local_dir: str,
    gtf_file: str,
    genome_fasta: str,
    mode: str = 'gene',
    seq_type: str = 'full',
    do_extract: bool = True,
    chromosomes: Optional[List[str]] = None,
    genes: Optional[List[str]] = None,  
    test_mode: bool = False,
    seq_format: str = 'parquet',
    single_sequence_file: bool = False,
    force_overwrite: bool = False,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Prepare genomic sequences for splice site prediction.
    
    Parameters
    ----------
    local_dir : str
        Directory to store output files
    gtf_file : str
        Path to GTF annotation file
    genome_fasta : str
        Path to genome FASTA file
    mode : str, default='gene'
        Mode for sequence extraction ('gene' or 'transcript')
    seq_type : str, default='full'
        Type of gene sequences to extract ('full' or 'minmax')
    do_extract : bool, default=True
        Whether to extract sequences or use existing files
    chromosomes : List[str], optional
        List of chromosomes to include
    genes : List[str], optional
        List of genes to include
    test_mode : bool, default=False
        Whether to run in test mode
    seq_format : str, default='parquet'
        Format for sequence files
    single_sequence_file : bool, default=False
        Whether to create a single file for all chromosomes
    force_overwrite : bool, default=False
        If True, re-extract sequences even when files exist
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with results including success status and file paths
    """
    result = {
        'success': False,
        'sequences_file': None,
        'sequences_df': None,
        'error': None
    }
    
    os.makedirs(local_dir, exist_ok=True)
    
    # Default chromosomes
    standard_chroms = [str(i) for i in range(1, 23)] + ['X', 'Y']
    if chromosomes is None:
        chromosomes = standard_chroms
    
    if verbosity >= 1:
        print(f"[info] Preparing sequences: mode={mode}, chromosomes={len(chromosomes)}")
    
    try:
        # Determine output file path
        if seq_type == 'minmax':
            seq_file = os.path.join(local_dir, f"gene_sequence_minmax.{seq_format}")
            chr_pattern = f"gene_sequence_minmax_*.{seq_format}"
        else:
            seq_file = os.path.join(local_dir, f"gene_sequence.{seq_format}")
            chr_pattern = f"gene_sequence_*.{seq_format}"
        
        # Check existing files
        chr_files = glob.glob(os.path.join(local_dir, chr_pattern))
        need_extraction = do_extract and (force_overwrite or len(chr_files) == 0)
        
        if need_extraction:
            if verbosity >= 1:
                print_emphasized(f"[action] Extracting {mode} sequences...")
            
            # First extract gene annotations
            gene_df = _extract_gene_annotations(
                gtf_file,
                chromosomes=chromosomes,
                verbosity=verbosity
            )
            
            # Filter by genes if specified
            if genes:
                gene_df = gene_df.filter(
                    pl.col('gene_id').is_in(genes) | 
                    pl.col('gene_name').is_in(genes)
                )
                if verbosity >= 1:
                    print(f"[info] Filtered to {len(gene_df)} target genes")
            
            # Extract sequences
            seq_df = _extract_gene_sequences(
                gene_df,
                genome_fasta,
                output_file=seq_file,
                verbosity=verbosity
            )
            
            result['sequences_df'] = seq_df
            result['sequences_file'] = seq_file
            result['success'] = True
            
        elif chr_files or os.path.exists(seq_file):
            # Load existing sequences
            if verbosity >= 1:
                print(f"[info] Loading existing sequence files...")
            
            if os.path.exists(seq_file):
                if seq_format == 'parquet':
                    seq_df = pl.read_parquet(seq_file)
                else:
                    seq_df = pl.read_csv(seq_file, separator='\t')
            else:
                # Load from chromosome files
                seq_df = _load_split_sequence_files(
                    chromosomes, local_dir, seq_type, seq_format, verbosity
                )
            
            if seq_df is not None:
                # Apply gene filtering
                if genes:
                    original_count = len(seq_df)
                    seq_df = seq_df.filter(
                        pl.col('gene_id').is_in(genes) | 
                        pl.col('gene_name').is_in(genes)
                    )
                    if verbosity >= 1:
                        print(f"[info] Filtered to {len(seq_df)} sequences for target genes")
                
                result['sequences_df'] = seq_df
                result['sequences_file'] = seq_file
                result['success'] = True
        else:
            result['error'] = "No sequence files found and extraction disabled"
            if verbosity >= 1:
                print(f"[warning] {result['error']}")
        
        return result
        
    except Exception as e:
        result['error'] = str(e)
        if verbosity >= 1:
            print(f"[error] Failed to prepare sequences: {e}")
        return result


def _load_split_sequence_files(
    chromosomes: List[str],
    local_dir: str,
    seq_type: str,
    seq_format: str,
    verbosity: int = 1
) -> Optional[pl.DataFrame]:
    """Load sequence data from per-chromosome files."""
    dfs = []
    
    for chrom in chromosomes:
        if seq_type == 'minmax':
            file_path = os.path.join(local_dir, f"gene_sequence_minmax_{chrom}.{seq_format}")
        else:
            file_path = os.path.join(local_dir, f"gene_sequence_{chrom}.{seq_format}")
        
        if not os.path.exists(file_path):
            continue
        
        try:
            if seq_format == 'parquet':
                df = pl.read_parquet(file_path)
            else:
                df = pl.read_csv(file_path, separator='\t')
            dfs.append(df)
        except Exception as e:
            if verbosity >= 1:
                print(f"[warning] Failed to load {file_path}: {e}")
    
    if not dfs:
        return None
    
    combined = pl.concat(dfs)
    if verbosity >= 1:
        print(f"[info] Loaded {len(combined)} sequences from {len(dfs)} files")
    
    return combined


def handle_overlapping_genes(
    local_dir: str,
    gtf_file: str,
    do_find: bool = True,
    min_exons: int = 2,
    filter_valid_splice_sites: bool = True,
    separator: str = '\t',
    output_format: str = 'pd',
    verbosity: int = 1,
    target_chromosomes: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Find or load overlapping gene information.
    
    This standalone version computes overlapping genes by analyzing
    gene coordinates from the GTF file.
    
    Parameters
    ----------
    local_dir : str
        Directory to store output files
    gtf_file : str
        Path to GTF file
    do_find : bool, default=True
        Whether to find overlapping genes if not already found
    min_exons : int, default=2
        Minimum number of exons for a gene to be considered
    filter_valid_splice_sites : bool, default=True
        Whether to filter genes for valid splice sites
    separator : str, default='\t'
        Separator for output TSV file
    output_format : str, default='pd'
        Output format ('pd' for pandas, 'pl' for polars)
    verbosity : int, default=1
        Verbosity level
    target_chromosomes : Optional[List[str]], default=None
        List of chromosomes to filter to
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing overlapping gene information
    """
    result = {
        'success': False,
        'overlapping_file': None,
        'overlapping_df': None
    }
    
    os.makedirs(local_dir, exist_ok=True)
    overlapping_file = os.path.join(local_dir, "overlapping_gene_counts.tsv")
    result['overlapping_file'] = overlapping_file
    
    if os.path.exists(overlapping_file) and not do_find:
        # Load existing file
        if verbosity >= 1:
            print(f"[info] Loading existing overlapping genes: {overlapping_file}")
        
        if output_format == 'pd':
            result['overlapping_df'] = pd.read_csv(overlapping_file, sep=separator)
        else:
            result['overlapping_df'] = pl.read_csv(
                overlapping_file, 
                separator=separator,
                schema_overrides={"chrom": pl.Utf8}
            )
        result['success'] = True
        return result
    
    if verbosity >= 1:
        print_emphasized("[action] Computing overlapping genes...")
    
    # Extract gene annotations
    gene_df = _extract_gene_annotations(
        gtf_file,
        chromosomes=target_chromosomes,
        verbosity=verbosity
    )
    
    # Extract exon counts per gene
    exon_df = extract_exon_annotations(
        gtf_file,
        chromosomes=target_chromosomes,
        verbosity=0
    )
    
    # Count exons per gene
    exon_counts = exon_df.group_by('gene_id').agg(
        pl.count().alias('num_exons')
    )
    
    # Join with gene info
    gene_df = gene_df.join(exon_counts, on='gene_id', how='left')
    gene_df = gene_df.with_columns(
        pl.col('num_exons').fill_null(0)
    )
    
    # Filter by minimum exons
    if min_exons > 0:
        gene_df = gene_df.filter(pl.col('num_exons') >= min_exons)
    
    # Find overlapping genes per chromosome
    overlap_records = []
    
    for (chrom,), chrom_genes in gene_df.group_by(['chrom']):
        genes = chrom_genes.sort('start').to_dicts()
        
        for i, gene in enumerate(genes):
            overlap_count = 0
            overlapping_genes = []
            
            for j, other in enumerate(genes):
                if i == j:
                    continue
                
                # Check for overlap
                if (gene['start'] <= other['end'] and gene['end'] >= other['start']):
                    overlap_count += 1
                    overlapping_genes.append(other['gene_id'])
            
            overlap_records.append({
                'gene_id': gene['gene_id'],
                'gene_name': gene.get('gene_name', ''),
                'chrom': chrom,
                'start': gene['start'],
                'end': gene['end'],
                'strand': gene['strand'],
                'num_exons': gene['num_exons'],
                'overlap_count': overlap_count,
                'overlapping_genes': ','.join(overlapping_genes[:10]) if overlapping_genes else '',
            })
    
    overlap_df = pl.DataFrame(overlap_records)
    
    # Save to file
    overlap_df.write_csv(overlapping_file, separator=separator)
    if verbosity >= 1:
        print(f"[info] Saved overlapping gene info to: {overlapping_file}")
        n_overlapping = overlap_df.filter(pl.col('overlap_count') > 0).height
        print(f"[info] Found {n_overlapping} genes with overlaps out of {len(overlap_df)}")
    
    if output_format == 'pd':
        result['overlapping_df'] = overlap_df.to_pandas()
    else:
        result['overlapping_df'] = overlap_df
    
    result['success'] = True
    return result


def load_spliceai_models(
    model_dir: Optional[str] = None,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Load SpliceAI models.
    
    Parameters
    ----------
    model_dir : str, optional
        Directory containing model files. If None, uses default location.
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded models and status
    """
    result = {
        'success': False,
        'models': None,
        'error': None
    }
    
    if verbosity >= 1:
        print_emphasized("[action] Loading SpliceAI models...")
    
    try:
        from keras.models import load_model
        from agentic_spliceai.splice_engine.resources import get_genomic_registry
        
        # Use registry to find model directory
        if model_dir is None:
            # Get registry and use its model weights directory
            registry = get_genomic_registry()
            model_dir = str(registry.get_model_weights_dir('spliceai'))
            
            # Fallback to common locations if registry path doesn't exist
            if not os.path.exists(model_dir):
                possible_dirs = [
                    os.path.expanduser("~/.spliceai/models"),
                    "/data/models/spliceai",
                ]
                for d in possible_dirs:
                    if os.path.exists(d):
                        model_dir = d
                        break
        
        if model_dir is None or not os.path.exists(model_dir):
            result['error'] = f"Model directory not found: {model_dir}"
            if verbosity >= 1:
                print(f"[error] {result['error']}")
            return result
        
        # Load models
        models = []
        model_files = sorted(glob.glob(os.path.join(model_dir, "*.h5")))
        
        if not model_files:
            result['error'] = f"No model files found in {model_dir}"
            if verbosity >= 1:
                print(f"[error] {result['error']}")
            return result
        
        for model_file in model_files:
            if verbosity >= 2:
                print(f"[info] Loading model: {model_file}")
            model = load_model(model_file, compile=False)
            models.append(model)
        
        if verbosity >= 1:
            print_with_indent(f"[info] Loaded {len(models)} SpliceAI models", indent=2)
        
        result['models'] = models
        result['success'] = True
        
    except ImportError as e:
        result['error'] = f"Keras not available: {e}"
        if verbosity >= 1:
            print(f"[error] {result['error']}")
    except Exception as e:
        result['error'] = str(e)
        if verbosity >= 1:
            print(f"[error] Failed to load models: {e}")
    
    return result
