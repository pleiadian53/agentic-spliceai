"""Data preparation for base layer predictions.

This module provides functions for preparing genomic data for splice site prediction,
including loading gene annotations from GTF files and extracting sequences from FASTA files.

Functions
---------
prepare_gene_data
    Main entry point - Load annotations and extract sequences
load_gene_annotations
    Load gene annotations from GTF file
extract_sequences
    Extract DNA sequences from FASTA file
filter_by_genes
    Filter DataFrame to specific genes
filter_by_chromosomes
    Filter DataFrame to specific chromosomes
normalize_chromosome_names
    Normalize chromosome names (chr1 vs 1)

Examples
--------
Basic usage:

>>> from agentic_spliceai.splice_engine.base_layer.data import prepare_gene_data
>>> 
>>> # Prepare data for specific genes
>>> gene_df = prepare_gene_data(
...     genes=['BRCA1', 'TP53'],
...     build='GRCh38'
... )
>>> 
>>> # Prepare data for entire chromosome
>>> chr21_df = prepare_gene_data(
...     chromosomes=['21'],
...     build='GRCh38'
... )

Advanced usage with custom paths:

>>> gene_df = prepare_gene_data(
...     genes=['BRCA1'],
...     gtf_path='/path/to/annotations.gtf',
...     fasta_path='/path/to/genome.fa'
... )
"""

import polars as pl
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import os


def prepare_gene_data(
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    build: str = 'GRCh38',
    annotation_source: str = 'mane',
    gtf_path: Optional[Union[str, Path]] = None,
    fasta_path: Optional[Union[str, Path]] = None,
    verbosity: int = 1
) -> pl.DataFrame:
    """Prepare gene data for splice site prediction.
    
    This is the main entry point for data preparation. It loads gene annotations
    and extracts sequences in a single call.
    
    Parameters
    ----------
    genes : List[str], optional
        List of gene symbols or IDs to include (e.g., ['BRCA1', 'TP53'])
    chromosomes : List[str], optional
        List of chromosomes to include (e.g., ['1', '21', 'X'])
        Accepts both 'chr1' and '1' formats
    build : str, default='GRCh38'
        Genome build ('GRCh38', 'GRCh37', 'GRCh38_MANE')
    annotation_source : str, default='mane'
        Annotation source ('mane', 'ensembl', 'gencode')
    gtf_path : str or Path, optional
        Custom path to GTF file (overrides build/source)
    fasta_path : str or Path, optional
        Custom path to FASTA file (overrides build/source)
    verbosity : int, default=1
        Verbosity level (0=silent, 1=normal, 2=verbose)
    
    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - seqname : str - Chromosome name
        - gene_id : str - Gene identifier (e.g., ENSG00000012048)
        - gene_name : str - Gene symbol (e.g., BRCA1)
        - start : int - Gene start position (1-based)
        - end : int - Gene end position (1-based)
        - strand : str - Strand ('+' or '-')
        - sequence : str - DNA sequence
    
    Raises
    ------
    ValueError
        If no genes or chromosomes specified and no default provided
    FileNotFoundError
        If GTF or FASTA files cannot be found
    
    Examples
    --------
    Prepare data for specific genes:
    
    >>> gene_df = prepare_gene_data(genes=['BRCA1', 'TP53'])
    >>> print(f"Loaded {len(gene_df)} genes")
    
    Prepare data for entire chromosome:
    
    >>> chr21_df = prepare_gene_data(chromosomes=['21'])
    >>> print(f"Loaded {len(chr21_df)} genes from chr21")
    
    Use custom annotation files:
    
    >>> gene_df = prepare_gene_data(
    ...     genes=['BRCA1'],
    ...     gtf_path='/data/custom/annotations.gtf',
    ...     fasta_path='/data/custom/genome.fa'
    ... )
    
    Notes
    -----
    - If both genes and chromosomes are None, all genes are loaded (slow!)
    - Gene names are matched case-insensitively
    - Sequences are returned in uppercase
    - Missing sequences result in null values
    """
    # Resolve paths if not provided
    if gtf_path is None or fasta_path is None:
        from ...resources import get_genomic_registry
        
        # Handle MANE special case
        if build == 'GRCh38' and annotation_source == 'mane':
            registry = get_genomic_registry(build='GRCh38_MANE', release='1.3')
        else:
            registry = get_genomic_registry(build=build)
        
        if gtf_path is None:
            gtf_path = registry.get_gtf_path(validate=True)
        if fasta_path is None:
            fasta_path = registry.get_fasta_path(validate=True)
    
    if verbosity >= 1:
        print(f"📂 Loading genomic resources:")
        print(f"  GTF: {gtf_path}")
        print(f"  FASTA: {fasta_path}")
    
    # Load gene annotations
    genes_df = load_gene_annotations(
        gtf_path=gtf_path,
        genes=genes,
        chromosomes=chromosomes,
        verbosity=verbosity
    )
    
    if genes_df.height == 0:
        if verbosity >= 1:
            print("❌ No genes found matching criteria")
        return genes_df
    
    if verbosity >= 1:
        print(f"✓ Loaded {genes_df.height} gene annotations")
    
    # Extract sequences
    if verbosity >= 1:
        print(f"\n🧬 Extracting sequences from FASTA...")
    
    genes_df = extract_sequences(
        gene_df=genes_df,
        fasta_path=fasta_path,
        verbosity=verbosity
    )
    
    # Ensure 'seqname' column exists (alias for 'chrom')
    if 'chrom' in genes_df.columns and 'seqname' not in genes_df.columns:
        genes_df = genes_df.with_columns(pl.col('chrom').alias('seqname'))
    
    if verbosity >= 1:
        n_with_seq = genes_df.filter(pl.col('sequence').is_not_null()).height
        print(f"✓ Extracted {n_with_seq}/{genes_df.height} sequences")
    
    # Filter to genes with sequences
    genes_df = genes_df.filter(pl.col('sequence').is_not_null())
    
    return genes_df


def prepare_splice_site_annotations(
    output_dir: Union[str, Path],
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    build: str = 'GRCh38',
    annotation_source: str = 'mane',
    gtf_path: Optional[Union[str, Path]] = None,
    output_filename: str = "splice_sites_enhanced.tsv",
    force_extract: bool = False,
    verbosity: int = 1
) -> Dict[str, Any]:
    """Prepare splice site annotations from GTF file.
    
    This function extracts splice sites (donors and acceptors) from gene annotations.
    It implements caching - if the output file already exists, it loads from cache
    unless force_extract=True.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to store splice site annotations
    genes : List[str], optional
        List of gene symbols or IDs to include (e.g., ['BRCA1', 'TP53'])
        If None, extracts all genes
    chromosomes : List[str], optional
        List of chromosomes to include (e.g., ['1', '21', 'X'])
        Accepts both 'chr1' and '1' formats
    build : str, default='GRCh38'
        Genome build ('GRCh38', 'GRCh37', 'GRCh38_MANE')
    annotation_source : str, default='mane'
        Annotation source ('mane', 'ensembl', 'gencode')
    gtf_path : str or Path, optional
        Custom path to GTF file (overrides build/source)
    output_filename : str, default='splice_sites_enhanced.tsv'
        Name of output file
    force_extract : bool, default=False
        If True, re-extract even if file exists
    verbosity : int, default=1
        Verbosity level (0=silent, 1=normal, 2=verbose)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'success': bool - Whether operation succeeded
        - 'splice_sites_file': str - Path to splice sites file
        - 'splice_sites_df': pl.DataFrame - Splice site annotations
        - 'n_sites': int - Number of splice sites
        - 'n_donors': int - Number of donor sites
        - 'n_acceptors': int - Number of acceptor sites
    
    Notes
    -----
    Output format (TSV with columns):
    - chrom: Chromosome name
    - start: Start position (for BED-style intervals)
    - end: End position (for BED-style intervals)
    - position: Exact splice site position (1-based)
    - strand: Strand ('+' or '-')
    - site_type: 'donor' or 'acceptor'
    - gene_id: Gene identifier (e.g., ENSG00000012048)
    - transcript_id: Transcript identifier
    - gene_name: Gene symbol (e.g., BRCA1)
    - gene_biotype: Gene biotype (e.g., protein_coding)
    - transcript_biotype: Transcript biotype
    - exon_id: Exon identifier
    - exon_number: Exon number
    - exon_rank: Exon rank
    
    Examples
    --------
    >>> # Extract all splice sites for a build
    >>> result = prepare_splice_site_annotations(
    ...     output_dir='data/ensembl/GRCh38',
    ...     build='GRCh38'
    ... )
    >>> 
    >>> # Extract splice sites for specific genes
    >>> result = prepare_splice_site_annotations(
    ...     output_dir='data/prepared',
    ...     genes=['BRCA1', 'TP53'],
    ...     build='GRCh38'
    ... )
    >>> 
    >>> # Extract splice sites for chromosome
    >>> result = prepare_splice_site_annotations(
    ...     output_dir='data/prepared',
    ...     chromosomes=['21'],
    ...     build='GRCh38'
    ... )
    """
    from ..data.genomic_extraction import extract_splice_sites
    
    # Initialize result
    result = {
        'success': False,
        'splice_sites_file': None,
        'splice_sites_df': None,
        'n_sites': 0,
        'n_donors': 0,
        'n_acceptors': 0
    }
    
    # Resolve paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not output_filename.endswith('.tsv'):
        output_filename = f"{output_filename}.tsv"
    
    output_file = output_dir / output_filename
    result['splice_sites_file'] = str(output_file)
    
    # Determine GTF path
    if gtf_path is None:
        from agentic_spliceai.splice_engine.resources import get_genomic_registry
        
        # Handle MANE special case (must match prepare_gene_data logic)
        if build == 'GRCh38' and annotation_source == 'mane':
            registry = get_genomic_registry(build='GRCh38_MANE', release='1.3')
        else:
            registry = get_genomic_registry(build=build)
        
        gtf_path = registry.resolve('gtf')
        if gtf_path is None:
            if verbosity >= 1:
                print(f"❌ GTF file not found for build={build}, source={annotation_source}")
            result['error'] = 'GTF file not found'
            return result
    
    if verbosity >= 1:
        print(f"📍 Preparing splice site annotations")
        print(f"  Build: {build}")
        print(f"  GTF: {gtf_path}")
        print(f"  Output: {output_file}")
    
    # Check if file exists and we're not forcing extraction
    if output_file.exists() and not force_extract:
        if verbosity >= 1:
            print(f"✓ Loading cached splice sites from: {output_file}")
        
        try:
            splice_sites_df = pl.read_csv(output_file, separator='\t')
            
            # Apply filtering if requested
            if genes:
                if verbosity >= 1:
                    orig_count = splice_sites_df.height
                    print(f"  Filtering to {len(genes)} genes...")
                splice_sites_df = filter_by_genes(splice_sites_df, genes)
                if verbosity >= 1:
                    print(f"  Filtered from {orig_count} to {splice_sites_df.height} sites")
            
            if chromosomes:
                if verbosity >= 1:
                    orig_count = splice_sites_df.height
                    print(f"  Filtering to {len(chromosomes)} chromosomes...")
                splice_sites_df = filter_by_chromosomes(splice_sites_df, chromosomes)
                if verbosity >= 1:
                    print(f"  Filtered from {orig_count} to {splice_sites_df.height} sites")
            
            # Calculate statistics
            result['splice_sites_df'] = splice_sites_df
            result['n_sites'] = splice_sites_df.height
            result['n_donors'] = splice_sites_df.filter(pl.col('site_type') == 'donor').height
            result['n_acceptors'] = splice_sites_df.filter(pl.col('site_type') == 'acceptor').height
            result['success'] = True
            
            if verbosity >= 1:
                print(f"✓ Loaded {result['n_sites']} splice sites ({result['n_donors']} donors, {result['n_acceptors']} acceptors)")
            
            return result
        except Exception as e:
            if verbosity >= 1:
                print(f"⚠ Failed to load cached file: {e}")
                print(f"  Re-extracting from GTF...")
    
    # Extract splice sites
    if verbosity >= 1:
        if force_extract:
            print(f"🔄 Force extracting splice sites from GTF...")
        else:
            print(f"🧬 Extracting splice sites from GTF...")
    
    try:
        # Extract splice sites from GTF
        splice_sites_df = extract_splice_sites(
            gtf_file=str(gtf_path),
            chromosomes=chromosomes,
            verbosity=verbosity
        )
        
        # Apply gene filtering if requested
        if genes:
            if verbosity >= 1:
                orig_count = splice_sites_df.height
                print(f"  Filtering to {len(genes)} genes...")
            splice_sites_df = filter_by_genes(splice_sites_df, genes)
            if verbosity >= 1:
                print(f"  Filtered from {orig_count} to {splice_sites_df.height} sites")
        
        # Save to file
        splice_sites_df.write_csv(output_file, separator='\t')
        
        # Calculate statistics
        result['splice_sites_df'] = splice_sites_df
        result['n_sites'] = splice_sites_df.height
        result['n_donors'] = splice_sites_df.filter(pl.col('site_type') == 'donor').height
        result['n_acceptors'] = splice_sites_df.filter(pl.col('site_type') == 'acceptor').height
        result['success'] = True
        
        if verbosity >= 1:
            print(f"✓ Extracted {result['n_sites']} splice sites ({result['n_donors']} donors, {result['n_acceptors']} acceptors)")
            print(f"✓ Saved to: {output_file}")
        
        return result
        
    except Exception as e:
        if verbosity >= 1:
            print(f"❌ Failed to extract splice sites: {e}")
        result['error'] = str(e)
        return result


def load_gene_annotations(
    gtf_path: Union[str, Path],
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    verbosity: int = 1
) -> pl.DataFrame:
    """Load gene annotations from GTF file.
    
    Extracts gene-level features from a GTF file and optionally filters
    to specific genes or chromosomes.
    
    Parameters
    ----------
    gtf_path : str or Path
        Path to GTF annotation file
    genes : List[str], optional
        List of gene symbols or IDs to filter to
    chromosomes : List[str], optional
        List of chromosomes to filter to
    verbosity : int, default=1
        Verbosity level
    
    Returns
    -------
    pl.DataFrame
        Gene annotations with columns:
        - seqname : str - Chromosome
        - gene_id : str - Gene ID
        - gene_name : str - Gene symbol
        - start : int - Start position
        - end : int - End position
        - strand : str - Strand
    
    Examples
    --------
    >>> genes_df = load_gene_annotations(
    ...     gtf_path='data/annotations.gtf',
    ...     genes=['BRCA1', 'TP53']
    ... )
    """
    from .genomic_extraction import extract_gene_annotations as _extract_genes
    
    gtf_path = Path(gtf_path)
    
    if not gtf_path.exists():
        raise FileNotFoundError(f"GTF file not found: {gtf_path}")
    
    if verbosity >= 2:
        print(f"[debug] Loading GTF: {gtf_path}")
    
    # Extract gene annotations using existing function
    genes_df = _extract_genes(
        gtf_file=str(gtf_path),
        chromosomes=chromosomes,
        verbosity=verbosity
    )
    
    # Add 'seqname' alias for 'chrom' if needed
    if 'chrom' in genes_df.columns and 'seqname' not in genes_df.columns:
        genes_df = genes_df.with_columns(pl.col('chrom').alias('seqname'))
    
    if verbosity >= 2:
        print(f"[debug] Loaded {genes_df.height} total genes from GTF")
    
    # Apply gene filter if specified
    if genes is not None:
        genes_df = filter_by_genes(genes_df, genes, verbosity=verbosity)
    
    return genes_df


def extract_sequences(
    gene_df: pl.DataFrame,
    fasta_path: Union[str, Path],
    verbosity: int = 1
) -> pl.DataFrame:
    """Extract DNA sequences for genes from FASTA file.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene annotations with columns: seqname, start, end, strand
    fasta_path : str or Path
        Path to genome FASTA file
    verbosity : int, default=1
        Verbosity level
    
    Returns
    -------
    pl.DataFrame
        Input DataFrame with added 'sequence' column
    
    Examples
    --------
    >>> genes_df = extract_sequences(genes_df, 'genome.fa')
    """
    from .sequence_extraction import extract_gene_sequences
    
    return extract_gene_sequences(
        gene_df=gene_df,
        fasta_file=str(fasta_path),
        verbosity=verbosity
    )


def filter_by_genes(
    df: pl.DataFrame,
    genes: List[str],
    gene_column: str = 'gene_name',
    verbosity: int = 1
) -> pl.DataFrame:
    """Filter DataFrame to specific genes.
    
    Performs case-insensitive matching on gene names or IDs.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with gene information
    genes : List[str]
        List of gene symbols or IDs to keep
    gene_column : str, default='gene_name'
        Column name to filter on ('gene_name' or 'gene_id')
    verbosity : int, default=1
        Verbosity level
    
    Returns
    -------
    pl.DataFrame
        Filtered DataFrame
    
    Examples
    --------
    >>> filtered_df = filter_by_genes(df, ['BRCA1', 'TP53'])
    """
    if gene_column not in df.columns:
        if verbosity >= 1:
            print(f"⚠️  Column '{gene_column}' not found, trying fallback...")
        # Try alternative column names
        if 'gene_name' in df.columns:
            gene_column = 'gene_name'
        elif 'gene_id' in df.columns:
            gene_column = 'gene_id'
        else:
            raise ValueError(f"No gene column found in DataFrame")
    
    # Convert gene list to uppercase for case-insensitive matching
    genes_upper = [g.upper() for g in genes]
    
    # Filter (case-insensitive)
    filtered_df = df.filter(
        pl.col(gene_column).str.to_uppercase().is_in(genes_upper)
    )
    
    if verbosity >= 2:
        print(f"[debug] Filtered from {df.height} to {filtered_df.height} genes")
    
    # Report missing genes
    if verbosity >= 1 and filtered_df.height < len(genes):
        found_genes = set(filtered_df[gene_column].str.to_uppercase().to_list())
        missing = [g for g in genes_upper if g not in found_genes]
        if missing:
            print(f"⚠️  {len(missing)} genes not found: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    
    return filtered_df


def filter_by_chromosomes(
    df: pl.DataFrame,
    chromosomes: List[str],
    chrom_column: str = 'seqname',
    verbosity: int = 1
) -> pl.DataFrame:
    """Filter DataFrame to specific chromosomes.
    
    Handles both 'chr1' and '1' formats automatically.
    
    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with chromosome information
    chromosomes : List[str]
        List of chromosomes to keep (e.g., ['1', '21', 'X'])
    chrom_column : str, default='seqname'
        Column name containing chromosome information (tries 'chrom' if not found)
    verbosity : int, default=1
        Verbosity level
    
    Returns
    -------
    pl.DataFrame
        Filtered DataFrame
    
    Examples
    --------
    >>> filtered_df = filter_by_chromosomes(df, ['1', '21', 'X'])
    """
    # Try alternative column names if specified column not found
    if chrom_column not in df.columns:
        if 'chrom' in df.columns:
            chrom_column = 'chrom'
        elif 'seqname' in df.columns:
            chrom_column = 'seqname'
        else:
            raise ValueError(f"No chromosome column found (tried: {chrom_column}, chrom, seqname)")
    
    # Normalize chromosome names
    normalized_chroms = normalize_chromosome_names(chromosomes)
    
    if verbosity >= 2:
        print(f"[debug] Normalized chromosomes: {normalized_chroms}")
    
    # Filter
    filtered_df = df.filter(pl.col(chrom_column).is_in(normalized_chroms))
    
    if verbosity >= 2:
        print(f"[debug] Filtered from {df.height} to {filtered_df.height} rows")
    
    return filtered_df


def normalize_chromosome_names(chromosomes: List[str]) -> List[str]:
    """Normalize chromosome names to standard format.
    
    Converts between 'chr1' and '1' formats to match the format used
    in the input data.
    
    Parameters
    ----------
    chromosomes : List[str]
        List of chromosome names in any format
    
    Returns
    -------
    List[str]
        Normalized chromosome names (both formats included)
    
    Examples
    --------
    >>> normalize_chromosome_names(['1', 'chr21', 'X'])
    ['1', 'chr1', '21', 'chr21', 'X', 'chrX']
    """
    normalized = []
    
    for chrom in chromosomes:
        chrom_str = str(chrom)
        
        # Add both formats to be safe
        if chrom_str.startswith('chr'):
            # Has 'chr' prefix
            normalized.append(chrom_str)  # chr1
            normalized.append(chrom_str[3:])  # 1
        else:
            # No prefix
            normalized.append(chrom_str)  # 1
            normalized.append(f'chr{chrom_str}')  # chr1
    
    return list(set(normalized))  # Remove duplicates


# ==== Helper functions for working with gene data ====

def get_gene_count(gene_df: pl.DataFrame) -> int:
    """Get number of genes in DataFrame.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene DataFrame
    
    Returns
    -------
    int
        Number of genes
    """
    return gene_df.height


def get_genes_by_chromosome(gene_df: pl.DataFrame) -> dict:
    """Get gene counts by chromosome.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene DataFrame with 'seqname' column
    
    Returns
    -------
    dict
        Dictionary mapping chromosome -> gene count
    
    Examples
    --------
    >>> counts = get_genes_by_chromosome(gene_df)
    >>> print(counts['chr1'])  # Number of genes on chr1
    """
    if 'seqname' not in gene_df.columns:
        raise ValueError("DataFrame must have 'seqname' column")
    
    counts = (
        gene_df
        .group_by('seqname')
        .agg(pl.count().alias('count'))
        .sort('seqname')
    )
    
    return dict(zip(counts['seqname'].to_list(), counts['count'].to_list()))


def get_missing_sequences(gene_df: pl.DataFrame) -> pl.DataFrame:
    """Get genes with missing sequences.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene DataFrame with 'sequence' column
    
    Returns
    -------
    pl.DataFrame
        Genes with null sequences
    """
    if 'sequence' not in gene_df.columns:
        raise ValueError("DataFrame must have 'sequence' column")
    
    return gene_df.filter(pl.col('sequence').is_null())


def validate_gene_data(gene_df: pl.DataFrame, verbosity: int = 1) -> bool:
    """Validate that gene DataFrame has required columns and data.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene DataFrame to validate
    verbosity : int, default=1
        Verbosity level
    
    Returns
    -------
    bool
        True if valid, False otherwise
    """
    required_columns = ['seqname', 'gene_id', 'gene_name', 'start', 'end', 'strand']
    
    # Check required columns
    missing = [col for col in required_columns if col not in gene_df.columns]
    if missing:
        if verbosity >= 1:
            print(f"❌ Missing required columns: {missing}")
        return False
    
    # Check for empty DataFrame
    if gene_df.height == 0:
        if verbosity >= 1:
            print("❌ DataFrame is empty")
        return False
    
    # Check for sequences (warning, not error)
    if 'sequence' in gene_df.columns:
        n_missing = gene_df.filter(pl.col('sequence').is_null()).height
        if n_missing > 0 and verbosity >= 1:
            print(f"⚠️  {n_missing}/{gene_df.height} genes missing sequences")
    
    if verbosity >= 1:
        print(f"✓ Gene data validated: {gene_df.height} genes")
    
    return True
