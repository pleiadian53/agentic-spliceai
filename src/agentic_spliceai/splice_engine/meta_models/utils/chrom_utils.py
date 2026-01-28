"""
Utility functions for chromosome handling and gene-to-chromosome mapping.

Ported from meta-spliceai for agentic-spliceai.
"""

import os
import re
import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Any

from ...utils.display import print_emphasized, print_with_indent


def normalize_chromosome_names(chromosomes: Optional[List[str]]) -> Optional[List[str]]:
    """
    Normalize chromosome names to handle both UCSC (chr1) and Ensembl (1) formats.
    
    Parameters
    ----------
    chromosomes : Optional[List[str]]
        List of chromosome names to normalize, or None
        
    Returns
    -------
    Optional[List[str]]
        List of normalized chromosome names including both formats for each chromosome,
        or None if input is None
        
    Notes
    -----
    This handles the common inconsistency between genomic data sources:
    - "chr1", "chr2", etc. (UCSC-style)
    - "1", "2", etc. (Ensembl-style)
    """
    if chromosomes is None:
        return None
        
    # Normalize chromosome names for matching
    # Create both versions (with and without "chr" prefix) for matching
    chr_variants = []
    for chrom in chromosomes:
        if not chrom:  # Skip empty strings
            continue
            
        # Strip "chr" prefix if present
        stripped_chrom = chrom[3:] if chrom.lower().startswith("chr") else chrom
        # Add both variants
        chr_variants.append(f"chr{stripped_chrom}")  # With "chr" prefix
        chr_variants.append(stripped_chrom)          # Without "chr" prefix
    
    # Remove duplicates while preserving order of first appearance
    seen = set()
    unique_chr_variants = [x for x in chr_variants if not (x in seen or seen.add(x))]
    
    return unique_chr_variants


def determine_target_chromosomes(
    local_dir: str, 
    gtf_file: str, 
    target_genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None, 
    test_mode: bool = False, 
    separator: str = '\t',
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Determine which chromosomes to process based on target genes.
    
    Parameters
    ----------
    local_dir : str
        Directory containing project data
    gtf_file : str
        Path to GTF file
    target_genes : Optional[List[str]], optional
        List of target genes to focus on, by default None
    chromosomes : Optional[List[str]], optional
        Initial list of chromosomes to consider, by default None
    test_mode : bool, optional
        Whether to run in test mode, by default False
    separator : str, optional
        Separator for files, by default '\t'
    verbosity : int, optional
        Verbosity level, by default 1
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing chromosome information and maps
    """
    result = {
        'chromosomes': chromosomes,
        'gene_chrom_map': {},
        'gene_name_map': {}
    }
    
    # Set up chromosomes to process
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y'] 
        # or add 'MT' for mitochondrial DNA
    
    result['chromosomes'] = chromosomes
    
    # If targeting specific genes, try to optimize which chromosomes to process
    if target_genes is not None and not test_mode and (chromosomes is None or len(chromosomes) > 5):
        # Only attempt optimization if we're looking at many chromosomes
        if verbosity >= 1:
            print_emphasized("[optimize] Determining which chromosomes contain target genes...")
        
        # First check if we already have gene-to-chromosome mapping info
        gene_chrom_file = os.path.join(local_dir, "gene_chromosome_map.tsv")
        if os.path.exists(gene_chrom_file):
            # Load existing mapping
            if verbosity >= 1:
                print_with_indent("Loading existing gene-chromosome mapping", indent_level=1)
            gene_chrom_df = pd.read_csv(gene_chrom_file, sep='\t')
            gene_chrom_map = dict(zip(gene_chrom_df['gene_id'], gene_chrom_df['chromosome']))
            gene_name_map = dict(zip(gene_chrom_df['gene_name'], gene_chrom_df['chromosome']))
        else:
            # Create mapping from annotations
            if verbosity >= 1:
                print_with_indent("Creating gene-chromosome mapping from annotations", indent_level=1)
            
            # Quick scan of gene annotations to build gene-to-chromosome map
            gene_chrom_map = {}
            gene_name_map = {}
            
            # Use GTF file to determine chromosomes for target genes
            with open(gtf_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 9:
                        continue
                    
                    if parts[2] == "gene":
                        chrom = parts[0]
                        attributes = parts[8]
                        
                        # Extract gene_id
                        gene_id_match = re.search(r'gene_id "([^"]+)"', attributes)
                        if gene_id_match:
                            gene_id = gene_id_match.group(1)
                            gene_chrom_map[gene_id] = chrom
                        
                        # Extract gene_name if available
                        gene_name_match = re.search(r'gene_name "([^"]+)"', attributes)
                        if gene_name_match:
                            gene_name = gene_name_match.group(1)
                            gene_name_map[gene_name] = chrom
            
            # Save mapping for future use
            gene_chrom_df = pd.DataFrame({
                'gene_id': list(gene_chrom_map.keys()),
                'gene_name': [gene_name_map.get(gene_id, "") for gene_id in gene_chrom_map.keys()],
                'chromosome': list(gene_chrom_map.values())
            })
            gene_chrom_df.to_csv(gene_chrom_file, sep='\t', index=False)
        
        # Store gene-chromosome mappings in result
        result['gene_chrom_map'] = gene_chrom_map
        result['gene_name_map'] = gene_name_map
        
        # Find which chromosomes contain our target genes
        target_chromosomes = set()
        for gene in target_genes:
            if gene in gene_chrom_map:
                target_chromosomes.add(gene_chrom_map[gene])
            elif gene in gene_name_map:
                target_chromosomes.add(gene_name_map[gene])
        
        if target_chromosomes:
            # Filter to only process chromosomes containing target genes
            filtered_chromosomes = sorted(list(target_chromosomes))
            if verbosity >= 1:
                print_emphasized(f"[optimize] Analysis focused on {len(filtered_chromosomes)} chromosome(s): {', '.join(filtered_chromosomes)}")
            result['chromosomes'] = filtered_chromosomes
        else:
            if verbosity >= 1:
                print_emphasized("[warn] Could not determine which chromosomes contain target genes")
                print_with_indent("Will process all specified chromosomes", indent_level=1)
    
    if test_mode:
        if verbosity >= 1:
            print("[mode] Test mode enabled - using subset of chromosomes")
        # If test_mode, use either the explicitly specified chromosomes or just chromosome 21
        if not chromosomes:
            result['chromosomes'] = ['21']
        if verbosity >= 1:
            print_with_indent(f"Processing chromosomes: {', '.join(result['chromosomes'])}", indent_level=1)
    
    return result
