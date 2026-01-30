"""
Genomic feature extraction utilities.

This module provides standalone functions for extracting genomic features
from GTF annotation files and FASTA genome files.

No external dependencies on meta-spliceai.
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path

import pandas as pd
import polars as pl


def parse_gtf_attributes(attribute_string: str) -> Dict[str, str]:
    """
    Parse GTF attribute string into a dictionary.
    
    Parameters
    ----------
    attribute_string : str
        GTF attribute field (9th column)
        
    Returns
    -------
    Dict[str, str]
        Dictionary of attribute key-value pairs
        
    Examples
    --------
    >>> attrs = parse_gtf_attributes('gene_id "ENSG00000223972"; gene_name "DDX11L1";')
    >>> attrs['gene_id']
    'ENSG00000223972'
    """
    attributes = {}
    # Pattern matches: key "value"; or key "value"
    pattern = r'(\w+)\s+"([^"]*)"'
    
    for match in re.finditer(pattern, attribute_string):
        key, value = match.groups()
        attributes[key] = value
    
    return attributes


def iter_gtf_records(
    gtf_file: str,
    feature_types: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None
) -> Iterator[Dict[str, Any]]:
    """
    Iterate over GTF records, yielding parsed dictionaries.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF file
    feature_types : List[str], optional
        Filter to specific feature types (e.g., ['gene', 'exon'])
    chromosomes : List[str], optional
        Filter to specific chromosomes
        
    Yields
    ------
    Dict[str, Any]
        Parsed GTF record with fields: seqname, source, feature, start, end,
        score, strand, frame, and parsed attributes
    """
    # Normalize chromosome filter to include both formats
    chrom_set = None
    if chromosomes:
        chrom_set = set()
        for c in chromosomes:
            chrom_set.add(c)
            if c.startswith('chr'):
                chrom_set.add(c[3:])
            else:
                chrom_set.add(f'chr{c}')
    
    feature_set = set(feature_types) if feature_types else None
    
    with open(gtf_file, 'r') as f:
        for line in f:
            # Skip comments
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            seqname, source, feature, start, end, score, strand, frame, attributes = parts
            
            # Apply filters
            if chrom_set and seqname not in chrom_set:
                continue
            if feature_set and feature not in feature_set:
                continue
            
            # Parse attributes
            attrs = parse_gtf_attributes(attributes)
            
            yield {
                'seqname': seqname,
                'chrom': seqname,  # Alias
                'source': source,
                'feature': feature,
                'start': int(start),
                'end': int(end),
                'score': score,
                'strand': strand,
                'frame': frame,
                **attrs
            }


def extract_gene_annotations(
    gtf_file: str,
    output_file: Optional[str] = None,
    chromosomes: Optional[List[str]] = None,
    separator: str = '\t',
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract gene annotations from a GTF file.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    output_file : str, optional
        Path to save output TSV file
    chromosomes : List[str], optional
        Filter to specific chromosomes
    separator : str, default='\t'
        Output file separator
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with gene annotations
        Columns: chrom, start, end, strand, gene_id, gene_name, gene_type
        
    Notes
    -----
    If the GTF file doesn't have 'gene' feature types (e.g., MANE GTF),
    gene information is derived from transcript records.
    """
    if verbosity >= 1:
        print(f"[extract] Extracting gene annotations from: {gtf_file}")
    
    # First try to extract from 'gene' features
    records = []
    for record in iter_gtf_records(gtf_file, feature_types=['gene'], chromosomes=chromosomes):
        records.append({
            'chrom': record['chrom'],
            'start': record['start'],
            'end': record['end'],
            'strand': record['strand'],
            'gene_id': record.get('gene_id', ''),
            'gene_name': record.get('gene_name', ''),
            'gene_type': record.get('gene_type', record.get('gene_biotype', '')),
        })
    
    # If no gene features found, derive from transcript features
    if len(records) == 0:
        if verbosity >= 1:
            print("[extract] No 'gene' features found, deriving from transcripts...")
        
        # Collect transcript info and aggregate to gene level
        gene_info = {}  # gene_id -> {chrom, start, end, strand, gene_name, gene_type}
        
        for record in iter_gtf_records(gtf_file, feature_types=['transcript'], chromosomes=chromosomes):
            gene_id = record.get('gene_id', '')
            if not gene_id:
                continue
            
            if gene_id not in gene_info:
                gene_info[gene_id] = {
                    'chrom': record['chrom'],
                    'start': record['start'],
                    'end': record['end'],
                    'strand': record['strand'],
                    'gene_name': record.get('gene_name', record.get('gene', '')),
                    'gene_type': record.get('gene_type', record.get('gene_biotype', '')),
                }
            else:
                # Expand gene boundaries to include all transcripts
                gene_info[gene_id]['start'] = min(gene_info[gene_id]['start'], record['start'])
                gene_info[gene_id]['end'] = max(gene_info[gene_id]['end'], record['end'])
        
        for gene_id, info in gene_info.items():
            records.append({
                'chrom': info['chrom'],
                'start': info['start'],
                'end': info['end'],
                'strand': info['strand'],
                'gene_id': gene_id,
                'gene_name': info['gene_name'],
                'gene_type': info['gene_type'],
            })
    
    df = pl.DataFrame(records)
    
    if verbosity >= 1:
        print(f"[extract] Found {len(df)} genes")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.write_csv(output_file, separator=separator)
        if verbosity >= 1:
            print(f"[extract] Saved to: {output_file}")
    
    return df


def extract_transcript_annotations(
    gtf_file: str,
    output_file: Optional[str] = None,
    chromosomes: Optional[List[str]] = None,
    separator: str = '\t',
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract transcript annotations from a GTF file.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    output_file : str, optional
        Path to save output TSV file
    chromosomes : List[str], optional
        Filter to specific chromosomes
    separator : str, default='\t'
        Output file separator
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with transcript annotations
        Columns: chrom, start, end, strand, gene_id, transcript_id, transcript_type
    """
    if verbosity >= 1:
        print(f"[extract] Extracting transcript annotations from: {gtf_file}")
    
    records = []
    for record in iter_gtf_records(gtf_file, feature_types=['transcript'], chromosomes=chromosomes):
        records.append({
            'chrom': record['chrom'],
            'start': record['start'],
            'end': record['end'],
            'strand': record['strand'],
            'gene_id': record.get('gene_id', ''),
            'gene_name': record.get('gene_name', ''),
            'transcript_id': record.get('transcript_id', ''),
            'transcript_type': record.get('transcript_type', record.get('transcript_biotype', '')),
        })
    
    df = pl.DataFrame(records)
    
    if verbosity >= 1:
        print(f"[extract] Found {len(df)} transcripts")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.write_csv(output_file, separator=separator)
        if verbosity >= 1:
            print(f"[extract] Saved to: {output_file}")
    
    return df


def extract_exon_annotations(
    gtf_file: str,
    output_file: Optional[str] = None,
    chromosomes: Optional[List[str]] = None,
    separator: str = '\t',
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract exon annotations from a GTF file.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    output_file : str, optional
        Path to save output TSV file
    chromosomes : List[str], optional
        Filter to specific chromosomes
    separator : str, default='\t'
        Output file separator
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with exon annotations
        Columns: chrom, start, end, strand, gene_id, transcript_id, exon_number
    """
    if verbosity >= 1:
        print(f"[extract] Extracting exon annotations from: {gtf_file}")
    
    records = []
    for record in iter_gtf_records(gtf_file, feature_types=['exon'], chromosomes=chromosomes):
        # Handle RefSeq/MANE GTF format which lacks some metadata
        # Default to protein_coding for MANE (curated protein-coding transcripts)
        gene_biotype = record.get('gene_biotype', record.get('gene_type', ''))
        if not gene_biotype:
            gene_biotype = 'protein_coding'  # MANE default
        
        transcript_biotype = record.get('transcript_biotype', record.get('transcript_type', ''))
        if not transcript_biotype:
            transcript_biotype = 'protein_coding'  # MANE default
        
        # exon_id: MANE doesn't provide, generate synthetic or leave empty
        exon_id = record.get('exon_id', '')
        if not exon_id:
            # Generate synthetic exon_id for MANE: transcript_id + exon coordinates
            exon_id = f"{record.get('transcript_id', 'unknown')}:{record['start']}-{record['end']}"
        
        records.append({
            'chrom': record['chrom'],
            'start': record['start'],
            'end': record['end'],
            'strand': record['strand'],
            'gene_id': record.get('gene_id', ''),
            'gene_name': record.get('gene_name', ''),
            'gene_biotype': gene_biotype,
            'transcript_id': record.get('transcript_id', ''),
            'transcript_biotype': transcript_biotype,
            'exon_id': exon_id,
            'exon_number': int(record.get('exon_number', 0)) if str(record.get('exon_number', '')).isdigit() else 0,
            'exon_rank': int(record.get('exon_rank', record.get('exon_number', 0)))
            if str(record.get('exon_rank', record.get('exon_number', ''))).isdigit() else 0,
        })
    
    df = pl.DataFrame(records)
    
    if verbosity >= 1:
        print(f"[extract] Found {len(df)} exons")
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.write_csv(output_file, separator=separator)
        if verbosity >= 1:
            print(f"[extract] Saved to: {output_file}")
    
    return df


def extract_splice_sites_from_exons(
    exon_df: pl.DataFrame,
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract splice sites from exon annotations.
    
    Splice sites are derived from exon boundaries:
    - Donor sites: 3' end of exons (except last exon)
    - Acceptor sites: 5' start of exons (except first exon)
    
    Parameters
    ----------
    exon_df : pl.DataFrame
        Exon annotations DataFrame
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with splice site annotations
        Columns: chrom, position, strand, gene_id, transcript_id, splice_type
    """
    if verbosity >= 1:
        print("[extract] Deriving splice sites from exon boundaries...")
    
    splice_sites = []
    
    # Group by transcript
    for (transcript_id,), group in exon_df.group_by(['transcript_id']):
        # Sort exons by position
        sorted_exons = group.sort('start')
        exons = sorted_exons.to_dicts()
        
        if len(exons) < 2:
            continue  # Need at least 2 exons for splice sites
        
        gene_id = exons[0].get('gene_id', '')
        gene_name = exons[0].get('gene_name', '')
        gene_biotype = exons[0].get('gene_biotype', '')
        transcript_biotype = exons[0].get('transcript_biotype', '')
        chrom = exons[0]['chrom']
        strand = exons[0]['strand']
        
        # Fix exon numbering for MANE/RefSeq (which doesn't provide explicit exon_number)
        # Assign exon numbers based on position in sorted list
        for i, exon in enumerate(exons):
            if exon.get('exon_number', 0) == 0:
                exon['exon_number'] = i + 1  # 1-based numbering
            if exon.get('exon_rank', 0) == 0:
                exon['exon_rank'] = i + 1
        
        for i, exon in enumerate(exons):
            # Donor site: end of exon (except last)
            if i < len(exons) - 1:
                if strand == '+':
                    donor_pos = exon['end']
                else:
                    donor_pos = exon['start']
                
                splice_sites.append({
                    'chrom': chrom,
                    'start': donor_pos,
                    'end': donor_pos,
                    'position': donor_pos,
                    'strand': strand,
                    'site_type': 'donor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id,
                    'gene_name': gene_name,
                    'gene_biotype': gene_biotype,
                    'transcript_biotype': transcript_biotype,
                    'exon_id': exon.get('exon_id', ''),
                    'exon_number': exon.get('exon_number', 0),
                    'exon_rank': exon.get('exon_rank', exon.get('exon_number', 0)),
                })
            
            # Acceptor site: start of exon (except first)
            if i > 0:
                if strand == '+':
                    acceptor_pos = exon['start']
                else:
                    acceptor_pos = exon['end']
                
                splice_sites.append({
                    'chrom': chrom,
                    'start': acceptor_pos,
                    'end': acceptor_pos,
                    'position': acceptor_pos,
                    'strand': strand,
                    'site_type': 'acceptor',
                    'gene_id': gene_id,
                    'transcript_id': transcript_id,
                    'gene_name': gene_name,
                    'gene_biotype': gene_biotype,
                    'transcript_biotype': transcript_biotype,
                    'exon_id': exon.get('exon_id', ''),
                    'exon_number': exon.get('exon_number', 0),
                    'exon_rank': exon.get('exon_rank', exon.get('exon_number', 0)),
                })
    
    df = pl.DataFrame(splice_sites)
    
    # Remove duplicates while preserving transcript/exon-level metadata
    # (same genomic position can occur across transcripts; keep per-transcript entries)
    if 'site_type' in df.columns:
        df = df.unique(subset=['chrom', 'position', 'strand', 'site_type', 'gene_id', 'transcript_id', 'exon_number'])
    else:
        df = df.unique(subset=['chrom', 'position', 'strand', 'gene_id', 'transcript_id'])
    
    if verbosity >= 1:
        type_col = 'site_type' if 'site_type' in df.columns else 'splice_type'
        n_donors = df.filter(pl.col(type_col) == 'donor').height
        n_acceptors = df.filter(pl.col(type_col) == 'acceptor').height
        print(f"[extract] Found {len(df)} unique splice sites ({n_donors} donors, {n_acceptors} acceptors)")
    
    return df


def extract_splice_sites(
    gtf_file: str,
    output_file: Optional[str] = None,
    chromosomes: Optional[List[str]] = None,
    separator: str = '\t',
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract splice sites from a GTF file.
    
    This is a convenience function that:
    1. Extracts exon annotations
    2. Derives splice sites from exon boundaries
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    output_file : str, optional
        Path to save output TSV file
    chromosomes : List[str], optional
        Filter to specific chromosomes
    separator : str, default='\t'
        Output file separator
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with splice site annotations
    """
    # Extract exons
    exon_df = extract_exon_annotations(
        gtf_file, 
        chromosomes=chromosomes,
        verbosity=verbosity
    )
    
    # Derive splice sites
    splice_sites_df = extract_splice_sites_from_exons(exon_df, verbosity=verbosity)
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        splice_sites_df.write_csv(output_file, separator=separator)
        if verbosity >= 1:
            print(f"[extract] Saved splice sites to: {output_file}")
    
    return splice_sites_df


def extract_all_annotations(
    gtf_file: str,
    output_file: Optional[str] = None,
    chromosomes: Optional[List[str]] = None,
    separator: str = '\t',
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract all transcript-level annotations from a GTF file.
    
    This extracts genes, transcripts, and exons into a single DataFrame
    suitable for downstream analysis.
    
    Parameters
    ----------
    gtf_file : str
        Path to GTF annotation file
    output_file : str, optional
        Path to save output TSV file
    chromosomes : List[str], optional
        Filter to specific chromosomes
    separator : str, default='\t'
        Output file separator
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with all annotations
    """
    if verbosity >= 1:
        print(f"[extract] Extracting all annotations from: {gtf_file}")
    
    records = []
    for record in iter_gtf_records(gtf_file, chromosomes=chromosomes):
        records.append({
            'chrom': record['chrom'],
            'source': record['source'],
            'feature': record['feature'],
            'start': record['start'],
            'end': record['end'],
            'strand': record['strand'],
            'gene_id': record.get('gene_id', ''),
            'gene_name': record.get('gene_name', ''),
            'transcript_id': record.get('transcript_id', ''),
        })
    
    df = pl.DataFrame(records)
    
    if verbosity >= 1:
        feature_counts = df.group_by('feature').count().sort('count', descending=True)
        print(f"[extract] Found {len(df)} total records")
        print(feature_counts.head(10))
    
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.write_csv(output_file, separator=separator)
        if verbosity >= 1:
            print(f"[extract] Saved to: {output_file}")
    
    return df
