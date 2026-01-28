"""
Sequence extraction utilities.

This module provides standalone functions for extracting genomic sequences
from FASTA files based on gene/transcript annotations.

No external dependencies on meta-spliceai.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Iterator
from pathlib import Path

import polars as pl


def read_fasta_index(fasta_file: str) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Read or create FASTA index for random access.
    
    Parameters
    ----------
    fasta_file : str
        Path to FASTA file
        
    Returns
    -------
    Dict[str, Tuple[int, int, int, int]]
        Dictionary mapping sequence names to (offset, length, bases_per_line, bytes_per_line)
    """
    fai_file = f"{fasta_file}.fai"
    
    if os.path.exists(fai_file):
        # Read existing index
        index = {}
        with open(fai_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    name = parts[0]
                    length = int(parts[1])
                    offset = int(parts[2])
                    bases_per_line = int(parts[3])
                    bytes_per_line = int(parts[4])
                    index[name] = (offset, length, bases_per_line, bytes_per_line)
        return index
    
    # Create index by scanning file
    index = {}
    with open(fasta_file, 'r') as f:
        current_name = None
        current_offset = 0
        current_length = 0
        bases_per_line = 0
        bytes_per_line = 0
        seq_start_offset = 0
        
        while True:
            line = f.readline()
            if not line:
                break
            
            if line.startswith('>'):
                # Save previous sequence
                if current_name:
                    index[current_name] = (seq_start_offset, current_length, bases_per_line, bytes_per_line)
                
                # Start new sequence
                current_name = line[1:].split()[0]
                current_offset = f.tell()
                seq_start_offset = current_offset
                current_length = 0
                bases_per_line = 0
                bytes_per_line = 0
            else:
                if bases_per_line == 0:
                    bases_per_line = len(line.strip())
                    bytes_per_line = len(line)
                current_length += len(line.strip())
        
        # Save last sequence
        if current_name:
            index[current_name] = (seq_start_offset, current_length, bases_per_line, bytes_per_line)
    
    return index


def extract_sequence_from_fasta(
    fasta_file: str,
    chrom: str,
    start: int,
    end: int,
    index: Optional[Dict] = None
) -> str:
    """
    Extract a sequence region from a FASTA file.
    
    Parameters
    ----------
    fasta_file : str
        Path to FASTA file
    chrom : str
        Chromosome/sequence name
    start : int
        Start position (1-based, inclusive)
    end : int
        End position (1-based, inclusive)
    index : Dict, optional
        Pre-computed FASTA index
        
    Returns
    -------
    str
        Extracted sequence (uppercase)
    """
    if index is None:
        index = read_fasta_index(fasta_file)
    
    # Try different chromosome name formats
    chrom_variants = [chrom]
    if chrom.startswith('chr'):
        chrom_variants.append(chrom[3:])
    else:
        chrom_variants.append(f'chr{chrom}')
    
    chrom_key = None
    for variant in chrom_variants:
        if variant in index:
            chrom_key = variant
            break
    
    if chrom_key is None:
        raise ValueError(f"Chromosome {chrom} not found in FASTA file")
    
    offset, length, bases_per_line, bytes_per_line = index[chrom_key]
    
    # Convert to 0-based coordinates
    start_0 = start - 1
    end_0 = end
    
    # Validate coordinates
    if start_0 < 0:
        start_0 = 0
    if end_0 > length:
        end_0 = length
    
    # Calculate file positions
    start_line = start_0 // bases_per_line
    start_col = start_0 % bases_per_line
    
    end_line = (end_0 - 1) // bases_per_line
    end_col = (end_0 - 1) % bases_per_line
    
    # Read sequence
    sequence = []
    with open(fasta_file, 'r') as f:
        for line_num in range(start_line, end_line + 1):
            line_offset = offset + line_num * bytes_per_line
            f.seek(line_offset)
            line = f.readline().strip()
            
            if line_num == start_line and line_num == end_line:
                sequence.append(line[start_col:end_col + 1])
            elif line_num == start_line:
                sequence.append(line[start_col:])
            elif line_num == end_line:
                sequence.append(line[:end_col + 1])
            else:
                sequence.append(line)
    
    return ''.join(sequence).upper()


def extract_gene_sequences(
    gene_df: pl.DataFrame,
    fasta_file: str,
    output_file: Optional[str] = None,
    context: int = 0,
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Extract sequences for genes from a FASTA file.
    
    Parameters
    ----------
    gene_df : pl.DataFrame
        Gene annotations with columns: chrom, start, end, strand, gene_id
    fasta_file : str
        Path to genome FASTA file
    output_file : str, optional
        Path to save output file
    context : int, default=0
        Additional context bases to include on each side
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with gene sequences
        Columns: gene_id, gene_name, chrom, start, end, strand, sequence
    """
    if verbosity >= 1:
        print(f"[extract] Extracting sequences for {len(gene_df)} genes...")
    
    # Build FASTA index
    index = read_fasta_index(fasta_file)
    
    sequences = []
    errors = []
    
    for row in gene_df.iter_rows(named=True):
        gene_id = row['gene_id']
        chrom = row['chrom']
        start = row['start'] - context
        end = row['end'] + context
        strand = row['strand']
        gene_name = row.get('gene_name', gene_id)
        
        try:
            seq = extract_sequence_from_fasta(fasta_file, chrom, start, end, index)
            
            # Reverse complement if on minus strand
            if strand == '-':
                seq = reverse_complement(seq)
            
            sequences.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'chrom': chrom,
                'start': start,
                'end': end,
                'strand': strand,
                'sequence': seq,
            })
        except Exception as e:
            errors.append((gene_id, str(e)))
            if verbosity >= 2:
                print(f"[warning] Failed to extract sequence for {gene_id}: {e}")
    
    if verbosity >= 1:
        print(f"[extract] Successfully extracted {len(sequences)} sequences")
        if errors:
            print(f"[warning] Failed to extract {len(errors)} sequences")
    
    result_df = pl.DataFrame(sequences)
    
    if output_file:
        # Determine format from extension
        ext = Path(output_file).suffix.lower()
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        if ext == '.parquet':
            result_df.write_parquet(output_file)
        else:
            result_df.write_csv(output_file, separator='\t')
        
        if verbosity >= 1:
            print(f"[extract] Saved to: {output_file}")
    
    return result_df


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.
    
    Parameters
    ----------
    seq : str
        DNA sequence
        
    Returns
    -------
    str
        Reverse complement sequence
    """
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                  'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n': 'n'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def one_hot_encode(seq: str) -> List[List[int]]:
    """
    One-hot encode a DNA sequence.
    
    Parameters
    ----------
    seq : str
        DNA sequence (A, C, G, T, N)
        
    Returns
    -------
    List[List[int]]
        One-hot encoded sequence, shape (len(seq), 4)
        Order: A, C, G, T
    """
    encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0],  # Unknown
    }
    return [encoding.get(base.upper(), [0, 0, 0, 0]) for base in seq]


def load_chromosome_sequences(
    fasta_file: str,
    chromosomes: Optional[List[str]] = None,
    verbosity: int = 1
) -> Dict[str, str]:
    """
    Load full chromosome sequences into memory.
    
    WARNING: This can use significant memory for large genomes.
    
    Parameters
    ----------
    fasta_file : str
        Path to genome FASTA file
    chromosomes : List[str], optional
        Specific chromosomes to load (None = all)
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping chromosome names to sequences
    """
    if verbosity >= 1:
        print(f"[load] Loading chromosome sequences from: {fasta_file}")
    
    # Normalize chromosome filter
    chrom_set = None
    if chromosomes:
        chrom_set = set()
        for c in chromosomes:
            chrom_set.add(c)
            if c.startswith('chr'):
                chrom_set.add(c[3:])
            else:
                chrom_set.add(f'chr{c}')
    
    sequences = {}
    current_name = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Save previous sequence
                if current_name and (chrom_set is None or current_name in chrom_set):
                    sequences[current_name] = ''.join(current_seq).upper()
                
                # Start new sequence
                current_name = line[1:].split()[0]
                current_seq = []
                
                # Skip if not in filter
                if chrom_set and current_name not in chrom_set:
                    current_name = None
            elif current_name:
                current_seq.append(line.strip())
        
        # Save last sequence
        if current_name and (chrom_set is None or current_name in chrom_set):
            sequences[current_name] = ''.join(current_seq).upper()
    
    if verbosity >= 1:
        total_bases = sum(len(seq) for seq in sequences.values())
        print(f"[load] Loaded {len(sequences)} chromosomes ({total_bases:,} bases)")
    
    return sequences
