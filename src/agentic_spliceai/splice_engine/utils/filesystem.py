"""File system utility functions for splice engine.

Provides utilities for file operations and path handling.

Ported from: meta_spliceai/splice_engine/utils_fs.py
"""

import os
from pathlib import Path
from typing import Union, Optional, List
import polars as pl
import pandas as pd


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.
    
    Parameters
    ----------
    path : str or Path
        Directory path
        
    Returns
    -------
    Path
        The directory path
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_splice_sites(
    file_path: Union[str, Path],
    separator: str = '\t',
    dtypes: Optional[dict] = None,
    use_polars: bool = True
) -> Optional[Union[pd.DataFrame, pl.DataFrame]]:
    """Read splice sites from a TSV file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to splice sites file
    separator : str, default='\t'
        Column separator
    dtypes : dict, optional
        Column dtypes (ignored, kept for compatibility)
    use_polars : bool, default=True
        If True, return Polars DataFrame; otherwise Pandas
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame or None
        Splice sites data, or None if file not found
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    # Determine separator from extension if not explicitly set
    suffix = file_path.suffix.lower()
    if separator == '\t' and suffix == '.csv':
        separator = ','
    
    if use_polars:
        # Ensure chromosome is read as string
        return pl.read_csv(
            file_path, 
            separator=separator,
            schema_overrides={'chrom': pl.Utf8, 'seqname': pl.Utf8}
        )
    else:
        return pd.read_csv(file_path, sep=separator, dtype={'chrom': str, 'seqname': str})


def save_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    file_path: Union[str, Path],
    format: str = 'tsv'
):
    """Save a DataFrame to file.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame to save
    file_path : str or Path
        Output file path
    format : str, default='tsv'
        Output format ('tsv', 'csv', 'parquet')
    """
    file_path = Path(file_path)
    ensure_directory(file_path.parent)
    
    if isinstance(df, pl.DataFrame):
        if format == 'parquet':
            df.write_parquet(file_path)
        elif format == 'csv':
            df.write_csv(file_path)
        else:  # tsv
            df.write_csv(file_path, separator='\t')
    else:
        if format == 'parquet':
            df.to_parquet(file_path)
        elif format == 'csv':
            df.to_csv(file_path, index=False)
        else:  # tsv
            df.to_csv(file_path, sep='\t', index=False)


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """List files in a directory matching a pattern.
    
    Parameters
    ----------
    directory : str or Path
        Directory to search
    pattern : str, default="*"
        Glob pattern to match
    recursive : bool, default=False
        If True, search recursively
        
    Returns
    -------
    List[Path]
        List of matching file paths
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Parameters
    ----------
    path : str or Path
        File path
        
    Returns
    -------
    int
        File size in bytes, or 0 if file doesn't exist
    """
    path = Path(path)
    if path.exists():
        return path.stat().st_size
    return 0


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Parameters
    ----------
    size_bytes : int
        Size in bytes
        
    Returns
    -------
    str
        Formatted size string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"
