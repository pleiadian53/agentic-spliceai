"""Display and printing utility functions for splice engine.

Provides utilities for formatted output and logging.

Ported from: meta_spliceai/splice_engine/utils_doc.py
"""

from typing import Union, Optional
import pandas as pd
import polars as pl


def print_emphasized(message: str, char: str = "=", width: int = 80):
    """Print a message with emphasis using surrounding characters.
    
    Parameters
    ----------
    message : str
        Message to print
    char : str, default="="
        Character to use for emphasis
    width : int, default=80
        Width of the emphasis line
    """
    print(char * width)
    print(message)
    print(char * width)


def print_with_indent(message: str, indent: int = 2):
    """Print a message with indentation.
    
    Parameters
    ----------
    message : str
        Message to print
    indent : int, default=2
        Number of spaces to indent
    """
    prefix = " " * indent
    for line in message.split('\n'):
        print(f"{prefix}{line}")


def print_section_separator(title: str = None, char: str = "-", width: int = 60):
    """Print a section separator with optional title.
    
    Parameters
    ----------
    title : str, optional
        Section title
    char : str, default="-"
        Character to use for separator
    width : int, default=60
        Width of the separator line
    """
    if title:
        padding = (width - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * padding}")
    else:
        print(char * width)


def display(df: Union[pd.DataFrame, pl.DataFrame], n: int = 10):
    """Display a DataFrame preview.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame to display
    n : int, default=10
        Number of rows to display
    """
    if df is None:
        print("(None)")
        return
    
    if isinstance(df, pl.DataFrame):
        print(df.head(n))
    elif isinstance(df, pd.DataFrame):
        print(df.head(n).to_string())
    else:
        print(df)


def display_dataframe_in_chunks(
    df: Union[pd.DataFrame, pl.DataFrame],
    chunk_size: int = 20,
    max_chunks: int = 5
):
    """Display a DataFrame in chunks for large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        DataFrame to display
    chunk_size : int, default=20
        Number of rows per chunk
    max_chunks : int, default=5
        Maximum number of chunks to display
    """
    if df is None:
        print("(None)")
        return
    
    total_rows = len(df) if isinstance(df, pd.DataFrame) else df.height
    
    for i in range(min(max_chunks, (total_rows + chunk_size - 1) // chunk_size)):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        
        print(f"\n--- Rows {start + 1} to {end} of {total_rows} ---")
        
        if isinstance(df, pl.DataFrame):
            print(df.slice(start, chunk_size))
        else:
            print(df.iloc[start:end].to_string())
        
        if end >= total_rows:
            break
    
    if total_rows > max_chunks * chunk_size:
        print(f"\n... ({total_rows - max_chunks * chunk_size} more rows not shown)")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    str
        Formatted time string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def format_count(count: int) -> str:
    """Format large numbers with K/M/B suffixes.
    
    Parameters
    ----------
    count : int
        Number to format
        
    Returns
    -------
    str
        Formatted string (e.g., "1.5M")
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    elif count < 1_000_000_000:
        return f"{count / 1_000_000:.1f}M"
    else:
        return f"{count / 1_000_000_000:.1f}B"
