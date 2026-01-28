"""DataFrame utility functions for splice engine.

Provides utilities for working with Pandas and Polars DataFrames.

Ported from: meta_spliceai/splice_engine/utils_df.py
"""

from typing import Union, List, Optional
import pandas as pd
import polars as pl


def is_dataframe_empty(df: Union[pd.DataFrame, pl.DataFrame]) -> bool:
    """Check if a DataFrame is empty.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input DataFrame
        
    Returns
    -------
    bool
        True if DataFrame is empty or None
    """
    if df is None:
        return True
    if isinstance(df, pd.DataFrame):
        return df.empty
    elif isinstance(df, pl.DataFrame):
        return df.height == 0
    return True


def smart_read_csv(file_path: str, use_polars: bool = True, **kwargs):
    """Read a CSV/TSV file with automatic separator detection.
    
    Parameters
    ----------
    file_path : str
        Path to the file to read
    use_polars : bool, default=True
        If True, use Polars; otherwise use Pandas
    **kwargs : dict
        Additional arguments to pass to the reader function
        
    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        Loaded DataFrame
    """
    # Try reading first few lines to detect separator
    with open(file_path, 'r') as f:
        first_line = f.readline().strip()
    
    # Guess the separator by counting occurrences
    potential_separators = [',', '\t', ';', '|']
    counts = {sep: first_line.count(sep) for sep in potential_separators}
    likely_separator = max(counts.items(), key=lambda x: x[1])[0]
    
    # Default to comma if no clear separator found
    separator = likely_separator if counts[likely_separator] > 0 else ','
    
    if use_polars:
        # Set default schema overrides for genomic data
        schema_overrides = kwargs.pop('schema_overrides', {})
        
        # Ensure chromosome columns are always read as strings
        for chrom_col in ['chrom', 'seqname', 'chromosome']:
            if chrom_col not in schema_overrides:
                schema_overrides[chrom_col] = pl.Utf8
        
        return pl.read_csv(file_path, separator=separator, schema_overrides=schema_overrides, **kwargs)
    else:
        return pd.read_csv(file_path, sep=separator, **kwargs)


def get_n_unique(df: Union[pd.DataFrame, pl.DataFrame], column_name: str) -> int:
    """Get the number of unique values in a specified column."""
    if isinstance(df, pd.DataFrame):
        return df[column_name].nunique()
    elif isinstance(df, pl.DataFrame):
        return df[column_name].n_unique()
    else:
        raise TypeError("Input must be a Pandas or Polars DataFrame")


def get_unique_values(df: Union[pd.DataFrame, pl.DataFrame], column_name: str):
    """Get the unique values in a specified column."""
    if isinstance(df, pd.DataFrame):
        return df[column_name].unique()
    elif isinstance(df, pl.DataFrame):
        return df.select(column_name).unique().to_series().to_list()
    else:
        raise TypeError("Input must be a Pandas or Polars DataFrame")


def drop_columns(df: Union[pd.DataFrame, pl.DataFrame], columns: List[str]):
    """Drop specified columns from a DataFrame."""
    if isinstance(df, pd.DataFrame):
        return df.drop(columns=columns, errors='ignore')
    elif isinstance(df, pl.DataFrame):
        existing = [c for c in columns if c in df.columns]
        return df.drop(existing) if existing else df
    else:
        raise ValueError("Unsupported DataFrame type")


def subsample_dataframe(
    df: Union[pd.DataFrame, pl.DataFrame],
    n: int = None,
    frac: float = None,
    random_state: int = 42
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Subsample rows from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input DataFrame
    n : int, optional
        Number of rows to sample
    frac : float, optional
        Fraction of rows to sample
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Subsampled DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return df.sample(n=n, frac=frac, random_state=random_state)
    elif isinstance(df, pl.DataFrame):
        if n is not None:
            n = min(n, df.height)
            return df.sample(n=n, seed=random_state)
        elif frac is not None:
            n = int(df.height * frac)
            return df.sample(n=n, seed=random_state)
    return df


def align_and_append(
    df1: Union[pd.DataFrame, pl.DataFrame],
    df2: Union[pd.DataFrame, pl.DataFrame]
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Align columns and append two DataFrames.
    
    Parameters
    ----------
    df1 : pd.DataFrame or pl.DataFrame
        First DataFrame
    df2 : pd.DataFrame or pl.DataFrame
        Second DataFrame to append
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Combined DataFrame
    """
    if isinstance(df1, pl.DataFrame):
        # Ensure df2 is also Polars
        if isinstance(df2, pd.DataFrame):
            df2 = pl.from_pandas(df2)
        
        # Get common columns
        common_cols = [c for c in df1.columns if c in df2.columns]
        
        # Select only common columns and concatenate
        return pl.concat([df1.select(common_cols), df2.select(common_cols)])
    
    elif isinstance(df1, pd.DataFrame):
        if isinstance(df2, pl.DataFrame):
            df2 = df2.to_pandas()
        
        # Get common columns
        common_cols = [c for c in df1.columns if c in df2.columns]
        
        return pd.concat([df1[common_cols], df2[common_cols]], ignore_index=True)
    
    raise TypeError("Input must be Pandas or Polars DataFrames")


def filter_by_column(
    df: Union[pd.DataFrame, pl.DataFrame],
    column: str,
    values: List
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Filter DataFrame by column values.
    
    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input DataFrame
    column : str
        Column name to filter on
    values : List
        Values to keep
        
    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Filtered DataFrame
    """
    if isinstance(df, pd.DataFrame):
        return df[df[column].isin(values)]
    elif isinstance(df, pl.DataFrame):
        return df.filter(pl.col(column).is_in(values))
    raise TypeError("Input must be Pandas or Polars DataFrame")
