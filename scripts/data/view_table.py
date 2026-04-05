#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.0.0",
#     "pyarrow>=14.0.0",
#     "tabulate>=0.9.0",
# ]
# ///
"""
Display TSV/CSV and Parquet files in a nicely formatted table.

Usage:
    view_table.py <file> [--rows N] [--randomize] [--sep SEPARATOR]

Examples:
    view_table.py data.tsv
    view_table.py data.csv --rows 20
    view_table.py data.parquet --rows 10 --randomize
    view_table.py data.csv --sep ','
"""

import argparse
import sys
from pathlib import Path


def load_data(filepath, separator=None):
    """Load data from TSV/CSV or Parquet file."""
    import pandas as pd
    
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found", file=sys.stderr)
        sys.exit(1)
    
    # Determine file type and load accordingly
    if filepath.suffix.lower() == '.parquet':
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            print(f"Error reading Parquet file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Handle TSV/CSV
        if separator is None:
            # Auto-detect separator
            if filepath.suffix.lower() == '.csv':
                separator = ','
            else:  # Default to tab for .tsv and unknown extensions
                separator = '\t'
        
        try:
            df = pd.read_csv(filepath, sep=separator)
        except Exception as e:
            print(f"Error reading CSV/TSV file: {e}", file=sys.stderr)
            sys.exit(1)
    
    return df


def display_table(df, num_rows=None, randomize=False):
    """Display dataframe in a nicely formatted table."""
    from tabulate import tabulate
    
    total_rows = len(df)
    
    # Sample or limit rows
    if randomize and num_rows and num_rows < total_rows:
        display_df = df.sample(n=num_rows, random_state=None)
        sample_msg = f"Showing {num_rows} random rows out of {total_rows} total rows"
    elif num_rows and num_rows < total_rows:
        display_df = df.head(num_rows)
        sample_msg = f"Showing first {num_rows} rows out of {total_rows} total rows"
    else:
        display_df = df
        sample_msg = f"Showing all {total_rows} rows"
    
    # Print summary
    print(f"\n{sample_msg}")
    print(f"Columns: {len(df.columns)}\n")
    
    # Display table with tabulate for nice formatting
    print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Display TSV/CSV and Parquet files in a formatted table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.tsv
  %(prog)s data.csv --rows 20
  %(prog)s data.parquet --rows 10 --randomize
  %(prog)s data.csv --sep ','
        """
    )
    
    parser.add_argument('file', help='Path to TSV/CSV or Parquet file')
    parser.add_argument('--rows', '-n', type=int, default=None,
                        help='Number of rows to display (default: all)')
    parser.add_argument('--randomize', '-r', action='store_true',
                        help='Display random sample of rows instead of first N rows')
    parser.add_argument('--sep', '-s', default=None,
                        help='Column separator (default: auto-detect from file extension)')
    
    args = parser.parse_args()
    
    # Load and display data
    df = load_data(args.file, args.sep)
    display_table(df, args.rows, args.randomize)


if __name__ == '__main__':
    main()
