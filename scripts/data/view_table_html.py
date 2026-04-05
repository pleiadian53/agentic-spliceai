#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas>=2.0.0",
#     "pyarrow>=14.0.0",
# ]
# ///
"""
Display TSV/CSV and Parquet files as beautiful interactive HTML tables in browser.

Usage:
    view_table_html.py <file> [--rows N] [--randomize] [--sep SEPARATOR] [--no-open]

Examples:
    view_table_html.py data.tsv
    view_table_html.py data.csv --rows 20
    view_table_html.py data.parquet --rows 10 --randomize
"""

import argparse
import sys
import webbrowser
import tempfile
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


def generate_html(df, filename, num_rows=None, randomize=False):
    """Generate beautiful HTML with interactive table."""
    
    total_rows = len(df)
    
    # Sample or limit rows
    if randomize and num_rows and num_rows < total_rows:
        display_df = df.sample(n=num_rows, random_state=None)
        sample_msg = f"Showing {num_rows} random rows out of {total_rows:,} total"
    elif num_rows and num_rows < total_rows:
        display_df = df.head(num_rows)
        sample_msg = f"Showing first {num_rows} rows out of {total_rows:,} total"
    else:
        display_df = df
        sample_msg = f"Showing all {total_rows:,} rows"
    
    # Get data types info
    dtypes_info = []
    for col in display_df.columns:
        dtype = str(display_df[col].dtype)
        dtypes_info.append(f"{col}: {dtype}")
    
    # Convert dataframe to HTML
    table_html = display_df.to_html(
        index=False,
        classes='data-table',
        border=0,
        na_rep='<span class="null-value">null</span>'
    )
    
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{filename} - Table Viewer</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 95%;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .header .filename {{
            font-size: 16px;
            opacity: 0.9;
            font-family: 'Monaco', 'Menlo', monospace;
        }}
        
        .stats {{
            display: flex;
            gap: 30px;
            padding: 20px 40px;
            background: #f8f9fa;
            border-bottom: 1px solid #e9ecef;
            flex-wrap: wrap;
        }}
        
        .stat {{
            display: flex;
            flex-direction: column;
        }}
        
        .stat-label {{
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 24px;
            font-weight: 600;
            color: #667eea;
        }}
        
        .table-container {{
            overflow-x: auto;
            padding: 20px 40px 40px;
        }}
        
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        .data-table thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            position: sticky;
            top: 0;
            z-index: 10;
        }}
        
        .data-table th {{
            padding: 15px 12px;
            text-align: left;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            white-space: nowrap;
        }}
        
        .data-table td {{
            padding: 12px;
            border-bottom: 1px solid #e9ecef;
            color: #212529;
        }}
        
        .data-table tbody tr {{
            transition: background-color 0.2s;
        }}
        
        .data-table tbody tr:hover {{
            background-color: #f8f9fa;
        }}
        
        .data-table tbody tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        
        .data-table tbody tr:nth-child(even):hover {{
            background-color: #e9ecef;
        }}
        
        .null-value {{
            color: #adb5bd;
            font-style: italic;
        }}
        
        .info-toggle {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 0 40px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }}
        
        .info-toggle:hover {{
            background: #5568d3;
        }}
        
        .dtypes-info {{
            display: none;
            padding: 20px 40px;
            background: #f8f9fa;
            border-top: 1px solid #e9ecef;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            line-height: 1.8;
        }}
        
        .dtypes-info.show {{
            display: block;
        }}
        
        .dtypes-info h3 {{
            margin-bottom: 15px;
            font-size: 14px;
            color: #495057;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Table Viewer</h1>
            <div class="filename">{filename}</div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Rows</div>
                <div class="stat-value">{total_rows:,}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Columns</div>
                <div class="stat-value">{len(df.columns)}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Displaying</div>
                <div class="stat-value">{len(display_df):,}</div>
            </div>
        </div>
        
        <button class="info-toggle" onclick="toggleDtypes()">
            Toggle Column Types
        </button>
        
        <div class="dtypes-info" id="dtypes">
            <h3>Column Data Types:</h3>
            {'<br>'.join(dtypes_info)}
        </div>
        
        <div class="table-container">
            {table_html}
        </div>
    </div>
    
    <script>
        function toggleDtypes() {{
            const dtypes = document.getElementById('dtypes');
            dtypes.classList.toggle('show');
        }}
    </script>
</body>
</html>
"""
    
    return html_template


def main():
    parser = argparse.ArgumentParser(
        description='Display TSV/CSV and Parquet files as beautiful HTML tables in browser',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('file', help='Path to TSV/CSV or Parquet file')
    parser.add_argument('--rows', '-n', type=int, default=None,
                        help='Number of rows to display (default: all)')
    parser.add_argument('--randomize', '-r', action='store_true',
                        help='Display random sample of rows instead of first N rows')
    parser.add_argument('--sep', '-s', default=None,
                        help='Column separator (default: auto-detect from file extension)')
    parser.add_argument('--no-open', action='store_true',
                        help='Do not automatically open browser')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.file, args.sep)
    
    # Generate HTML
    filename = Path(args.file).name
    html_content = generate_html(df, filename, args.rows, args.randomize)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        temp_path = f.name
    
    print(f"✓ Generated HTML table: {temp_path}")
    
    # Open in browser
    if not args.no_open:
        webbrowser.open(f'file://{temp_path}')
        print(f"✓ Opening in browser...")
    else:
        print(f"  To view: open {temp_path}")


if __name__ == '__main__':
    main()
