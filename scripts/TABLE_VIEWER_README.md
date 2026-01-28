# Table Viewer Scripts

Beautiful and easy-to-use tools for viewing TSV, CSV, and Parquet files.

## Available Commands

### `vt` - View Table (Browser) 🌐
Opens an **aesthetic, interactive HTML table** in your browser with:
- Beautiful gradient design
- Sticky headers
- Row hover effects
- Column type information
- Data statistics

**Usage:**
```bash
vt gene_manifest.tsv                    # View entire file
vt data.csv --rows 10                   # View first 10 rows
vt results.parquet --rows 20 -r         # View 20 random rows
vt data.txt --sep '|' --rows 50         # Custom separator
```

### `vtt` - View Table Terminal 📟
Quick terminal display using grid format (no browser).

**Usage:**
```bash
vtt gene_manifest.tsv                   # Quick terminal view
vtt data.csv --rows 5                   # First 5 rows in terminal
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--rows N` | `-n` | Limit number of rows to display |
| `--randomize` | `-r` | Show random sample instead of first N rows |
| `--sep 'X'` | `-s` | Specify custom separator |
| `--no-open` | | Generate HTML but don't open browser (vt only) |

## Examples

```bash
# Quick preview of first 10 rows
vt my_data.tsv -n 10

# Random sample for large files
vt large_dataset.parquet -n 100 -r

# Terminal output for scripting
vtt results.csv --rows 20

# Custom separator
vt pipe_delimited.txt --sep '|'
```

## Features

### Browser View (vt)
- ✨ Beautiful gradient design
- 📊 Interactive sticky headers
- 🔍 Hover effects on rows
- 📈 Data type information (toggle)
- 📉 Statistics (rows, columns, displaying)
- 🎨 Zebra striping
- 💡 Null value highlighting

### Terminal View (vtt)
- ⚡ Fast, no browser needed
- 📋 Clean grid format
- 🔢 Row/column counts

## Supported File Types

- `.tsv` - Tab-separated values
- `.csv` - Comma-separated values
- `.parquet` - Apache Parquet format
- Any text file with custom separator

## Installation

These scripts are already set up in your PATH. After opening a new terminal or running:

```bash
source ~/.zshrc
```

You can use `vt` and `vtt` from any directory!
