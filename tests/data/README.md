# Data Setup Scripts (Private)

**⚠️ NOT FOR PUBLIC SHARING - Contains local paths**

This directory contains private setup scripts with hardcoded local paths for development.

## Setup Scripts

### `setup_data_symlinks.sh`

Creates symlinks from `agentic-spliceai/data/` to `meta-spliceai/data/` for genomic datasets.

**Usage**:
```bash
cd /Users/pleiadian53/work/agentic-spliceai
chmod +x tests/data/setup_data_symlinks.sh
./tests/data/setup_data_symlinks.sh
```

**Privacy Note**: This script contains hardcoded paths specific to the local development environment and should not be shared publicly.

**What it does**:
- Links genomic annotations (ensembl, mane, GRCh38_MANE)
- Links pre-trained models (spliceai, openspliceai)
- Links training/test datasets
- Links reference files (uORF data, GTEx data)
- Preserves existing local data in `data/llm/`

**Benefits**:
- No data duplication (~20-30GB saved)
- Single source of truth in meta-spliceai
- Automatic updates when meta-spliceai data changes

## Data Structure

See `data/README.md` for public-facing documentation of the data structure.

## Notes

- All symlinks use absolute paths
- Re-run setup script if meta-spliceai data structure changes
- Symlinks are explicitly ignored in `.gitignore`
- See `data/README.md` for detailed documentation on the data structure
