# Agentic-SpliceAI Data Directory

This directory contains both local datasets and symlinked genomic datasets from `meta-spliceai`.

## Directory Structure

```
data/
├── llm/                          # Local: LLM-related data
├── splice_sites_enhanced.tsv     # Local: Example splice site data
├── splice_sites_enhanced_summary.md  # Local: Data summary
│
├── ensembl/          -> symlink  # Ensembl annotations (GRCh37, GRCh38)
├── mane/             -> symlink  # MANE Select transcripts
├── GRCh38_MANE/      -> symlink  # GRCh38 MANE-specific data
├── models/           -> symlink  # Pre-trained SpliceAI models
├── spliceai_analysis/-> symlink  # SpliceAI analysis results
├── spliceai_eval/    -> symlink  # SpliceAI evaluation datasets
│
├── train_pc_1000_3mers/         -> symlink  # Training data
├── train_pc_100_3mers_diverse/  -> symlink  # Diverse training data
├── test_pc_1000_3mers/          -> symlink  # Test data
├── test_quick/                  -> symlink  # Quick test data
│
├── query_uorfs.fa               -> symlink  # uORF sequences
├── query_uorfs.gtf              -> symlink  # uORF annotations
├── gtex_uORFconnected_txs.csv   -> symlink  # GTEx uORF data
└── supplementary-tables.xlsx    -> symlink  # Supplementary data
```

## Dataset Categories

### 1. Local Datasets (Agentic-SpliceAI)

**LLM Data** (`llm/`):
- LLM-specific configurations and prompts
- Agent workflow data

**Example Data**:
- `splice_sites_enhanced.tsv` - Sample splice site dataset
- `splice_sites_enhanced_summary.md` - Dataset documentation

### 2. Genomic Reference Data (Symlinked)

**Genome Annotations**:
- `ensembl/` - Ensembl gene annotations
  - `GRCh37/` - Human genome build 37
  - `GRCh38/` - Human genome build 38
- `mane/` - MANE Select transcripts (clinical-grade)
- `GRCh38_MANE/` - GRCh38-specific MANE data

**Pre-trained Models**:
- `models/spliceai/` - Base SpliceAI models
- `models/openspliceai/` - OpenSpliceAI models

### 3. Analysis & Evaluation Data (Symlinked)

**SpliceAI Analysis**:
- `spliceai_analysis/` - Analysis outputs and results
- `spliceai_eval/` - Evaluation datasets and benchmarks
  - `meta_models/` - Meta-learning model evaluations

### 4. Training & Test Data (Symlinked)

**Training Datasets**:
- `train_pc_1000_3mers/` - Large training set (1000 protein-coding genes)
- `train_pc_100_3mers_diverse/` - Diverse training set (100 genes)

**Test Datasets**:
- `test_pc_1000_3mers/` - Large test set
- `test_quick/` - Quick test set for development

### 5. Reference Files (Symlinked)

**uORF Data**:
- `query_uorfs.fa` - uORF sequences (FASTA)
- `query_uorfs.gtf` - uORF annotations (GTF)
- `gtex_uORFconnected_txs.csv` - GTEx uORF-connected transcripts
- `gtex_uORFconnected_txs.w_utrs.csv` - With UTR information

**Supplementary**:
- `supplementary-tables.xlsx` - Supplementary tables from publications

## Setup

### Create Symlinks

Run the setup script to create symlinks:

```bash
cd /Users/pleiadian53/work/agentic-spliceai
chmod +x tests/data/setup_data_symlinks.sh
./tests/data/setup_data_symlinks.sh
```

**Note**: The setup script is in `tests/data/` (private, not in git) as it contains local paths.

### Verify Setup

```bash
# Check symlinks
ls -lh data/

# Verify genomic data access
ls -lh data/ensembl/GRCh38/

# Check model availability
ls -lh data/models/spliceai/
```

## Data Access in Code

### Python Examples

```python
from pathlib import Path

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Access genomic annotations
ensembl_grch38 = DATA_DIR / "ensembl" / "GRCh38"
mane_transcripts = DATA_DIR / "mane" / "GRCh38"

# Access pre-trained models
spliceai_models = DATA_DIR / "models" / "spliceai"

# Access training data
train_data = DATA_DIR / "train_pc_1000_3mers"

# Access reference files
uorf_fasta = DATA_DIR / "query_uorfs.fa"
gtex_data = DATA_DIR / "gtex_uORFconnected_txs.csv"
```

### Configuration

The `agentic_spliceai` package automatically detects data paths:

```python
from agentic_spliceai.data_access import get_data_path

# Get specific dataset paths
ensembl_path = get_data_path("ensembl/GRCh38")
model_path = get_data_path("models/spliceai")
```

## Storage Considerations

### Disk Usage

**Local files** (~7 KB):
- Small example datasets
- LLM configurations

**Symlinked data** (varies):
- Genomic annotations: ~1-10 GB per genome build
- Pre-trained models: ~100 MB - 1 GB
- Training datasets: ~100 MB - 10 GB
- Reference files: ~10-100 MB

**Total**: No duplication - symlinks point to meta-spliceai data

### Benefits of Symlinks

✅ **No data duplication** - Single source of truth  
✅ **Automatic updates** - Changes in meta-spliceai reflected here  
✅ **Disk space efficient** - Symlinks are tiny (bytes)  
✅ **Organized structure** - Clear separation of local vs. shared data  

## Maintenance

### Updating Symlinks

If meta-spliceai data structure changes, re-run:

```bash
./tests/data/setup_data_symlinks.sh
```

### Adding New Datasets

To add new symlinks manually:

```bash
# Symlink a directory
ln -s /Users/pleiadian53/work/meta-spliceai/data/new_dataset data/new_dataset

# Symlink a file
ln -s /Users/pleiadian53/work/meta-spliceai/data/new_file.tsv data/new_file.tsv
```

### Removing Symlinks

```bash
# Remove symlink (does NOT delete source data)
rm data/symlink_name

# Verify it's a symlink first
ls -lh data/symlink_name
```

## Genome Builds

### GRCh37 (hg19)
- Older reference genome
- Still used in many clinical applications
- Path: `data/ensembl/GRCh37/`

### GRCh38 (hg38)
- Current reference genome
- Recommended for new analyses
- Path: `data/ensembl/GRCh38/`

### MANE Select
- Clinical-grade transcript set
- One transcript per gene
- Path: `data/mane/GRCh38/`

## Notes

- **Symlinks are relative to the repository root**
- **Do not commit large data files** - Only symlinks and small examples
- **Source data lives in meta-spliceai** - This is the single source of truth
- **Local data is for agentic-spliceai-specific** - LLM configs, agent data, etc.

## Troubleshooting

### Broken Symlinks

If symlinks are broken:

```bash
# Find broken symlinks
find data/ -type l ! -exec test -e {} \; -print

# Re-run setup
./setup_data_symlinks.sh
```

### Permission Issues

```bash
# Ensure read permissions on source data
chmod -R a+r /Users/pleiadian53/work/meta-spliceai/data/
```

### Path Issues

Symlinks use absolute paths. If you move repositories:

```bash
# Remove old symlinks
find data/ -type l -delete

# Re-create with new paths
./setup_data_symlinks.sh
```
