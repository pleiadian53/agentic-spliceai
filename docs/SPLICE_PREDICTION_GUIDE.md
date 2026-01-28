# Splice Site Prediction Guide

**Agentic-SpliceAI Base Layer Integration**

This guide explains how to use the splice site prediction capabilities in agentic-spliceai, which integrates the base layer from the meta-spliceai project.

---

## Overview

The agentic-spliceai system now includes true splice site prediction capabilities through integration with meta-spliceai's base layer. This provides:

- **Base Model Support**: SpliceAI and OpenSpliceAI models
- **Flexible Interface**: CLI and Python API
- **Production-Ready**: Memory-efficient processing with checkpointing
- **Model-Agnostic**: Easy to extend with custom models

---

## Quick Start

### Prerequisites

1. **Install meta-spliceai** (required dependency):
   ```bash
   # From the meta-spliceai project directory
   cd /Users/pleiadian53/work/meta-spliceai
   mamba activate agentic-spliceai
   pip install -e .
   ```

2. **Verify installation**:
   ```bash
   python -c "import meta_spliceai; print('meta-spliceai installed')"
   ```

### CLI Usage

```bash
# Activate environment
mamba activate agentic-spliceai

# Predict for specific genes
agentic-spliceai-predict --genes BRCA1 TP53 UNC13A

# Predict for a chromosome
agentic-spliceai-predict --chromosomes 21

# Use SpliceAI (GRCh37) instead of OpenSpliceAI (GRCh38)
agentic-spliceai-predict --base-model spliceai --genes BRCA1

# Test mode with sample data
agentic-spliceai-predict --mode test --coverage sample --genes BRCA1
```

### Python API Usage

```python
from agentic_spliceai.splice_engine import predict_splice_sites

# Simple prediction
results = predict_splice_sites(genes=["BRCA1", "TP53"])

# Access results
positions = results["positions"]  # Polars DataFrame
print(f"Found {positions.height} splice positions")

# Get high-confidence predictions
high_conf = positions.filter(
    (pl.col("donor_score") > 0.9) | (pl.col("acceptor_score") > 0.9)
)
```

---

## Entry Points

### Option 1: CLI Entry Point (Simplest)

**Command**: `agentic-spliceai-predict`

**Pros**:
- ✅ Zero code needed
- ✅ Automatic data path resolution
- ✅ Production-ready
- ✅ Clear CLI interface

**Usage**:
```bash
agentic-spliceai-predict --base-model openspliceai --mode test --coverage sample
```

### Option 2: Python API Entry Point (Most Flexible)

**Module**: `agentic_spliceai.splice_engine`

**Function**: `predict_splice_sites()` or `run_base_model_predictions()`

**Pros**:
- ✅ Full programmatic control
- ✅ Integration with custom workflows
- ✅ Access to in-memory results
- ✅ Gene-level filtering

**Usage**:
```python
from agentic_spliceai.splice_engine import run_base_model_predictions

results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['UNC13A', 'STMN2'],
    verbosity=1
)
```

### Option 3: High-Level API (Object-Oriented)

**Module**: `agentic_spliceai.splice_engine.api`

**Class**: `SplicePredictionAPI`

**Pros**:
- ✅ Clean object-oriented interface
- ✅ Built-in filtering and export methods
- ✅ Convenient for interactive use

**Usage**:
```python
from agentic_spliceai.splice_engine.api import SplicePredictionAPI

# Initialize API
api = SplicePredictionAPI(base_model="openspliceai")

# Predict for genes
results = api.predict_genes(["BRCA1", "TP53"])

# Get high-confidence predictions
high_conf = api.get_high_confidence_predictions(results, threshold=0.9)

# Export to file
api.export_predictions(results, "predictions.csv", format="csv")
```

---

## Examples

### Example 1: Basic Gene Prediction

```python
from agentic_spliceai.splice_engine import predict_splice_sites

# Predict splice sites for ALS-related genes
results = predict_splice_sites(
    genes=["UNC13A", "STMN2", "TARDBP"],
    base_model="openspliceai"
)

# Access predictions
positions = results["positions"]
print(f"Total positions: {positions.height}")

# Filter to high-confidence donors
import polars as pl
donors = positions.filter(
    (pl.col("splice_type") == "donor") & 
    (pl.col("donor_score") > 0.9)
)
print(f"High-confidence donors: {donors.height}")
```

### Example 2: Chromosome-Wide Analysis

```python
from agentic_spliceai.splice_engine.api import SplicePredictionAPI

# Initialize API
api = SplicePredictionAPI(base_model="openspliceai", verbosity=1)

# Predict for chromosome 21
results = api.predict_chromosomes(["21"])

# Get error positions (false positives and false negatives)
errors = api.get_error_positions(results)
print(f"Total errors: {errors.height}")

# Export results
api.export_predictions(results, "chr21_predictions.tsv", format="tsv")
```

### Example 3: Quick Interactive Analysis

```python
from agentic_spliceai.splice_engine.api import quick_predict, predict_and_filter

# Quick prediction
results = quick_predict(genes=["BRCA1"])

# Or predict and filter in one step
high_conf = predict_and_filter(
    genes=["BRCA1", "TP53"],
    confidence_threshold=0.95
)

print(high_conf.head())
```

### Example 4: Custom Configuration

```python
from agentic_spliceai.splice_engine import run_base_model_predictions

# Advanced configuration
results = run_base_model_predictions(
    base_model="openspliceai",
    target_genes=["UNC13A"],
    mode="test",
    coverage="gene_subset",
    threshold=0.5,
    verbosity=2,
    no_tn_sampling=False,  # Sample true negatives to reduce memory
    save_nucleotide_scores=False  # Don't save per-nucleotide scores
)

# Access different result components
positions = results["positions"]  # All positions
errors = results["error_analysis"]  # TP/FP/FN classifications
sequences = results["analysis_sequences"]  # Sequence windows
manifest = results["gene_manifest"]  # Processing metadata
```

---

## CLI Reference

### Basic Options

```bash
agentic-spliceai-predict [OPTIONS]
```

**Target Selection** (mutually exclusive):
- `--genes GENE1 GENE2 ...` - Gene symbols or IDs
- `--chromosomes CHR1 CHR2 ...` - Chromosomes to process

**Model Selection**:
- `--base-model {openspliceai,spliceai}` - Base model (default: openspliceai)

**Mode and Coverage**:
- `--mode {test,production}` - Execution mode (default: test)
- `--coverage {gene_subset,chromosome,full_genome,sample}` - Coverage level

**Output Control**:
- `--output-dir DIR` - Output directory
- `--verbosity {0,1,2}` - Output verbosity (default: 1)
- `--format {summary,json,paths}` - Output format

**Advanced Options**:
- `--threshold FLOAT` - Score threshold (default: 0.5)
- `--no-tn-sampling` - Keep all true negatives (memory intensive)
- `--save-nucleotide-scores` - Save per-nucleotide scores (large files)

### Examples

```bash
# Basic gene prediction
agentic-spliceai-predict --genes BRCA1 TP53

# Chromosome analysis with detailed output
agentic-spliceai-predict --chromosomes 21 --verbosity 2

# Production run with custom threshold
agentic-spliceai-predict \
  --mode production \
  --coverage full_genome \
  --threshold 0.5 \
  --output-dir /path/to/output

# Test mode with JSON output
agentic-spliceai-predict \
  --genes UNC13A STMN2 \
  --mode test \
  --format json > results.json
```

---

## Python API Reference

### Core Functions

#### `predict_splice_sites()`

```python
def predict_splice_sites(
    genes: Optional[List[str]] = None,
    chromosomes: Optional[List[str]] = None,
    base_model: str = "openspliceai",
    **kwargs
) -> Dict[str, Any]
```

Convenience wrapper for splice site prediction.

**Parameters**:
- `genes`: Gene symbols or IDs
- `chromosomes`: Chromosomes to process
- `base_model`: "openspliceai" or "spliceai"
- `**kwargs`: Additional configuration

**Returns**: Results dictionary

#### `run_base_model_predictions()`

```python
def run_base_model_predictions(
    base_model: str = "openspliceai",
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    verbosity: int = 1,
    **kwargs
) -> Dict[str, Any]
```

Full-featured prediction function with complete configuration control.

### SplicePredictionAPI Class

```python
class SplicePredictionAPI:
    def __init__(self, base_model: str = "openspliceai", verbosity: int = 1, **config_kwargs)
    def predict_genes(self, genes: List[str], **kwargs) -> Dict[str, Any]
    def predict_chromosomes(self, chromosomes: List[str], **kwargs) -> Dict[str, Any]
    
    @staticmethod
    def get_high_confidence_predictions(results: Dict, threshold: float = 0.9) -> DataFrame
    
    @staticmethod
    def get_error_positions(results: Dict, error_type: Optional[str] = None) -> DataFrame
    
    @staticmethod
    def export_predictions(results: Dict, output_path: str, format: str = "csv")
```

---

## Data Requirements

### Genomic Resources

The system requires:

1. **Reference Genome** (FASTA)
2. **Gene Annotations** (GTF/GFF3)
3. **Base Model Weights**

These are automatically managed by meta-spliceai's genomic resources system.

### Directory Structure

```
data/
├── mane/GRCh38/                 # For OpenSpliceAI
│   ├── MANE.GRCh38.v1.3.refseq_genomic.gff
│   ├── GCF_000001405.40_GRCh38.p14_genomic.fna
│   └── openspliceai_eval/       # Output directory
│
└── ensembl/GRCh37/              # For SpliceAI
    ├── Homo_sapiens.GRCh37.87.gtf
    ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa
    └── spliceai_eval/           # Output directory
```

---

## Results Structure

The prediction functions return a dictionary with:

```python
{
    'success': bool,                    # Whether workflow completed
    'positions': pl.DataFrame,          # All analyzed positions
    'error_analysis': pl.DataFrame,     # TP/FP/FN classifications
    'analysis_sequences': pl.DataFrame, # Sequence windows (±250nt)
    'gene_manifest': pl.DataFrame,      # Processing metadata
    'nucleotide_scores': pl.DataFrame,  # Per-nucleotide scores (if enabled)
    'paths': {                          # Output file paths
        'eval_dir': str,
        'artifacts_dir': str,
        'positions_artifact': str,
        'errors_artifact': str
    },
    'manifest_summary': {               # Processing statistics
        'processed_genes': int,
        'total_positions': int
    }
}
```

### Key DataFrames

**positions**: All splice site predictions
- Columns: `gene_id`, `position`, `splice_type`, `donor_score`, `acceptor_score`, `error_type`, etc.

**error_analysis**: Error classifications
- Columns: `error_type` (TP/FP/FN/TN), `gene_id`, `position`, `splice_type`, `strand`

**analysis_sequences**: Sequence windows around each position
- Columns: `gene_id`, `position`, `sequence`, `splice_type`, `error_type`

---

## Troubleshooting

### Issue: ImportError for meta_spliceai

**Error**: `ImportError: meta-spliceai is required for splice prediction`

**Solution**: Install meta-spliceai:
```bash
cd /Users/pleiadian53/work/meta-spliceai
mamba activate agentic-spliceai
pip install -e .
```

### Issue: Out of Memory

**Solution 1**: Reduce batch size (via meta-spliceai config)
**Solution 2**: Process fewer genes/chromosomes at once
**Solution 3**: Disable nucleotide score collection (default)

### Issue: Wrong Environment

**Error**: Module not found or import errors

**Solution**: Always activate the correct environment:
```bash
mamba activate agentic-spliceai
# OR use mamba run
mamba run -n agentic-spliceai agentic-spliceai-predict --genes BRCA1
```

---

## Integration with Nexus Research Agent

The splice prediction capabilities can be used by the Nexus Research Agent for:

1. **Literature-Guided Analysis**: Research splice mechanisms, then predict sites
2. **Validation**: Validate predictions against published findings
3. **Discovery**: Identify novel splice patterns for further investigation

Example workflow:
```python
# 1. Research splice mechanisms
from nexus.agents.research import ResearchAgent
agent = ResearchAgent()
report = agent.research("UNC13A cryptic exon in ALS")

# 2. Predict splice sites
from agentic_spliceai.splice_engine import predict_splice_sites
results = predict_splice_sites(genes=["UNC13A"])

# 3. Analyze predictions in context of research
high_conf = results["positions"].filter(pl.col("donor_score") > 0.9)
# Compare with literature findings...
```

---

## Next Steps

1. **Explore Examples**: Try the examples in this guide
2. **Read Meta-SpliceAI Docs**: See `/Users/pleiadian53/work/meta-spliceai/docs/base_models/BASE_LAYER_INTEGRATION_GUIDE.md`
3. **Customize**: Extend with custom models or analysis workflows
4. **Integrate**: Combine with Nexus Research Agent for literature-guided analysis

---

## Additional Resources

- **Meta-SpliceAI Project**: `/Users/pleiadian53/work/meta-spliceai`
- **Base Layer Guide**: `meta-spliceai/docs/base_models/BASE_LAYER_INTEGRATION_GUIDE.md`
- **Agentic-SpliceAI README**: `README.md`
- **Nexus Documentation**: `src/nexus/README.md`

---

**Questions?** Open an issue or consult the meta-spliceai documentation.
