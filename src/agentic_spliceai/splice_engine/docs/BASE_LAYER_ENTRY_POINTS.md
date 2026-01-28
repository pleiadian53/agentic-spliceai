# Base Layer Entry Points

This document describes the entry points for the base layer splice site prediction pipeline in `agentic-spliceai`, and how they map to the equivalent functionality in `meta-spliceai`.

## Overview

The base layer provides splice site predictions using pre-trained models (SpliceAI, OpenSpliceAI). It generates per-nucleotide scores (donor, acceptor, neither) for gene sequences.

## Entry Points Comparison

| Purpose | meta-spliceai | agentic-spliceai |
|---------|---------------|------------------|
| **Main API** | `meta_spliceai/run_base_model.py` | `splice_engine/meta_models/workflows/splice_prediction_workflow.py` |
| **CLI** | `run_base_model_cli.py` | `splice_engine/cli.py` |
| **Core Prediction** | `splice_engine/run_spliceai_workflow.py` | `splice_engine/base_layer/prediction/core.py` |
| **Configuration** | `splice_engine/meta_models/core/data_types.py` | `splice_engine/base_layer/models/config.py` |

## Primary Entry Points

### 1. Python API (Recommended)

**Module**: `agentic_spliceai.splice_engine.meta_models.workflows.splice_prediction_workflow`

**Function**: `run_base_model_predictions()`

```python
from agentic_spliceai.splice_engine import run_base_model_predictions

# Simple usage
results = run_base_model_predictions(
    base_model='openspliceai',
    target_genes=['BRCA1', 'TP53'],
    save_nucleotide_scores=True,  # Full coverage mode
    verbosity=1
)

# Access results
positions_df = results['positions']           # High-scoring splice sites
nucleotide_scores = results['nucleotide_scores']  # All nucleotide scores
```

**Parameters**:
- `base_model`: `'spliceai'` or `'openspliceai'`
- `target_genes`: List of gene symbols (e.g., `['BRCA1', 'TP53']`)
- `target_chromosomes`: List of chromosomes (e.g., `['21', '22']`)
- `eval_dir`: Output directory for results
- `mode`: `'test'` or `'production'`
- `coverage`: `'gene_subset'`, `'chromosome'`, or `'full_genome'`
- `save_nucleotide_scores`: Enable full coverage mode (scores for every nucleotide)
- `verbosity`: 0 (minimal), 1 (normal), 2 (detailed)

### 2. Command-Line Interface

**Module**: `agentic_spliceai.splice_engine.cli`

```bash
# Predict for specific genes
agentic-spliceai-predict --genes BRCA1 TP53 --base-model openspliceai

# Predict for a chromosome
agentic-spliceai-predict --chromosomes 21 22

# Full coverage mode
agentic-spliceai-predict --genes BRCA1 --base-model openspliceai --save-nucleotide-scores

# Full coverage + test mode (auto-routes outputs to data/test_runs/... if --output-dir is omitted)
agentic-spliceai-predict --mode test --genes BRCA1 TP53 --base-model openspliceai --save-nucleotide-scores

# Test mode with sample data
agentic-spliceai-predict --mode test --coverage sample --genes BRCA1
```

### 3. Low-Level Prediction API

**Module**: `agentic_spliceai.splice_engine.base_layer.prediction.core`

For direct access to prediction functions:

```python
from agentic_spliceai.splice_engine.base_layer.prediction import (
    predict_splice_sites_for_genes,
    load_spliceai_models,
    prepare_input_sequence,
)

# Load models
models = load_spliceai_models(model_type='spliceai', verbosity=1)

# Prepare gene DataFrame with sequences
import polars as pl
gene_df = pl.DataFrame({
    'gene_id': ['gene-BRCA1'],
    'gene_name': ['BRCA1'],
    'seqname': ['chr17'],
    'start': [43044295],
    'end': [43125364],
    'strand': ['-'],
    'sequence': ['ACGT...']  # Actual sequence
})

# Run prediction
predictions = predict_splice_sites_for_genes(
    gene_df=gene_df,
    models=models,
    context=10000,
    output_format='dict',
    verbosity=1
)
```

### 4. Configuration Classes

**Module**: `agentic_spliceai.splice_engine.base_layer`

```python
from agentic_spliceai.splice_engine.base_layer import (
    BaseModelConfig,
    SpliceAIConfig,
    OpenSpliceAIConfig,
    create_config,
)

# Create config for SpliceAI (GRCh37/Ensembl)
spliceai_config = SpliceAIConfig(
    mode='test',
    coverage='gene_subset',
    threshold=0.5,
    save_nucleotide_scores=True
)

# Create config for OpenSpliceAI (GRCh38/MANE)
openspliceai_config = OpenSpliceAIConfig(
    mode='production',
    coverage='full_genome'
)

# Factory function
config = create_config(base_model='openspliceai', mode='test')

# Access auto-resolved paths
print(config.gtf_file)       # GTF annotation file
print(config.genome_fasta)   # Reference genome FASTA
print(config.eval_dir)       # Output directory
print(config.genomic_build)  # 'GRCh37' or 'GRCh38'
```

## Package Structure

```text
splice_engine/
├── __init__.py              # Top-level exports
├── cli.py                   # Command-line interface
├── api.py                   # High-level API
├── base_layer/              # Base model prediction
│   ├── __init__.py          # Exports: BaseModelConfig, etc.
│   ├── models/              # Configuration classes
│   │   └── config.py        # BaseModelConfig, SpliceAIConfig, OpenSpliceAIConfig
│   ├── prediction/          # Core prediction logic
│   │   ├── core.py          # predict_splice_sites_for_genes, load_spliceai_models
│   │   └── evaluation.py    # Evaluation functions
│   └── data/                # Data types and extraction
│       ├── types.py         # GeneManifest, PredictionResult
│       └── genomic_extraction.py  # GTF/FASTA extraction
├── meta_models/             # Prediction workflows
│   └── workflows/
│       ├── splice_prediction_workflow.py  # run_base_model_predictions
│       └── data_preparation.py            # Data preparation utilities
├── config/                  # Configuration management
│   └── genomic_config.py    # Config loading, path resolution
└── resources/               # Resource registry
    └── registry.py          # Genomic resource paths
```

## Validation Methodology

The base layer predictions were validated against meta-spliceai using the following approach:

### Test Setup
1. Run predictions on the same gene sequences using both implementations
2. Compare position sets and probability values

### Validation Criteria
- **Position sets**: Must match exactly (same genomic positions)
- **Probabilities**: Must match within floating-point tolerance (< 1e-6)

### Validated Genes

| Gene | Strand | Length | Positions Match | Max Prob Diff | Status |
|------|--------|--------|-----------------|---------------|--------|
| BRCA1 | - | 81,070 | ✅ | 1.49e-07 | ✅ PASS |
| TP53 | - | 19,070 | ✅ | 1.64e-07 | ✅ PASS |
| EGFR | + | 192,612 | ✅ | 1.49e-07 | ✅ PASS |
| MYC | + | 6,721 | ✅ | 1.19e-07 | ✅ PASS |

### Validation Code

```python
import numpy as np
import polars as pl

# Load predictions from both implementations
agentic_df = pl.read_parquet('agentic_BRCA1_scores.parquet')
meta_positions = np.load('meta_BRCA1_positions.npy')
meta_donor = np.load('meta_BRCA1_donor.npy')

# Create position-to-probability mappings
agentic_donor = {row['genomic_position']: row['donor_score'] 
                 for row in agentic_df.iter_rows(named=True)}
meta_donor_map = {pos: meta_donor[i] for i, pos in enumerate(meta_positions)}

# Compare at common positions
common = set(agentic_donor.keys()) & set(meta_donor_map.keys())
diffs = [abs(agentic_donor[p] - meta_donor_map[p]) for p in common]

assert max(diffs) < 1e-6, f"Max diff: {max(diffs)}"
print("✅ Predictions match within tolerance")
```

## Output Format

### Full Coverage Mode (`save_nucleotide_scores=True`)

Output file: `nucleotide_scores.tsv`

| Column | Description |
|--------|-------------|
| `gene_id` | Gene identifier |
| `gene_name` | Gene symbol |
| `chrom` | Chromosome |
| `strand` | Strand (+/-) |
| `position` | Relative position (1-indexed, 5'→3' in transcription space) |
| `genomic_position` | Absolute genomic coordinate |
| `donor_score` | Donor splice site probability |
| `acceptor_score` | Acceptor splice site probability |
| `neither_score` | Neither probability |

**Position Mapping**:

- **Positive strand**: Position 1 = gene_start (5' end)
- **Negative strand**: Position 1 = gene_end (5' end)

### High-Scoring Positions (Default)

Output file: `full_splice_positions_enhanced.tsv`

Only positions with donor or acceptor score > threshold (default 0.1).

## Differences from meta-spliceai

| Aspect | meta-spliceai | agentic-spliceai |
|--------|---------------|------------------|
| **SpliceAI default build** | GRCh38 | GRCh37 (original training) |
| **OpenSpliceAI build** | GRCh38/MANE | GRCh38/MANE |
| **Config properties** | N/A | `genomic_build`, `annotation_source` |
| **Import style** | Relative (`...`) | Absolute |
| **Dependencies** | Requires meta-spliceai | Standalone |

## See Also

- `splice_engine/README.md` - Package overview
- `docs/STAGE_6_PORTING_PLAN.md` - Porting plan from meta-spliceai
- `base_layer/prediction/core.py` - Core prediction implementation
