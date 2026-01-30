# Base Layer Data Module

Data loading and preparation utilities for splice site prediction.

---

## Quick Start

### Load Gene Data (Phase 2 - Complete)

```python
from agentic_spliceai.splice_engine.base_layer.data import prepare_gene_data

# Load specific genes
gene_df = prepare_gene_data(genes=['BRCA1', 'TP53'])

# Load entire chromosome
chr21_df = prepare_gene_data(chromosomes=['21'])
```

**Returns**: Polars DataFrame with columns:
- `seqname` - Chromosome (e.g., 'chr17')
- `gene_id` - Gene ID (e.g., 'ENSG00000012048')
- `gene_name` - Gene symbol (e.g., 'BRCA1')
- `start` - Start position (1-based)
- `end` - End position (1-based)
- `strand` - Strand ('+' or '-')
- `sequence` - DNA sequence (uppercase)

### Extract Splice Sites (Phase 2 - Complete) ⭐ NEW

```python
from agentic_spliceai.splice_engine.base_layer.data import prepare_splice_site_annotations

# Extract splice sites for specific genes
result = prepare_splice_site_annotations(
    output_dir='data/prepared',
    genes=['BRCA1', 'TP53'],
    build='GRCh38'
)

if result['success']:
    splice_df = result['splice_sites_df']
    print(f"Extracted {result['n_sites']} splice sites")
    print(f"  File: {result['splice_sites_file']}")
```

**Returns**: Dictionary with:
- `success` - bool - Whether extraction succeeded
- `splice_sites_df` - Polars DataFrame - Splice site annotations
- `splice_sites_file` - str - Path to TSV file
- `n_sites` - int - Total splice sites
- `n_donors` - int - Number of donor sites
- `n_acceptors` - int - Number of acceptor sites

**DataFrame columns**:
- `chrom` - Chromosome name
- `position` - Exact splice site position (1-based)
- `site_type` - 'donor' or 'acceptor'
- `gene_id` - Gene identifier
- `gene_name` - Gene symbol
- `strand` - '+' or '-'
- `transcript_id` - Transcript identifier
- `exon_number` - Exon number
- (plus additional metadata)

---

## Modules

### `preparation.py` (Phase 2 ✅)

**Main entry point for data preparation**

Functions:
- `prepare_gene_data()` - Load annotations + extract sequences
- `prepare_splice_site_annotations()` - Extract splice sites ⭐ NEW
- `load_gene_annotations()` - Load from GTF
- `extract_sequences()` - Extract from FASTA
- `filter_by_genes()` - Filter to specific genes
- `filter_by_chromosomes()` - Filter to chromosomes
- Helper utilities

### `genomic_extraction.py`

**Low-level GTF parsing** (used by preparation.py)

Functions:
- `extract_gene_annotations()` - Parse gene features
- `extract_transcript_annotations()` - Parse transcripts
- `extract_exon_annotations()` - Parse exons
- `extract_splice_sites()` - Parse splice sites

### `sequence_extraction.py`

**FASTA sequence extraction** (used by preparation.py)

Functions:
- `extract_gene_sequences()` - Extract sequences for genes
- `one_hot_encode()` - Encode sequences for models
- `reverse_complement()` - Reverse complement

### `types.py`

**Data types and containers**

Classes:
- `GeneManifest` - Gene processing manifest
- `PredictionResult` - Prediction results container
- `SplicePosition` - Splice site position

### `position_types.py`

**Position coordinate conversions**

Functions:
- `absolute_to_relative()` - Convert coordinates
- `relative_to_absolute()` - Convert back
- `validate_position_range()` - Validate coordinates

---

## Examples

### Example 1: Load and Explore

```python
from agentic_spliceai.splice_engine.base_layer.data import (
    prepare_gene_data,
    get_genes_by_chromosome
)

# Load data
df = prepare_gene_data(chromosomes=['21'])

# Explore
print(f"Loaded {len(df)} genes")
print(get_genes_by_chromosome(df))
```

### Example 2: Custom Paths

```python
# Use custom annotation files
df = prepare_gene_data(
    genes=['BRCA1'],
    gtf_path='/data/custom/annotations.gtf',
    fasta_path='/data/custom/genome.fa'
)
```

### Example 3: Step-by-Step

```python
from agentic_spliceai.splice_engine.base_layer.data import (
    load_gene_annotations,
    extract_sequences,
    filter_by_genes
)

# Load annotations
genes = load_gene_annotations(
    gtf_path='annotations.gtf',
    chromosomes=['1', '21']
)

# Extract sequences
genes = extract_sequences(genes, 'genome.fa')

# Filter
brca = filter_by_genes(genes, ['BRCA1'])
```

### Example 4: Integration with Prediction

```python
from agentic_spliceai.splice_engine.base_layer.data import prepare_gene_data
from agentic_spliceai.splice_engine.base_layer.prediction import (
    load_spliceai_models,
    predict_splice_sites_for_genes
)

# Prepare data
genes_df = prepare_gene_data(genes=['BRCA1'])

# Load models
models = load_spliceai_models(model_type='openspliceai')

# Predict
predictions = predict_splice_sites_for_genes(
    gene_df=genes_df,
    models=models,
    context=10000
)
```

---

## Testing

Run integration tests:

```bash
python tests/integration/base_layer/test_phase2_data_preparation.py
```

---

## Architecture

### Data Flow

```
GTF File → load_gene_annotations() → Gene DataFrame
                                          ↓
FASTA File → extract_sequences() → Gene DataFrame + Sequences
                                          ↓
                                    Ready for Prediction
```

### Dependencies

**Internal**:
- `resources.registry` - Path resolution
- Shared between all data modules

**External**:
- `polars` - DataFrame operations
- `pyfaidx` - FASTA reading

---

## Notes

### Genome Builds

Supported:
- `GRCh38` - Human genome build 38
- `GRCh37` - Human genome build 37
- `GRCh38_MANE` - MANE annotations on GRCh38

### Chromosome Names

Both formats are supported:
- With prefix: `chr1`, `chr21`, `chrX`
- Without prefix: `1`, `21`, `X`

Filtering automatically handles both formats.

### Performance

- First GTF load: ~2-3s (creates index)
- Subsequent loads: ~1s
- FASTA extraction: ~1s per 100 genes
- Memory: ~100MB for chr21 (214 genes)

---

## See Also

- `docs/base_layer/PHASE2_DATA_PREPARATION_COMPLETE.md` - Phase 2 completion report
- `tests/integration/base_layer/test_phase2_data_preparation.py` - Integration tests
- Phase 1 complete: `BaseModelRunner` working
- Phase 3 planned: Workflow orchestration
- Phase 4 planned: Artifact management
