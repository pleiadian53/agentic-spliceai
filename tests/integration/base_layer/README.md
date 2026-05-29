# Base Layer Integration Tests

This directory contains integration tests for the base layer (base model predictions).

## Test Scripts

### `test_phase1.py`
**Purpose**: Phase 1 integration test - single gene prediction  
**What it tests**:
- BaseModelRunner integration with prediction pipeline
- Gene loading and filtering
- Sequence extraction
- Model loading
- Single gene prediction (BRCA1)

**Usage**:
```bash
cd ~/work/agentic-spliceai
mamba activate agentic-spliceai
python tests/integration/base_layer/test_phase1.py
```

**Expected output**: ~81,070 positions predicted for BRCA1

---

### `test_chromosome.py`
**Purpose**: Full chromosome prediction test  
**What it tests**:
- BaseModelRunner on full chromosome (chr21)
- Multiple gene processing
- Large-scale prediction performance

**Usage**:
```bash
cd ~/work/agentic-spliceai
mamba activate agentic-spliceai
python tests/integration/base_layer/test_chromosome.py
```

**Expected output**: ~200-300 genes from chromosome 21

---

## Output Locations

### Test Mode Artifacts
When running in `test` mode with a specific `test_name`, outputs go to:

```
data/mane/GRCh38/openspliceai_eval/base_layer/tests/{test_name}/
```

**Example** (for `test_name='phase1_brca1'`):
```
data/mane/GRCh38/openspliceai_eval/base_layer/tests/phase1_brca1/
├── analysis_sequences_chr17.tsv    # 501nt windows with features
├── splice_errors_chr17.tsv         # TP/FP/FN/TN classifications  
├── nucleotide_scores_chr17.tsv     # Per-nucleotide probabilities (if enabled)
└── gene_manifest.tsv               # Gene processing status
```

### Development Mode Artifacts
When running in `development` mode, outputs go to timestamped directories:

```
data/mane/GRCh38/openspliceai_eval/base_layer/dev/{timestamp}/
```

**Example**:
```
data/mane/GRCh38/openspliceai_eval/base_layer/dev/20260128_030124/
└── (same artifacts as test mode)
```

### Production Mode Artifacts
When running in `production` mode, outputs go to:

```
data/mane/GRCh38/openspliceai_eval/base_layer/
├── analysis_sequences_chr*.tsv     # Per chromosome
├── splice_errors_chr*.tsv
├── nucleotide_scores_chr*.tsv      # If enabled
└── gene_manifest.tsv
```

**Note**: Production artifacts are immutable and cannot be overwritten.

---

## Test Results

Tests return `BaseModelResult` with:
- `success`: Boolean indicating success
- `runtime_seconds`: Execution time
- `positions`: Polars DataFrame with predictions
- `processed_genes`: Set of successfully processed gene IDs
- `missing_genes`: Set of genes that weren't found
- `error`: Error message if failed

---

## Adding New Tests

When adding new base layer integration tests:

1. **Create test script** in this directory
2. **Update this README** with test description
3. **Use descriptive test names**: `test_{feature}_{scenario}.py`
4. **Document expected outputs**
5. **Include usage examples**

---

**Last Updated**: January 28, 2026  
**Test Status**: Phase 1 Complete ✅
