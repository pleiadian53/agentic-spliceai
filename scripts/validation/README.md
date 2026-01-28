# Validation Scripts

Cross-project validation scripts for verifying that agentic-spliceai produces 
identical results to meta-spliceai after porting.

## Scripts

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `test_base_layer_comparison.py` | Compare base layer predictions between both systems | After changes to base layer code |
| `compare_evaluation.py` | Compare evaluation metrics between both systems | After changes to evaluation code |

## Running Validation

### Base Layer Comparison (Both Models)

Tests that both systems produce identical splice site predictions on randomly 
selected genes for **both SpliceAI and OpenSpliceAI**:

```bash
cd /Users/pleiadian53/work/agentic-spliceai
mamba activate metaspliceai
python scripts/validation/test_base_layer_comparison.py
```

**Models Tested:**
- **SpliceAI** (GRCh37/Ensembl) - Original SpliceAI model
- **OpenSpliceAI** (GRCh38/MANE) - Open-source retrained model

**Expected Output:**
- Position sets should match exactly
- Probability scores should match within floating-point tolerance (< 1e-5)
- All genes should show "✅ PASS" for both models

### Evaluation Metrics Comparison

Compares the evaluation pipeline outputs:

```bash
python scripts/validation/compare_evaluation.py
```

## Validation Criteria

| Metric | Tolerance | Description |
|--------|-----------|-------------|
| Position sets | Exact match | Same genomic positions identified |
| Donor scores | < 1e-5 | Donor splice site probabilities |
| Acceptor scores | < 1e-5 | Acceptor splice site probabilities |
| TP/FP/FN/TN | Exact match | Classification metrics |
| Precision/Recall/F1 | < 0.001 | Performance metrics |

## Artifact Protection

The validation scripts protect production artifacts:

| Protected Directory | Description |
|---------------------|-------------|
| `data/mane/GRCh38/openspliceai_eval/meta_models` | OpenSpliceAI production outputs |
| `data/mane/GRCh38/openspliceai_eval/predictions` | OpenSpliceAI predictions |

Tests use isolated directories with unique timestamps (e.g., `tests/validation_test_20251213_010000/`).

## Validation Documentation

Comprehensive validation reports are stored in `docs/`:

| Document | Description |
|----------|-------------|
| [`docs/BASE_LAYER_VALIDATION_REPORT.md`](docs/BASE_LAYER_VALIDATION_REPORT.md) | Full validation report with test results, bugs fixed, and new modules created |

## Previously Validated Genes

### December 13, 2025 - Full Validation (Seed=42)

Both SpliceAI (GRCh37) and OpenSpliceAI (GRCh38) tested with **345,397 total positions**:

| Gene | Strand | Positions | Max Diff | Status |
|------|--------|-----------|----------|--------|
| HTT | + | 169,277 | 0.00e+00 | ✅ PASS |
| MYC | + | 6,721 | 0.00e+00 | ✅ PASS |
| BRCA1 | - | 81,070 | 0.00e+00 | ✅ PASS |
| APOB | - | 42,645 | 0.00e+00 | ✅ PASS |
| KRAS | - | 45,684 | 0.00e+00 | ✅ PASS |

**Result:** All predictions match EXACTLY between meta-spliceai and agentic-spliceai.

### Earlier Validation

| Gene | Strand | Length | Max Prob Diff | Status |
|------|--------|--------|---------------|--------|
| BRCA1 | - | 81,070 | 1.49e-07 | ✅ PASS |
| TP53 | - | 19,070 | 1.64e-07 | ✅ PASS |
| EGFR | + | 192,612 | 1.49e-07 | ✅ PASS |
| MYC | + | 6,721 | 1.19e-07 | ✅ PASS |

## Adding New Validation Scripts

When adding new validation scripts:

1. Follow the naming convention: `test_<component>_comparison.py` or `compare_<aspect>.py`
2. Include clear pass/fail criteria
3. Use the `FLOAT_TOLERANCE` constant for numerical comparisons
4. Print a clear summary at the end with ✅/❌ status
5. Update this README with the new script
6. **Always protect production artifacts** - use unique test names

## See Also

- [`docs/BASE_LAYER_VALIDATION_REPORT.md`](docs/BASE_LAYER_VALIDATION_REPORT.md) - Full validation report
- `src/agentic_spliceai/splice_engine/docs/BASE_LAYER_ENTRY_POINTS.md` - Entry point documentation
- `src/agentic_spliceai/splice_engine/base_layer/data/position_types.py` - Position coordinate utilities
- `dev/meta_spliceai/porting_and_refactoring/` - Porting notes and decisions
