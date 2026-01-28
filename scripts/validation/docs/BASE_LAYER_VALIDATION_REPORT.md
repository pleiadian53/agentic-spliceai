# Base Layer Validation Report

**Date:** December 13, 2025  
**Purpose:** Validate that agentic-spliceai produces identical base layer predictions as meta-spliceai  
**Status:** ✅ PASSED

---

## Executive Summary

This document records the validation testing performed to ensure that the base layer predictions from **agentic-spliceai** exactly match those from **meta-spliceai**. Both systems were tested with two base models (SpliceAI and OpenSpliceAI) across multiple genes.

### Key Finding

**✅ When both systems compute predictions for the same genes, the results are IDENTICAL (max difference = 0.0).**

---

## Test Configuration

### Systems Under Test

| System | Version | Description |
|--------|---------|-------------|
| **meta-spliceai** | Current | Original implementation (gold standard) |
| **agentic-spliceai** | Current | Ported/refactored implementation |

### Base Models Tested

| Model | Genome Build | Annotation Source | Training Data |
|-------|--------------|-------------------|---------------|
| **SpliceAI** | GRCh37 | Ensembl v87 | Original Illumina training |
| **OpenSpliceAI** | GRCh38 | MANE v1.3 | Retrained on MANE |

### Test Script

- **Location:** `scripts/validation/test_base_layer_comparison.py`
- **Random Seed:** 42
- **Float Tolerance:** 1e-5

---

## Test Results

### Successful Test Run (Seed=42)

**Test Genes:** HTT, MYC, BRCA1, APOB, KRAS

#### SpliceAI (GRCh37) Results

| Gene | Positions | Max Donor Diff | Max Acceptor Diff | Status |
|------|-----------|----------------|-------------------|--------|
| HTT | 169,277 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| MYC | 6,721 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| BRCA1 | 81,070 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| APOB | 42,645 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| KRAS | 45,684 | 0.00e+00 | 0.00e+00 | ✅ PASS |

**Total Positions Compared:** 345,397  
**Max Difference:** 0.00e+00  
**Result:** ✅ ALL PASSED

#### OpenSpliceAI (GRCh38/MANE) Results

| Gene | Positions | Max Donor Diff | Max Acceptor Diff | Status |
|------|-----------|----------------|-------------------|--------|
| HTT | 169,277 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| MYC | 6,721 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| BRCA1 | 81,070 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| APOB | 42,645 | 0.00e+00 | 0.00e+00 | ✅ PASS |
| KRAS | 45,684 | 0.00e+00 | 0.00e+00 | ✅ PASS |

**Total Positions Compared:** 345,397  
**Max Difference:** 0.00e+00  
**Result:** ✅ ALL PASSED

---

## Bugs Fixed During Validation

### Bug 1: Nucleotide Scores Position Doubling (meta-spliceai)

**File:** `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`

**Symptom:** The `genomic_position` column in nucleotide_scores output was approximately 2× the expected value (e.g., BRCA1 showing 82M instead of 41M for GRCh37).

**Root Cause:** The `positions` array from `predict_splice_sites_for_genes` already contains ABSOLUTE genomic coordinates, but the nucleotide_scores generation code treated them as RELATIVE positions and added `gene_start` again:

```python
# BEFORE (BUG): 
positions = pred_data.get('positions', ...)
'genomic_position': [gene_start + p - 1 for p in positions]  # DOUBLED!
```

**Fix:**
```python
# AFTER (FIXED):
absolute_positions = pred_data.get('positions', ...)
relative_positions = absolute_to_relative(absolute_positions, gene_start, gene_end, strand)
'position': relative_positions,  # RELATIVE (1-indexed, 5' to 3')
'genomic_position': absolute_positions  # ABSOLUTE (already correct)
```

### Bug 2: Registry Search Order (meta-spliceai)

**File:** `meta_spliceai/system/genomic_resources/registry.py`

**Symptom:** meta-spliceai was loading an incomplete `gene_features.tsv` from root `data/ensembl/` (900 genes) instead of build-specific `data/ensembl/GRCh37/gene_features.tsv` (63K+ genes).

**Root Cause:** Search order `[self.top, self.stash, self.legacy]` checked root directory first.

**Fix:** Changed search order to prioritize build-specific directories:
```python
# For derived datasets, search stash first
search_order = [self.stash, self.legacy, self.top]
```

### Bug 3: SpliceAIConfig Not Using Correct Paths (meta-spliceai)

**File:** `meta_spliceai/splice_engine/meta_models/core/data_types.py`

**Symptom:** SpliceAI (GRCh37) was getting GRCh38 paths because `__post_init__` only overrode paths for non-spliceai models.

**Fix:** Explicitly set correct Registry for ALL base models:
```python
if base_model_lower == 'openspliceai':
    registry = Registry(build='GRCh38_MANE', release='1.3')
elif base_model_lower == 'spliceai':
    registry = Registry(build='GRCh37', release='87')
```

---

## New Module: Position Types

A new module was created to explicitly handle coordinate system semantics and prevent future bugs.

### Files Created

| Project | Path |
|---------|------|
| meta-spliceai | `meta_models/core/position_types.py` |
| agentic-spliceai | `base_layer/data/position_types.py` |

### Coordinate Systems

| Type | Description | Example Use |
|------|-------------|-------------|
| **ABSOLUTE** | Genomic coordinates from reference | GTF files, splice_sites_enhanced.tsv |
| **RELATIVE** | Strand-dependent gene positions (1-indexed) | nucleotide_scores.tsv, meta-model training data |

### Key Functions

```python
from position_types import PositionType, absolute_to_relative, relative_to_absolute

# Convert absolute → relative
rel = absolute_to_relative(41277500, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: 1 (5' end for negative strand)

# Convert relative → absolute
abs_pos = relative_to_absolute(1, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: 41277500
```

### Strand-Dependent Position Mapping

```
Positive strand (+):
  Position 1 = gene_start (lowest coordinate, 5' end)
  Position N = gene_end (highest coordinate, 3' end)

Negative strand (-):
  Position 1 = gene_end (highest coordinate, 5' end)
  Position N = gene_start (lowest coordinate, 3' end)
```

---

## How to Run Validation Tests

```bash
# Navigate to agentic-spliceai
cd /Users/pleiadian53/work/agentic-spliceai

# Activate environment
mamba activate metaspliceai

# Run validation
python scripts/validation/test_base_layer_comparison.py

# To test different genes, modify RANDOM_SEED in the script
# RANDOM_SEED = 42  # Default: HTT, MYC, BRCA1, APOB, KRAS
```

---

## Output Files

The test script generates isolated test directories to avoid overwriting production artifacts:

```
data/<annotation>/<build>/<model>_eval/tests/validation_test_<timestamp>_<model>/
  └── meta_models/
      └── predictions/
          ├── nucleotide_scores.tsv    # Full nucleotide-level scores
          └── gene_manifest.tsv        # Processing status per gene
```

### Protected Production Directories

The following directories are NEVER overwritten by tests:
- `data/mane/GRCh38/openspliceai_eval/meta_models/`
- `data/ensembl/GRCh37/spliceai_eval/meta_models/`

---

## Troubleshooting

### Issue: "Position mismatch" or "No nucleotide scores"

**Cause:** meta-spliceai may be loading cached checkpoints without nucleotide scores.

**Solution:**
1. Clear checkpoints for the genes being tested (script does this automatically)
2. Ensure `save_nucleotide_scores=True` is passed
3. Use `force_overwrite=True` if needed

### Issue: Different genes selected with same seed

**Cause:** Gene pool or seed changed.

**Solution:** Verify RANDOM_SEED and GENE_POOL in the script match expected values.

---

## Conclusion

The porting of base layer functionality from meta-spliceai to agentic-spliceai is **VALIDATED**. Both systems produce identical predictions when:

1. Using the same base model (SpliceAI or OpenSpliceAI)
2. Using the correct genome build paths
3. Generating fresh predictions (not cached checkpoints)

The new `position_types` module provides explicit handling of coordinate systems to prevent the position-doubling bug from recurring.

---

## References

- `meta-spliceai/docs/base_models/NUCLEOTIDE_SCORES_DESIGN_RATIONALE.md`
- `meta-spliceai/docs/base_models/GENOME_BUILD_COMPATIBILITY.md`
- `meta-spliceai/docs/base_models/COMPARE_BASE_MODELS_ROBUST_USAGE.md`
- `agentic-spliceai/src/agentic_spliceai/splice_engine/docs/BASE_LAYER_ENTRY_POINTS.md`

