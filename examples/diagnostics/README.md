# Diagnostic Tools

**Purpose**: Tools for troubleshooting, validation, and quality assurance

---

## Overview

This directory contains diagnostic scripts that help users:
- 🔍 **Identify issues** with model performance
- ✅ **Validate** model/data consistency
- 🐛 **Debug** coordinate mismatches
- 📊 **Analyze** prediction quality

These are **user-facing tools** meant to be run interactively, not automated tests.

---

## Available Diagnostics

### 1. `check_coordinate_consistency.py` ⭐⚠️

**Purpose**: Detect position offset issues between model predictions and annotations

**⚠️ KNOWN LIMITATION**: This tool tests POSITION offsets (shifting detected peaks), not SCORE ARRAY adjustments (rolling arrays before peak detection). For SpliceAI specifically, this may not detect known offsets that require array rolling. See warning in tool output and `dev/diagnostics/DIAGNOSTIC_TOOL_BUG_REPORT.md` for details.

**When to Use**:
- ✅ Adding a new base model
- ✅ Model shows high windowed recall but low exact recall
- ✅ Switching genome builds
- ✅ Poor performance on a specific gene

**Quick Start**:
```bash
# Check SpliceAI on TP53
python check_coordinate_consistency.py --model spliceai --genes TP53

# Check OpenSpliceAI with random sample
python check_coordinate_consistency.py --model openspliceai --sample 20

# Save adjustment file for future use
python check_coordinate_consistency.py --model spliceai --genes TP53 BRCA1 --save-adjustments
```

**What It Does**:
1. Runs predictions on sample genes
2. Compares to ground truth annotations
3. Tests position offsets from -5bp to +5bp
4. Finds optimal offset per site type and strand
5. Reports metrics before/after adjustment
6. Optionally saves adjustment file

**Example Output**:
```
📐 Detected Position Offsets:
   Donor sites:
      + strand: +0 bp ✅ (aligned)
      - strand: +0 bp ✅ (aligned)
   Acceptor sites:
      + strand: +0 bp ✅ (aligned)
      - strand: +0 bp ✅ (aligned)

📈 Performance Metrics:
   Before adjustment:
      Exact Recall:    40.0%
      Windowed Recall: 82.3% (±2bp)
   
   After adjustment:
      Exact Recall:    89.5% (+49.5% ✅)

💡 Recommendation: ✅ Coordinates are well-aligned
```

**Interpretation**:

| Windowed Recall | Exact Recall | Diagnosis | Action |
|----------------|--------------|-----------|--------|
| High (>80%) | High (>80%) | ✅ Perfect alignment | None needed |
| High (>80%) | Low (<50%) | ⚠️ Coordinate offset | Apply detected adjustments |
| Low (<50%) | Low (<50%) | ❌ Model issue | Check model/data compatibility |

---

## Common Use Cases

### Use Case 1: New Model Validation

**Scenario**: You trained/downloaded a new SpliceAI variant

**Steps**:
```bash
# 1. Run diagnostic
python check_coordinate_consistency.py \
    --model my_new_model \
    --sample 20 \
    --save-adjustments

# 2. Check output
# - If offsets are 0: ✅ Model is aligned
# - If offsets are ±1-3bp: Apply adjustments
# - If offsets are >3bp: ⚠️ Check model training build
```

---

### Use Case 2: Poor Performance Debugging

**Scenario**: Model shows poor metrics

**Decision Tree**:
```
1. Run: check_coordinate_consistency.py --model X --genes Y

2. Check output:
   
   a) Windowed recall HIGH + Exact recall LOW
      → ✅ Coordinate offset (easy fix)
      → Apply detected adjustments
   
   b) Both recalls LOW
      → ❌ Not a coordinate issue
      → Check:
         - Is genome build correct?
         - Is annotation source correct?
         - Is gene in training set?
```

---

### Use Case 3: Genome Build Verification

**Scenario**: Switched from GRCh38 to GRCh37

**Steps**:
```bash
# Before: Running on GRCh38 (wrong build for SpliceAI)
python check_coordinate_consistency.py --model spliceai --genes TP53
# Expected: Large offsets or low recall

# After: Running on GRCh37 (correct build)
python check_coordinate_consistency.py --model spliceai --genes TP53
# Expected: Zero offsets, high recall
```

---

## Interpreting Results

### Healthy Alignment ✅

```
Detected offsets: All 0bp
Exact recall: 85-95%
Windowed recall: 90-98%
```

**Meaning**: Model and annotations are perfectly aligned

**Action**: None needed, proceed with predictions

---

### Coordinate Offset ⚠️

```
Detected offsets: ±1-3bp (varies by site/strand)
Exact recall: 40-60% (low)
Windowed recall: 80-90% (high)
```

**Meaning**: Model is predicting ~2bp off from annotations

**Action**: 
1. Apply detected adjustments
2. Re-run evaluation
3. Expect recall to improve to 85-95%

---

### Severe Misalignment ❌

```
Detected offsets: >±5bp
Exact recall: <30%
Windowed recall: <50%
```

**Meaning**: Fundamental compatibility issue

**Action**:
1. ⚠️ Verify genome build matches model training
2. ⚠️ Check if using correct annotation source
3. ⚠️ Consult model documentation

---

## ⚠️ Known Limitations

### Coordinate Consistency Tool Methodology

**Current Implementation**: Tests POSITION offsets (shifts detected peak positions after peak detection)

**Limitation**: Does not test SCORE ARRAY adjustments (rolling arrays with `np.roll` before peak detection)

**Impact**: May fail to detect offsets that require score array rolling, specifically:
- SpliceAI's known systematic offsets (+2bp donor/+strand, +1bp donor/-strand, -1bp acceptor/-strand)
- Any model where the score at the true position is below threshold, but rolling would create a peak there

**How to Detect This Issue**:
```
If you see:
  - Exact recall = Windowed recall (SAME value)
  - Low recall (<70%)
  - Tool says "no offset needed"
  - You're using SpliceAI

Then: The tool's result may be incorrect!
```

**Workaround**: Use MetaSpliceAI's full adjustment system (porting in progress, see `dev/base_layer/SPLICE_COORDINATE_ADJUSTMENT_TODO.md`)

**Tool Behavior**: The tool now WARNS when it detects this pattern and explicitly tells you not to trust the "0bp offset" result for SpliceAI.

---

## Future Diagnostics (Planned)

### `check_model_health.py`

Comprehensive model validation:
- ✅ Model files exist and load correctly
- ✅ Genomic resources are available
- ✅ Predictions run without errors
- ✅ Output format is correct

### `compare_annotations.py`

Compare annotations between builds/sources:
- MANE vs Ensembl
- GRCh37 vs GRCh38
- Different releases

### `profile_prediction_speed.py`

Benchmark prediction performance:
- Throughput (genes/minute)
- Memory usage
- Optimal batch size

---

## Development Notes

**Directory Purpose**: User-facing diagnostics, not automated tests

**Related Directories**:
- `tests/` - Automated pytest tests
- `scripts/validation/` - One-off validation scripts
- `examples/base_layer/` - Full prediction examples

**When to Add Here**:
- ✅ Tool helps users troubleshoot issues
- ✅ Interactive, human-readable output
- ✅ Run manually on user's data
- ✅ Diagnostic/investigative purpose

**When to Put Elsewhere**:
- ❌ Automated testing (use `tests/`)
- ❌ Data preprocessing (use `examples/data_preparation/`)
- ❌ Full prediction workflow (use `examples/base_layer/`)

---

## See Also

- `examples/base_layer/` - Full prediction and evaluation examples
- `examples/data_preparation/` - Data pipeline examples
- `tests/base_layer/` - Automated unit tests
- `dev/base_layer/SPLICE_COORDINATE_ADJUSTMENT_TODO.md` - Full implementation plan

---

**Created**: February 5, 2026  
**Status**: Initial diagnostic tool created, more coming soon!
