# Examples: Driver Scripts

**Purpose**: Quick-start driver scripts for common workflows and demonstrations

**Organization**: Topic-specific subdirectories matching the multi-layer architecture

## 📖 Important Documentation

- **[README_PATH_RESOLUTION.md](README_PATH_RESOLUTION.md)** - How to properly import from `src/`
  - No more fragile `parent.parent.parent` patterns!
  - Uses marker-based root finding via `_example_utils.py`
  - Roadmap for system-wide path resolution (Phase 3/4)

---

## 📁 Directory Structure

```
examples/
├── README.md                          ← This file
│
├── base_layer/                        ← Base model prediction examples
│   ├── 01_phase1_prediction.py       ← Phase 1: Single gene prediction
│   ├── 02_chromosome_prediction.py   ← Phase 1: Chromosome-wide prediction
│   ├── 03_prediction_with_evaluation.py  ← Phase 1: Prediction + Evaluation
│   └── README.md
│
├── data_preparation/                  ← Data prep workflows
│   ├── 01_prepare_gene_data.py       ← Phase 2: Gene data extraction
│   ├── 02_prepare_splice_sites.py    ← Phase 2: Splice site annotation
│   ├── 03_full_data_pipeline.py      ← Phase 2: Complete pipeline
│   ├── validate_mane_metadata.py     ← Validation: MANE vs Ensembl
│   └── README.md
│
├── diagnostics/                       ← Diagnostic & troubleshooting tools
│   ├── check_coordinate_consistency.py  ← Detect position offset issues
│   └── README.md
│
├── meta_layer/                        ← Meta-layer examples (Phase 5)
│   └── README.md
│
└── variant_analysis/                  ← Variant analysis (Phase 6)
    └── README.md
```

---

## 🚀 Quick Start Examples

### Base Layer Prediction

**Phase 1: Single Gene Prediction**
```bash
cd examples/base_layer
python 01_phase1_prediction.py --gene BRCA1
```

**Phase 1: Chromosome Prediction**
```bash
cd examples/base_layer
python 02_chromosome_prediction.py --chromosome chr21 --genes 10
```

---

### Data Preparation

**Phase 2: Gene Data Extraction**
```bash
cd examples/data_preparation
python 01_prepare_gene_data.py --genes BRCA1 TP53 --output /tmp/gene_data/
```

**Phase 2: Splice Site Annotation**
```bash
cd examples/data_preparation
python 02_prepare_splice_sites.py --genes BRCA1 --annotation-source mane
```

**Phase 2: Complete Pipeline**
```bash
cd examples/data_preparation
python 03_full_data_pipeline.py --genes BRCA1 TP53 EGFR --output /tmp/full_pipeline/
```

**Validation: MANE Metadata**
```bash
cd examples/data_preparation
python validate_mane_metadata.py
```

---

### Diagnostics

**Check Coordinate Consistency**
```bash
cd examples/diagnostics

# Check specific genes
python check_coordinate_consistency.py --model spliceai --genes TP53 BRCA1

# Quick check with sample
python check_coordinate_consistency.py --model openspliceai --sample 20
```

---

## 📊 What vs Where

### "I want to predict splice sites for a gene"
→ `base_layer/01_phase1_prediction.py`

### "I want to prepare data for meta-layer training"
→ `data_preparation/03_full_data_pipeline.py`

### "I want to validate MANE metadata inference"
→ `data_preparation/validate_mane_metadata.py`

### "I want to test the complete Phase 1 workflow"
→ `base_layer/01_phase1_prediction.py` (single gene)  
→ `base_layer/02_chromosome_prediction.py` (multiple genes)

### "I want to test the complete Phase 2 workflow"
→ `data_preparation/03_full_data_pipeline.py`

### "My model shows poor performance - is it a coordinate issue?"
→ `diagnostics/check_coordinate_consistency.py`

### "I'm adding a new base model - how do I validate it?"
→ `diagnostics/check_coordinate_consistency.py`

---

## 🎯 Organization Philosophy

### Examples (This Directory)

**Purpose**: Driver scripts for quick iteration and testing

**Characteristics**:
- Standalone Python scripts
- Command-line arguments
- Fast execution
- Print results to console
- Useful for development and testing

**Target Audience**: Developers, contributors, researchers

---

### Notebooks (`notebooks/`)

**Purpose**: Illustrative, educational, step-by-step explanations

**Characteristics**:
- Jupyter notebooks (.ipynb)
- Detailed explanations
- Visualizations
- Step-by-step walkthrough
- Useful for learning and documentation

**Target Audience**: New users, students, documentation

---

### Scripts (`scripts/`)

**Purpose**: Production utilities and batch processing

**Characteristics**:
- Complex pipelines
- Batch processing
- Production-ready
- Error handling
- Logging

**Target Audience**: Production users, automation

---

## 📝 Adding New Examples

### Guidelines

1. **Name**: Use numbered prefixes for ordering (e.g., `01_`, `02_`)
2. **Topic**: Place in appropriate subdirectory
3. **Documentation**: Add docstring and usage examples
4. **README**: Update topic README with new example
5. **Test**: Ensure example runs successfully

### Template

```python
#!/usr/bin/env python
"""Brief description of what this example does.

Usage:
    python example_script.py --arg1 value1 --arg2 value2

Example:
    python example_script.py --gene BRCA1 --output /tmp/results/
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Example script")
    parser.add_argument("--gene", required=True, help="Gene symbol")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    # Your code here
    print(f"Processing {args.gene}...")

if __name__ == "__main__":
    main()
```

---

## 🔗 Related Directories

- **`notebooks/`**: Jupyter notebooks with detailed explanations
- **`scripts/`**: Production utilities and batch processing
- **`tests/`**: Unit and integration tests
- **`docs/`**: User-facing documentation

---

**Last Updated**: January 30, 2026  
**Status**: Phase 1-2 examples available, Phase 3+ coming soon
