# Tests Directory

Organized test structure for agentic-spliceai.

## Directory Structure

```
tests/
├── README.md                    # This file
│
├── unit/                        # Unit tests (fast, isolated)
│   ├── base_layer/             # Base layer unit tests
│   ├── meta_layer/             # Meta layer unit tests
│   ├── resources/              # Resource management tests
│   └── utils/                  # Utility function tests
│
├── integration/                 # Integration tests (slower, end-to-end)
│   ├── base_layer/             # Base model prediction pipeline
│   │   ├── README.md           # Base layer test documentation
│   │   ├── test_phase1.py      # Single gene test
│   │   └── test_chromosome.py  # Full chromosome test
│   ├── meta_layer/             # Meta-learning pipeline
│   └── full_pipeline/          # Complete end-to-end tests
│
└── fixtures/                    # Test data and fixtures
    ├── data/                   # Small test datasets
    ├── configs/                # Test configurations
    └── expected_outputs/       # Expected outputs for validation
```

## Test Categories

### Unit Tests (`unit/`)
- **Purpose**: Fast, isolated tests of individual functions/classes
- **Scope**: Single function or class
- **Duration**: < 1 second per test
- **Dependencies**: Minimal, use mocks/fixtures
- **Example**: Test `one_hot_encode()` function

### Integration Tests (`integration/`)
- **Purpose**: Test component interactions and workflows
- **Scope**: Multiple components working together
- **Duration**: Seconds to minutes
- **Dependencies**: Real data, real models
- **Example**: Test full base model prediction pipeline

### End-to-End Tests (`integration/full_pipeline/`)
- **Purpose**: Test complete workflows from start to finish
- **Scope**: Entire pipeline (base → meta → output)
- **Duration**: Minutes to hours
- **Dependencies**: Full data, all models
- **Example**: Full chromosome prediction with meta-layer refinement

---

## Running Tests

### Quick Start

```bash
# Activate environment
cd ~/work/agentic-spliceai
mamba activate agentic-spliceai

# Run specific test
python tests/integration/base_layer/test_phase1.py

# Run all unit tests (when pytest is set up)
pytest tests/unit/

# Run specific integration test
pytest tests/integration/base_layer/test_chromosome.py
```

### With pytest (Future)

```bash
# All tests
pytest

# Specific category
pytest tests/unit/
pytest tests/integration/

# Specific module
pytest tests/unit/base_layer/

# With coverage
pytest --cov=agentic_spliceai tests/
```

---

## Test Naming Conventions

### Files
- `test_{feature}_{scenario}.py` - Integration tests
- `test_{module_name}.py` - Unit tests for specific module

### Functions
- `test_{function_name}_{condition}()` - Unit test
- `test_{workflow}_{scenario}()` - Integration test

### Examples
- `test_predict_single_gene()` - Basic single gene test
- `test_predict_chromosome_21()` - Chr21 test
- `test_one_hot_encode_valid_sequence()` - Unit test
- `test_load_models_gpu_available()` - Conditional test

---

## Test Data Locations

### Fixtures (Small Test Data)
```
tests/fixtures/data/
├── test_genes.gtf              # Small GTF with 5-10 genes
├── test_sequence.fa            # Small FASTA segment
└── test_variants.vcf           # Small variant set
```

### Real Data (via symlink)
```
data/ -> ../meta-spliceai/data/
├── mane/GRCh38/               # Full genomic resources
└── models/openspliceai/       # Model checkpoints
```

---

## Output Locations

### Test Artifacts
Test mode outputs go to isolated directories:
```
data/mane/GRCh38/openspliceai_eval/base_layer/tests/{test_name}/
```

This prevents:
- Overwriting production data
- Test interference
- Accidental commits of test outputs

---

## Best Practices

### DO ✅
- **Organize by topic**: `tests/{category}/{topic}/`
- **Use descriptive names**: `test_predict_brca1_with_variants.py`
- **Document expected outputs**: In test docstrings
- **Clean up after tests**: Remove temp files
- **Use fixtures**: Share test data across tests
- **Test both success and failure**: Edge cases matter

### DON'T ❌
- **Don't put tests in project root**: Use organized structure
- **Don't commit large test data**: Use fixtures or symlinks
- **Don't hardcode paths**: Use Path objects and relative paths
- **Don't skip error cases**: Test failures too
- **Don't test in production dirs**: Use test mode

---

## Adding New Tests

1. **Choose category**: unit vs integration
2. **Choose topic**: base_layer, meta_layer, etc.
3. **Create test file**: Follow naming convention
4. **Document in README**: Update relevant README.md
5. **Run locally first**: Verify it works
6. **Add to CI** (future): Include in automation

---

## Current Status

### Implemented ✅
- `integration/base_layer/test_phase1.py` - Single gene test
- `integration/base_layer/test_chromosome.py` - Chromosome test
- `integration/base_layer/README.md` - Documentation

### Planned 📋
- Unit tests for prediction functions
- Unit tests for data extraction
- Integration tests for meta layer
- End-to-end pipeline tests
- pytest configuration
- CI/CD integration

---

**Last Updated**: January 28, 2026  
**Test Coverage**: Phase 1 integration tests complete
