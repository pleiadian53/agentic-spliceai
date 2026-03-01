# Position Coordinate Systems

**Module:** `base_layer/data/position_types.py`  
**Ported from:** `meta_spliceai/splice_engine/meta_models/core/position_types.py`  
**Created:** December 2025  
**Purpose:** Explicit handling of position column semantics to prevent coordinate misinterpretation bugs

---

## The Problem

The `position` column appears in multiple contexts with **different semantics**:

| Context | Position Meaning | Example |
|---------|-----------------|---------|
| GTF/GFF annotations | Absolute genomic coordinates | chr17:41,196,312 |
| `splice_sites_enhanced.tsv` | Absolute genomic coordinates | 41196312 |
| `nucleotide_scores.tsv` | Strand-dependent relative position | 1 (5' end) |
| Meta-model training artifacts | Strand-dependent relative position | 1, 2, 3, ... |

This module provides explicit types and conversion functions to prevent mixing these coordinate systems.

---

## Coordinate System Definitions

### ABSOLUTE Coordinates

Genomic coordinates from the reference assembly.

```
Characteristics:
- Independent of strand orientation
- Always increasing from lower to higher values on the reference
- Used in annotation files (GTF, GFF, BED)
- Example: BRCA1 on chr17 spans 41,196,312 - 41,277,500 (GRCh37)
```

### RELATIVE Coordinates

Strand-dependent positions within a gene, 1-indexed from the 5' end.

```
Characteristics:
- Position 1 = transcription start (5' end)
- Increases in transcription direction (5' → 3')
- Strand-dependent mapping to absolute coordinates
- Used in prediction outputs and training artifacts
```

### Strand-Dependent Mapping

```
POSITIVE STRAND (+):
  Transcription: 5' -----> 3'
  Genomic:       gene_start -----> gene_end
  
  Position 1 = gene_start (lowest coordinate)
  Position N = gene_end (highest coordinate)
  
  Formula: absolute = gene_start + relative - 1
           relative = absolute - gene_start + 1

NEGATIVE STRAND (-):
  Transcription: 5' -----> 3'
  Genomic:       gene_end <----- gene_start
  
  Position 1 = gene_end (highest coordinate)
  Position N = gene_start (lowest coordinate)
  
  Formula: absolute = gene_end - relative + 1
           relative = gene_end - absolute + 1
```

---

## API Reference

### PositionType Enum

```python
from agentic_spliceai.splice_engine.base_layer.data import PositionType

PositionType.ABSOLUTE  # Genomic coordinates
PositionType.RELATIVE  # Strand-dependent gene positions
```

### Core Conversion Functions

#### `absolute_to_relative()`

Convert absolute genomic coordinate(s) to relative position(s).

```python
from agentic_spliceai.splice_engine.base_layer.data import absolute_to_relative

# Single position
rel = absolute_to_relative(
    41277500,           # absolute position
    gene_start=41196312,
    gene_end=41277500,
    strand='-'
)
# Returns: 1 (5' end for negative strand)

# Batch conversion
positions = [41277500, 41277499, 41277498]
rel_batch = absolute_to_relative(positions, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: [1, 2, 3]
```

#### `relative_to_absolute()`

Convert relative position(s) to absolute genomic coordinate(s).

```python
from agentic_spliceai.splice_engine.base_layer.data import relative_to_absolute

# Single position
abs_pos = relative_to_absolute(
    1,                  # relative position
    gene_start=41196312,
    gene_end=41277500,
    strand='-'
)
# Returns: 41277500

# Batch conversion
positions = [1, 2, 3]
abs_batch = relative_to_absolute(positions, gene_start=41196312, gene_end=41277500, strand='-')
# Returns: [41277500, 41277499, 41277498]
```

### Helper Classes and Functions

#### `GeneCoordinates` Dataclass

```python
from agentic_spliceai.splice_engine.base_layer.data import GeneCoordinates

coords = GeneCoordinates(
    gene_start=41196312,
    gene_end=41277500,
    strand='-',
    gene_id='ENSG00000012048'  # optional
)

# Properties
coords.length  # 81189 (gene length in nucleotides)
```

#### `validate_position_range()`

Validate that a position is within expected range.

```python
from agentic_spliceai.splice_engine.base_layer.data import (
    validate_position_range, PositionType
)

# Returns True/False
is_valid = validate_position_range(
    position=50000,
    position_type=PositionType.RELATIVE,
    gene_start=41196312,
    gene_end=41277500
)

# With strict=True, raises ValueError for invalid positions
validate_position_range(position=100000, ..., strict=True)
# Raises: ValueError("Relative position 100000 outside range [1, 81189]")
```

#### `infer_position_type()`

Heuristic helper to detect coordinate type (for debugging/migration).

```python
from agentic_spliceai.splice_engine.base_layer.data import infer_position_type

# Detect from position values
pos_type, confidence = infer_position_type(
    positions=[41196312, 41196313, 41196314],
    gene_start=41196312,
    gene_end=41277500
)
# Returns: (PositionType.ABSOLUTE, 1.0)

pos_type, confidence = infer_position_type(
    positions=[1, 2, 3, 4, 5],
    gene_start=41196312,
    gene_end=41277500
)
# Returns: (PositionType.RELATIVE, 1.0)
```

---

## Usage in agentic-spliceai

### splice_prediction_workflow.py

```python
from agentic_spliceai.splice_engine.base_layer.data import absolute_to_relative

# In nucleotide scores generation
for pos in sorted(merged_results.keys()):  # pos is ABSOLUTE
    # Convert ABSOLUTE → RELATIVE for position column
    rel_position = absolute_to_relative(
        pos, 
        gene_start=gene_start, 
        gene_end=gene_end, 
        strand=strand
    )
    
    nucleotide_scores.append({
        'position': rel_position,   # RELATIVE (1-indexed, 5' to 3')
        'genomic_position': pos,    # ABSOLUTE genomic coordinate
        ...
    })
```

---

## Real-World Example: BRCA1

BRCA1 is on the **negative strand** of chromosome 17.

```
Gene coordinates (GRCh37):
  gene_start = 41,196,312 (lower coordinate, but 3' end!)
  gene_end = 41,277,500 (higher coordinate, and 5' end!)
  strand = '-'
  length = 81,189 bp

Position mapping:
  Relative 1     → Absolute 41,277,500 (5' end, transcription start)
  Relative 2     → Absolute 41,277,499
  Relative 3     → Absolute 41,277,498
  ...
  Relative 81189 → Absolute 41,196,312 (3' end)
```

---

## Validation

This module was validated as part of the base layer porting:

- **Test:** `scripts/validation/test_base_layer_comparison.py`
- **Result:** Both meta-spliceai and agentic-spliceai produce identical outputs
- **Documentation:** `scripts/validation/docs/BASE_LAYER_VALIDATION_REPORT.md`

---

## See Also

- `position_types.py` - Source code with full docstrings
- `splice_prediction_workflow.py` - Primary user of these utilities
- `scripts/validation/docs/BASE_LAYER_VALIDATION_REPORT.md` - Validation report
- `docs/base_layer/PROCESSING_ARCHITECTURE.md` - Base layer architecture overview

