# Path Resolution for Examples

**Date**: 2026-01-30  
**Status**: Active

## Problem

Previously, examples used fragile `parent.parent.parent` patterns:

```python
# ❌ BAD: Fragile, breaks if script moves
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
```

**Problems**:
- Breaks if script depth changes
- Hard to maintain
- Doesn't handle experimental packages

---

## Solution: Use `_example_utils`

**New pattern** (recommended):

```python
#!/usr/bin/env python
"""Example script."""

import sys
from pathlib import Path

# Setup imports - no more parent.parent.parent!
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

# Now import normally
from agentic_spliceai.splice_engine.base_layer import ...
```

**Benefits**:
- ✅ Marker-based root finding (`.git`, `pyproject.toml`)
- ✅ Works at any script depth
- ✅ Handles experimental packages (`foundation_models/`)
- ✅ Easy to maintain

---

## How It Works

**`_example_utils.py`**:
- Searches for `.git`, `pyproject.toml`, `setup.py` to find project root
- Adds `src/` to `sys.path`
- Optionally adds experimental packages

```python
def get_project_root() -> Path:
    """Find project root using marker files."""
    markers = ['.git', 'pyproject.toml', 'setup.py']
    # Search upward from current location...
    
def setup_example_environment():
    """Add src/ and experimental packages to sys.path."""
    project_root = get_project_root()
    sys.path.insert(0, str(project_root / 'src'))
    # Also add foundation_models/ if it exists
```

---

## Migration Guide

### Update Existing Examples

**Before**:
```python
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))
```

**After**:
```python
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()
```

### For New Examples

Use this template:

```python
#!/usr/bin/env python
"""Example: <description>."""

import sys
from pathlib import Path

# Setup imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

# Your imports
from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
import polars as pl

def main():
    """Run example."""
    # Your code here
    pass

if __name__ == "__main__":
    main()
```

---

## Future: System-Wide Path Resolution

### Phase 2: Environment Variables (Phase 3)

**Goal**: Support environment-based configuration

```bash
# Set in ~/.zshrc
export AGENTIC_SPLICEAI_ROOT="/Users/pleiadian53/work/agentic-spliceai"

# For pod deployments
export AGENTIC_SPLICEAI_ROOT="/workspace/agentic-spliceai"
```

**Usage**:
```python
from agentic_spliceai.splice_engine.config import get_project_root

# Will use AGENTIC_SPLICEAI_ROOT if set, otherwise markers
project_root = get_project_root()
```

### Phase 3: Artifact Manager (Phase 4)

**Goal**: Unified output management like `meta-spliceai`

```python
from agentic_spliceai.splice_engine.outputs import ArtifactManager

manager = ArtifactManager(
    model='openspliceai',
    mode='test',
    test_name='brca1_validation'
)

# Auto-generates paths following conventions
manager.save_positions(positions_df)
manager.save_metrics(metrics_dict)
```

**Output Structure**:
```
data/<annotation>/<build>/base_model_eval/
├── openspliceai/
│   ├── production/           # Immutable
│   │   └── full_genome_20260130/
│   └── test/                 # Overwritable
│       └── brca1_validation/
└── spliceai/
```

---

## See Also

- `dev/planning/infrastructure/PATH_RESOLUTION_ROADMAP.md` - Full roadmap
- `src/agentic_spliceai/splice_engine/config/` - Config module with `get_project_root()`
- `src/agentic_spliceai/splice_engine/resources/` - Resource registry

---

*This temporary solution (using `_example_utils`) will be replaced by system-wide path resolution in Phase 3/4.*
