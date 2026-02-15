# Configuration System Design

**Purpose**: Type-safe, environment-aware configuration management  
**Component**: `splice_engine/config/`  
**Status**: ✅ Implemented, documented

---

## Overview

The configuration system provides a **single source of truth** for all project settings, combining:
- **YAML files** for human-readable defaults
- **Environment variables** for deployment flexibility  
- **Dataclasses** for type safety
- **Automatic path resolution** for portability

---

## Core Principles

### 1. Configuration, Not Code

Settings should be **configurable**, not hardcoded:

```python
# BAD - Hardcoded
GENOME_BUILD = "GRCh38"
DATA_ROOT = "/Users/me/data"

# GOOD - Configured
from agentic_spliceai.splice_engine.config import config
build = config.build
data_root = config.data_root
```

### 2. Environment Hierarchy

Configuration cascades from multiple sources:

```
1. Defaults (settings.yaml)
   ↓
2. Environment variables (highest priority)
   ↓
3. Runtime overrides (programmatic)
```

### 3. Type Safety

Use Python type hints for safety and IDE support:

```python
@dataclass
class Config:
    species: str
    build: str
    release: str
    data_root: Path  # Type-checked!
```

---

## Implementation

See [Resource Management](resource_management.md) for complete implementation details.

### Quick Reference

**Load configuration**:
```python
from agentic_spliceai.splice_engine.config import config

# Access settings
print(config.build)        # "GRCh38"
print(config.data_root)    # Path("/path/to/data")
```

**Environment overrides**:
```bash
export SS_BUILD=GRCh37
export SS_RELEASE=87
export SS_DATA_ROOT=/mnt/shared/data
```

**Access in code**:
```python
# Environment overrides automatically applied
config.build  # "GRCh37" (from SS_BUILD)
```

---

## Configuration File Structure

### settings.yaml

**Location**: `src/agentic_spliceai/splice_engine/config/settings.yaml`

```yaml
# Global defaults
species: homo_sapiens
default_build: GRCh38
default_release: "112"
data_root: data

# Derived datasets
derived_datasets:
  splice_sites: "splice_sites_enhanced.tsv"
  gene_features: "gene_features.tsv"

# Base models
base_models:
  spliceai:
    training_build: "GRCh37"
    annotation_source: "ensembl"
  openspliceai:
    training_build: "GRCh38"
    annotation_source: "mane"

# Build specifications
builds:
  GRCh38:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
```

---

## Environment Variables

### Naming Convention

Use prefix `SS_` (SpliceAI System):

```bash
SS_BUILD          # Genome build
SS_RELEASE        # Annotation release
SS_DATA_ROOT      # Data root directory
SS_SPECIES        # Species (default: homo_sapiens)
```

### Deployment Examples

**Local development**:
```bash
# .env
SS_BUILD=GRCh38
SS_DATA_ROOT=./data
```

**Remote server**:
```bash
# .env.production
SS_BUILD=GRCh38
SS_DATA_ROOT=/mnt/shared/genomic_data
```

**Docker**:
```dockerfile
ENV SS_BUILD=GRCh38
ENV SS_DATA_ROOT=/data
```

**CI/CD**:
```yaml
# GitHub Actions
env:
  SS_BUILD: GRCh37
  SS_DATA_ROOT: /tmp/test_data
```

---

## Multi-Environment Support

### Local Development

```bash
# Use local data symlink
export SS_DATA_ROOT=./data

# Or point to meta-spliceai
export SS_DATA_ROOT=../meta-spliceai/data
```

### Testing

```python
# Override config for tests
def test_with_custom_config():
    config = load_config()
    config.data_root = Path("tests/fixtures/data")
    # Test with isolated data
```

### Production

```bash
# Production server
export SS_DATA_ROOT=/mnt/shared/genomic_data
export SS_BUILD=GRCh38
```

---

## Best Practices

### DO ✅

**Use environment variables for deployment**:
```bash
# Production
export SS_DATA_ROOT=/mnt/shared/data

# Development
export SS_DATA_ROOT=./data
```

**Access via config object**:
```python
from agentic_spliceai.splice_engine.config import config
data_dir = config.get_data_dir()
```

**Use type hints**:
```python
def process_genome(build: str, data_root: Path):
    ...
```

### DON'T ❌

**Don't hardcode paths**:
```python
# BAD
DATA_DIR = "/Users/me/data"
```

**Don't scatter configuration**:
```python
# BAD - Multiple config files
config1.yaml, config2.yaml, settings.ini, ...
```

**Don't ignore type hints**:
```python
# BAD
data_root = config.data_root  # type: ignore
```

---

## Testing

### Unit Tests

```python
def test_config_loading():
    """Should load config with defaults."""
    cfg = load_config()
    assert cfg.species == "homo_sapiens"
    assert cfg.build in ["GRCh37", "GRCh38"]

def test_env_override():
    """Should override with environment variables."""
    os.environ["SS_BUILD"] = "GRCh37"
    cfg = load_config()
    assert cfg.build == "GRCh37"
```

---

## Integration Points

### Resource Management

Configuration system provides paths for resource management:

```python
# In ArtifactManager
from ..config import config
self.data_root = config.data_root
```

### Base Layer

Base models use config for build-specific settings:

```python
# Get appropriate build for base model
build = config.get_base_model_build("spliceai")  # "GRCh37"
```

### Meta Layer

Meta layer uses config for training parameters:

```python
# Access configuration
batch_size = config.meta_layer_config.get("batch_size", 32)
```

---

## Summary

### Key Benefits

✅ **Single Source**: All settings in one place  
✅ **Type Safe**: Dataclass with type hints  
✅ **Environment Aware**: Overrides for deployment  
✅ **Portable**: Same code, different environments  
✅ **Validated**: Checks settings on load  

### Design Principles Applied

1. ✅ **Configuration Over Code**  
2. ✅ **Explicit Over Implicit**  
3. ✅ **Type Safety**  
4. ✅ **Fail Fast** (validation on load)  

---

**Related**: [Resource Management](resource_management.md), [Output Management](output_management.md)  
**Implementation**: [src/agentic_spliceai/splice_engine/config/](../../src/agentic_spliceai/splice_engine/config/)  
**Last Updated**: February 15, 2026
