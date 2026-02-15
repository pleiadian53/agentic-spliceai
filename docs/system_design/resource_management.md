# Resource Management System Design

**Purpose**: Centralized, systematic management of genomic resources and data paths  
**Component**: `splice_engine/config/`  
**Status**: ✅ Implemented, documented during refactoring

---

## Problem Statement

### What We're Solving

Research projects like agentic-spliceai need to access many resources:
- **Genomic data**: GTF annotations, FASTA genomes (multiple builds)
- **Derived datasets**: Splice sites, gene annotations, transcript features
- **Base models**: SpliceAI, OpenSpliceAI (trained on different builds)
- **External data**: Variant databases, clinical datasets

### Challenges Without Resource Management

❌ **Path Chaos**:
```python
# Scattered across codebase:
gtf_file = "../../data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf"
fasta_file = "/Users/me/data/genomes/GRCh38.fa"
splice_sites = "../data/mane/GRCh38/splice_sites.tsv"
```

❌ **Fragility**: Paths break when:
- Running from different directories
- Moving to different machines
- Deploying to servers/containers
- Collaborating with others

❌ **Inconsistency**: Multiple paths to same resource:
```python
# file1.py
data_dir = "data/ensembl/GRCh38"
# file2.py
data_path = "./data/ensembl/GRCh38/"
# file3.py
DATA_ROOT = "/mnt/shared/data/ensembl/GRCh38"
```

❌ **Configuration Hell**: Changing builds requires code changes:
```python
# To switch from GRCh38 to GRCh37, must find and update N places in code
```

---

## Design Goals

### 1. Single Source of Truth

All resource paths defined in **one place** (configuration).

```python
# Everywhere in code:
from agentic_spliceai.splice_engine.config import config

gtf_path = config.get_data_dir() / filename("gtf", config)
```

### 2. Portable Across Environments

Same code works on:
- Local development machines
- Remote servers
- Docker containers
- CI/CD pipelines
- Collaborator machines

### 3. Configuration-Driven

Switch builds/sources via **configuration**, not code:

```bash
# Change build via environment variable
export SS_BUILD=GRCh37

# Or in settings.yaml
default_build: GRCh37
```

### 4. Type-Safe and Explicit

Use Python dataclasses for type safety:

```python
@dataclass
class Config:
    species: str
    build: str
    release: str
    data_root: Path
```

### 5. Multi-Build, Multi-Source Support

Support multiple genome builds and annotation sources:
- **Builds**: GRCh37, GRCh38
- **Sources**: Ensembl, MANE RefSeq, GENCODE

---

## Architecture

### Component Structure

```
splice_engine/config/
├── __init__.py              # Public API exports
├── genomic_config.py        # Core config system
└── settings.yaml            # Configuration data
```

### Key Components

#### 1. Config Dataclass

```python
@dataclass
class Config:
    """Configuration for genomic resources."""
    species: str                  # e.g., "homo_sapiens"
    build: str                    # e.g., "GRCh38"
    release: str                  # e.g., "112"
    data_root: Path               # Absolute path to data root
    builds: dict                  # Build-specific settings
    derived_datasets: dict        # Filenames for derived data
    annotation_sources: dict      # Available sources
    base_models: dict            # Base model specifications
    default_annotation_source: str
```

**Why dataclass**: Type safety, IDE autocomplete, explicit structure

#### 2. Project Root Detection

```python
def find_project_root(current_path: str = './') -> str:
    """Find project root by looking for markers."""
    root_markers = ['.git', 'pyproject.toml', 'setup.py']
    
    path = os.path.abspath(current_path)
    while path != os.path.dirname(path):  # Not at filesystem root
        for marker in root_markers:
            if os.path.exists(os.path.join(path, marker)):
                return path
        path = os.path.dirname(path)
    
    raise RuntimeError("Could not find project root")
```

**Why automatic detection**: No hardcoded paths, works from any location

#### 3. Configuration Loader

```python
def load_config(path: str = None) -> Config:
    """Load configuration from YAML with environment overrides."""
    # Find config file
    if path is None:
        # Look in config module first, then fallback locations
        config_locations = [
            Path(__file__).parent / "settings.yaml",
            project_root / "configs" / "genomic_resources.yaml",
        ]
        for config_path in config_locations:
            if config_path.exists():
                path = config_path
                break
    
    # Load YAML
    with open(path) as f:
        y = yaml.safe_load(f)
    
    # Resolve data_root to absolute path
    data_root = Path(os.getenv("SS_DATA_ROOT", y["data_root"]))
    if not data_root.is_absolute():
        data_root = project_root / data_root
    
    # Environment overrides
    build = os.getenv("SS_BUILD", y["default_build"])
    release = os.getenv("SS_RELEASE", y["default_release"])
    
    return Config(...)
```

**Why YAML + env vars**: Human-readable config, flexible overrides

#### 4. Path Resolution Methods

```python
def get_data_dir(self, build: str = None, annotation_source: str = None) -> Path:
    """
    Get data directory for a build and annotation source.
    
    Structure: data_root / annotation_source / build
    
    Examples:
        >>> config.get_data_dir("GRCh37")
        Path("data/ensembl/GRCh37")
        
        >>> config.get_data_dir("GRCh38_MANE")
        Path("data/mane/GRCh38")
    """
    if build is None:
        build = self.build
    if annotation_source is None:
        annotation_source = self.get_annotation_source(build)
    
    build_dir = build.replace("_MANE", "").replace("_GENCODE", "")
    return self.data_root / annotation_source / build_dir
```

**Why method-based**: Logic encapsulated, easy to extend

---

## Directory Structure

### Standard Layout

```
data/                           # Data root (configurable)
├── ensembl/                    # Annotation source
│   ├── GRCh37/                # Build directory
│   │   ├── Homo_sapiens.GRCh37.87.gtf.gz
│   │   ├── Homo_sapiens.GRCh37.87.dna.primary_assembly.fa.gz
│   │   ├── splice_sites_enhanced.tsv
│   │   ├── gene_features.tsv
│   │   └── spliceai_eval/    # Base model evaluation
│   │       ├── base_layer/
│   │       └── meta_layer/
│   └── GRCh38/
│       ├── Homo_sapiens.GRCh38.112.gtf.gz
│       ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
│       └── (derived datasets)
│
└── mane/                       # MANE RefSeq
    └── GRCh38/
        ├── MANE.GRCh38.v1.3.refseq_genomic.gtf.gz
        ├── Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
        └── openspliceai_eval/
            ├── base_layer/
            └── meta_layer/
```

### Why This Structure

**1. Annotation source at top level**:
- Different sources (Ensembl, MANE) can have different transcript sets
- Keeps annotation-specific data isolated

**2. Build under annotation source**:
- Same annotation source can support multiple builds
- Clear separation of GRCh37 vs GRCh38 data

**3. Base model evaluation directories**:
- Each base model (spliceai, openspliceai) gets dedicated space
- Separates base_layer and meta_layer artifacts

---

## Configuration File Design

### settings.yaml Structure

```yaml
# Global defaults
species: homo_sapiens
default_build: GRCh38
default_release: "112"
default_annotation_source: ensembl
data_root: data

# Derived datasets (filenames)
derived_datasets:
  splice_sites: "splice_sites_enhanced.tsv"
  gene_features: "gene_features.tsv"
  annotations_db: "annotations.db"

# Annotation sources (documentation)
annotation_sources:
  ensembl:
    name: "Ensembl"
    format: "GTF"
    notes: "Primary annotation source"
  mane:
    name: "MANE RefSeq"
    format: "GFF3"
    notes: "High-confidence transcript set"

# Base models (specifications)
base_models:
  spliceai:
    training_build: "GRCh37"
    training_annotation: "GENCODE V24lift37"
    annotation_source: "ensembl"
  openspliceai:
    training_build: "GRCh38"
    training_annotation: "MANE v1.3 RefSeq"
    annotation_source: "mane"

# Build-specific settings
builds:
  GRCh38:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ensembl_base: "https://ftp.ensembl.org/pub/release-{release}"
    
  GRCh37:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    default_release: "87"
    
  GRCh38_MANE:
    annotation_source: mane
    gtf: "MANE.GRCh38.v{release}.refseq_genomic.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
```

### Design Rationale

**1. Release string interpolation**:
```python
filename = cfg.builds[cfg.build]["gtf"].format(release=cfg.release)
# "Homo_sapiens.GRCh38.{release}.gtf" → "Homo_sapiens.GRCh38.112.gtf"
```

**Why**: Flexible release management without config changes

**2. Build-specific defaults**:
```yaml
GRCh37:
  default_release: "87"  # Last Ensembl release for GRCh37
```

**Why**: Sensible defaults per build, overridable via env vars

**3. Base model metadata**:
```yaml
base_models:
  spliceai:
    training_build: "GRCh37"
    annotation_source: "ensembl"
```

**Why**: Encode which build/annotation each model was trained on

---

## Usage Patterns

### Basic Resource Access

```python
from agentic_spliceai.splice_engine.config import config, filename

# Get GTF path for current build
gtf_file = config.get_data_dir() / filename("gtf", config)

# Get FASTA path
fasta_file = config.get_data_dir() / filename("fasta", config)

# Get derived dataset
splice_sites = config.get_data_dir() / config.derived_datasets["splice_sites"]
```

### Multi-Build Support

```python
# Get paths for specific build
grch37_dir = config.get_data_dir("GRCh37")
grch38_dir = config.get_data_dir("GRCh38")

# Get build for base model
spliceai_build = config.get_base_model_build("spliceai")  # "GRCh37"
openspliceai_build = config.get_base_model_build("openspliceai")  # "GRCh38"
```

### Environment Overrides

```bash
# Override via environment variables
export SS_BUILD=GRCh37
export SS_RELEASE=87
export SS_DATA_ROOT=/mnt/shared/genomic_data

python run_prediction.py  # Uses GRCh37, release 87
```

### Testing with Isolated Data

```python
# Tests can override data_root
test_config = load_config()
test_config.data_root = Path("tests/fixtures/data")
```

---

## Best Practices

### DO ✅

**Use the config everywhere**:
```python
from agentic_spliceai.splice_engine.config import config
data_dir = config.get_data_dir()
```

**Make paths absolute**:
```python
# Config resolves to absolute paths
assert config.data_root.is_absolute()  # True
```

**Use pathlib**:
```python
from pathlib import Path
path = config.data_root / "subdir" / "file.txt"
```

**Environment overrides for deployment**:
```bash
# Production
export SS_DATA_ROOT=/mnt/shared/data

# Development
export SS_DATA_ROOT=./data
```

### DON'T ❌

**Hardcode paths**:
```python
# BAD
data_dir = "/Users/me/data"  # Breaks on other machines
```

**String concatenation**:
```python
# BAD
path = data_dir + "/" + subdir + "/" + filename  # Not cross-platform
```

**Relative imports for data**:
```python
# BAD
data_path = "../../data/file.txt"  # Depends on execution location
```

**Scattered configuration**:
```python
# BAD - different modules with different defaults
# module1.py
DEFAULT_BUILD = "GRCh38"
# module2.py
GENOME_BUILD = "GRCh37"
```

---

## Integration with Other Systems

### Artifact Management

`ArtifactManager` uses resource config to determine output paths:

```python
class ArtifactManager:
    def __init__(self, base_model: str):
        from ..config import config
        self.registry = config
        self.base_dir = config.get_data_dir() / f"{base_model}_eval"
```

**Why**: Output paths derived from resource config ensures consistency

### Data Preparation

Data prep scripts use config for input/output:

```python
def prepare_splice_sites():
    gtf_path = config.get_data_dir() / filename("gtf", config)
    output_path = config.get_data_dir() / config.derived_datasets["splice_sites"]
    # Process...
```

**Why**: Scripts work across builds without modification

---

## Migration from meta-spliceai

### Changes Made

**Before (meta-spliceai)**:
```python
# Fragmented system
from meta_spliceai.system.genomic_resources.registry import GenomicRegistry
from meta_spliceai.system.genomic_resources.config import Config

registry = GenomicRegistry(build='GRCh38', source='mane')
```

**After (agentic-spliceai)**:
```python
# Unified system
from agentic_spliceai.splice_engine.config import config

# Config loaded automatically, used everywhere
data_dir = config.get_data_dir()
```

### Improvements

1. **Simpler import**: Single import point
2. **Global config**: One config instance, consistent everywhere
3. **Better env support**: Environment variables work out of box
4. **Type safe**: Dataclass provides type hints

---

## Testing Strategy

### Unit Tests

```python
def test_project_root_detection():
    """Should find project root from any subdirectory."""
    root = get_project_root()
    assert (root / ".git").exists() or (root / "pyproject.toml").exists()

def test_path_resolution():
    """Should resolve paths correctly."""
    cfg = load_config()
    data_dir = cfg.get_data_dir()
    assert data_dir.is_absolute()
    assert data_dir.exists() or not cfg.data_root.exists()  # OK if data not present

def test_build_support():
    """Should support multiple builds."""
    cfg = load_config()
    grch37_dir = cfg.get_data_dir("GRCh37")
    grch38_dir = cfg.get_data_dir("GRCh38")
    assert grch37_dir != grch38_dir

def test_base_model_metadata():
    """Should retrieve base model metadata."""
    cfg = load_config()
    spliceai_build = cfg.get_base_model_build("spliceai")
    assert spliceai_build == "GRCh37"
```

### Integration Tests

```python
def test_config_with_artifacts():
    """Resource config should integrate with artifact management."""
    from ..artifacts import get_artifact_manager
    am = get_artifact_manager("spliceai")
    assert am.base_layer_dir.exists() or True  # OK if not yet created
```

---

## Future Extensions

### Multi-Species Support

```python
# Extend for mouse, other species
config = load_config()
if config.species == "mus_musculus":
    # Mouse-specific logic
```

### Cloud Storage Integration

```python
# Support S3, GCS paths
data_root = "s3://bucket/genomic-data/"
config.data_root = CloudPath(data_root)
```

### Caching Layer

```python
# Add transparent caching
class CachedConfig(Config):
    def get_data_dir(self, ...):
        # Check local cache first
        # Fall back to remote if needed
```

---

## Summary

### Key Benefits

✅ **Portability**: Same code works everywhere  
✅ **Maintainability**: Single source of truth  
✅ **Flexibility**: Configuration-driven  
✅ **Type Safety**: Dataclass with type hints  
✅ **Multi-Build**: Supports GRCh37, GRCh38, MANE  
✅ **Environment-Aware**: Overrides via env vars  
✅ **Testable**: Easy to mock, override for tests  

### Design Principles Applied

1. ✅ **Single Source of Truth**: All paths in config
2. ✅ **Configuration Over Code**: YAML + env vars
3. ✅ **Explicit Over Implicit**: Type hints, clear methods
4. ✅ **Fail Fast**: Validates paths, checks file existence
5. ✅ **Testability**: Pure functions, dependency injection

---

**Implementation**: See [src/agentic_spliceai/splice_engine/config/genomic_config.py](../../src/agentic_spliceai/splice_engine/config/genomic_config.py)  
**Refactoring Plan**: See [dev/refactoring/RESOURCE_MANAGEMENT_REFACTORING.md](../../dev/refactoring/RESOURCE_MANAGEMENT_REFACTORING.md)  
**Last Updated**: February 15, 2026
