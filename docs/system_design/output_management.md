# Output & Artifact Management System Design

**Purpose**: Systematic organization of experimental outputs, model artifacts, and predictions  
**Component**: `splice_engine/resources/artifacts/`  
**Status**: 📝 Design complete, implementation in progress

---

## Problem Statement

### What We're Solving

Research and ML projects generate many artifacts:
- **Base layer outputs**: Splice site predictions, analysis sequences, error metrics
- **Meta layer outputs**: Meta-model predictions, training checkpoints
- **Model artifacts**: Trained weights, architectures, hyperparameters
- **Experimental results**: Metrics, plots, logs, evaluation results
- **Training artifacts**: Checkpoints, validation curves, intermediate states

### Challenges Without Output Management

❌ **Output Chaos**:
```python
# Scattered outputs
model.save("final_model.pkl")  # Where did this go?
results.to_csv("results.csv")  # Which results?
plt.savefig("plot.png")        # Which experiment?
```

❌ **Non-Reproducible Experiments**:
- "Where are the results from experiment X?"
- "Which model weights correspond to these metrics?"
- "What hyperparameters produced this plot?"

❌ **Overwrite Disasters**:
```python
# Production artifacts overwritten by mistake
predictions.save("predictions.tsv")  # Oops, overwrote last week's run!
```

❌ **No Experiment Tracking**:
- Can't compare experiments
- Can't reproduce results
- Can't find "that good model from last month"

---

## Design Goals

### 1. Systematic Organization

Clear, predictable structure for all outputs:

```
data/ensembl/GRCh38/spliceai_eval/
├── base_layer/                    # Base model predictions
│   ├── analysis_sequences_chr*.tsv
│   ├── splice_errors_chr*.tsv
│   ├── nucleotide_scores_chr*.parquet
│   └── gene_manifest.tsv
└── meta_layer/                    # Meta model outputs
    ├── checkpoints/
    │   ├── epoch_001.pkl
    │   ├── epoch_002.pkl
    │   └── latest.pkl
    ├── predictions/
    │   └── meta_predictions.tsv
    └── metrics/
        ├── training_metrics.json
        └── evaluation_metrics.json
```

### 2. Reproducibility

Track what, when, how:
- **What**: Artifact type, content, format
- **When**: Timestamp, experiment ID
- **How**: Configuration, code version, parameters

### 3. Mode-Based Isolation

Different modes for different use cases:

| Mode | Purpose | Overwrite? | Location |
|------|---------|------------|----------|
| **Production** | Immutable results | ❌ No | `base_layer/` |
| **Development** | Iterative work | ✅ Yes | `base_layer/dev/timestamp/` |
| **Test** | Isolated testing | ✅ Yes | `base_layer/tests/test_name/` |

### 4. Type-Safe Artifacts

Enum-based artifact types prevent typos:

```python
class ArtifactType(Enum):
    ANALYSIS_SEQUENCES = "analysis_sequences"
    SPLICE_ERRORS = "splice_errors"
    NUCLEOTIDE_SCORES = "nucleotide_scores"
```

### 5. Format Consistency

Standardized formats:
- **Tabular data**: TSV for readability, Parquet for performance
- **Models**: Pickle (Python) or framework-specific (PyTorch, TF)
- **Metrics**: JSON for structure, human-readable
- **Plots**: PNG (high DPI) or PDF (vector)

---

## Architecture

### Component Structure

```
splice_engine/resources/
├── artifacts/
│   ├── __init__.py          # Public API
│   ├── manager.py           # ArtifactManager class
│   ├── types.py             # ArtifactType enum
│   └── paths.py             # Path utilities
└── genomic/
    └── (resource management)
```

### Key Components

#### 1. ArtifactType Enum

```python
class ArtifactType(Enum):
    """Types of artifacts with standard names."""
    
    # Base layer outputs
    ANALYSIS_SEQUENCES = "analysis_sequences"
    SPLICE_ERRORS = "splice_errors"
    NUCLEOTIDE_SCORES = "nucleotide_scores"
    GENE_MANIFEST = "gene_manifest"
    
    # Meta layer outputs
    META_PREDICTIONS = "meta_predictions"
    META_CHECKPOINTS = "meta_checkpoints"
    META_METRICS = "meta_metrics"
    
    # Training artifacts
    TRAINING_DATA = "training_data"
    VALIDATION_DATA = "validation_data"
```

**Why enum**: Prevents typos, IDE autocomplete, explicit inventory

#### 2. ArtifactManager Class

```python
class ArtifactManager:
    """
    Unified artifact management for all pipeline stages.
    
    Responsibilities:
    - Path resolution for artifacts
    - Save/load with format handling
    - Overwrite policy enforcement
    - Mode-based directory isolation
    
    Supports:
    - Production mode (immutable)
    - Development mode (timestamped)
    - Test mode (isolated)
    """
    
    def __init__(
        self,
        base_model: str,                    # e.g., "spliceai"
        mode: str = 'production',           # 'production', 'development', 'test'
        test_name: Optional[str] = None     # Required if mode='test'
    ):
        self.base_model = base_model
        self.mode = mode
        self.test_name = test_name
        self._setup_paths()
```

#### 3. Path Resolution

```python
def get_artifact_path(
    self,
    artifact_type: ArtifactType,
    chromosome: Optional[str] = None,
    layer: str = 'base'
) -> Path:
    """
    Get path for specific artifact.
    
    Examples:
        >>> am.get_artifact_path(ArtifactType.ANALYSIS_SEQUENCES, "chr1")
        Path("data/ensembl/GRCh38/spliceai_eval/base_layer/analysis_sequences_chr1.tsv")
        
        >>> am.get_artifact_path(ArtifactType.META_CHECKPOINTS, layer="meta")
        Path("data/ensembl/GRCh38/spliceai_eval/meta_layer/checkpoints/meta_checkpoints.pkl")
    """
    base_dir = self.base_layer_dir if layer == 'base' else self.meta_layer_dir
    
    # Build filename
    filename = artifact_type.value
    if chromosome:
        filename += f"_chr{chromosome}"
    filename += self._get_extension(artifact_type)
    
    return base_dir / filename
```

#### 4. Save with Overwrite Policy

```python
def save_artifact(
    self,
    data: pl.DataFrame,
    artifact_type: ArtifactType,
    chromosome: Optional[str] = None,
    layer: str = 'base'
) -> Path:
    """
    Save artifact with overwrite policy enforcement.
    
    Raises:
        FileExistsError: If production artifact exists
    """
    path = self.get_artifact_path(artifact_type, chromosome, layer)
    
    # Check overwrite policy
    if path.exists() and not self.should_overwrite(artifact_type):
        raise FileExistsError(
            f"Artifact exists and overwrite not allowed: {path}\n"
            f"Mode: {self.mode} (production artifacts are immutable)"
        )
    
    # Save based on format
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.tsv':
        data.write_csv(path, separator='\t')
    else:
        data.write_parquet(path)
    
    return path
```

#### 5. Load with Validation

```python
def load_artifact(
    self,
    artifact_type: ArtifactType,
    chromosome: Optional[str] = None,
    layer: str = 'base'
) -> pl.DataFrame:
    """
    Load artifact with existence checking.
    
    Raises:
        FileNotFoundError: If artifact doesn't exist
    """
    path = self.get_artifact_path(artifact_type, chromosome, layer)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            f"Expected in {layer} layer for {self.base_model}"
        )
    
    if path.suffix == '.tsv':
        return pl.read_csv(path, separator='\t')
    else:
        return pl.read_parquet(path)
```

---

## Mode-Based Isolation

### Production Mode

**Purpose**: Immutable, authoritative results

**Behavior**:
- ❌ Cannot overwrite existing artifacts
- ✅ Standard directory structure
- ✅ Used for published results, deployed models

**Location**:
```
data/ensembl/GRCh38/spliceai_eval/base_layer/
```

**Example**:
```python
am = ArtifactManager("spliceai", mode="production")
am.save_artifact(predictions, ArtifactType.ANALYSIS_SEQUENCES, "chr1")
# Saves to: data/.../base_layer/analysis_sequences_chr1.tsv

# Second save fails
am.save_artifact(predictions2, ArtifactType.ANALYSIS_SEQUENCES, "chr1")
# Raises: FileExistsError (production artifacts immutable)
```

### Development Mode

**Purpose**: Iterative experimentation

**Behavior**:
- ✅ Can overwrite within same session
- ✅ Timestamped directories prevent cross-session conflicts
- ✅ Easy to compare different runs

**Location**:
```
data/ensembl/GRCh38/spliceai_eval/base_layer/dev/20260215_143022/
data/ensembl/GRCh38/spliceai_eval/base_layer/dev/20260215_151035/
```

**Example**:
```python
am = ArtifactManager("spliceai", mode="development")
am.save_artifact(predictions, ArtifactType.ANALYSIS_SEQUENCES, "chr1")
# Saves to: data/.../dev/20260215_143022/analysis_sequences_chr1.tsv

# Can overwrite within this session
am.save_artifact(predictions2, ArtifactType.ANALYSIS_SEQUENCES, "chr1")
# Overwrites same file (same timestamp directory)
```

### Test Mode

**Purpose**: Isolated unit/integration testing

**Behavior**:
- ✅ Isolated from production/dev
- ✅ Named test directories
- ✅ Easy to clean up

**Location**:
```
data/ensembl/GRCh38/spliceai_eval/base_layer/tests/test_prediction/
data/ensembl/GRCh38/spliceai_eval/base_layer/tests/test_integration/
```

**Example**:
```python
am = ArtifactManager("spliceai", mode="test", test_name="test_prediction")
am.save_artifact(predictions, ArtifactType.ANALYSIS_SEQUENCES, "chr1")
# Saves to: data/.../tests/test_prediction/analysis_sequences_chr1.tsv

# Tests can clean up easily
shutil.rmtree(am.base_layer_dir)
```

---

## Format Standards

### TSV (Tab-Separated Values)

**Use for**: Human-readable, line-oriented data

**Artifacts**:
- Analysis sequences
- Splice errors
- Gene manifests

**Advantages**:
- Human-readable
- Line-oriented (easy diffs)
- Standard tool support (grep, awk, etc.)

**Example**:
```tsv
gene_id	chromosome	strand	start	end	transcript_count
ENSG00000000003	chrX	-	100627108	100639991	3
ENSG00000000005	chrX	+	100584936	100599885	2
```

### Parquet

**Use for**: Large, high-performance data

**Artifacts**:
- Nucleotide scores (millions of rows)
- Training data (large features)

**Advantages**:
- Fast read/write
- Columnar compression
- Schema preservation
- Efficient filtering

**Example**:
```python
# Efficient reading
scores = pl.scan_parquet("nucleotide_scores_chr1.parquet") \
    .filter(pl.col("position") > 1000000) \
    .collect()
```

### JSON

**Use for**: Structured metadata, metrics

**Artifacts**:
- Training metrics
- Evaluation results
- Experiment configuration

**Advantages**:
- Human-readable
- Nested structure
- Language-agnostic
- Schema-less flexibility

**Example**:
```json
{
  "experiment_id": "exp_20260215_143022",
  "model": "meta_spliceai_v1",
  "metrics": {
    "train_loss": 0.023,
    "val_loss": 0.031,
    "val_accuracy": 0.947
  },
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32
  }
}
```

### Pickle

**Use for**: Python objects, model weights

**Artifacts**:
- Model checkpoints
- Complex Python objects
- Trained models

**Advantages**:
- Preserves Python objects
- Fast serialize/deserialize
- Handles complex types

**⚠️ Warning**: Not portable across Python versions, security risk for untrusted data

---

## Usage Patterns

### Basic Save/Load

```python
from agentic_spliceai.splice_engine.resources.artifacts import (
    get_artifact_manager,
    ArtifactType
)

# Get manager
am = get_artifact_manager("spliceai", mode="production")

# Save artifact
predictions = pl.DataFrame(...)
path = am.save_artifact(
    predictions,
    ArtifactType.ANALYSIS_SEQUENCES,
    chromosome="chr1"
)
print(f"Saved to: {path}")

# Load artifact
loaded = am.load_artifact(
    ArtifactType.ANALYSIS_SEQUENCES,
    chromosome="chr1"
)
```

### Per-Chromosome Processing

```python
# Save per-chromosome artifacts
for chrom in ["chr1", "chr2", ..., "chrX"]:
    predictions = process_chromosome(chrom)
    am.save_artifact(
        predictions,
        ArtifactType.ANALYSIS_SEQUENCES,
        chromosome=chrom
    )

# Load specific chromosome
chr1_data = am.load_artifact(
    ArtifactType.ANALYSIS_SEQUENCES,
    chromosome="chr1"
)
```

### Meta Layer Artifacts

```python
# Save training checkpoint
checkpoint = {
    "epoch": 42,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "metrics": {"val_loss": 0.031}
}
am.save_artifact(
    checkpoint,
    ArtifactType.META_CHECKPOINTS,
    layer="meta"
)

# Save metrics
metrics = {"train_loss": 0.023, "val_accuracy": 0.947}
am.save_artifact(
    metrics,
    ArtifactType.META_METRICS,
    layer="meta"
)
```

### Development Workflow

```python
# Development: iterate freely
am_dev = get_artifact_manager("spliceai", mode="development")

for version in range(10):
    predictions = experiment(version)
    am_dev.save_artifact(predictions, ArtifactType.ANALYSIS_SEQUENCES)
    # Each save overwrites in the timestamped directory

# Promote to production when satisfied
am_prod = get_artifact_manager("spliceai", mode="production")
final_predictions = best_experiment()
am_prod.save_artifact(final_predictions, ArtifactType.ANALYSIS_SEQUENCES)
```

---

## Integration with Experiments

### OutputManager Pattern

For complex experiments, extend ArtifactManager:

```python
class OutputManager(ArtifactManager):
    """Enhanced artifact manager with experiment tracking."""
    
    def __init__(self, experiment_name: str, base_model: str):
        super().__init__(base_model, mode="development")
        self.experiment_name = experiment_name
        self.created_at = datetime.now()
    
    def save_experiment_metadata(self, config: dict, metrics: dict):
        """Save experiment metadata."""
        metadata = {
            "experiment_name": self.experiment_name,
            "created_at": self.created_at.isoformat(),
            "config": config,
            "metrics": metrics
        }
        metadata_path = self.base_layer_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_plot(self, fig, name: str):
        """Save matplotlib figure."""
        plots_dir = self.base_layer_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        fig.savefig(plots_dir / f"{name}.png", dpi=300, bbox_inches='tight')
```

**Usage**:
```python
om = OutputManager("splice_analysis_v3", "spliceai")

# Run experiment
results = run_experiment(config)

# Save everything
om.save_artifact(results, ArtifactType.ANALYSIS_SEQUENCES)
om.save_experiment_metadata(config, metrics)
om.save_plot(confusion_matrix_fig, "confusion_matrix")
```

---

## Best Practices

### DO ✅

**Use artifact types**:
```python
am.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)  # Type-safe
```

**Check mode before production saves**:
```python
if am.mode == "production":
    # Extra validation before immutable save
    validate_predictions(data)
am.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
```

**Add metadata**:
```python
metadata = {
    "created_at": datetime.now().isoformat(),
    "config": config_dict,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"])
}
# Save alongside artifact
```

**Use appropriate formats**:
```python
# Small, human-readable → TSV
gene_manifest.write_csv("gene_manifest.tsv", separator='\t')

# Large, performance-critical → Parquet
nucleotide_scores.write_parquet("scores.parquet")
```

### DON'T ❌

**Don't hardcode artifact paths**:
```python
# BAD
predictions.save("data/spliceai_eval/predictions.tsv")

# GOOD
am.save_artifact(predictions, ArtifactType.ANALYSIS_SEQUENCES)
```

**Don't use arbitrary names**:
```python
# BAD
am.save_artifact(data, "my_predictions")  # Not a valid ArtifactType

# GOOD
am.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
```

**Don't mix modes accidentally**:
```python
# BAD - Easy to overwrite production
am = ArtifactManager("spliceai", mode="production")  # Think you're in dev!
am.save_artifact(data, ...)  # Fails if exists (good!)
```

---

## Testing Strategy

### Unit Tests

```python
def test_save_load_artifact():
    """Should save and load artifacts correctly."""
    am = ArtifactManager("spliceai", mode="test", test_name="test_save_load")
    
    data = pl.DataFrame({"col1": [1, 2, 3]})
    path = am.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
    
    loaded = am.load_artifact(ArtifactType.ANALYSIS_SEQUENCES)
    assert loaded.equals(data)
    
    # Cleanup
    shutil.rmtree(am.base_layer_dir)

def test_overwrite_policy():
    """Production should not overwrite, development should."""
    # Production: no overwrite
    am_prod = ArtifactManager("spliceai", mode="production")
    data = pl.DataFrame({"col1": [1]})
    am_prod.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
    
    with pytest.raises(FileExistsError):
        am_prod.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
    
    # Development: can overwrite
    am_dev = ArtifactManager("spliceai", mode="development")
    am_dev.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)
    am_dev.save_artifact(data, ArtifactType.ANALYSIS_SEQUENCES)  # OK
```

---

## Future Extensions

### Version Control Integration

```python
class VersionedArtifactManager(ArtifactManager):
    """Artifact manager with automatic versioning."""
    
    def save_artifact(self, data, artifact_type, ...):
        # Add git commit to metadata
        git_commit = get_git_commit()
        metadata = {"git_commit": git_commit, ...}
        # Save metadata alongside artifact
```

### Cloud Storage Support

```python
class CloudArtifactManager(ArtifactManager):
    """Artifact manager with cloud storage backend."""
    
    def save_artifact(self, data, artifact_type, ...):
        # Save locally
        local_path = super().save_artifact(...)
        # Upload to S3/GCS
        self.upload_to_cloud(local_path)
```

### Artifact Catalog

```python
class CatalogedArtifactManager(ArtifactManager):
    """Maintains searchable catalog of artifacts."""
    
    def save_artifact(self, data, artifact_type, ...):
        path = super().save_artifact(...)
        # Register in catalog
        self.catalog.register(artifact_type, path, metadata)
```

---

## Summary

### Key Benefits

✅ **Organization**: Predictable structure for all outputs  
✅ **Reproducibility**: Track what, when, how  
✅ **Safety**: Immutable production artifacts  
✅ **Flexibility**: Development mode for iteration  
✅ **Isolation**: Test mode for testing  
✅ **Type Safety**: Enum-based artifact types  
✅ **Format Consistency**: Standard formats for each type  

### Design Principles Applied

1. ✅ **Single Source of Truth**: ArtifactManager for all artifacts
2. ✅ **Separation of Concerns**: Artifact management isolated
3. ✅ **Fail Fast**: Overwrite policy enforced
4. ✅ **Explicit Over Implicit**: Typed artifact types
5. ✅ **Testability**: Mode-based isolation for tests

---

**Related**: [Resource Management](resource_management.md), [Configuration System](configuration_system.md)  
**Implementation**: `src/agentic_spliceai/splice_engine/config/`  
**Last Updated**: February 15, 2026
