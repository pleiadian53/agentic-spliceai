# Stage 4: Genomic Resources System Analysis

**Purpose**: Understand the genomic resources Registry for consistent path resolution  
**Date**: December 10, 2025

---

## Overview

The `genomic_resources` module is the **central nervous system** for path resolution in Meta-SpliceAI. It ensures that all components use consistent, build-specific paths for genomic data files.

**Location**: `meta_spliceai/system/genomic_resources/`

**Key Files**:
| File | Purpose |
|------|---------|
| `registry.py` | Path resolution for all genomic resources |
| `config.py` | Configuration management with YAML + env vars |
| `__init__.py` | Public API exports |
| `derive.py` | Generate derived datasets from GTF |
| `validators.py` | Data validation utilities |
| `schema.py` | Column standardization |

---

## 1. Registry Class Purpose

The `Registry` class provides a **unified interface** for locating genomic resource files across multiple possible locations. It solves the critical problem of:

> **"Where is my data?"** - Given a build (GRCh37 vs GRCh38) and annotation source (Ensembl vs MANE), find the correct paths for GTF, FASTA, and derived files.

### Key Responsibilities

1. **Build-specific path resolution** - Maps build names to directory structures
2. **Multi-location search** - Searches multiple directories in priority order
3. **Environment variable overrides** - Allows runtime path customization
4. **Derived file discovery** - Finds generated files (splice sites, sequences, etc.)
5. **Base model support** - Provides model-specific evaluation directories

---

## 2. Build Name → Directory Mapping

### The Mapping Logic

```python
# In Registry.__init__():
self.annotation_source = self.cfg.get_annotation_source(self.cfg.build)
self.top = self.cfg.data_root / self.annotation_source
build_dir = self.cfg.build.replace("_MANE", "").replace("_GENCODE", "")
self.stash = self.top / build_dir
```

### Build Name Mappings

| Build Name | Annotation Source | Directory Path |
|------------|-------------------|----------------|
| `GRCh37` | `ensembl` | `data/ensembl/GRCh37/` |
| `GRCh38` | `ensembl` | `data/ensembl/GRCh38/` |
| `GRCh38_MANE` | `mane` | `data/mane/GRCh38/` |
| `GRCh38_GENCODE` | `gencode` | `data/gencode/GRCh38/` |

### Configuration Source (genomic_resources.yaml)

```yaml
builds:
  GRCh37:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    fasta: "Homo_sapiens.GRCh37.dna.primary_assembly.fa"
    
  GRCh38:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh38.{release}.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    
  GRCh38_MANE:
    annotation_source: mane
    gtf: "MANE.GRCh38.v{release}.refseq_genomic.gtf"
    fasta: "Homo_sapiens.GRCh38.dna.primary_assembly.fa"

base_models:
  spliceai:
    training_build: "GRCh37"
    annotation_source: "ensembl"
    
  openspliceai:
    training_build: "GRCh38"
    annotation_source: "mane"
```

---

## 3. Registry Methods and Return Values

### Core Path Resolution Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `resolve(kind)` | `str` or `None` | Generic resolver for any resource kind |
| `get_gtf_path()` | `Path` | Path to GTF/GFF3 annotation file |
| `get_fasta_path()` | `Path` | Path to reference genome FASTA |
| `get_annotations_db_path()` | `Path` | Path to SQLite annotations database |
| `get_overlapping_genes_path()` | `Path` | Path to overlapping genes TSV |
| `get_chromosome_sequence_path(chr)` | `Path` | Path to per-chromosome Parquet file |

### Directory Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `get_local_dir()` | `Path` | Build-specific directory (`data_dir`) |
| `get_eval_dir()` | `Path` | Evaluation output directory |
| `get_analysis_dir()` | `Path` | Analysis output directory |
| `get_base_model_eval_dir(model)` | `Path` | Model-specific eval directory |
| `get_meta_models_artifact_dir(model)` | `Path` | Meta-model artifacts directory |

### List All Resources

```python
registry.list_all()
# Returns:
{
    'gtf': '/path/to/Homo_sapiens.GRCh38.112.gtf',
    'fasta': '/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa',
    'fasta_index': '/path/to/Homo_sapiens.GRCh38.dna.primary_assembly.fa.fai',
    'splice_sites': '/path/to/splice_sites_enhanced.tsv',
    'gene_features': '/path/to/gene_features.tsv',
    'transcript_features': None,  # Not found
    'exon_features': None,
    'junctions': None
}
```

---

## 4. Example Usage

### Example 1: SpliceAI (GRCh37/Ensembl)

```python
from meta_spliceai.system.genomic_resources import Registry

# Create registry for SpliceAI's training build
registry = Registry(build='GRCh37', release='87')

# Get paths
gtf_path = registry.get_gtf_path()
# → data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf

fasta_path = registry.get_fasta_path()
# → data/ensembl/GRCh37/Homo_sapiens.GRCh37.dna.primary_assembly.fa

splice_sites = registry.resolve('splice_sites')
# → data/ensembl/GRCh37/splice_sites_enhanced.tsv

eval_dir = registry.get_base_model_eval_dir('spliceai')
# → data/ensembl/GRCh37/spliceai_eval/

meta_models_dir = registry.get_meta_models_artifact_dir('spliceai')
# → data/ensembl/GRCh37/spliceai_eval/meta_models/
```

### Example 2: OpenSpliceAI (GRCh38/MANE)

```python
from meta_spliceai.system.genomic_resources import Registry

# Create registry for OpenSpliceAI's training build
registry = Registry(build='GRCh38_MANE', release='1.3')

# Get paths
gtf_path = registry.get_gtf_path()
# → data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf

fasta_path = registry.get_fasta_path()
# → data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa

splice_sites = registry.resolve('splice_sites')
# → data/mane/GRCh38/splice_sites_enhanced.tsv

eval_dir = registry.get_base_model_eval_dir('openspliceai')
# → data/mane/GRCh38/openspliceai_eval/

meta_models_dir = registry.get_meta_models_artifact_dir('openspliceai')
# → data/mane/GRCh38/openspliceai_eval/meta_models/
```

### Example 3: Using Config to Get Base Model Build

```python
from meta_spliceai.system.genomic_resources import load_config, Registry

# Load config
cfg = load_config()

# Get build for a base model
spliceai_build = cfg.get_base_model_build('spliceai')
# → 'GRCh37'

openspliceai_build = cfg.get_base_model_build('openspliceai')
# → 'GRCh38'

# Create registry for the correct build
registry = Registry(build=openspliceai_build)
```

---

## 5. Why Build-Specific Path Resolution is Critical

### The Problem: Cross-Build Contamination

Without proper path resolution, it's easy to accidentally mix data from different builds:

```python
# ❌ WRONG: Hard-coded paths ignore build differences
gtf_file = "data/annotations.gtf"  # Which build? GRCh37? GRCh38?
splice_sites = "data/splice_sites.tsv"  # Coordinates for which build?
```

**Consequences of mixing builds:**
- **Coordinate mismatch**: GRCh37 position 12345 ≠ GRCh38 position 12345
- **Gene ID mismatch**: Ensembl IDs may differ between releases
- **Performance degradation**: Up to 44% drop in PR-AUC when using wrong annotations

### The Solution: Registry Enforces Correctness

```python
# ✅ CORRECT: Registry ensures build-specific paths
registry = Registry(build='GRCh37')  # For SpliceAI
gtf_file = registry.get_gtf_path()   # Guaranteed GRCh37 GTF
splice_sites = registry.resolve('splice_sites')  # Guaranteed GRCh37 splice sites
```

### Directory Structure Prevents Errors

```text
data/
├── ensembl/
│   ├── GRCh37/                    # SpliceAI data
│   │   ├── Homo_sapiens.GRCh37.87.gtf
│   │   ├── splice_sites_enhanced.tsv
│   │   └── spliceai_eval/
│   │       └── meta_models/
│   │
│   └── GRCh38/                    # Ensembl GRCh38 data
│       ├── Homo_sapiens.GRCh38.112.gtf
│       └── ...
│
└── mane/
    └── GRCh38/                    # OpenSpliceAI data
        ├── MANE.GRCh38.v1.3.refseq_genomic.gtf
        ├── splice_sites_enhanced.tsv
        └── openspliceai_eval/
            └── meta_models/
```

**Key insight**: The physical directory structure makes it **impossible** to accidentally load GRCh37 data when working with GRCh38.

---

## 6. Search Order and Fallbacks

The Registry searches for files in this priority order:

```python
# In resolve():
for root in [self.stash, self.top, self.legacy]:
    p = Path(root) / name
    if p.exists():
        return str(p.resolve())
```

| Priority | Location | Example | Purpose |
|----------|----------|---------|---------|
| 1 | `stash` (build-specific) | `data/ensembl/GRCh37/` | Primary location |
| 2 | `top` (annotation source) | `data/ensembl/` | Shared files |
| 3 | `legacy` | `data/ensembl/spliceai_analysis/` | Backward compatibility |

### Environment Variable Overrides

```bash
# Override any path at runtime
export SS_GTF_PATH=/custom/path/to/annotations.gtf
export SS_FASTA_PATH=/custom/path/to/genome.fa
```

---

## 7. Integration with Model Configs

The Registry integrates with `SpliceAIConfig` and `OpenSpliceAIConfig`:

```python
# In SpliceAIConfig.__post_init__():
from meta_spliceai.system.genomic_resources import Registry

# Auto-resolve paths for GRCh37
analyzer = Analyzer()  # Uses Registry internally
self.genome_fasta = analyzer.genome_fasta
self.gtf_file = analyzer.gtf_file

# In OpenSpliceAIConfig.__post_init__():
registry = Registry(build='GRCh38_MANE', release='1.3')
self.genome_fasta = str(registry.get_fasta_path())
self.gtf_file = str(registry.get_gtf_path())
```

---

## 8. Complete Module Structure

```text
meta_spliceai/system/genomic_resources/
├── __init__.py          # Public API (40+ exports)
├── config.py            # Config dataclass + load_config()
├── registry.py          # Registry class (path resolution)
├── derive.py            # GenomicDataDeriver (generate derived files)
├── splice_sites.py      # Splice site extraction
├── validators.py        # Data validation
├── schema.py            # Column standardization
├── gene_mapper.py       # Gene ID mapping
├── gene_mapper_enhanced.py  # Enhanced mapping with strategies
├── external_id_mapper.py    # MANE ↔ Ensembl mapping
├── gene_selection.py    # Gene sampling utilities
├── build_naming.py      # Build name standardization
├── download.py          # Download Ensembl files
└── cli.py               # Command-line interface
```

---

## 9. Key Takeaways for Porting

### Must Port (Essential)

| Component | Reason |
|-----------|--------|
| `Registry` class | Core path resolution |
| `Config` dataclass | Configuration management |
| `load_config()` | YAML loading |
| `genomic_resources.yaml` | Build definitions |

### Should Port (Recommended)

| Component | Reason |
|-----------|--------|
| `GenomicDataDeriver` | Generate derived files |
| `standardize_*_schema()` | Column consistency |
| `validators.py` | Data validation |

### Can Defer (Optional)

| Component | Reason |
|-----------|--------|
| `GeneMapper` | Only if gene ID mapping needed |
| `GeneSelector` | Only if gene sampling needed |
| `download.py` | Only if auto-download needed |

### Porting Considerations

1. **Config file location**: Decide where `genomic_resources.yaml` lives in agentic-spliceai
2. **Project root detection**: Adapt `get_project_root()` for new project structure
3. **Data directory**: Configure `data_root` for agentic-spliceai's data location
4. **Environment variables**: Keep `SS_*` prefix or change to `AS_*` (Agentic-SpliceAI)

---

## 10. Summary Diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GENOMIC RESOURCES SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Code                                                                 │
│       │                                                                     │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         Registry(build='GRCh37')                    │  │
│   │                                    │                                │  │
│   │                                    ▼                                │  │
│   │                         ┌─────────────────────┐                     │  │
│   │                         │    load_config()    │                     │  │
│   │                         │         │           │                     │  │
│   │                         │         ▼           │                     │  │
│   │                         │  genomic_resources  │                     │  │
│   │                         │       .yaml         │                     │  │
│   │                         └─────────────────────┘                     │  │
│   │                                    │                                │  │
│   │                    ┌───────────────┼───────────────┐                │  │
│   │                    │               │               │                │  │
│   │                    ▼               ▼               ▼                │  │
│   │             get_gtf_path()  get_fasta_path()  resolve('splice_sites')│  │
│   │                    │               │               │                │  │
│   │                    ▼               ▼               ▼                │  │
│   │         data/ensembl/GRCh37/   data/ensembl/GRCh37/   data/ensembl/ │  │
│   │         Homo_sapiens.GRCh37   Homo_sapiens.GRCh37    GRCh37/splice_ │  │
│   │         .87.gtf               .dna.primary_         sites_enhanced  │  │
│   │                               assembly.fa           .tsv            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 4, Step 2: Schema Standardization

### 1. What Problem Does Schema Standardization Solve?

Different data sources use **different column names for the same concept**. This creates integration problems:

```python
# From Ensembl GTF parser:
df1 = pl.DataFrame({'site_type': ['donor'], 'seqname': ['chr1']})

# From MANE GFF3 parser:
df2 = pl.DataFrame({'splice_type': ['donor'], 'chrom': ['chr1']})

# From internal processing:
df3 = pl.DataFrame({'type': ['donor'], 'chromosome': ['chr1']})
```

Without standardization, downstream code must handle all variants:

```python
# ❌ FRAGILE: Must check multiple column names
if 'splice_type' in df.columns:
    col = 'splice_type'
elif 'site_type' in df.columns:
    col = 'site_type'
elif 'type' in df.columns:
    col = 'type'
```

With standardization, code is clean and consistent:

```python
# ✅ ROBUST: Always use standard name
df = standardize_splice_sites_schema(df)
col = 'splice_type'  # Guaranteed to exist
```

---

### 2. Synonymous Column Name Mappings

#### Splice Sites Schema

| Non-Standard (Input) | Standard (Output) | Origin |
|---------------------|-------------------|--------|
| `site_type` | `splice_type` | GTF convention → biological terminology |
| `type` | `splice_type` | Generic → specific |

#### Gene Features Schema

| Non-Standard (Input) | Standard (Output) | Origin |
|---------------------|-------------------|--------|
| `seqname` | `chrom` | GTF convention → genomics convention |
| `gene_type` | `gene_biotype` | Alternative naming |
| `biotype` | `gene_biotype` | Short form → full form |

#### Transcript Features Schema

| Non-Standard (Input) | Standard (Output) | Origin |
|---------------------|-------------------|--------|
| `seqname` | `chrom` | GTF convention |
| `transcript_type` | `transcript_biotype` | Alternative naming |
| `biotype` | `transcript_biotype` | Short form |

#### Exon Features Schema

| Non-Standard (Input) | Standard (Output) | Origin |
|---------------------|-------------------|--------|
| `seqname` | `chrom` | GTF convention |

---

### 3. Standardization Functions

| Function | Purpose |
|----------|---------|
| `standardize_splice_sites_schema(df)` | Standardize splice site annotations |
| `standardize_gene_features_schema(df)` | Standardize gene features |
| `standardize_transcript_features_schema(df)` | Standardize transcript features |
| `standardize_exon_features_schema(df)` | Standardize exon features |
| `standardize_all_schemas(...)` | Batch standardize multiple datasets |
| `get_standard_column_mapping(schema_type)` | Get mapping dict for a schema type |
| `print_standard_schemas()` | Print all mappings for reference |

#### Design Principles

1. **Non-destructive**: Columns are renamed, not replaced
2. **Idempotent**: Safe to call multiple times
3. **Flexible**: Works with both Polars and Pandas DataFrames
4. **Conflict-aware**: Only renames if target column doesn't exist

---

### 4. Code Example: Using `standardize_splice_sites_schema()`

```python
import polars as pl
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema

# Input data with non-standard column names
df = pl.DataFrame({
    'chrom': ['1', '1', '2'],
    'position': [1000, 2000, 3000],
    'site_type': ['donor', 'acceptor', 'donor'],  # ← Non-standard
    'gene_id': ['ENSG00000000001', 'ENSG00000000001', 'ENSG00000000002']
})

print("Before standardization:")
print(df.columns)
# ['chrom', 'position', 'site_type', 'gene_id']

# Standardize the schema
df_standardized = standardize_splice_sites_schema(df, verbose=True)
# [schema] Standardizing splice_sites columns:
#   site_type → splice_type

print("\nAfter standardization:")
print(df_standardized.columns)
# ['chrom', 'position', 'splice_type', 'gene_id']

# Now downstream code can reliably use 'splice_type'
donors = df_standardized.filter(pl.col('splice_type') == 'donor')
```

#### Batch Standardization

```python
from meta_spliceai.system.genomic_resources import standardize_all_schemas

# Standardize multiple datasets at once
result = standardize_all_schemas(
    splice_sites=ss_df,
    gene_features=gf_df,
    verbose=True
)

standardized_ss = result['splice_sites']
standardized_gf = result['gene_features']
```

---

### 5. When Must Schema Standardization Be Called?

Schema standardization is called **immediately after loading splice site annotations**, before any processing begins.

#### Location in Workflow

```python
# In splice_prediction_workflow.py, lines 296-306:

# CRITICAL: Standardize splice site schema for consistency across the system
# This handles synonymous column names (e.g., site_type → splice_type)
from meta_spliceai.system.genomic_resources import standardize_splice_sites_schema

if verbosity >= 1:
    print_with_indent("[schema] Standardizing splice site annotations before workflow...", indent_level=1)
    print_with_indent(f"[schema] Columns before: {ss_annotations_df.columns}", indent_level=2)

ss_annotations_df = standardize_splice_sites_schema(
    ss_annotations_df, 
    verbose=True  # Always verbose for debugging
)
```

#### Workflow Position

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                         WORKFLOW SEQUENCE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   1. prepare_gene_annotations()                                             │
│      └── Returns: gene_annotations_df                                       │
│                                                                             │
│   2. prepare_splice_site_annotations()                                      │
│      └── Returns: ss_annotations_df (may have 'site_type' column)          │
│                                                                             │
│   3. ★ standardize_splice_sites_schema(ss_annotations_df) ★                │
│      └── Renames: 'site_type' → 'splice_type'                              │
│      └── Returns: ss_annotations_df (now has 'splice_type' column)         │
│                                                                             │
│   4. prepare_genomic_sequences()                                            │
│      └── Uses standardized ss_annotations_df                               │
│                                                                             │
│   5. Prediction loop                                                        │
│      └── All code expects 'splice_type' column                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why This Timing is Critical

1. **Before any filtering**: Code that filters by splice type needs consistent column name
2. **Before any joins**: Joining datasets requires matching column names
3. **Before prediction loop**: The nested loop expects standardized schema
4. **After loading, before processing**: Ensures all downstream code sees consistent schema

---

### 6. Porting Considerations

#### Must Port

| Component | Reason |
|-----------|--------|
| `schema.py` | Core standardization logic |
| Column mappings | Define synonyms for each schema type |
| `standardize_splice_sites_schema()` | Used in main workflow |

#### Integration Point

In the ported workflow, add standardization call immediately after loading splice sites:

```python
# After loading splice site annotations
ss_annotations_df = prepare_splice_site_annotations(...)

# CRITICAL: Standardize before any processing
from agentic_spliceai.splice_engine.genomic_resources import standardize_splice_sites_schema
ss_annotations_df = standardize_splice_sites_schema(ss_annotations_df)
```

---

## Stage 4, Step 3: Standard Directory Layout

### Complete Directory Tree

```text
data/
│
├── ════════════════════════════════════════════════════════════════════════════
│   BASE MODEL DATA (Build-Specific Reference Data)
│   ════════════════════════════════════════════════════════════════════════════
│
├── ensembl/                                    # Ensembl annotation source
│   │
│   └── GRCh37/                                 # ★ SPLICEAI DATA ★
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ INPUT FILES (User Provides)                                    │
│       │   └─────────────────────────────────────────────────────────────────┘
│       ├── Homo_sapiens.GRCh37.87.gtf                    # Ensembl GTF
│       ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa   # Reference genome
│       ├── Homo_sapiens.GRCh37.dna.primary_assembly.fa.fai  # FASTA index
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ DERIVED FILES (System Generates)                               │
│       │   └─────────────────────────────────────────────────────────────────┘
│       ├── annotations.db                                # SQLite database
│       ├── splice_sites_enhanced.tsv                     # Ground truth splice sites
│       ├── gene_features.tsv                             # Gene-level features
│       ├── gene_chromosome_map.tsv                       # Gene→chromosome mapping
│       ├── chromosome_sizes.tsv                          # Chromosome lengths
│       │
│       ├── gene_sequence_1.parquet                       # Per-chromosome sequences
│       ├── gene_sequence_2.parquet
│       ├── gene_sequence_*.parquet
│       │   └── ... (one per chromosome)
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ OUTPUT ARTIFACTS (Workflow Produces)                           │
│       │   └─────────────────────────────────────────────────────────────────┘
│       └── spliceai_eval/                                # SpliceAI outputs
│           ├── gene_manifest.json                        # Processing manifest
│           └── meta_models/                              # Meta-model artifacts
│               ├── splice_positions_1_chunk_1_500.tsv    # Per-chunk predictions
│               ├── error_analysis_1_chunk_1_500.tsv      # Per-chunk errors
│               ├── analysis_sequences_1_chunk_1_500.tsv  # Per-chunk sequences
│               ├── ... (more chunks)
│               │
│               ├── splice_positions_aggregated.tsv       # Aggregated predictions
│               ├── error_analysis_aggregated.tsv         # Aggregated errors
│               ├── analysis_sequences_aggregated.tsv     # Aggregated sequences
│               │
│               └── inference/                            # Inference outputs
│                   ├── predictions_<run_id>.parquet
│                   └── delta_scores_<run_id>.parquet
│
├── mane/                                       # MANE annotation source
│   │
│   └── GRCh38/                                 # ★ OPENSPLICEAI DATA ★
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ INPUT FILES (User Provides)                                    │
│       │   └─────────────────────────────────────────────────────────────────┘
│       ├── MANE.GRCh38.v1.3.refseq_genomic.gff           # MANE GFF3 (not GTF!)
│       ├── GCF_000001405.40_GRCh38.p14_genomic.fna       # RefSeq genome
│       ├── GCF_000001405.40_GRCh38.p14_genomic.fna.fai   # FASTA index
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ DERIVED FILES (System Generates)                               │
│       │   └─────────────────────────────────────────────────────────────────┘
│       ├── annotations.db                                # SQLite database
│       ├── splice_sites_enhanced.tsv                     # Ground truth splice sites
│       ├── gene_features.tsv                             # Gene-level features
│       ├── gene_chromosome_map.tsv                       # Gene→chromosome mapping
│       ├── chromosome_sizes.tsv                          # Chromosome lengths
│       │
│       ├── gene_sequence_1.parquet                       # Per-chromosome sequences
│       ├── gene_sequence_2.parquet
│       ├── gene_sequence_*.parquet
│       │   └── ... (one per chromosome)
│       │
│       │   ┌─────────────────────────────────────────────────────────────────┐
│       │   │ OUTPUT ARTIFACTS (Workflow Produces)                           │
│       │   └─────────────────────────────────────────────────────────────────┘
│       └── openspliceai_eval/                            # OpenSpliceAI outputs
│           ├── gene_manifest.json                        # Processing manifest
│           └── meta_models/                              # Meta-model artifacts
│               ├── splice_positions_1_chunk_1_500.tsv
│               ├── error_analysis_1_chunk_1_500.tsv
│               ├── analysis_sequences_1_chunk_1_500.tsv
│               ├── ... (more chunks)
│               │
│               ├── splice_positions_aggregated.tsv
│               ├── error_analysis_aggregated.tsv
│               ├── analysis_sequences_aggregated.tsv
│               │
│               └── inference/
│                   ├── predictions_<run_id>.parquet
│                   └── delta_scores_<run_id>.parquet
│
├── ════════════════════════════════════════════════════════════════════════════
│   MODEL WEIGHTS (Pre-trained Neural Networks)
│   ════════════════════════════════════════════════════════════════════════════
│
├── models/
│   │
│   ├── spliceai/                               # SpliceAI ensemble (Keras)
│   │   ├── spliceai_1.h5                       # Model 1 of 5
│   │   ├── spliceai_2.h5
│   │   ├── spliceai_3.h5
│   │   ├── spliceai_4.h5
│   │   ├── spliceai_5.h5
│   │   └── config.json                         # Model configuration
│   │
│   └── openspliceai/                           # OpenSpliceAI ensemble (PyTorch)
│       ├── openspliceai_1.pt                   # Model 1 of 5
│       ├── openspliceai_2.pt
│       ├── openspliceai_3.pt
│       ├── openspliceai_4.pt
│       ├── openspliceai_5.pt
│       └── config.json                         # Model configuration
│
├── ════════════════════════════════════════════════════════════════════════════
│   TRAINING/TEST DATASETS (Meta-Model Training)
│   ════════════════════════════════════════════════════════════════════════════
│
├── train_pc_7000_3mers_opt/                    # Production training set
│   ├── master/
│   │   ├── training_data.parquet               # Main training data
│   │   └── feature_manifest.csv                # Feature descriptions
│   └── README.md
│
├── train_pc_100_3mers_diverse/                 # Quick testing set
│   └── ...
│
└── test_pc_1000_3mers/                         # Held-out test set
    └── ...
```

---

### Summary: Where Data Lives

| Data Category | SpliceAI (GRCh37) | OpenSpliceAI (GRCh38) |
|---------------|-------------------|----------------------|
| **Input Files** | `data/ensembl/GRCh37/` | `data/mane/GRCh38/` |
| **Derived Files** | `data/ensembl/GRCh37/` | `data/mane/GRCh38/` |
| **Model Weights** | `data/models/spliceai/` | `data/models/openspliceai/` |
| **Output Artifacts** | `data/ensembl/GRCh37/spliceai_eval/` | `data/mane/GRCh38/openspliceai_eval/` |
| **Meta-Model Data** | `.../spliceai_eval/meta_models/` | `.../openspliceai_eval/meta_models/` |

---

### Why This Structure Prevents Data Mixing

#### 1. Physical Separation by Build

```text
data/
├── ensembl/GRCh37/    ← SpliceAI ONLY (GRCh37 coordinates)
└── mane/GRCh38/       ← OpenSpliceAI ONLY (GRCh38 coordinates)
```

**Key insight**: It's **physically impossible** to accidentally load GRCh37 splice sites when working in the `mane/GRCh38/` directory.

#### 2. Annotation Source in Path

The path includes the annotation source, preventing mixing of different annotation systems:

| Path Component | Meaning |
|----------------|---------|
| `ensembl/` | Ensembl annotations (GTF format) |
| `mane/` | MANE RefSeq annotations (GFF3 format) |
| `gencode/` | GENCODE annotations (GTF format) |

#### 3. Model-Specific Output Directories

Each base model has its own evaluation directory:

```text
data/ensembl/GRCh37/spliceai_eval/       ← SpliceAI outputs only
data/mane/GRCh38/openspliceai_eval/      ← OpenSpliceAI outputs only
```

This prevents mixing predictions from different models.

#### 4. Registry Enforces Correctness

The `Registry` class automatically resolves to the correct directory:

```python
# SpliceAI → data/ensembl/GRCh37/
registry = Registry(build='GRCh37')
gtf = registry.get_gtf_path()  # Guaranteed GRCh37 GTF

# OpenSpliceAI → data/mane/GRCh38/
registry = Registry(build='GRCh38_MANE')
gtf = registry.get_gtf_path()  # Guaranteed GRCh38 MANE GFF3
```

---

### Consequences of Data Mixing

| Model | Correct Data | Wrong Data | PR-AUC Drop |
|-------|--------------|------------|-------------|
| SpliceAI | GRCh37 (0.97) | GRCh38 (0.541) | **-44%** |
| OpenSpliceAI | GRCh38 MANE (0.98) | GRCh37 (0.523) | **-47%** |

**Root cause**: Genomic coordinates differ between builds. A splice site at position 12345 in GRCh37 is at a **different position** in GRCh38.

---

### Directory Naming Conventions

#### Build Names

| Name | Annotation Source | Format | Used By |
|------|-------------------|--------|---------|
| `GRCh37` | Ensembl | GTF | SpliceAI |
| `GRCh38` | Ensembl | GTF | General use |
| `GRCh38_MANE` | MANE RefSeq | GFF3 | OpenSpliceAI |

#### Dataset Naming

```text
{purpose}_{gene_type}_{count}_{features}_{version}
```

| Component | Values | Example |
|-----------|--------|---------|
| Purpose | `train`, `test`, `eval` | `train` |
| Gene Type | `pc` (protein-coding), `nc`, `all` | `pc` |
| Count | Number of genes | `7000` |
| Features | Feature set | `3mers` |
| Version | Variant | `opt` |

**Example**: `train_pc_7000_3mers_opt` = Training, protein-coding, 7000 genes, 3-mers, optimized

---

### Porting Considerations

#### Must Replicate

1. **Directory structure**: `data/<annotation_source>/<build>/`
2. **Model-specific eval dirs**: `<model>_eval/meta_models/`
3. **Registry path resolution**: Automatic mapping from build name to directory

#### Configuration

In `genomic_resources.yaml`:

```yaml
builds:
  GRCh37:
    annotation_source: ensembl
    gtf: "Homo_sapiens.GRCh37.{release}.gtf"
    
  GRCh38_MANE:
    annotation_source: mane
    gtf: "MANE.GRCh38.v{release}.refseq_genomic.gff"
```

---

**Next**: Stage 5 - Identify Essential vs Optional Components
