# Agentic-SpliceAI — Project Structure

**Complete overview of the agentic-spliceai project organization**

> **Last Updated**: March 2026

## 📁 Directory Structure

```
agentic-spliceai/
├── src/                             # Core production code (pip installable)
│   ├── agentic_spliceai/
│   │   ├── splice_engine/           # 🧬 Core splice prediction engine
│   │   │   ├── config/                  # Configuration management
│   │   │   │   ├── genomic_config.py        # Config dataclass & loader
│   │   │   │   └── settings.yaml            # Default settings
│   │   │   ├── resources/               # Genomic resource management
│   │   │   │   ├── registry.py              # Path resolution for GTF/FASTA/models
│   │   │   │   └── schema.py                # Column standardization (splice_type, chrom)
│   │   │   ├── utils/                   # Shared utilities
│   │   │   │   ├── dataframe.py             # DataFrame operations
│   │   │   │   ├── display.py               # Printing & formatting
│   │   │   │   ├── filesystem.py            # File I/O helpers
│   │   │   │   └── memory_monitor.py        # Background RSS monitor + graceful abort
│   │   │   ├── base_layer/              # Base model predictions
│   │   │   │   ├── models/                  # Model configs + runner
│   │   │   │   ├── prediction/              # Core prediction logic
│   │   │   │   ├── workflows/               # Chunked prediction pipeline
│   │   │   │   ├── io/                      # Artifact management
│   │   │   │   └── data/                    # Data types & preparation
│   │   │   ├── features/                # 🎨 Multimodal feature engineering
│   │   │   │   ├── pipeline.py              # FeaturePipeline (dependency resolution)
│   │   │   │   ├── workflow.py              # FeatureWorkflow (genome-scale)
│   │   │   │   ├── modality.py              # Modality protocol (ABC)
│   │   │   │   ├── sampling.py              # Early position sampling
│   │   │   │   ├── verification.py          # Position alignment verification
│   │   │   │   └── modalities/              # 9 registered modalities:
│   │   │   │       ├── base_scores.py           # 43 engineered features
│   │   │   │       ├── annotation.py            # Ground truth labels (3)
│   │   │   │       ├── sequence.py              # DNA context via pyfaidx (3)
│   │   │   │       ├── genomic.py               # GC content, CpG, dinucs (4)
│   │   │   │       ├── conservation.py          # PhyloP/PhastCons bigWig (9)
│   │   │   │       ├── epigenetic.py            # H3K36me3/H3K4me3 ChIP-seq (12)
│   │   │   │       ├── junction.py              # GTEx RNA-seq junctions (12)
│   │   │   │       ├── rbp_eclip.py             # ENCODE RBP eCLIP binding (8)
│   │   │   │       └── chrom_access.py          # ENCODE ATAC-seq accessibility (6)
│   │   │   ├── eval/                    # 📊 Cross-layer evaluation
│   │   │   │   ├── metrics.py               # TP/FP/FN, sensitivity, specificity
│   │   │   │   ├── splitting.py             # Chromosome-based train/test splits
│   │   │   │   ├── calibration.py           # ECE, reliability curves
│   │   │   │   ├── output.py                # EvaluationOutputWriter
│   │   │   │   └── display.py               # Result visualization
│   │   │   ├── data/                    # Cross-layer data utilities
│   │   │   │   └── sampling.py              # Balanced train/test sampling
│   │   │   ├── meta_layer/              # 🧠 Meta-learning layer
│   │   │   │   ├── core/                    # Configuration & schema
│   │   │   │   │   ├── config.py                # MetaLayerConfig
│   │   │   │   │   └── feature_schema.py        # Feature definitions (9 column groups)
│   │   │   │   ├── models/                  # Neural network models
│   │   │   │   ├── training/                # Training pipeline
│   │   │   │   └── workflows/               # Meta-layer workflows
│   │   │   └── cli/                     # CLI entry points
│   │   │       ├── predict.py               # agentic-spliceai-predict
│   │   │       └── prepare.py               # agentic-spliceai-prepare
│   │   ├── agents/                      # 🤖 Agentic workflows (Phase 7)
│   │   ├── server/                      # FastAPI splice service
│   │   └── analysis/                    # Analysis tools & templates
│   │
│   └── nexus/                           # 📚 Research agent package
│       ├── agents/                          # Multi-agent pipeline
│       │   ├── research/                        # Research orchestrator
│       │   ├── planner/                         # Research planning
│       │   ├── researcher/                      # Information gathering
│       │   ├── writer/                          # Report writing
│       │   └── editor/                          # Report refinement
│       ├── core/                            # Core utilities
│       ├── cli/                             # CLI interface
│       └── templates/                       # Report templates
│
├── foundation_models/               # 🔬 Experimental sub-project (own pyproject.toml)
│   ├── foundation_models/
│   │   ├── evo2/                        # Evo2-based exon classifier
│   │   ├── classifiers/                 # Splice classifiers
│   │   └── utils/                       # Quantization, chunking
│   ├── configs/                         # GPU + SkyPilot configs
│   └── docs/                            # Sub-project documentation
│
├── server/                          # 🌐 Standalone FastAPI services
│   ├── bio/                             # Bioinformatics Lab UI (port 8005)
│   ├── splice_service/                  # Splice prediction API (port 8004)
│   └── chart_service/                   # Chart/viz API (port 8003)
│
├── examples/                        # 📖 Learning path examples
│   ├── base_layer/                      # 5 scripts: prediction → precomputation
│   ├── features/                        # Multimodal feature engineering
│   │   ├── 06_multimodal_genome_workflow.py  # YAML-driven genome-scale workflow
│   │   ├── configs/                     # 4 YAML profiles
│   │   ├── docs/                        # Per-modality tutorials
│   │   └── verify_feature_alignment.py  # Position alignment verification
│   ├── meta_layer/                      # Meta-layer training scripts
│   │   ├── 01_xgboost_baseline.py       # M1 XGBoost baseline
│   │   ├── 02_calibration_analysis.py   # Calibration evaluation
│   │   ├── 03_modality_ablation.py      # Leave-one-out ablation + SHAP
│   │   └── docs/                        # M1-M4 model variants guide
│   ├── foundation_models/              # 7+ scripts: resource check → orchestrate
│   └── data_preparation/               # Data prep & ground truth generation
│
├── scripts/                         # Utility scripts
│   ├── aggregate_gtex_junctions.py      # GTEx v8 junction aggregation
│   └── aggregate_eclip_peaks.py         # ENCODE eCLIP peak aggregation
│
├── data/                            # Data directory (symlinked, not in git)
│   ├── ensembl/GRCh37/                  # Ensembl annotations
│   └── mane/GRCh38/                     # MANE annotations + derived data
│       ├── junction_data/                   # GTEx junction parquets
│       ├── rbp_data/                        # ENCODE eCLIP parquets
│       └── openspliceai_eval/               # Predictions + feature artifacts
│
├── notebooks/                       # Jupyter analysis & demos
├── docs/                            # Public documentation (MkDocs)
├── dev/                             # Private development notes
│   ├── sessions/                        # Date-stamped session logs
│   ├── planning/                        # Roadmap, wishlist
│   ├── tasks/                           # todo.md, lessons.md
│   └── refactoring/                     # Refactoring plans & logs
├── tests/                           # Unit & integration tests
└── pyproject.toml                   # Package configuration
```

## 📚 Documentation Hierarchy

### 1. Public Documentation (`docs/`)

Topic-based documentation for users and contributors. Organized by architectural layer and concern:

| Directory | Content |
|-----------|---------|
| `docs/architecture/` | System design, package organization, structure |
| `docs/system_design/` | Resource management, configuration, output patterns |
| `docs/base_layer/` | Data preparation, processing architecture, coordinates |
| `docs/meta_layer/` | Foundation-Adaptor architecture, methods, experiments |
| `docs/multimodal_feature_engineering/` | Feature catalog (100 columns, 9 modalities) |
| `docs/applications/` | Domain-specific workflows (oncology, VUS, neurology) |
| `docs/api/` | REST endpoints, data format, configuration |
| `docs/tutorials/` | Splice prediction guide |
| `docs/foundation_models/` | Evo2, SpliceBERT, DeepSpeed training |
| `docs/biology/` | Splice site biology background |

### 2. Per-Modality Tutorials (`examples/features/docs/`)

Detailed tutorials for each external data modality:
- `epigenetic-marks-tutorial.md` — H3K36me3/H3K4me3 ChIP-seq
- `rbp-eclip-tutorial.md` — ENCODE RBP eCLIP binding
- `chromatin-accessibility-tutorial.md` — ENCODE ATAC-seq

### 3. Private Development Notes (`dev/`)

Session logs, refactoring plans, task tracking. Not published.

## 🔧 Key Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, CLI entry points, linting config |
| `environment.yml` | Mamba/Conda environment definition |
| `src/.../config/settings.yaml` | Default genomic settings (builds, chromosomes, paths) |
| `examples/features/configs/*.yaml` | Feature engineering workflow profiles |
| `foundation_models/configs/gpu_config.yaml` | GPU infrastructure defaults |

## 🎯 CLI Entry Points

Defined in `pyproject.toml [project.scripts]`:

| Command | Purpose |
|---------|---------|
| `agentic-spliceai` | Main CLI |
| `agentic-spliceai-predict` | Splice site prediction |
| `agentic-spliceai-prepare` | Data preparation |
| `agentic-spliceai-server` | FastAPI service |
| `nexus` | Research agent CLI |
| `nexus-server` | Research agent server |

## 📦 Installation

```bash
# Core only (local development)
pip install -e ".[dev]"

# Run predictions
agentic-spliceai-predict --genes BRCA1 TP53

# Foundation models (pod with GPU)
pip install -e ./foundation_models
```

## 🔗 Related Documentation

- [Package Organization](PACKAGE_ORGANIZATION.md) — Experimental package strategy
- [Architecture Overview](README.md) — Three-layer pipeline diagram
- [System Design](../system_design/README.md) — Design principles and patterns
- [Roadmap](../ROADMAP.md) — Development phases and status
