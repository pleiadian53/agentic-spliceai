# Agentic-SpliceAI Documentation

Central documentation hub for **Agentic-SpliceAI** — Context-Aware Novel Isoform Discovery for Drug Target Identification.

---

## 📁 Topic Index

### Getting Started

| Document | Description |
| --- | --- |
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5 minutes |
| [SETUP.md](SETUP.md) | Full environment setup |
| [getting_started/RUNPODS_SETUP.md](getting_started/RUNPODS_SETUP.md) | Cloud GPU (RunPods) workflow |
| [getting_started/LATEX_SETUP.md](getting_started/LATEX_SETUP.md) | LaTeX / PDF generation setup |

### Architecture

| Document | Description |
| --- | --- |
| [architecture/README.md](architecture/README.md) | System architecture overview |
| [architecture/STRUCTURE.md](architecture/STRUCTURE.md) | Full project directory structure |
| [architecture/PACKAGE_ORGANIZATION.md](architecture/PACKAGE_ORGANIZATION.md) | Package organization guide |

### System Design

| Document | Description |
| --- | --- |
| [system_design/README.md](system_design/README.md) | System design overview |
| [system_design/resource_management.md](system_design/resource_management.md) | Genomic resource registry |
| [system_design/output_management.md](system_design/output_management.md) | Output & artifact management |
| [system_design/configuration_system.md](system_design/configuration_system.md) | Configuration (Pydantic) |
| [system_design/base_layer_architecture.md](system_design/base_layer_architecture.md) | Base layer architecture |

### Base Layer

| Document | Description |
| --- | --- |
| [base_layer/DATA_PREPARATION_CLI.md](base_layer/DATA_PREPARATION_CLI.md) | Data preparation CLI |
| [base_layer/PROCESSING_ARCHITECTURE.md](base_layer/PROCESSING_ARCHITECTURE.md) | Chunking & processing architecture |
| [base_layer/POSITION_COORDINATE_SYSTEMS.md](base_layer/POSITION_COORDINATE_SYSTEMS.md) | Position / coordinate system reference |

### Meta Layer

| Document | Description |
| --- | --- |
| [meta_layer/README.md](meta_layer/README.md) | Meta layer overview |
| [meta_layer/ARCHITECTURE.md](meta_layer/ARCHITECTURE.md) | Foundation-Adaptor architecture |
| [meta_layer/methods/README.md](meta_layer/methods/README.md) | Methods overview |
| [meta_layer/methods/ROADMAP.md](meta_layer/methods/ROADMAP.md) | Development roadmap |
| [meta_layer/methods/PAIRED_DELTA_PREDICTION.md](meta_layer/methods/PAIRED_DELTA_PREDICTION.md) | Paired delta prediction method |
| [meta_layer/methods/VALIDATED_DELTA_PREDICTION.md](meta_layer/methods/VALIDATED_DELTA_PREDICTION.md) | Validated delta prediction method |
| [meta_layer/experiments/README.md](meta_layer/experiments/README.md) | Experiments index |

### Foundation Models

| Document | Description |
| --- | --- |
| [foundation_models/README.md](foundation_models/README.md) | Foundation models overview |
| [foundation_models/evo2/junction_support_labels.md](foundation_models/evo2/junction_support_labels.md) | Evo2 junction support labels |
| [foundation_models/training/deepspeed_training.md](foundation_models/training/deepspeed_training.md) | DeepSpeed training guide |

### Isoform Discovery & Variant Analysis

| Document | Description |
| --- | --- |
| [isoform_discovery/README.md](isoform_discovery/README.md) | Novel isoform discovery overview |
| [variant_analysis/README.md](variant_analysis/README.md) | Variant analysis overview |

### Agentic Workflows

| Document | Description |
| --- | --- |
| [agency/README.md](agency/README.md) | Agentic workflow overview |
| [agency/AGENTIC_MEMORY_TUTORIAL.md](agency/AGENTIC_MEMORY_TUTORIAL.md) | Memory system tutorial |
| [agency/MEMORY_PATTERNS.md](agency/MEMORY_PATTERNS.md) | Memory design patterns |

### Tutorials & Workflows

| Document | Description |
| --- | --- |
| [tutorials/README.md](tutorials/README.md) | Tutorials overview |
| [tutorials/SPLICE_PREDICTION_GUIDE.md](tutorials/SPLICE_PREDICTION_GUIDE.md) | Splice site prediction guide |
| [workflows/README.md](workflows/README.md) | Analysis workflows |

### Domain Knowledge

| Document | Description |
| --- | --- |
| [biology/README.md](biology/README.md) | Splice site biology background |
| [bioinformatics/README.md](bioinformatics/README.md) | Bioinformatics methods |

### API Reference

| Document | Description |
| --- | --- |
| [api/README.md](api/README.md) | REST & Python API reference |

---

## 📖 Documentation Conventions

- **`docs/`** — Official, centralized documentation (mkdocs + Obsidian). All new tutorial and reference notes go here.
- **Package-level `docs/`** (e.g., `src/agentic_spliceai/docs/`) — Temporary working notes close to the code. Promoted to `docs/<topic>/` when mature.
- **`dev/`** — Private development notes, session logs, refactoring plans. Not in mkdocs or Obsidian.

---

## 🎯 Find What You Need

| I want to... | Go to |
|---|---|
| Get running fast | [QUICKSTART.md](QUICKSTART.md) |
| Understand the pipeline | [architecture/README.md](architecture/README.md) |
| Learn about the base layer | [base_layer/](base_layer/) |
| Understand meta-learning | [meta_layer/](meta_layer/) |
| Work with foundation models | [foundation_models/](foundation_models/) |
| Discover novel isoforms | [isoform_discovery/README.md](isoform_discovery/README.md) |
| Use agentic validation | [agency/README.md](agency/README.md) |
| Follow a tutorial | [tutorials/SPLICE_PREDICTION_GUIDE.md](tutorials/SPLICE_PREDICTION_GUIDE.md) |
| Read splice biology | [biology/README.md](biology/README.md) |
| Use the API | [api/README.md](api/README.md) |

---

**Last Updated:** February 2026
