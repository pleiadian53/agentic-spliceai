# Base Layer Porting Documentation

This directory contains detailed documentation for porting the meta-spliceai base layer to agentic-spliceai.

---

## Overview

The base layer is a **model-agnostic framework** for splice site prediction that:
- Runs predictions with any compatible base model (SpliceAI, OpenSpliceAI, custom models)
- Evaluates predictions against reference annotations (TP/FP/FN/TN)
- Extracts training features for meta-learning
- Manages artifacts with intelligent checkpointing
- Uses memory-efficient mini-batch processing

---

## Porting Process (6 Stages)

We follow a systematic approach outlined in the meta-spliceai AI Agent Porting Guide:

### ✅ Stage 1: Understand Entry Points
**Status**: Complete  
**Document**: [STAGE_1_ENTRY_POINTS_ANALYSIS.md](STAGE_1_ENTRY_POINTS_ANALYSIS.md)

**Key Findings**:
- Both Python API and CLI converge to `run_enhanced_splice_prediction_workflow()`
- Configuration uses ABC pattern: `BaseModelConfig` → `SpliceAIConfig`, `OpenSpliceAIConfig`
- Entry points are thin wrappers (~200-400 lines)

**Deliverables**:
- Entry point flow diagrams (Mermaid)
- Configuration architecture diagram
- Parameter mapping tables
- Convergence point identification

---

### ⏳ Stage 2: Trace Core Workflow
**Status**: Pending  
**Document**: `STAGE_2_CORE_WORKFLOW_ANALYSIS.md` (to be created)

**Goals**:
- Analyze `run_enhanced_splice_prediction_workflow()`
- Understand data preparation steps
- Map the processing loop structure (chunks → mini-batches)
- Identify evaluation logic
- Understand artifact management

---

### ⏳ Stage 3: Map Data Dependencies
**Status**: Pending  
**Document**: `STAGE_3_DATA_DEPENDENCIES.md` (to be created)

**Goals**:
- Identify required data files (FASTA, GTF, model weights)
- Understand derived files (splice sites, sequences, databases)
- Map data flow through the system
- Document file formats and schemas

---

### ⏳ Stage 4: Understand Genomic Resources System
**Status**: Pending  
**Document**: `STAGE_4_GENOMIC_RESOURCES.md` (to be created)

**Goals**:
- Understand the Registry system for path resolution
- Learn schema standardization (column name mapping)
- Document directory conventions
- Understand build-specific paths (GRCh37 vs GRCh38)

---

### ⏳ Stage 5: Identify Essential vs Optional Components
**Status**: Pending  
**Document**: `STAGE_5_COMPONENT_ANALYSIS.md` (to be created)

**Goals**:
- List essential files (~20 core files)
- Identify optional components
- Create dependency graph
- Prioritize porting order

---

### ⏳ Stage 6: Create Minimal Port
**Status**: Pending  
**Document**: `STAGE_6_IMPLEMENTATION.md` (to be created)

**Goals**:
- Port essential components to agentic-spliceai
- Adapt imports and structure
- Create entry points
- Verify end-to-end functionality

---

## Quick Reference

### Key Files in meta-spliceai

**Entry Points**:
- `meta_spliceai/run_base_model.py` - Python API
- `meta_spliceai/cli/run_base_model_cli.py` - CLI

**Core Workflow**:
- `meta_spliceai/splice_engine/meta_models/workflows/splice_prediction_workflow.py`
- `meta_spliceai/splice_engine/meta_models/workflows/data_preparation.py`

**Configuration**:
- `meta_spliceai/splice_engine/meta_models/core/model_config.py`
- `meta_spliceai/splice_engine/meta_models/core/data_types.py`

**Genomic Resources**:
- `meta_spliceai/system/genomic_resources/registry.py`
- `meta_spliceai/system/genomic_resources/schema.py`

### Key Concepts

**Mini-Batching**:
- Outer loop: Chromosomes
- Middle loop: 500-gene chunks
- Inner loop: 50-gene mini-batches

**Model-Agnostic Design**:
- Works with any model producing per-nucleotide splice scores
- Polymorphic configuration via `BaseModelConfig` ABC

**Checkpointing**:
- Chunk-level artifacts
- Resume from interruption
- Production vs. test modes

---

## Diagrams

All stage documents include Mermaid diagrams for:
- Flow diagrams
- Sequence diagrams
- Class hierarchies
- Data flow

View them in any Markdown viewer that supports Mermaid (GitHub, VS Code, etc.)

---

## Progress Tracking

| Stage | Status | Document | Key Deliverable |
|-------|--------|----------|-----------------|
| 1 | ✅ Complete | [STAGE_1](STAGE_1_ENTRY_POINTS_ANALYSIS.md) | Entry point analysis |
| 2 | ⏳ Pending | TBD | Workflow trace |
| 3 | ⏳ Pending | TBD | Data dependency map |
| 4 | ⏳ Pending | TBD | Genomic resources guide |
| 5 | ⏳ Pending | TBD | Component list |
| 6 | ⏳ Pending | TBD | Working implementation |

---

## Related Documentation

**In meta-spliceai**:
- `docs/base_models/AI_AGENT_PORTING_GUIDE.md` - Systematic porting guide
- `docs/base_models/BASE_LAYER_INTEGRATION_GUIDE.md` - Integration guide
- `docs/data/DATA_LAYOUT_MASTER_GUIDE.md` - Data organization

**In agentic-spliceai**:
- `docs/SPLICE_PREDICTION_GUIDE.md` - User guide (wrapper approach)
- `docs/base_layer/BASE_LAYER_INTEGRATION_SUMMARY.md` - Integration summary

---

## Notes

- **Genomic Data**: Symlinked from meta-spliceai to `agentic-spliceai/data/`
- **Environment**: Use `agentic-spliceai` mamba environment
- **Testing**: Test each stage incrementally before proceeding

---

**Last Updated**: November 27, 2025  
**Current Stage**: 1 (Complete)  
**Next Stage**: 2 (Trace Core Workflow)
