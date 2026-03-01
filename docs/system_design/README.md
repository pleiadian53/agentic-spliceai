# System Design Documentation

**Purpose**: Architectural principles and design patterns for agentic-spliceai  
**Audience**: Developers, contributors, and future maintainers  
**Status**: Living documentation - updated during refactoring

---

## Overview

This directory documents the **architectural foundations** of agentic-spliceai. These are design decisions that affect the entire system and should be understood before making significant changes.

System design documentation differs from implementation details:
- **System design** (here): Why we made architectural choices, the principles behind them
- **Implementation**: How we're implementing those designs during refactoring (see source code)

---

## Core System Design Areas

### 1. [Resource Management](resource_management.md)

**Why it matters**: All code needs to access data, models, and genomic resources consistently

**What it covers**:
- Centralized path resolution
- Project root detection
- Data directory structure
- Configuration-driven resource access
- Environment-agnostic portability

**Key principle**: Single source of truth for all paths

---

### 2. [Output & Artifact Management](output_management.md)

**Why it matters**: Research code produces models, predictions, metrics, and experimental artifacts

**What it covers**:
- Artifact lifecycle management
- Experiment organization
- Reproducibility tracking
- Mode-based isolation (production/dev/test)
- Immutability policies

**Key principle**: Systematic organization prevents "where did I save that?" chaos

---

### 3. [Configuration System](configuration_system.md)

**Why it matters**: Supports multiple genome builds, base models, and execution environments

**What it covers**:
- YAML-based configuration
- Environment variable overrides
- Type-safe config with dataclasses
- Build-specific settings
- Multi-environment support (local, remote, CI/CD)

**Key principle**: Configuration, not code changes, for different setups

---

### 4. [Base Layer Architecture](base_layer_architecture.md)

**Why it matters**: Foundation for splice site prediction with extensible base model support

**What it covers**:
- Abstract base model interface
- Coordinate system conventions
- Prediction workflow pipeline
- Data preparation architecture
- Chunking and checkpointing

**Key principle**: Abstract interface allows pluggable base models

---

### 5. [Meta Layer Architecture](meta_layer_architecture.md)

**Why it matters**: Multimodal meta-learning for adaptive splice prediction

**What it covers**:
- Foundation-Adaptor framework
- Context integration patterns
- Training and inference pipelines
- Feature engineering
- Performance optimization

**Key principle**: Meta-learning adapts base predictions to specific contexts

---

## Design Principles

These principles guide all architectural decisions:

### 1. **Separation of Concerns**

Each component has a single, well-defined responsibility:
- `genomic_config.py` → Path resolution
- `ArtifactManager` → Output management  
- `BaseModel` → Prediction interface

**Why**: Makes code easier to understand, test, and modify

---

### 2. **Single Source of Truth**

Critical information has exactly one authoritative location:
- Paths → Configuration system
- Schemas → Type definitions
- Defaults → YAML config files

**Why**: Prevents inconsistencies and "which version is correct?" confusion

---

### 3. **Configuration Over Code**

Use configuration files for environment-specific settings:
- Data paths → `settings.yaml`
- Model versions → Environment variables
- Build selection → Config, not hardcode

**Why**: Same code runs everywhere without modification

---

### 4. **Explicit Over Implicit**

Make dependencies and requirements visible:
- Type annotations everywhere
- Explicit imports
- Named parameters over positionals
- Clear error messages

**Why**: Code is self-documenting and errors are understandable

---

### 5. **Progressive Disclosure**

Expose simple interfaces, hide complex implementation:
- Public API → Simple functions
- Internal → Complex logic
- Defaults → Common cases
- Options → Advanced use

**Why**: Easy to start, powerful when needed

---

### 6. **Fail Fast, Fail Clearly**

Detect problems early with helpful messages:
- Validate inputs immediately
- Check file existence before processing
- Descriptive error messages with solutions
- Type checking at boundaries

**Why**: Problems are caught early when they're easy to fix

---

### 7. **Testability**

Design for easy testing:
- Pure functions where possible
- Dependency injection
- Mock-friendly interfaces
- Isolated test environments

**Why**: Confidence in changes, faster development

---

## Refactoring Context

These design documents inform the ongoing refactoring from **meta-spliceai** to **agentic-spliceai**.

### Current Phase: Base Layer Refactoring

We're applying these principles to:
- Extract base layer from monolithic workflow
- Implement resource management system
- Create artifact management infrastructure
- Establish configuration foundations

**Status**: In progress — see source code under `src/agentic_spliceai/`

---

## How to Use This Documentation

### Before Making Architectural Changes

1. **Read relevant design document** to understand current architecture
2. **Understand the principles** behind the design
3. **Propose changes** that align with principles
4. **Update documentation** if design evolves

### When Adding New Components

1. **Follow established patterns** documented here
2. **Apply design principles** consistently
3. **Document significant decisions** in relevant design doc
4. **Add examples** to help future developers

### When Debugging Design Issues

1. **Check if issue violates** a design principle
2. **Review design docs** for intended architecture
3. **Consider if refactoring** would prevent similar issues
4. **Update docs** with lessons learned

---

## Document Status

| Document | Status | Last Updated |
|----------|--------|--------------|
| [Resource Management](resource_management.md) | ✅ Complete | Feb 15, 2026 |
| [Output Management](output_management.md) | ✅ Complete | Feb 15, 2026 |
| [Configuration System](configuration_system.md) | ✅ Complete | Feb 15, 2026 |
| [Base Layer Architecture](base_layer_architecture.md) | 📝 Draft | TBD |
| [Meta Layer Architecture](meta_layer_architecture.md) | 📝 Draft | TBD |

---

## Related Documentation

### Code Documentation
- `src/agentic_spliceai/splice_engine/config/` - Configuration implementation
- [STRUCTURE.md](../architecture/STRUCTURE.md) - Project structure overview

### User Guides
- [SETUP.md](../SETUP.md) - Setup and configuration
- [QUICKSTART.md](../QUICKSTART.md) - Getting started

---

## Contributing

When contributing architectural changes:

1. **Discuss design** in issues/PRs before implementing
2. **Follow existing patterns** documented here
3. **Update design docs** to reflect changes
4. **Add examples** for new patterns
5. **Explain rationale** for design decisions

---

## Questions?

- **Design questions**: See relevant design document in this directory
- **Implementation questions**: See source code under `src/agentic_spliceai/`
- **Usage questions**: See [QUICKSTART.md](../QUICKSTART.md) or [SETUP.md](../SETUP.md)

---

**Last Updated**: February 15, 2026  
**Maintainer**: agentic-spliceai team  
**Status**: ✅ Core design documents complete, specialized docs in progress
