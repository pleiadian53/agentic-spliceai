# Splice Agent - Project Structure

Complete overview of the agentic-spliceai project organization.

## 📁 Directory Structure

```
agentic-spliceai/
├── README.md                    # Project overview
├── QUICKSTART.md                # 5-minute getting started
├── MIGRATION.md                 # Moving to new projects
├── STRUCTURE.md                 # This file
├── LICENSE                      # MIT License
│
├── environment.yml              # Mamba environment definition
├── pyproject.toml               # Poetry project configuration
├── requirements.txt             # Pip requirements (legacy)
├── .env.example                 # Environment variable template
├── .gitignore                   # Git ignore rules
│
├── agentic_spliceai/                # Main package
│   ├── __init__.py              # Package initialization
│   ├── data_access.py           # Dataset loading and querying
│   ├── planning.py              # Chart code generation
│   ├── llm_client.py            # LLM API client
│   ├── utils.py                 # Utility functions
│   ├── splice_analysis.py       # Splice-specific analysis
│   │
│   ├── server/                  # FastAPI service
│   │   ├── splice_service.py    # Main API service
│   │   ├── config.py            # Configuration
│   │   ├── schemas.py           # Pydantic models
│   │   └── manage.py            # Service management
│   │
│   └── docs/                    # Package-level docs
│       └── README.md            # Package documentation
│
├── docs/                        # Global documentation
│   ├── README.md                # Documentation index
│   ├── architecture/            # System architecture
│   ├── tutorials/               # Step-by-step guides
│   ├── installation/            # Setup guides
│   ├── api/                     # API reference
│   ├── biology/                 # Biological background
│   └── workflows/               # Analysis workflows
│
├── examples/                    # Example scripts
│   ├── quick_start.py           # Quick examples
│   └── analyze_splice_sites.py  # Full CLI tool
│
├── tests/                       # Unit tests
│   ├── test_data_access.py
│   ├── test_planning.py
│   ├── test_splice_analysis.py
│   └── conftest.py              # Pytest configuration
│
├── data/                        # Data directory
│   ├── README.md                # Data documentation
│   └── .gitkeep                 # Keep directory in git
│
└── output/                      # Generated outputs
    └── splice_charts/           # Generated charts
```

## 📚 Documentation Hierarchy

### 1. Global Documentation (`docs/`)

**Purpose:** High-level, topic-based documentation for users and contributors.

**Structure:**
```
docs/
├── README.md                    # Documentation index
├── architecture/                # System design
│   ├── overview.md
│   ├── components.md
│   └── data_flow.md
├── tutorials/                   # Learning guides
│   ├── getting_started.md
│   └── advanced_usage.md
├── installation/                # Setup guides
│   ├── environment.md
│   └── troubleshooting.md
├── api/                         # API reference
│   ├── rest_api.md
│   └── python_api.md
├── biology/                     # Domain knowledge
│   ├── splice_sites.md
│   └── alternative_splicing.md
└── workflows/                   # Analysis patterns
    ├── predefined.md
    └── custom.md
```

### 2. Package Documentation (`agentic_spliceai/docs/`)

**Purpose:** Code-specific documentation close to implementation.

**Contents:**
- Module API references
- Implementation details
- Code examples
- Internal architecture

## 🔧 Configuration Files

### `environment.yml`
Mamba/Conda environment definition with all dependencies.

```yaml
name: agentic-spliceai
channels:
  - conda-forge
dependencies:
  - python=3.11
  - openai
  - fastapi
  - duckdb
  # ... more dependencies
```

### `pyproject.toml`
Poetry project configuration with metadata and dependencies.

```toml
[project]
name = "agentic-spliceai"
version = "0.1.0"
dependencies = [
    "openai",
    "fastapi",
    # ... more dependencies
]
```

### `.gitignore`
Excludes from version control:
- `.env` - Environment variables
- `output/` - Generated files
- `data/*.tsv` - Data files
- Python artifacts

## 🎯 Module Responsibilities

### Core Modules

**`data_access.py`**
- Dataset loading (TSV, CSV, Parquet, SQLite)
- SQL query execution
- Schema introspection
- Data validation

**`planning.py`**
- Chart code generation
- Code critique and refinement
- Reflection loops
- Prompt engineering

**`llm_client.py`**
- OpenAI API integration
- Retry logic
- Error handling
- Response parsing

**`utils.py`**
- Model recommendations
- File operations
- Logging utilities
- Helper functions

**`splice_analysis.py`**
- Domain context
- Analysis templates
- Template-based generation
- Exploratory analysis

### Server Modules

**`server/splice_service.py`**
- FastAPI application
- REST endpoints
- Request handling
- Response formatting

**`server/config.py`**
- Path resolution
- Configuration management
- Environment variables
- Constants

**`server/schemas.py`**
- Pydantic models
- Request validation
- Response schemas
- Type definitions

## 🚀 Development Workflow

### 1. Environment Setup

```bash
# Create environment
mamba env create -f environment.yml
mamba activate agentic-spliceai

# Or use poetry
poetry install
```

### 2. Development

```bash
# Run tests
pytest

# Format code
black .
ruff check .

# Type checking
mypy agentic_spliceai/
```

### 3. Running Server

```bash
cd agentic_spliceai/server
python splice_service.py
# Visit http://localhost:8004/docs
```

### 4. Documentation

- **Add global docs** → `docs/<topic>/`
- **Add package docs** → `agentic_spliceai/docs/`

## 📦 Package Distribution

### Local Installation

```bash
# Development mode
pip install -e .

# With optional dependencies
pip install -e ".[dev,bio]"
```

### Building

```bash
# Build distribution
python -m build

# Install from wheel
pip install dist/agentic_spliceai-0.1.0-py3-none-any.whl
```

## 🔗 Integration Points

### With agentic-ai-public

- Shares core patterns (reflection, planning)
- Can use same environment
- Independent git repositories
- Manual code syncing

### With External Tools

- OpenAI API for LLM
- DuckDB for data access
- FastAPI for web service
- Matplotlib/Seaborn for visualization

## 🎨 Design Principles

1. **Modularity** - Clear separation of concerns
2. **Documentation** - Three-tier documentation structure
3. **Testability** - Comprehensive test coverage
4. **Extensibility** - Easy to add new analyses
5. **Portability** - Self-contained and relocatable

## 📝 File Naming Conventions

- **Python files:** `snake_case.py`
- **Documentation:** `UPPERCASE.md` or `lowercase.md`
- **Tests:** `test_*.py`
- **Examples:** `descriptive_name.py`

## 🔍 Finding Things

**I want to...**

- **Understand the architecture** → `docs/architecture/`
- **Learn how to use it** → `docs/tutorials/`
- **Check API reference** → `docs/api/` or `agentic_spliceai/docs/`
- **See examples** → `examples/`
- **Run tests** → `tests/`

---

**Last Updated:** 2025-11-19  
**Version:** 0.1.0
