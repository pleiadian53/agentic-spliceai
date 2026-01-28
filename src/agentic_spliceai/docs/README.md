# Agentic-SpliceAI Package Documentation

Package-level documentation for the `agentic_spliceai` Python package.

## 📦 Package Structure

```
agentic_spliceai/
├── __init__.py              # Package initialization
├── data_access.py           # Dataset loading and querying
├── planning.py              # Chart code generation
├── llm_client.py            # LLM API client
├── utils.py                 # Utility functions
├── splice_analysis.py       # Splice-specific analysis templates
│
├── splice_engine/           # Core splice prediction engine
│   ├── base_layer/         # Base model wrappers (SpliceAI, OpenSpliceAI)
│   ├── meta_layer/         # Meta-learning layer (recalibration)
│   ├── meta_models/        # Prediction workflows
│   └── resources/          # Genomic resource management
│
└── docs/                    # This directory
    ├── README.md            # This file
    └── meta_layer/          # Meta-layer documentation
        ├── ARCHITECTURE.md
        ├── experiments/     # Experiment logs
        └── methods/         # Methodology docs
```

---

## 📚 Documentation Index

### Meta-Layer Documentation

The meta-layer is a multimodal deep learning system for splice site prediction recalibration.

| Document | Description |
|----------|-------------|
| [meta_layer/README.md](meta_layer/README.md) | Meta-layer overview |
| [meta_layer/ARCHITECTURE.md](meta_layer/ARCHITECTURE.md) | System architecture |
| [meta_layer/methods/](meta_layer/methods/) | Methodology documentation |
| [meta_layer/experiments/](meta_layer/experiments/) | Experiment logs |

**Best Approach**: [Validated Delta Prediction](meta_layer/experiments/004_validated_delta/) (r=0.41)

---

## 📚 Module Documentation

### Core Modules

#### `data_access.py`
Dataset loading and SQL querying interface.

**Key Classes:**
- `ChartDataset` - Abstract base for datasets
- `DuckDBDataset` - DuckDB-based dataset (TSV, CSV, Parquet)
- `SQLiteDataset` - SQLite database support

**Usage:**
```python
from splice_agent import create_dataset

dataset = create_dataset("data/splice_sites.tsv")
schema = dataset.get_schema_description()
```

#### `planning.py`
LLM-based chart code generation with reflection.

**Key Functions:**
- `generate_chart_code()` - Generate chart code from natural language
- `critique_code()` - Critique and improve generated code
- `reflect_and_improve()` - Iterative refinement

**Usage:**
```python
from splice_agent.planning import generate_chart_code

result = generate_chart_code(
    dataset=dataset,
    user_request="Show top genes by splice sites",
    client=openai_client
)
```

#### `llm_client.py`
OpenAI API client with retry logic and error handling.

**Key Classes:**
- `LLMClient` - Wrapper for OpenAI API
- `ResponsesAPIClient` - Responses API support

**Usage:**
```python
from splice_agent.llm_client import LLMClient

client = LLMClient(api_key="sk-...")
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    model="gpt-4o-mini"
)
```

#### `splice_analysis.py`
Domain-specific splice site analysis templates.

**Key Components:**
- `SPLICE_SITE_CONTEXT` - Biological domain context
- `ANALYSIS_TEMPLATES` - Predefined analysis templates
- `generate_analysis_insight()` - Template-based analysis
- `generate_exploratory_insight()` - Custom research questions

**Usage:**
```python
from splice_agent.splice_analysis import generate_analysis_insight

result = generate_analysis_insight(
    dataset=dataset,
    analysis_type="high_alternative_splicing",
    client=client
)
```

### Server Modules

#### `server/splice_service.py`
FastAPI REST API service for splice analysis.

**Endpoints:**
- `GET /` - Service info
- `GET /health` - Health check
- `GET /analyses` - List available analyses
- `POST /analyze/template` - Template-based analysis
- `POST /analyze/exploratory` - Custom research questions

**Usage:**
```bash
cd server
python splice_service.py
# Visit http://localhost:8004/docs
```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
SPLICE_AGENT_PORT=8004
SPLICE_AGENT_HOST=0.0.0.0
SPLICE_AGENT_DATA_DIR=data/
SPLICE_AGENT_OUTPUT_DIR=output/splice_charts
```

### Model Selection

```python
from splice_agent.utils import get_recommended_models

models = get_recommended_models()
# Returns:
# {
#   "fast_prototyping": ["gpt-5-mini", "gpt-4o-mini", ...],
#   "code_generation": ["gpt-5-codex", "gpt-5.1", ...],
#   ...
# }
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_splice_analysis.py

# Run with coverage
pytest --cov=splice_agent --cov-report=html
```

## 📖 API Reference

For detailed API documentation, see:
- [REST API Docs](../../docs/api/) - HTTP API reference
- [Python API Docs](../../docs/api/python.md) - Python API reference

## 🔗 Related Documentation

- [Global Docs](../../docs/) - High-level documentation
- [Development Docs](../../dev/) - Private development notes
- [Main README](../../README.md) - Project overview

---

**Package Version:** 0.1.0  
**Last Updated:** 2025-11-19
