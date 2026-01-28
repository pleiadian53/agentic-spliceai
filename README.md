# Agentic-SpliceAI

**Autonomous AI Agents for Intelligent Splice Site Prediction & Validation**

**Agentic-SpliceAI** builds upon the [Meta-SpliceAI](https://github.com/pleiadian53/meta-spliceai) framework, inheriting its powerful base layer and meta-learning capabilities while adding **agentic AI workflows** to revolutionize alternative splice site prediction.

---

## The Key Innovation: Autonomous AI Agents

Traditional splice site prediction is a one-shot process: input sequence → output scores. But real biological validation requires context, evidence, and iteration.

**Agentic-SpliceAI introduces autonomous AI agents that can:**

- 🔬 **Validate predictions** by cross-referencing literature, gene expression databases, and clinical resources
- 📚 **Research context** by automatically gathering relevant biological knowledge
- 🧠 **Self-improve** by learning from the latest splicing research and experimental data
- 🔄 **Iterate intelligently** through multi-agent pipelines for comprehensive analysis

---

## 🎯 Key Features

### 🧩 Inherited from Meta-SpliceAI (Base + Meta Layers)

| Component | Description |
|-----------|-------------|
| **Base Layer** | SpliceAI & OpenSpliceAI integration with per-nucleotide splice scores |
| **Meta Layer** | Multimodal deep learning (DNA sequences + base model scores) |
| **Tabular Meta-Models** | XGBoost with SHAP interpretability |
| **Smart Checkpointing** | Chunk-level resumption for genome-scale analysis |
| **Memory Efficiency** | Mini-batch processing for stable memory usage |

### 🤖 NEW: Agentic Workflow Enhancements

| Feature | Description |
|---------|-------------|
| **Literature Validation Agent** | Cross-reference predictions with PubMed, arXiv, and splicing databases |
| **Expression Evidence Agent** | Query GTEx, ENCODE, and tissue-specific expression data |
| **Clinical Annotation Agent** | Check ClinVar, SpliceVarDB, and disease associations |
| **Research Report Generator** | Comprehensive PDF reports with citations and biological context |
| **Self-Improving Pipeline** | Learn from validation feedback to refine predictions |

### 📊 Splice Analysis Tools

- **🧬 Domain-Specific Analysis** - Predefined templates for common splice site analyses
- **🤖 AI-Powered Insights** - LLM-generated visualizations with biological context
- **📊 Publication-Ready Charts** - High-quality plots using matplotlib/seaborn
- **🔬 Exploratory Research** - Ask custom questions about your splice site data
- **🚀 REST API** - FastAPI service for integration with other tools

### 📚 Nexus Research Agent

- **Literature Search** - Automated research on splicing mechanisms
- **Research Reports** - Comprehensive reports with LaTeX equations and citations
- **Multi-Source Integration** - arXiv, PubMed, Europe PMC, Wikipedia
- **Publication-Quality Output** - PDF generation with proper formatting
- **Iterative Refinement** - Multi-agent pipeline (Planner → Researcher → Writer → Editor)

---

## 🏗️ Architecture

Agentic SpliceAI is organized into three main layers:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AGENTIC LAYER (WIP)                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   Literature    │  │   Expression    │  │      Clinical           │  │
│  │   Validation    │  │    Evidence     │  │     Annotation          │  │
│  │     Agent       │  │     Agent       │  │       Agent             │  │
│  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘  │
│           │                    │                       │                │
│           └────────────────────┼───────────────────────┘                │
│                                ▼                                        │
│                    ┌───────────────────────┐                            │
│                    │   Nexus Research      │                            │
│                    │   Agent (Orchestrator)│                            │
│                    └───────────┬───────────┘                            │
└────────────────────────────────┼────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────┐
│                        META LAYER                                       │
│  ┌─────────────────────────────▼─────────────────────────────────────┐  │
│  │              Multimodal Deep Learning Fusion                      │  │
│  │   ┌─────────────────┐              ┌─────────────────────────┐    │  │
│  │   │  DNA Sequence   │              │   Base Model Scores     │    │  │
│  │   │  Encoder (CNN/  │──────┬───────│   Encoder (MLP)         │    │  │
│  │   │  HyenaDNA)      │      │       │                         │    │  │
│  │   └─────────────────┘      ▼       └─────────────────────────┘    │  │
│  │                    ┌───────────────┐                              │  │
│  │                    │    Fusion     │                              │  │
│  │                    │   Classifier  │                              │  │
│  │                    └───────────────┘                              │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                 │
┌────────────────────────────────┼────────────────────────────────────────┐
│                        BASE LAYER                                       │
│  ┌─────────────────────────────▼─────────────────────────────────────┐  │
│  │                    Base Model Runner                              │  │
│  │   ┌─────────────────┐              ┌─────────────────────────┐    │  │
│  │   │    SpliceAI     │              │     OpenSpliceAI        │    │  │
│  │   │   (GRCh37)      │              │      (GRCh38)           │    │  │
│  │   └─────────────────┘              └─────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Genomic Resources                              │  │
│  │   Registry │ Config │ Schema Standardization │ Utilities          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | Purpose | Status |
|-------|---------|--------|
| **Base Layer** | Per-nucleotide splice scores from pre-trained models | ✅ Ported |
| **Meta Layer** | Recalibrate predictions using multimodal deep learning | ✅ Ported |
| **Agentic Layer** | Validate & enrich predictions with external knowledge | 🚧 WIP |

---

## 🧬 What is Splice Site Analysis?

Splice sites are genomic positions where introns are removed during RNA splicing. Understanding splice site patterns is crucial for:

- **Alternative Splicing Research** - Protein diversity and gene regulation
- **Transcript Annotation** - Identifying and validating transcript structures
- **Disease Genomics** - Splice site mutations in genetic disorders
- **Drug Discovery** - Targeting splicing mechanisms
- **Evolutionary Biology** - Splice site conservation across species

---

## 🚀 Quick Start

### Splice Analysis

#### Option 1: REST API Service (Recommended)

**Start the service:**

```bash
cd agentic_spliceai/server
mamba run -n agentic-spliceai python splice_service.py
```

**Access the API:**
- Swagger UI: http://localhost:8004/docs
- API Root: http://localhost:8004

**Example API call:**

```bash
curl -X POST http://localhost:8004/analyze/template \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/splice_sites_enhanced.tsv",
    "analysis_type": "high_alternative_splicing",
    "model": "gpt-4o-mini"
  }'
```

#### Option 2: Python Library

```python
from agentic_spliceai import create_dataset
from agentic_spliceai.splice_analysis import generate_analysis_insight
from openai import OpenAI

# Load dataset
dataset = create_dataset("data/splice_sites_enhanced.tsv")

# Generate analysis
client = OpenAI()
result = generate_analysis_insight(
    dataset=dataset,
    analysis_type="high_alternative_splicing",
    client=client,
    model="gpt-4o-mini"
)

# Save and execute code
with open("analysis.py", "w") as f:
    f.write(result["chart_code"])

# Execute to generate chart
exec(result["chart_code"])
```

### Nexus Research Agent

Generate comprehensive research reports on splicing topics:

```bash
# Generate research report on splicing mechanisms
nexus "Alternative Splicing Mechanisms in Cancer" --pdf

# Research specific splicing topics
nexus "SpliceAI Deep Learning Architecture" \
  --model openai:gpt-4o \
  --length comprehensive

# Quick literature review
nexus "Recent advances in splice site prediction" \
  --model openai:gpt-4o-mini \
  --length brief
```

**Python API:**

```python
from nexus.agents.research import ResearchAgent
from nexus.core.config import Config

# Initialize research agent
config = Config()
agent = ResearchAgent(config)

# Generate research report
result = agent.research(
    topic="Splice Site Recognition by U1 snRNP",
    length="standard",
    generate_pdf=True
)

print(f"Report saved to: {result['output_path']}")
```

**Use Cases:**
- Research latest splicing mechanisms before analysis
- Generate literature reviews for grant proposals
- Stay updated on splice prediction methods
- Validate analysis approaches with current research
- Generate comprehensive background sections

### Splice Site Prediction

Predict splice sites using state-of-the-art models:

```bash
# CLI: Predict for genes
agentic-spliceai-predict --genes BRCA1 TP53 UNC13A

# CLI: Predict for chromosome
agentic-spliceai-predict --chromosomes 21 --base-model openspliceai
```

**Python API:**

```python
from agentic_spliceai.splice_engine import predict_splice_sites

# Simple prediction
results = predict_splice_sites(genes=["BRCA1", "TP53"])
positions = results["positions"]

# High-confidence predictions
import polars as pl
high_conf = positions.filter(pl.col("donor_score") > 0.9)
```

**Use Cases:**
- Predict splice sites for genes of interest
- Genome-wide splice site analysis
- Validate predictions against annotations
- Generate training data for meta-models

**See**: [Splice Prediction Guide](docs/SPLICE_PREDICTION_GUIDE.md) for complete documentation

## 📊 Predefined Analysis Templates

### 1. High Alternative Splicing
**Identifies genes with the most splice sites**

Genes with many splice sites often undergo extensive alternative splicing, crucial for protein diversity.

```python
analysis_type = "high_alternative_splicing"
```

### 2. Genomic Distribution
**Visualizes splice site distribution across chromosomes**

Reveals gene density and transcript complexity patterns across the genome.

```python
analysis_type = "splice_site_genomic_view"
```

### 3. Exon Complexity
**Analyzes transcript structure by exon count**

Shows relationship between exon count and splice site density.

```python
analysis_type = "exon_complexity"
```

### 4. Strand Bias
**Analyzes strand distribution of splice sites**

Reveals genomic organization and potential annotation biases.

```python
analysis_type = "strand_bias"
```

### 5. Transcript Diversity
**Identifies genes with most transcript isoforms**

High transcript diversity indicates complex alternative splicing patterns.

```python
analysis_type = "gene_transcript_diversity"
```

## 🔬 Exploratory Analysis

Ask custom research questions:

```python
from agentic_spliceai.splice_analysis import generate_exploratory_insight

result = generate_exploratory_insight(
    dataset=dataset,
    research_question="What is the relationship between gene length and splice site density?",
    client=client
)
```

**Example questions:**
- "How do splice sites distribute across different gene biotypes?"
- "Which chromosomes have the highest alternative splicing rates?"
- "What is the average exon count for highly expressed genes?"
- "Are there differences in splice site patterns between coding and non-coding RNAs?"

## 📦 Installation

### Prerequisites

```bash
# Python 3.9+
python --version

# Create environment for agentic-spliceai
mamba env create -f environment.yml
mamba activate agenticspliceai
```

### Install Dependencies

```bash
# Install package in development mode
cd agentic-spliceai
pip install -e .
```

> **Note**: This project is designed to run independently with its own environment and dependencies.

### Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

## 🗂️ Project Structure

```text
agentic-spliceai/
├── src/
│   └── agentic_spliceai/
│       │
│       ├── splice_engine/           # 🧬 Core splice prediction (ported from meta-spliceai)
│       │   │
│       │   ├── config/              # Configuration management
│       │   │   ├── genomic_config.py    # Config dataclass & loader
│       │   │   └── settings.yaml        # Default settings
│       │   │
│       │   ├── resources/           # Genomic resource management
│       │   │   ├── registry.py          # Path resolution for GTF/FASTA/models
│       │   │   └── schema.py            # Column standardization
│       │   │
│       │   ├── utils/               # Shared utilities
│       │   │   ├── dataframe.py         # DataFrame operations
│       │   │   ├── display.py           # Printing & formatting
│       │   │   └── filesystem.py        # File I/O helpers
│       │   │
│       │   ├── base_layer/          # Base model predictions
│       │   │   ├── models/              # SpliceAI, OpenSpliceAI configs
│       │   │   │   ├── config.py            # BaseModelConfig
│       │   │   │   └── runner.py            # BaseModelRunner
│       │   │   ├── data/                # Data types
│       │   │   │   └── types.py             # GeneManifest, PredictionResult
│       │   │   ├── prediction/          # Prediction workflows (WIP)
│       │   │   └── io/                  # I/O handlers (WIP)
│       │   │
│       │   └── meta_layer/          # Meta-learning layer
│       │       ├── core/                # Configuration & schema
│       │       │   ├── config.py            # MetaLayerConfig
│       │       │   └── feature_schema.py    # Feature definitions
│       │       ├── models/              # Neural network models (WIP)
│       │       ├── training/            # Training pipeline (WIP)
│       │       └── inference/           # Inference pipeline (WIP)
│       │
│       ├── agents/                  # 🤖 Agentic workflows (WIP)
│       │   ├── validation/              # Literature validation agent
│       │   ├── expression/              # Expression evidence agent
│       │   └── clinical/                # Clinical annotation agent
│       │
│       ├── server/                  # FastAPI service
│       │   └── splice_service.py
│       │
│       └── analysis/                # Analysis tools
│           ├── data_access.py           # Dataset loading
│           ├── splice_analysis.py       # Analysis templates
│           └── planning.py              # Chart code generation
│
├── nexus/                           # 📚 Research agent package
│   ├── agents/                      # Multi-agent pipeline
│   │   ├── research/                    # Research orchestrator
│   │   ├── planner/                     # Research planning
│   │   ├── researcher/                  # Information gathering
│   │   ├── writer/                      # Report writing
│   │   └── editor/                      # Report refinement
│   ├── core/                        # Core utilities
│   ├── cli/                         # CLI interface
│   └── templates/                   # Report templates
│
├── data/                            # Data directory (symlinked)
│   ├── ensembl/GRCh37/              # Ensembl annotations
│   ├── mane/GRCh38/                 # MANE annotations
│   └── models/                      # Pre-trained model weights
│       ├── spliceai/
│       └── openspliceai/
│
├── docs/                            # Documentation
├── tests/                           # Unit tests
└── pyproject.toml                   # Package configuration
```

## 🧪 Example Workflow

### 1. Load Your Data

```python
from agentic_spliceai import create_dataset

# Supports TSV, CSV, Parquet, SQLite
dataset = create_dataset("data/splice_sites_enhanced.tsv")

# Inspect schema
print(dataset.get_schema_description())
```

### 2. Generate Analysis

```python
from agentic_spliceai.splice_analysis import generate_analysis_insight
from openai import OpenAI

client = OpenAI()

# Use predefined template
result = generate_analysis_insight(
    dataset=dataset,
    analysis_type="high_alternative_splicing",
    client=client,
    model="gpt-4o-mini"
)

print(f"Title: {result['title']}")
print(f"Description: {result['description']}")
print(f"\nGenerated code:\n{result['chart_code']}")
```

### 3. Execute and Visualize

```python
# Save code
with open("splice_analysis.py", "w") as f:
    f.write(result["chart_code"])

# Execute
exec(result["chart_code"])
# This generates and displays the chart
```

### 4. Customize and Iterate

```python
# Review the generated code
# Modify as needed
# Re-execute

# Or use reflection for automated refinement
result = generate_analysis_insight(
    dataset=dataset,
    analysis_type="high_alternative_splicing",
    client=client,
    enable_reflection=True,
    max_iterations=3
)
```

## 🌐 API Endpoints

### GET /analyses
List available analysis templates

```bash
curl http://localhost:8004/analyses
```

### POST /analyze/template
Generate analysis using predefined template

```bash
curl -X POST http://localhost:8004/analyze/template \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/splice_sites_enhanced.tsv",
    "analysis_type": "high_alternative_splicing",
    "model": "gpt-4o-mini"
  }'
```

### POST /analyze/exploratory
Generate custom exploratory analysis

```bash
curl -X POST http://localhost:8004/analyze/exploratory \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/splice_sites_enhanced.tsv",
    "research_question": "What is the distribution of splice sites by chromosome?",
    "model": "gpt-4o-mini"
  }'
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

### Supported Models

- `gpt-4o-mini` - Fast and cost-effective (default)
- `gpt-4o` - More capable for complex analyses
- `gpt-5-mini` - Latest fast model
- `gpt-5` - Latest capable model
- `gpt-5.1-codex-mini` - Optimized for code generation

## 📚 Data Format

Splice Agent expects datasets with the following columns:

**Required:**
- `chrom` - Chromosome (e.g., chr1, chr2, ..., chrX, chrY)
- `position` - Genomic position (0-based or 1-based)
- `site_type` - Splice site type (donor or acceptor)
- `strand` - Strand (+ or -)

**Optional but recommended:**
- `gene_name` - Gene symbol (e.g., TP53, BRCA1)
- `gene_id` - Gene identifier
- `transcript_id` - Transcript identifier
- `exon_rank` - Exon number in transcript

**Example data:**

```tsv
chrom	position	site_type	strand	gene_name	transcript_id	exon_rank
chr1	12345	donor	+	TP53	NM_000546.6	5
chr1	12678	acceptor	+	TP53	NM_000546.6	6
```

## 🎓 Learning Resources

### Documentation
- [API Documentation](docs/API.md) - Complete API reference
- [Biology Background](docs/BIOLOGY.md) - Splice site biology primer
- [Tutorial](docs/TUTORIAL.md) - Step-by-step guide

### Examples
- [analyze_splice_sites.py](examples/analyze_splice_sites.py) - Full CLI tool
- [quick_start.py](examples/quick_start.py) - Quick examples

### Related Projects
- [Meta-SpliceAI](https://github.com/pleiadian53/meta-spliceai) - Original research implementation with base and meta layers
- [Agentic AI Lab](https://github.com/pleiadian53/agentic-ai-lab) - Nexus Research Agent and agentic workflows

## 🤝 Contributing

Splice Agent is designed to be extensible. Contributions welcome!

**Add new analysis templates:**
1. Add template to `splice_analysis.py::ANALYSIS_TEMPLATES`
2. Include SQL query, chart prompt, and biological context
3. Test with sample data
4. Submit PR

**Add new data sources:**
1. Implement `ChartDataset` interface in `data_access.py`
2. Add format detection logic
3. Test with real data
4. Submit PR

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Built on [Meta-SpliceAI](https://github.com/pleiadian53/meta-spliceai) foundation
- Nexus Research Agent from [Agentic AI Lab](https://github.com/pleiadian53/agentic-ai-lab)
- Powered by OpenAI GPT models
- Inspired by genomics research community

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: your.email@example.com

## 🚀 Roadmap

**Goal**: Refactor meta-spliceai into a cleaner, more maintainable architecture, then add agentic validation capabilities.

### Phase 1: Foundation ✅ **COMPLETE**

**Status**: Base layer core prediction working, tested on full chromosomes

- [x] Port base model prediction functions (SpliceAI, OpenSpliceAI)
- [x] Set up genomic resources system (Registry, Config, Schema)
- [x] Create utility modules (DataFrame, Display, Filesystem)
- [x] Integrate Nexus Research Agent from agentic-ai-lab
- [x] Wire up BaseModelRunner with data loading and prediction
- [x] Test on single genes (BRCA1) and full chromosome (chr21)
- [x] Achieve independence from meta-spliceai (100% standalone)

**Deliverable**: Can run single-gene and chromosome predictions ✅

---

### Phase 2: Base Layer Complete (Next - Weeks 3-4)

**Goal**: Port full prediction workflow with data preparation and artifact management

- [ ] Port main workflow orchestration (1,200+ lines from meta-spliceai)
- [ ] Implement data preparation pipeline (annotations, sequences)
- [ ] Add chunking and checkpointing for memory management
- [ ] Integrate artifact management system
- [ ] Port evaluation and error analysis
- [ ] Add CLI commands for prediction (`agentic-spliceai predict`)
- [ ] Comprehensive testing suite

**Deliverable**: Full production base layer with CLI

---

### Phase 3: Meta Layer Integration (Weeks 5-6)

**Goal**: Port multimodal deep learning meta-learning layer

- [ ] Port meta_layer directory from meta-spliceai
- [ ] Update imports to new package structure
- [ ] Adapt configuration to new resource system
- [ ] Integrate with refactored base layer
- [ ] Port training pipeline and model architectures
- [ ] Port inference workflows
- [ ] Add meta model CLI commands

**Deliverable**: Can train and run meta models for prediction refinement

---

### Phase 4: Agentic Validation Layer (Future - After Meta Layer)

**Goal**: Add autonomous AI agents for intelligent validation

- [ ] **Literature Validation Agent**: Cross-reference predictions with PubMed/arXiv
- [ ] **Expression Evidence Agent**: Query GTEx, ENCODE for splice event evidence  
- [ ] **Clinical Annotation Agent**: Check ClinVar, SpliceVarDB for disease associations
- [ ] **Conservation Agent**: Analyze cross-species splice site conservation
- [ ] **Research Report Generator**: Comprehensive reports with Nexus Research Agent
- [ ] **Self-Improvement Pipeline**: Learn from validation feedback

**Deliverable**: Intelligent validation system with agentic workflows

---

### Phase 5: Advanced Features (Long-term)

- [ ] Variant impact prediction with agentic validation
- [ ] Tissue-specific splice site prediction  
- [ ] Interactive dashboards with real-time validation
- [ ] Integration with clinical pipelines
- [ ] Active learning for continuous model improvement

---

**See**: `dev/refactoring/` for detailed refactoring plans and implementation guides

---

## 🔬 Use Cases

### Alternative Splicing Research

Identify genes with complex splicing patterns and analyze isoform diversity.

### Disease Genomics
Analyze splice site mutations and their impact on gene expression.

### Drug Discovery
Target splicing mechanisms for therapeutic intervention.

### Evolutionary Biology
Study splice site conservation and evolution across species.

---

**Ready to analyze splice sites?** Start with the [Quick Start](#-quick-start) guide or explore the [API documentation](docs/API.md)!
