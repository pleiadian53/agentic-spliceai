# Agentic-SpliceAI Setup Guide

## Overview

**Agentic-SpliceAI** combines three powerful capabilities:

1. **Nexus Research Agent** - Literature review and synthesis of SoA splice site prediction
2. **Meta-SpliceAI** - Advanced splice site prediction via meta-learning
3. **Agentic Workflow** - LLM-powered analysis and insight generation

## Environment Setup

### 1. Create Dedicated Conda Environment

```bash
cd /Users/pleiadian53/work/agentic-spliceai

# Create environment from yml
mamba env create -f environment.yml

# Activate environment
mamba activate agentic-spliceai
```

### 2. Install Package in Development Mode

```bash
# Install agentic-spliceai package
pip install -e .

# Verify installation
python -c "import agentic_spliceai; print('✓ Package installed')"
```

### 3. Test CLI Commands

```bash
# Test CLI entry points
agentic-spliceai --help
agentic-spliceai-server --help
```

## Architecture

### Three-Layer System

```
┌─────────────────────────────────────────────────────────────┐
│                    AGENTIC-SPLICEAI                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────┐ │
│  │ Nexus Research   │  │ Meta-SpliceAI    │  │ Agentic  │ │
│  │ Agent            │  │ Prediction       │  │ Workflow │ │
│  ├──────────────────┤  ├──────────────────┤  ├──────────┤ │
│  │ • Literature     │  │ • Base SpliceAI  │  │ • LLM    │ │
│  │   synthesis      │  │ • Meta-learning  │  │   agents │ │
│  │ • arXiv/PubMed   │  │ • Ensemble       │  │ • Tool   │ │
│  │ • LaTeX reports  │  │   models         │  │   use    │ │
│  │ • Web search     │  │ • Feature eng.   │  │ • Multi- │ │
│  │                  │  │                  │  │   agent  │ │
│  └──────────────────┘  └──────────────────┘  └──────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Breakdown

**From Nexus (Research)**:
- LLM clients: `openai`, `anthropic`, `mistralai`, `aisuite`
- Research tools: `tavily-python`, `wikipedia`, `beautifulsoup4`
- PDF generation: `weasyprint`, `pandoc`, `tectonic`, `pypandoc`
- Web framework: `fastapi`, `uvicorn`, `pydantic`

**From Meta-SpliceAI (Prediction)**:
- Deep learning: `tensorflow`, `pytorch`, `keras`
- SpliceAI: `spliceai==1.3.1`
- Genomics: `biopython`, `pysam`, `pybedtools`, `pybigwig`
- Genomic formats: `bcbio-gff`, `gffutils`, `gtftools`, `pyfastx`
- ML optimization: `optuna`, `hyperopt`, `hpbandster`
- Feature engineering: `category-encoders`, `feature-engine`

**Shared (Both)**:
- Data: `pandas`, `polars`, `numpy`, `duckdb`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- ML: `scikit-learn`, `lightgbm`, `xgboost`
- Utilities: `tqdm`, `joblib`, `requests`

## Development Workflow

### Phase 1: Port Nexus Research Capability

```bash
# Copy Nexus research agent code
cp -r /Users/pleiadian53/work/agentic-ai-lab/src/nexus/agents/research/* \
      /Users/pleiadian53/work/agentic-spliceai/agentic_spliceai/research/

# Test research capability
python -c "from agentic_spliceai.research import pipeline; print('✓ Research imported')"
```

### Phase 2: Integrate Meta-SpliceAI Prediction

```bash
# Copy meta-spliceai core modules
cp -r /Users/pleiadian53/work/meta-spliceai/meta_spliceai/splice_engine/* \
      /Users/pleiadian53/work/agentic-spliceai/agentic_spliceai/prediction/

# Test prediction capability
python -c "from agentic_spliceai.prediction import base_model; print('✓ Prediction imported')"
```

### Phase 3: Build Agentic Workflow

```bash
# Combine research + prediction with LLM agents
# Create workflow that:
# 1. Uses Nexus to research latest splice site methods
# 2. Uses Meta-SpliceAI to make predictions
# 3. Uses LLM to interpret results and generate insights
```

## Environment Comparison

| Feature | agentic-ai | agentic-spliceai |
|---------|-----------|------------------|
| **Purpose** | General research | Splice site research |
| **Nexus** | ✓ Full | ✓ Ported |
| **Deep Learning** | ✗ | ✓ TensorFlow/PyTorch |
| **Genomics** | ✗ | ✓ Full bioinformatics |
| **SpliceAI** | ✗ | ✓ Base + Meta |
| **Size** | ~2GB | ~5GB (with models) |

## Next Steps

1. ✅ Create environment: `mamba env create -f environment.yml`
2. ✅ Install package: `pip install -e .`
3. ⏳ Port Nexus research agent
4. ⏳ Integrate Meta-SpliceAI prediction
5. ⏳ Build unified agentic workflow
6. ⏳ Test end-to-end pipeline
7. ⏳ Create example notebooks
8. ⏳ Write documentation
9. ⏳ First commit to GitHub

## Troubleshooting

### TensorFlow/PyTorch Conflicts

If you encounter GPU/CUDA issues:
```bash
# CPU-only versions (lighter)
pip install tensorflow-cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Genomics Tools

Some genomics tools require system dependencies:
```bash
# macOS
brew install htslib bedtools

# Ubuntu/Debian
sudo apt-get install libhts-dev bedtools
```

### Memory Issues

Large models may require significant RAM:
- Minimum: 16GB RAM
- Recommended: 32GB RAM
- With GPU: 64GB+ RAM for large-scale training

## Resources

- **Nexus Research**: `/Users/pleiadian53/work/agentic-ai-lab/src/nexus/agents/research/`
- **Meta-SpliceAI**: `/Users/pleiadian53/work/meta-spliceai/`
- **Documentation**: See `QUICKSTART.md` and `README.md`
