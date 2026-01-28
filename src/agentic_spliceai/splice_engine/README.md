# Splice Engine - Base Layer Integration

This module provides splice site prediction capabilities by integrating the base layer from the meta-spliceai project.

## Overview

The splice engine enables agentic-spliceai to perform true splice site prediction using state-of-the-art models like SpliceAI and OpenSpliceAI. It wraps the meta-spliceai base layer with a simplified, user-friendly interface.

## Features

- **Multiple Base Models**: Support for SpliceAI (GRCh37) and OpenSpliceAI (GRCh38)
- **Flexible Interface**: CLI and Python API
- **Memory Efficient**: Handles large-scale genome analysis
- **Production Ready**: Robust checkpointing and error handling

## Quick Start

### CLI

```bash
# Predict for genes
agentic-spliceai-predict --genes BRCA1 TP53

# Predict for chromosome
agentic-spliceai-predict --chromosomes 21
```

### Python API

```python
from agentic_spliceai.splice_engine import predict_splice_sites

# Simple prediction
results = predict_splice_sites(genes=["BRCA1", "TP53"])
positions = results["positions"]
```

## Requirements

This module requires **meta-spliceai** to be installed:

```bash
cd /Users/pleiadian53/work/meta-spliceai
mamba activate agentic-spliceai
pip install -e .
```

## Documentation

See the comprehensive guide: `/docs/SPLICE_PREDICTION_GUIDE.md`

## Module Structure

```
splice_engine/
├── __init__.py       # Main API (predict_splice_sites, run_base_model_predictions)
├── cli.py            # CLI interface (agentic-spliceai-predict)
├── api.py            # High-level Python API (SplicePredictionAPI)
└── README.md         # This file
```

## Entry Points

1. **CLI**: `agentic-spliceai-predict` - Command-line interface
2. **Simple API**: `predict_splice_sites()` - Quick predictions
3. **Full API**: `run_base_model_predictions()` - Complete control
4. **OOP API**: `SplicePredictionAPI` - Object-oriented interface

## Examples

See `/examples/splice_prediction_example.py` for complete examples.

## Integration with Meta-SpliceAI

This module is a thin wrapper around meta-spliceai's base layer:

- **meta-spliceai**: Provides the core prediction engine
- **agentic-spliceai**: Provides simplified interface and integration

The design follows the integration guidelines from:
`/Users/pleiadian53/work/meta-spliceai/docs/base_models/BASE_LAYER_INTEGRATION_GUIDE.md`

## License

MIT License - Same as parent project
