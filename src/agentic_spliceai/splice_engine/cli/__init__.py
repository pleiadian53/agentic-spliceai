"""Command-line interfaces for splice site prediction and data preparation.

This package provides CLI tools for:
- Data preparation: Extract genomic data from GTF/FASTA files
- Prediction: Run splice site predictions with base models

Modules:
    prepare: Data preparation CLI (agentic-spliceai-prepare)
    predict: Prediction CLI (agentic-spliceai-predict)
"""

from . import prepare
from . import predict

__all__ = ['prepare', 'predict']
