"""Base Layer Application — pluggable splice-site prediction.

Packaged version of the use cases demonstrated in ``examples/base_layer/``.
Supports any predictor that satisfies the ``BasePredictor`` protocol:
per-nucleotide 3-class scores (neither / acceptor / donor).

Registered predictors are discovered via ``registry.py`` and exposed
through the ``agentic-spliceai-base`` CLI.

Quick start
-----------

    from agentic_spliceai.applications.base_layer import get_predictor

    predictor = get_predictor("openspliceai")
    scores = predictor.predict_genes(["BRCA1", "TP53"])

CLI
---

    agentic-spliceai-base list-predictors
    agentic-spliceai-base predict --predictor openspliceai --genes BRCA1 TP53
    agentic-spliceai-base evaluate --predictor openspliceai --chromosomes chr21

See also
--------
- ``protocol.BasePredictor`` — the per-nt 3-class output contract
- ``registry`` — discovery and plugin registration
- ``docs/applications/canonical_splice_prediction/README.md``
"""

from .protocol import BasePredictor, PredictionResult
from .registry import get_predictor, list_predictors, register_predictor

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "get_predictor",
    "list_predictors",
    "register_predictor",
]
