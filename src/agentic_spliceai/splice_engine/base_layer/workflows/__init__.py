"""Workflow orchestration for base layer predictions.

This package provides:
- PredictionWorkflow: Chunked prediction pipeline with checkpointing

Exports:
    PredictionWorkflow: Orchestrate genome-scale predictions with chunking and resume
    WorkflowResult: Container for workflow execution results
"""

from .prediction import PredictionWorkflow, WorkflowResult

__all__ = [
    "PredictionWorkflow",
    "WorkflowResult",
]
