"""Input/output handling for base layer.

This package provides:
- ArtifactManager: Structured output for chunked prediction workflows
- File handlers for reading/writing predictions (future)

Exports:
    ArtifactManager: Manage chunked prediction artifacts with overwrite policies
"""

from .artifacts import ArtifactManager

__all__ = [
    "ArtifactManager",
]
