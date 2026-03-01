"""Nexus CLI entry point.

Routes to the research agent CLI. This module exists so that the
``nexus`` console-script entry point defined in pyproject.toml
resolves correctly.
"""

from nexus.agents.research.run import main

__all__ = ["main"]
