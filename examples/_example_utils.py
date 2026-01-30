"""Utilities for example scripts.

This module provides helper functions for examples and notebooks to
properly import from the source code without fragile parent.parent.parent patterns.

Usage in examples:
    from _example_utils import setup_example_environment
    
    setup_example_environment()
    
    # Now import normally
    from agentic_spliceai.splice_engine.base_layer import ...
"""

import sys
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory using marker files.
    
    Looks for common project markers (.git, pyproject.toml, setup.py)
    to reliably identify the project root regardless of script depth.
    
    Returns
    -------
    Path
        Absolute path to project root
        
    Raises
    ------
    RuntimeError
        If project root cannot be found
    """
    # Start from this file's location
    current = Path(__file__).resolve().parent
    
    # Project markers in priority order
    markers = ['.git', 'pyproject.toml', 'setup.py']
    
    # Search upward
    for parent in [current] + list(current.parents):
        for marker in markers:
            if (parent / marker).exists():
                return parent
    
    # Last resort: if we're in examples/, go up one level
    if current.name == 'examples' or current.parent.name == 'examples':
        root = current.parent if current.name != 'examples' else current.parent.parent
        if (root / 'src').exists():
            return root
    
    raise RuntimeError(
        f"Could not find project root from {current}. "
        f"Expected to find one of {markers}"
    )


def setup_example_environment():
    """Configure sys.path for examples to import from src/.
    
    This function:
    1. Finds the project root using marker files (no fragile parent.parent.parent!)
    2. Adds src/ to sys.path
    3. Optionally adds experimental packages if they exist
    
    Call this at the start of example scripts BEFORE any project imports.
    
    Example
    -------
    >>> from _example_utils import setup_example_environment
    >>> setup_example_environment()
    >>> from agentic_spliceai.splice_engine.base_layer import ...
    """
    project_root = get_project_root()
    
    # Add src/ to path
    src_path = project_root / 'src'
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Optionally add experimental packages (e.g., foundation_models/)
    experimental_packages = [
        'foundation_models',  # For Evo2, etc.
    ]
    
    for pkg in experimental_packages:
        pkg_path = project_root / pkg
        if pkg_path.exists() and str(pkg_path) not in sys.path:
            sys.path.insert(0, str(pkg_path))


def get_data_dir() -> Path:
    """Get the data directory.
    
    Returns
    -------
    Path
        Path to data/ directory (project_root/data)
        
    Notes
    -----
    This is a convenience function. For production use cases,
    use the full resource registry from:
    agentic_spliceai.splice_engine.resources.registry
    """
    return get_project_root() / 'data'


def get_output_dir() -> Path:
    """Get the default output directory.
    
    Returns
    -------
    Path
        Path to default output directory (project_root/outputs)
        Creates if it doesn't exist.
        
    Notes
    -----
    This is a convenience function. For production use cases with
    artifact management, use the full output resource manager (Phase 4).
    """
    output_dir = get_project_root() / 'outputs'
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
