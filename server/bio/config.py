"""Configuration and path resolution for AgenticSpliceAI Lab."""

from pathlib import Path

# Project root (3 levels up from server/bio/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Template directory
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

# Default output directories for evaluation results
EXAMPLES_OUTPUT_DIR = PROJECT_ROOT / "examples" / "base_layer" / "output"

# Gene cache directory (Parquet files for fast loading)
CACHE_DIR = PROJECT_ROOT / "output" / "bio_cache"

# Server settings
HOST = "0.0.0.0"
PORT = 8005

# Pagination defaults
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

# Prediction cache (LRU): max number of (gene, model) entries to keep in memory.
# Each entry is ~1-10 MB depending on gene length.  50 entries ≈ 50-500 MB worst case.
MAX_CACHED_PREDICTIONS = 50
