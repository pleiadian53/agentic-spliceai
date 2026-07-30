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

# ── Novel Site Explorer (M3) ──────────────────────────────────────────────────
# M3 ranks candidate *unannotated* sites per gene. It serves from the eval gene
# cache, whose universe is the held-out SpliceAI test chromosomes (1/3/5/7/9) —
# every inspectable gene is therefore one the model never trained on.
M3_MODEL_NAME = "m3_v1"
M3_EVAL_DIR = PROJECT_ROOT / "output" / "meta_layer" / "m3_eval_d1"
M3_GENE_CACHE_DIR = M3_EVAL_DIR / "gene_cache"
M3_EVAL_GENES = M3_EVAL_DIR / "eval_genes.parquet"

# Label / truth tables used for the novelty post-filter and evidence badges.
M3_LABELS_DIR = PROJECT_ROOT / "data" / "mane" / "GRCh38" / "m3_labels"
M3_ANNOTATION_MASK = M3_LABELS_DIR / "annotation_mask.parquet"       # novelty post-filter
M3_DISEASE_ANCHORS = M3_LABELS_DIR / "disease_anchors.parquet"       # D2 evidence
M3_LONGREAD_TRUTH = (
    PROJECT_ROOT / "data" / "encode_longread" / "GRCh38" / "longread_truth_novel.parquet"
)  # D1 evidence

# Ranking defaults. The score floor and peak suppression are not cosmetic: the
# model emits a score for every position, so without them a top-k list is padded
# with ~1e-9 noise and with the shoulders of peaks already listed.
DEFAULT_TOP_K = 20
MAX_TOP_K = 100
DEFAULT_MIN_PROB = 0.01
PEAK_MIN_SEPARATION = 3
