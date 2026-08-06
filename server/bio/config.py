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

# ── Annotation registry (Gene Browser) ────────────────────────────────────────
# Browsing genes needs an annotation, not a model. The model dropdown still sets
# the DEFAULT annotation (that is how we record which dataset trained which base
# model), but the browser may look at any annotation registered here.
#
# Key is `<source>.<build>`, matching `get_model_resources(m).annotation_source`
# + `.build` so a model resolves to its own annotation with no lookup table.
# Entries whose GTF is absent are filtered out at request time — the menu must
# only offer what the system can actually load.
ANNOTATIONS: dict[str, dict] = {
    "mane.GRCh38": {
        "name": "MANE (GRCh38)",
        "source": "mane",
        "build": "GRCh38",
        "gtf": PROJECT_ROOT / "data/mane/GRCh38/MANE.GRCh38.v1.3.refseq_genomic.gtf",
        "sites": PROJECT_ROOT / "data/mane/GRCh38/splice_sites_track.parquet",
        "notes": "Canonical one-transcript-per-gene set. Trains M1-S/M3-S; OpenSpliceAI's annotation.",
    },
    "ensembl.GRCh38": {
        "name": "Ensembl 112 (GRCh38)",
        "source": "ensembl",
        "build": "GRCh38",
        "gtf": PROJECT_ROOT / "data/ensembl/Homo_sapiens.GRCh38.112.gtf",
        "sites": PROJECT_ROOT / "data/ensembl/GRCh38/splice_sites_track.parquet",
        "notes": "All transcripts. Trains M2-S; Ensembl \\ MANE is the alternative-site delta set.",
    },
    "gencode.GRCh38": {
        "name": "GENCODE v47 (GRCh38)",
        "source": "gencode",
        "build": "GRCh38",
        "gtf": PROJECT_ROOT / "data/gencode/GRCh38/gencode.v47.annotation.gtf",
        "sites": PROJECT_ROOT / "data/gencode/GRCh38/splice_sites_track.parquet",
        "notes": "Near-superset of Ensembl (+136,858 splice sites), mostly outside protein-coding genes.",
    },
    "ensembl.GRCh37": {
        "name": "Ensembl 87 (GRCh37)",
        "source": "ensembl",
        "build": "GRCh37",
        "gtf": PROJECT_ROOT / "data/ensembl/GRCh37/Homo_sapiens.GRCh37.87.gtf",
        "sites": None,
        "notes": "Legacy build, for the SpliceAI base model. Not comparable to GRCh38 coordinates.",
    },
}

DEFAULT_ANNOTATION = "mane.GRCh38"

# Base model preselected in the UI. Not merely cosmetic: openspliceai is the
# GRCh38/MANE model every promoted meta model refines, so it is the only choice
# where the meta overlay and the Novel Site Explorer are immediately meaningful.
# Falls back to the first servable model if this one is unavailable.
DEFAULT_MODEL = "openspliceai"

# Prediction cache (LRU): max number of (gene, model) entries to keep in memory.
# Each entry is ~1-10 MB depending on gene length.  50 entries ≈ 50-500 MB worst case.
MAX_CACHED_PREDICTIONS = 50

# ── Novel Site Explorer (M3) ──────────────────────────────────────────────────
# M3 ranks candidate *unannotated* sites per gene. It serves from the eval gene
# cache, whose universe is the held-out SpliceAI test chromosomes (1/3/5/7/9) —
# every inspectable gene is therefore one the model never trained on.
M3_MODEL_NAME = "m3s.concat_fusion.cleanannot"
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
