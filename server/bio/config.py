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
#
# `release` is the single place a version is written per entry. The display name
# and the GTF filename are both DERIVED from it, so a label can no longer claim
# one version while the path loads another. `scripts/check_annotation_registry.py`
# then checks these paths against what the core registry (settings.yaml) resolves
# for the same `<source>.<build>`, since the two are independent config surfaces.

#: Per-source GTF filename template and how the version reads in prose.
#: Templates mirror `builds.<key>.gtf` in settings.yaml; the check script is what
#: keeps the two honest.
_ANNOTATION_FORMATS = {
    "mane":    {"gtf": "MANE.{build}.v{release}.refseq_genomic.gtf", "label": "MANE v{release}"},
    "ensembl": {"gtf": "Homo_sapiens.{build}.{release}.gtf",         "label": "Ensembl {release}"},
    "gencode": {"gtf": "gencode.v{release}.annotation.gtf",          "label": "GENCODE v{release}"},
}


def _annotation(source: str, build: str, release: str, gtf_dir: str,
                sites: str | None, notes: str) -> dict:
    """One annotation registry entry, with name and GTF derived from `release`."""
    fmt = _ANNOTATION_FORMATS[source]
    return {
        "name": f"{fmt['label'].format(release=release)} ({build})",
        "source": source,
        "build": build,
        "release": release,
        "gtf": PROJECT_ROOT / gtf_dir / fmt["gtf"].format(build=build, release=release),
        "sites": PROJECT_ROOT / sites if sites else None,
        "notes": notes,
    }


ANNOTATIONS: dict[str, dict] = {
    "mane.GRCh38": _annotation(
        "mane", "GRCh38", "1.3", "data/mane/GRCh38",
        "data/mane/GRCh38/splice_sites_track.parquet",
        "Canonical one-transcript-per-gene set. Trains M1-S/M3-S; OpenSpliceAI's annotation.",
    ),
    # Note the flat directory: this GTF sits at data/ensembl/, not data/ensembl/GRCh38/.
    # The core registry finds it because it searches <source>/<build>/ then <source>/.
    "ensembl.GRCh38": _annotation(
        "ensembl", "GRCh38", "112", "data/ensembl",
        "data/ensembl/GRCh38/splice_sites_track.parquet",
        "All transcripts. Trains M2-S; Ensembl \\ MANE is the alternative-site delta set.",
    ),
    "gencode.GRCh38": _annotation(
        "gencode", "GRCh38", "47", "data/gencode/GRCh38",
        "data/gencode/GRCh38/splice_sites_track.parquet",
        "Near-superset of Ensembl (+136,858 splice sites), mostly outside protein-coding genes.",
    ),
    # `sites=None` is load-bearing: with no track parquet, this annotation cannot
    # serve as a ground truth, which is what keeps the genome view from offering
    # SpliceAI a GRCh38 yardstick. See annotation_tracks.truth_sets_for_build.
    "ensembl.GRCh37": _annotation(
        "ensembl", "GRCh37", "87", "data/ensembl/GRCh37", None,
        "Legacy build, for the SpliceAI base model. Not comparable to GRCh38 coordinates.",
    ),
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
