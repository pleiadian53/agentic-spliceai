"""AgenticSpliceAI Lab — FastAPI service for bioinformatics UI.

Gene browsing, metrics visualization, and splice site analysis.
Serves on port 8005 alongside chart_service (8003) and splice_service (8004).
"""

import asyncio
import json
import logging
import math
import sys
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import polars as pl
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from agentic_spliceai.splice_engine.resources import (
    get_model_resources,
    list_available_models,
    list_available_meta_models,
    get_meta_model_config,
    resolve_meta_model_name,
)
from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_splice_site_annotations,
    prepare_gene_data,
)
from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
    evaluate_splice_site_predictions,
    filter_annotations_by_transcript,
)
from . import config
from . import m3_inference
from . import meta_metrics
from . import variant_inference
from .base_inference import is_servable, predict_gene as base_predict_gene
from .gene_cache import (
    get_genes, get_gene_stats, get_chromosomes,
    get_genes_for_annotation, available_annotations, annotation_for_model,
)
from . import annotation_tracks
from .annotation_tracks import gene_annotation_tracks
from .model_cache import is_cached as is_model_cached
from .meta_inference import build_overlay_predictions
from .schemas import (
    GeneRecord, GeneListResponse, GeneStatsResponse, ModelInfo,
    GenomeResponse, SpliceSiteMarker,
)

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory=str(config.TEMPLATES_DIR))


# =========================
# Lifespan Management
# =========================

def servable_models() -> list[str]:
    """Base models the UI both offers and can actually run.

    The menu is declared in settings.yaml but predictions are served by two
    different loaders, so a model can be declared without being loadable. Every
    place that lists or validates a model goes through here, which makes the
    invariant "if it's in the menu, a click works" structural rather than a
    convention someone has to remember.
    """
    return [name for name in list_available_models() if is_servable(name)]


def default_model(models: list[str]) -> str | None:
    """The base model to preselect: ``config.DEFAULT_MODEL`` when servable.

    Previously this was ``models[0]``, i.e. whatever the registry happened to
    list first (spliceai, a GRCh37 model), so the UI opened on a build that none
    of the meta models can be overlaid on.
    """
    if not models:
        return None
    return config.DEFAULT_MODEL if config.DEFAULT_MODEL in models else models[0]


def models_for_template() -> tuple[list[str], str | None]:
    """Servable models with the default first, plus that default."""
    models = servable_models()
    chosen = default_model(models)
    if chosen:
        models = [chosen] + [m for m in models if m != chosen]
    return models, chosen


def _log_annotation_bindings() -> None:
    """Record which annotation version each model resolves to, and flag drift.

    settings.yaml and ``config.ANNOTATIONS`` are independent surfaces over the
    same files; drift between them does not raise, it just scores predictions
    against a different annotation than the page names. Printing the binding at
    boot makes the version visible in any log that accompanies a result, and the
    cross-registry check runs here so divergence surfaces without anyone
    remembering to run ``scripts/check_annotation_registry.py``.
    """
    for m in servable_models():
        try:
            r = get_model_resources(m)
            truths = annotation_tracks.truth_sets_for_build(r.build)
            logger.info("  %-22s %s / %s v%s   truth sets: %s",
                        m, r.build, r.annotation_source, r.release or "(default)",
                        list(truths) or "none (no track parquet on this build)")
        except Exception as e:                       # never block startup on a log line
            logger.warning("  %-22s could not resolve resources: %s", m, e)

    try:
        sys.path.insert(0, str(config.PROJECT_ROOT / "scripts"))
        from check_annotation_registry import check_bio_lab_registry
        problems = check_bio_lab_registry()
        for p in problems:
            logger.warning("Annotation registry drift: %s", p)
        if not problems:
            logger.info("Annotation registries agree (settings.yaml <-> Bio Lab).")
    except Exception as e:
        logger.debug("Annotation registry check skipped: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting AgenticSpliceAI Lab...")
    logger.info(f"Templates: {config.TEMPLATES_DIR}")
    logger.info(f"Cache dir: {config.CACHE_DIR}")

    declared = list_available_models()
    models = servable_models()
    if len(models) != len(declared):
        hidden = sorted(set(declared) - set(models))
        logger.warning(f"Declared but not servable, hidden from menu: {hidden}")
    logger.info(f"Available models: {models}")
    _log_annotation_bindings()

    yield

    logger.info("Shutting down AgenticSpliceAI Lab...")


# =========================
# FastAPI App
# =========================

app = FastAPI(
    title="AgenticSpliceAI Lab",
    description="Bioinformatics UI for splice site analysis",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Sub-routers
# =========================

# Ingestion-layer readiness endpoints (read-only wrappers over
# data_preparation.get_status / multimodal_features.get_status).
from . import ingest_api  # noqa: E402

app.include_router(ingest_api.router)


# =========================
# Page Routes
# =========================

@app.get("/", response_class=HTMLResponse)
async def gene_browser_page(request: Request):
    """Gene browser page."""
    models, chosen = models_for_template()
    return templates.TemplateResponse("gene_browser.html", {
        "request": request,
        "models": models,
        "default_model": chosen,
    })


# =========================
# API Routes — Models
# =========================

@app.get("/api/models", response_model=list[ModelInfo])
async def get_models():
    """List available models with build info."""
    result = []
    for name in servable_models():
        resources = get_model_resources(name)
        result.append(ModelInfo(
            name=name,
            build=resources.build,
            annotation_source=resources.annotation_source,
        ))
    return result


# =========================
# API Routes — Genes
# =========================

def _genes_for(model: str, annotation: str | None):
    """Gene table for an explicit annotation, else the model's own.

    Omitting *annotation* reproduces the pre-selector behaviour exactly, which
    is what keeps the model->dataset mapping the default rather than a setting
    the user has to know about.
    """
    if annotation:
        return get_genes_for_annotation(annotation)
    return get_genes(model)


@app.get("/api/annotations")
async def list_annotations(
    model: str | None = Query(None, description="Mark this model's default annotation"),
):
    """Annotations the Gene Browser can load, with the model's default flagged."""
    default = None
    if model and model in servable_models():
        try:
            default = annotation_for_model(model)
        except Exception:
            default = None
    items = available_annotations()
    for it in items:
        it["is_model_default"] = it["key"] == default
    return {"annotations": items, "model_default": default}


@app.get("/api/genes", response_model=GeneListResponse)
async def get_gene_list(
    model: str = Query(..., description="Model name"),
    chr: str | None = Query(None, description="Filter by chromosome"),
    annotation: str | None = Query(
        None,
        description="Annotation key (e.g. ensembl.GRCh38). Defaults to the model's own.",
    ),
    search: str | None = Query(None, description="Search gene name, ID, synonym or description"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(
        config.DEFAULT_PAGE_SIZE,
        ge=1,
        le=config.MAX_PAGE_SIZE,
        description="Results per page",
    ),
):
    """Paginated gene list with optional filtering."""
    # Validate model
    available = servable_models()
    if model not in available:
        return GeneListResponse(
            genes=[], total=0, page=page, per_page=per_page, total_pages=0
        )

    try:
        df = _genes_for(model, annotation)
    except (KeyError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Apply chromosome filter
    if chr:
        df = df.filter(pl.col("chrom") == chr)

    # Apply search filter (case-insensitive on gene_name, gene_id and description).
    # Description matters because gene symbols are not what users know a gene by:
    # the ALS gene TDP-43 is filed as `TARDBP`, findable only via its description
    # ("TAR DNA binding protein"). Symbol-only search returns the unrelated TDP1/TDP2.
    if search:
        search_lower = search.lower()
        match = (
            pl.col("gene_name").str.to_lowercase().str.contains(search_lower, literal=True)
            | pl.col("gene_id").str.to_lowercase().str.contains(search_lower, literal=True)
        )
        for col in ("description", "aliases"):
            if col in df.columns:
                match = match | (
                    pl.col(col)
                    .fill_null("")
                    .str.to_lowercase()
                    .str.contains(search_lower, literal=True)
                )
        df = df.filter(match)

        # Rank by relevance, else a symbol query is buried by its own relatives:
        # searching "BRCA1" matches BRIP1/BARD1/BABAM1 via their descriptions
        # ("BRCA1 interacting helicase", "BRCA1 associated RING domain"), and
        # plain alphabetical order would put BRAP above the exact hit.
        # An exact *alias* hit ranks just under an exact symbol hit, so searching
        # "TDP-43" surfaces TARDBP ahead of genes that merely mention it.
        name_lower = pl.col("gene_name").str.to_lowercase()
        alias_exact = (
            pl.col("aliases").fill_null("").str.to_lowercase()
            .str.split(",").list.contains(search_lower)
            if "aliases" in df.columns else pl.lit(False)
        )
        df = df.with_columns(
            pl.when(name_lower == search_lower).then(0)
            .when(alias_exact).then(1)
            .when(name_lower.str.starts_with(search_lower)).then(2)
            .when(name_lower.str.contains(search_lower, literal=True)).then(3)
            .otherwise(4)
            .alias("_rank")
        ).sort(["_rank", "gene_name"]).drop("_rank")

    total = df.height
    total_pages = max(1, math.ceil(total / per_page))

    # Paginate
    offset = (page - 1) * per_page
    page_df = df.slice(offset, per_page)

    genes = [
        GeneRecord(**row)
        for row in page_df.to_dicts()
    ]

    return GeneListResponse(
        genes=genes,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
    )


@app.get("/api/genes/stats", response_model=GeneStatsResponse)
async def get_genes_stats(
    model: str = Query(..., description="Model name"),
    annotation: str | None = Query(None, description="Annotation key; defaults to the model's own"),
):
    """Summary statistics for the browsed gene set."""
    available = servable_models()
    if model not in available:
        return GeneStatsResponse(
            model=model, build="unknown", annotation_source="unknown",
            total_genes=0, per_chromosome={},
        )

    if not annotation:
        return GeneStatsResponse(**get_gene_stats(model))

    spec = config.ANNOTATIONS.get(annotation)
    if spec is None:
        raise HTTPException(status_code=400, detail=f"Unknown annotation: {annotation}")
    try:
        df = get_genes_for_annotation(annotation)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    counts = df.group_by("chrom").agg(pl.len().alias("n")).sort("n", descending=True)
    return GeneStatsResponse(
        model=model,
        build=spec["build"],
        annotation_source=spec["source"],
        total_genes=df.height,
        per_chromosome=dict(zip(counts["chrom"].to_list(), counts["n"].to_list())),
    )


@app.get("/api/genes/chromosomes")
async def get_chromosome_list(
    model: str = Query(..., description="Model name"),
    annotation: str | None = Query(None, description="Annotation key; defaults to the model's own"),
):
    """Get sorted list of chromosomes for the browsed gene set."""
    available = servable_models()
    if model not in available:
        return []
    if not annotation:
        return get_chromosomes(model)
    try:
        df = get_genes_for_annotation(annotation)
    except (KeyError, FileNotFoundError):
        return []
    return sorted(df["chrom"].unique().to_list())


# =========================
# Page Routes — Metrics
# =========================

@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """Metrics dashboard page."""
    return templates.TemplateResponse("metrics.html", {"request": request})


# =========================
# API Routes — Metrics
# =========================

def _scan_metrics_runs() -> list[dict]:
    """Scan output directories for metrics.json files."""
    runs = []
    output_dir = config.EXAMPLES_OUTPUT_DIR

    if not output_dir.exists():
        return runs

    for metrics_path in sorted(output_dir.rglob("metrics.json")):
        try:
            data = json.loads(metrics_path.read_text())
            meta = data.get("metadata", {})

            # run_id = parent directory name (e.g., "openspliceai")
            run_id = metrics_path.parent.name

            runs.append({
                "run_id": run_id,
                "model": meta.get("model", run_id),
                "build": meta.get("build", ""),
                "annotation_source": meta.get("annotation_source", ""),
                "n_genes": meta.get("n_genes", 0),
                "threshold": meta.get("threshold", 0.5),
                "timestamp": meta.get("timestamp", ""),
                "runtime_seconds": meta.get("runtime_seconds"),
                "genes": meta.get("genes", []),
                "path": str(metrics_path),
            })
        except Exception as e:
            logger.warning(f"Could not read {metrics_path}: {e}")

    return runs


@app.get("/api/metrics/runs")
async def list_metrics_runs():
    """List available evaluation runs."""
    return _scan_metrics_runs()


@app.get("/api/metrics/compare")
async def compare_metrics(
    runs: str = Query(..., description="Comma-separated run IDs"),
):
    """Compare metrics across multiple runs."""
    run_ids = [r.strip() for r in runs.split(",") if r.strip()]
    output_dir = config.EXAMPLES_OUTPUT_DIR

    results = {}
    for run_id in run_ids:
        metrics_path = output_dir / run_id / "metrics.json"
        if metrics_path.exists():
            results[run_id] = json.loads(metrics_path.read_text())

    if not results:
        raise HTTPException(status_code=404, detail="No matching runs found")

    return results


@app.get("/api/metrics/{run_id}")
async def get_metrics_run(run_id: str):
    """Get full metrics data for a specific run."""
    output_dir = config.EXAMPLES_OUTPUT_DIR
    metrics_path = output_dir / run_id / "metrics.json"

    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return json.loads(metrics_path.read_text())


# =========================
# API Routes — Meta-Layer Metrics (base vs meta)
# =========================

@app.get("/api/meta-metrics/runs")
async def list_meta_metrics_runs():
    """List available meta-layer comparison runs (base vs meta)."""
    return meta_metrics.list_meta_runs()


@app.get("/api/meta-metrics/{run_id}")
async def get_meta_metrics_run(run_id: str):
    """Full base-vs-meta comparison payload for one meta-layer run."""
    run = meta_metrics.get_meta_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Meta run '{run_id}' not found")
    return run


# =========================
# Novel Site Explorer (M3)
# =========================
# M3 ranks candidate *unannotated* sites per gene — a sparse ranked list, not a
# dense overlay — so it gets its own route, cache and response model.

# LRU: gene_name -> the expensive part (full ranking at MAX_TOP_K, min_prob=0).
# top_k / min_prob are cheap re-slices and stay OUT of the key, the same split
# _genome_predict_meta uses for `threshold`.
_m3_cache: OrderedDict[str, dict] = OrderedDict()


def _m3_cache_put(key: str, value: dict) -> None:
    _m3_cache[key] = value
    _m3_cache.move_to_end(key)
    while len(_m3_cache) > config.MAX_CACHED_PREDICTIONS:
        evicted, _ = _m3_cache.popitem(last=False)
        logger.info(f"M3 cache evicted: {evicted}")


def _m3_cache_get(key: str) -> dict | None:
    if key in _m3_cache:
        _m3_cache.move_to_end(key)
        return _m3_cache[key]
    return None


@app.get("/novel/{gene_name}", response_class=HTMLResponse)
async def novel_sites_page(request: Request, gene_name: str):
    """Novel Site Explorer page for a specific gene."""
    return templates.TemplateResponse("novel_sites.html", {
        "request": request,
        "gene_name": gene_name,
        "meta_model": config.M3_MODEL_NAME,
        "default_top_k": config.DEFAULT_TOP_K,
        "default_min_prob": config.DEFAULT_MIN_PROB,
    })


@app.get("/api/novel/genes")
async def novel_sites_genes():
    """The inspectable gene universe (held-out chromosomes), flagged for D2 anchors."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, m3_inference.list_inspectable_genes)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/novel/{gene_name}/candidates")
async def novel_sites_candidates(
    gene_name: str,
    top_k: int = Query(config.DEFAULT_TOP_K, ge=1, le=config.MAX_TOP_K),
    min_prob: float = Query(config.DEFAULT_MIN_PROB, ge=0.0, le=1.0),
):
    """Top-k novel candidates for a gene, M3-ranked with the novelty post-filter."""
    loop = asyncio.get_event_loop()
    cache_key = gene_name.upper()

    full = _m3_cache_get(cache_key)
    if full is None:
        try:
            full = await loop.run_in_executor(
                None,
                lambda: m3_inference.rank_novel_candidates(
                    gene_name, top_k=config.MAX_TOP_K, min_prob=0.0
                ),
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception(f"M3 ranking failed for {gene_name}")
            raise HTTPException(status_code=500, detail=str(e))
        _m3_cache_put(cache_key, full)

    # Cheap re-slice of the cached full ranking.
    kept = [c for c in full["candidates"] if c["meta_prob"] >= min_prob][:top_k]
    for i, c in enumerate(kept, start=1):
        c = dict(c)
        c["rank"] = i
        kept[i - 1] = c

    return {**full, "candidates": kept, "top_k": top_k, "min_prob": min_prob}


# =========================
# Page Routes — Variant Effect
# =========================

#: Worked examples for the variant page. Two groups, and the split is the point.
#:
#: `canonical` are invariant-dinucleotide disruptions from
#: examples/variant_analysis/test_variants.yaml, where every model agrees. They
#: are here so a reader can calibrate on an easy case first.
#:
#: `rescued` are the harder ones: RNA-seq-validated MutSpliceDB variants that
#: the base model scores below 0.1 while M2-S recovers. Measured 2026-08-12 from
#: examples/variant_analysis/results/mutsplicedb_analysis_v4_resolver_mane_fixed
#: (base misses 12 of 434; M2-S recovers 10, M1-S 4). Alleles are transcript
#: orientation, matching the HGVS, and the runner resolves them against the FASTA.
VARIANT_EXAMPLES = [
    {"group": "Canonical splice-site disruption", "gene": "PSMD2", "chrom": "chr3",
     "pos": 184300445, "ref": "G", "alt": "A", "strand": "+",
     "note": "G of the invariant GT donor. Every model fires."},
    {"group": "Canonical splice-site disruption", "gene": "MYBPC3", "chrom": "chr11",
     "pos": 47333193, "ref": "C", "alt": "A", "strand": "-",
     "note": "Minus-strand donor. Checks the strand handling end to end."},
    {"group": "Canonical splice-site disruption", "gene": "CFTR", "chrom": "chr7",
     "pos": 117480148, "ref": "G", "alt": "A", "strand": "+",
     "note": "Cystic fibrosis donor."},
    {"group": "Base model misses, M2-S recovers", "gene": "CDKN2C", "chrom": "chr1",
     "pos": 50970498, "ref": "G", "alt": "T", "strand": "+",
     "hgvs": "NM_001262.2:c.129+1G>T",
     "note": "Invariant donor +1, yet the base model scores it 0.011. Select M2-S."},
    {"group": "Base model misses, M2-S recovers", "gene": "ARID1A", "chrom": "chr1",
     "pos": 26775575, "ref": "A", "alt": "T", "strand": "+",
     "hgvs": "NM_006015.6:c.4994-2A>T",
     "note": "Invariant acceptor -2; base 0.084. M1-S misses it too, only M2-S recovers."},
    {"group": "Base model misses, M2-S recovers", "gene": "KLF3", "chrom": "chr4",
     "pos": 38689884, "ref": "G", "alt": "A", "strand": "+",
     "hgvs": "NM_016531.6:c.695+5G>A",
     "note": "Intron +5, outside the invariant dinucleotide; base 0.040."},
    {"group": "Base model misses, M2-S recovers", "gene": "DNMT3A", "chrom": "chr2",
     "pos": 25243899, "ref": "A", "alt": "T", "strand": "-",
     "hgvs": "NM_022552.5:c.1935A>T",
     "note": "Exonic, not a splice-site position at all; base 0.058. The two models "
             "disagree on the kind of change here, which is the interesting part."},
]


@app.get("/variant", response_class=HTMLResponse)
async def variant_effect_page(request: Request):
    """Variant Effect page: ref vs alt delta scoring for a single nucleotide change."""
    meta_models = [
        {"key": key, "label": get_meta_model_config(key).get("name", key)}
        for key in list_available_meta_models()
    ]
    return templates.TemplateResponse("variant_effect.html", {
        "request": request,
        "meta_models": meta_models,
        "default_meta": meta_models[0]["key"] if meta_models else "",
        "examples": VARIANT_EXAMPLES,
    })


@app.get("/api/variant/score")
async def variant_score(
    chrom: str = Query(..., description="Chromosome, with or without the chr prefix"),
    pos: int = Query(..., ge=1, description="1-based variant position"),
    ref: str = Query(..., min_length=1, max_length=1, description="Reference allele"),
    alt: str = Query(..., min_length=1, max_length=1, description="Alternate allele"),
    strand: str = Query("+", pattern="^[+-]$"),
    gene: str | None = Query(None, description="Gene symbol, for display only"),
    meta: str | None = Query(None, description="Meta model key; defaults to the first"),
):
    """Ref-vs-alt delta scores for one SNV, from the base model and a meta model.

    Single nucleotide changes only. Indels would need a different alignment
    between the ref and alt coordinate frames, and the delta convention this
    serves (position-wise ``alt - ref``) does not survive an insertion.
    """
    if ref.upper() == alt.upper():
        raise HTTPException(status_code=400, detail="ref and alt are identical")
    for allele, name in ((ref, "ref"), (alt, "alt")):
        if allele.upper() not in {"A", "C", "G", "T"}:
            raise HTTPException(status_code=400, detail=f"{name} allele must be A/C/G/T")

    available = list_available_meta_models()
    if not available:
        raise HTTPException(status_code=404, detail="No meta models are configured")
    meta_model = meta or available[0]
    if meta_model not in available:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown meta model '{meta_model}'. Expected one of {available}.",
        )

    try:
        return await variant_inference.score_variant(
            chrom, pos, ref, alt, strand, gene, meta_model,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Variant scoring failed for %s:%d %s>%s", chrom, pos, ref, alt)
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Page Routes — Genome View
# =========================

@app.get("/genome/{gene_name}", response_class=HTMLResponse)
async def genome_view_page(request: Request, gene_name: str):
    """Genome view page for a specific gene."""
    models, chosen = models_for_template()
    # Show the human-readable `name` ("M1-S (canonical)") in the dropdown while
    # keeping the canonical <variant>.<arch>.<corpus> key as the option value —
    # the key is precise but not what a demo audience should be reading.
    meta_models = [
        {"key": key, "label": get_meta_model_config(key).get("name", key)}
        for key in list_available_meta_models()
    ]
    # Truth sets are per genome build: a GRCh37 model has no GRCh38 track it can
    # honestly be scored against. Offer only what each model supports rather
    # than letting the selector present a choice the API will reject.
    truth_sets = {
        m: list(annotation_tracks.truth_sets_for_build(get_model_resources(m).build))
        for m in models
    }
    return templates.TemplateResponse("genome_view.html", {
        "request": request,
        "gene_name": gene_name,
        "models": models,
        "default_model": chosen,
        "meta_models": meta_models,
        "truth_sets": truth_sets,
    })


# =========================
# API Routes — Genome View
# =========================

MAX_PLOT_POINTS = 10_000

# LRU prediction cache: (gene_name, model) -> (predictions_dict, annotations_df)
# Threshold only affects classification, not raw predictions, so we cache
# the expensive parts and re-run the cheap evaluation on threshold change.
# OrderedDict gives us O(1) move-to-end on hit + O(1) pop-oldest on eviction.
_prediction_cache: OrderedDict[tuple[str, str], tuple[dict, pl.DataFrame]] = OrderedDict()


def _cache_put(key: tuple[str, str], value: tuple[dict, pl.DataFrame]) -> None:
    """Insert into LRU cache, evicting oldest entry if over capacity."""
    _prediction_cache[key] = value
    _prediction_cache.move_to_end(key)
    while len(_prediction_cache) > config.MAX_CACHED_PREDICTIONS:
        evicted_key, _ = _prediction_cache.popitem(last=False)
        logger.info(f"Prediction cache evicted: {evicted_key[0]}/{evicted_key[1]}")


def _cache_get(key: tuple[str, str]) -> tuple[dict, pl.DataFrame] | None:
    """Retrieve from LRU cache, promoting to most-recent on hit."""
    if key in _prediction_cache:
        _prediction_cache.move_to_end(key)
        return _prediction_cache[key]
    return None


def _rescore_to_truth(
    gene_name: str,
    chrom: str,
    truth: str | None,
    default_truth: str,
    gene_start: int,
    gene_end: int,
    positions: list,
    tracks: list[tuple[list, list, float]],
    build: str = "GRCh38",
) -> tuple[str, str | None, list[dict] | None, list[int] | None, list[str] | None]:
    """Re-derive TP/FP/FN for one or more score tracks against an explicit truth set.

    Shared by the base-only and base-vs-meta paths so a switch of ground truth
    means the same thing on both. ``tracks`` is a list of
    ``(donor_prob, acceptor_prob, threshold)``; every track is scored under the
    same rule, keeping a base-vs-meta comparison apples-to-apples.

    Returns ``(truth_used, note, scored, gt_positions, gt_site_types)``. ``scored``
    is ``None`` — meaning "keep what the caller already computed" — when the
    request is for the model's own annotation (nothing to redo) or when that
    annotation's track parquet has not been built.
    """
    if not truth or truth == default_truth:
        return default_truth, None, None, None, None

    sites = annotation_tracks.truth_sites(gene_name, chrom, truth, build)
    if sites is None:
        note = f"Track for '{truth}' is not built; showing {default_truth} instead."
        return default_truth, note, None, None, None

    scored = [
        annotation_tracks.score_against_truth(
            positions, donor, acceptor, sites, gene_start, gene_end, thr)
        for donor, acceptor, thr in tracks
    ]
    # Clipped to the scored window, matching what score_against_truth counted.
    # Another annotation's gene is frequently longer; returning sites the model
    # was never run over would contradict the TP/FP/FN it sits beside.
    gt_positions = sorted(p for p in sites['donor'] | sites['acceptor']
                          if gene_start <= p <= gene_end)
    gt_site_types = ['donor' if p in sites['donor'] else 'acceptor' for p in gt_positions]
    return truth, None, scored, gt_positions, gt_site_types


def _truth_markers(scored: dict) -> list:
    """``score_against_truth`` markers as response models.

    Scores are zeroed: the re-derived path counts from the track parquets and
    has no per-site model score to carry, and the plot reads probabilities from
    the dense arrays rather than from markers.
    """
    return [SpliceSiteMarker(**k, donor_score=0.0, acceptor_score=0.0)
            for k in scored['markers']]


def _build_genome_response(
    gene_name: str,
    model_name: str,
    predictions: dict,
    annotations_df: pl.DataFrame,
    positions_df: pl.DataFrame,
    threshold: float,
    truth: str | None = None,
) -> dict:
    """Build genome view JSON response from prediction + evaluation data."""
    gene_id = next(iter(predictions))
    pred = predictions[gene_id]

    positions = pred['positions']
    donor_prob = pred['donor_prob']
    acceptor_prob = pred['acceptor_prob']
    n_total = len(positions)

    # Downsample probability tracks for large genes, preserving peaks.
    # Naive every-Nth slicing skips sharp 1-2 position peaks, so we always
    # include positions where either probability exceeds a small floor,
    # then fill remaining budget with evenly-spaced background points.
    factor = max(1, n_total // MAX_PLOT_POINTS)
    if factor <= 1:
        ds_positions = positions
        ds_donor = donor_prob
        ds_acceptor = acceptor_prob
    else:
        import numpy as np
        donor_arr = np.asarray(donor_prob)
        acceptor_arr = np.asarray(acceptor_prob)
        # Peak indices: any position with non-trivial probability
        peak_mask = (donor_arr > 0.01) | (acceptor_arr > 0.01)
        peak_idx = set(np.where(peak_mask)[0].tolist())
        # Evenly-spaced background indices
        bg_idx = set(range(0, n_total, factor))
        # Merge and sort
        all_idx = sorted(peak_idx | bg_idx)
        ds_positions = [positions[i] for i in all_idx]
        ds_donor = [donor_prob[i] for i in all_idx]
        ds_acceptor = [acceptor_prob[i] for i in all_idx]

    # Classification markers (TP/FP/FN only, never downsampled)
    markers = []
    classified = positions_df.filter(
        pl.col('pred_type').is_in(['TP', 'FP', 'FN'])
    )
    for row in classified.iter_rows(named=True):
        markers.append(SpliceSiteMarker(
            position=row['position'],
            site_type=row['splice_type'],
            pred_type=row['pred_type'],
            donor_score=row.get('donor_score', 0.0),
            acceptor_score=row.get('acceptor_score', 0.0),
        ))

    # Ground truth positions (never downsampled)
    # Filter annotations to this gene
    if 'gene_name' in annotations_df.columns:
        gene_annot = annotations_df.filter(pl.col('gene_name') == gene_name)
    else:
        gene_annot = annotations_df
    gt_positions = gene_annot['position'].to_list() if gene_annot.height > 0 else []
    gt_site_types = gene_annot['splice_type'].to_list() if gene_annot.height > 0 else []

    # Counts
    pred_types = positions_df['pred_type'].to_list() if positions_df.height > 0 else []
    n_tp = pred_types.count('TP')
    n_fp = pred_types.count('FP')
    n_fn = pred_types.count('FN')

    # Re-score against the requested truth set when it is not this model's own
    # annotation. Without this the Ground truth selector is inert whenever no
    # meta model is overlaid.
    _res = get_model_resources(model_name)
    truth_used, gt_note, scored, re_gt, re_types = _rescore_to_truth(
        gene_name, str(pred.get('chrom', pred.get('seqname', ''))), truth,
        _res.annotation_source,
        pred['gene_start'], pred['gene_end'], ds_positions,
        [(ds_donor, ds_acceptor, threshold)], build=_res.build,
    )
    if scored is not None:
        markers = _truth_markers(scored[0])
        n_tp, n_fp, n_fn = scored[0]['n_tp'], scored[0]['n_fp'], scored[0]['n_fn']
        gt_positions, gt_site_types = re_gt, re_types

    return GenomeResponse(
        gene_name=pred.get('gene_name', gene_name),
        gene_id=gene_id,
        chrom=pred.get('chrom', pred.get('seqname')),
        strand=pred['strand'],
        gene_start=pred['gene_start'],
        gene_end=pred['gene_end'],
        model=model_name,
        threshold=threshold,
        positions=ds_positions,
        donor_prob=ds_donor,
        acceptor_prob=ds_acceptor,
        gt_positions=gt_positions,
        gt_site_types=gt_site_types,
        markers=markers,
        n_tp=n_tp,
        n_fp=n_fp,
        n_fn=n_fn,
        downsample_factor=factor,
        total_positions=n_total,
        truth=truth_used,
        truth_note=gt_note,
    ).model_dump()


# =========================
# Meta-layer overlay (Phase B–E): base vs meta prediction
# =========================

# Meta-overlay LRU cache: (gene_name, meta_model) -> (base_pred, meta_pred, annotations_df)
_meta_prediction_cache: OrderedDict[tuple[str, str], tuple[dict, dict, pl.DataFrame]] = OrderedDict()


def _meta_cache_put(key: tuple[str, str], value: tuple[dict, dict, pl.DataFrame]) -> None:
    _meta_prediction_cache[key] = value
    _meta_prediction_cache.move_to_end(key)
    while len(_meta_prediction_cache) > config.MAX_CACHED_PREDICTIONS:
        ek, _ = _meta_prediction_cache.popitem(last=False)
        logger.info(f"Meta prediction cache evicted: {ek[0]}/{ek[1]}")


def _meta_cache_get(key: tuple[str, str]) -> tuple[dict, dict, pl.DataFrame] | None:
    if key in _meta_prediction_cache:
        _meta_prediction_cache.move_to_end(key)
        return _meta_prediction_cache[key]
    return None


def _markers_and_counts(positions_df: pl.DataFrame):
    """Build SpliceSiteMarkers + (n_tp, n_fp, n_fn) from an evaluated positions_df."""
    markers = []
    if positions_df.height > 0:
        for row in positions_df.filter(
            pl.col('pred_type').is_in(['TP', 'FP', 'FN'])
        ).iter_rows(named=True):
            markers.append(SpliceSiteMarker(
                position=row['position'], site_type=row['splice_type'],
                pred_type=row['pred_type'],
                donor_score=row.get('donor_score', 0.0),
                acceptor_score=row.get('acceptor_score', 0.0),
            ))
        pred_types = positions_df['pred_type'].to_list()
    else:
        pred_types = []
    return markers, pred_types.count('TP'), pred_types.count('FP'), pred_types.count('FN')


def _build_overlay_response(
    gene_name: str, base_model_name: str, meta_model_name: str,
    base_pred: dict, meta_pred: dict, annotations_df: pl.DataFrame,
    base_positions_df: pl.DataFrame, meta_positions_df: pl.DataFrame,
    threshold: float, meta_threshold: float,
    truth: str | None = None,
) -> dict:
    """Build a base-vs-meta overlay response (shared, peak-preserving downsample).

    ``threshold`` classifies the base model and ``meta_threshold`` the meta model.
    They are separate because the two score distributions are: a cutoff that suits
    one puts the other at the wrong operating point.
    """
    import numpy as np
    gene_id = next(iter(base_pred))
    bp, mp = base_pred[gene_id], meta_pred[gene_id]
    positions = bp['positions']
    n_total = len(positions)
    bd, ba = np.asarray(bp['donor_prob']), np.asarray(bp['acceptor_prob'])
    md, ma = np.asarray(mp['donor_prob']), np.asarray(mp['acceptor_prob'])

    # Shared downsample indices: peaks in EITHER base or meta + background.
    # The meta layer fires on far more positions than the (sparse) base model,
    # so a 0.01 floor floods the point set; use a higher 0.05 floor and a hard
    # cap, but ALWAYS keep called sites (called by EITHER model at ITS OWN
    # threshold) so no TP/FP peak is dropped from the line.
    factor = max(1, n_total // MAX_PLOT_POINTS)
    if factor <= 1:
        idx = list(range(n_total))
    else:
        called = (bd > threshold) | (ba > threshold) | (md > meta_threshold) | (ma > meta_threshold)
        called_idx = set(np.where(called)[0].tolist())
        peak = (bd > 0.05) | (ba > 0.05) | (md > 0.05) | (ma > 0.05)
        peak_idx = set(np.where(peak)[0].tolist())
        bg_idx = set(range(0, n_total, factor))
        idx_set = called_idx | peak_idx | bg_idx
        cap = 3 * MAX_PLOT_POINTS
        if len(idx_set) > cap:
            # subsample the non-called points; never drop a called site
            droppable = sorted(idx_set - called_idx)
            step = max(1, len(droppable) // max(1, cap - len(called_idx)))
            idx_set = called_idx | set(droppable[::step])
        idx = sorted(idx_set)

    base_markers, b_tp, b_fp, b_fn = _markers_and_counts(base_positions_df)
    meta_markers, m_tp, m_fp, m_fn = _markers_and_counts(meta_positions_df)

    if 'gene_name' in annotations_df.columns:
        gene_annot = annotations_df.filter(pl.col('gene_name') == gene_name)
    else:
        gene_annot = annotations_df
    gt_positions = gene_annot['position'].to_list() if gene_annot.height > 0 else []
    gt_site_types = gene_annot['splice_type'].to_list() if gene_annot.height > 0 else []

    ds_positions = [positions[i] for i in idx]
    ds_bd = [float(bd[i]) for i in idx]
    ds_ba = [float(ba[i]) for i in idx]
    ds_md = [float(md[i]) for i in idx]
    ds_ma = [float(ma[i]) for i in idx]

    # Re-score against the resolved truth set when it is not the base model's own
    # annotation. Both models go through the same rule so the comparison stays
    # apples-to-apples, each at its own threshold.
    _res = get_model_resources(base_model_name)
    truth_used, gt_note, scored, re_gt, re_types = _rescore_to_truth(
        gene_name, str(bp.get('chrom', '')), truth,
        _res.annotation_source,
        bp['gene_start'], bp['gene_end'], ds_positions,
        [(ds_bd, ds_ba, threshold), (ds_md, ds_ma, meta_threshold)],
        build=_res.build,
    )
    if scored is not None:
        b, m = scored
        base_markers, meta_markers = _truth_markers(b), _truth_markers(m)
        b_tp, b_fp, b_fn = b['n_tp'], b['n_fp'], b['n_fn']
        m_tp, m_fp, m_fn = m['n_tp'], m['n_fp'], m['n_fn']
        gt_positions, gt_site_types = re_gt, re_types

    return GenomeResponse(
        gene_name=bp.get('gene_name', gene_name), gene_id=gene_id,
        chrom=bp.get('chrom', bp.get('seqname')), strand=bp['strand'],
        gene_start=bp['gene_start'], gene_end=bp['gene_end'],
        model=base_model_name, threshold=threshold,
        positions=ds_positions, donor_prob=ds_bd, acceptor_prob=ds_ba,
        gt_positions=gt_positions, gt_site_types=gt_site_types,
        markers=base_markers, n_tp=b_tp, n_fp=b_fp, n_fn=b_fn,
        downsample_factor=factor, total_positions=n_total,
        meta_model=meta_model_name, meta_threshold=meta_threshold,
        meta_donor_prob=ds_md, meta_acceptor_prob=ds_ma,
        meta_markers=meta_markers, meta_n_tp=m_tp, meta_n_fp=m_fp, meta_n_fn=m_fn,
        truth=truth_used, truth_note=gt_note,
    ).model_dump()


async def _genome_predict_meta(gene_name: str, meta_model_name: str,
                               threshold: float, loop,
                               meta_threshold: float | None = None,
                               truth: str | None = None) -> dict:
    """Genome prediction with a meta-layer overlay (base vs meta).

    ``meta_threshold`` defaults to ``threshold``, which is the historical
    single-cutoff behaviour. Pass it to score each model at its own operating
    point.
    """
    if meta_threshold is None:
        meta_threshold = threshold
    base_model_name = get_meta_model_config(meta_model_name).get('base_model', 'openspliceai')
    cache_key = (gene_name, meta_model_name)

    cached = _meta_cache_get(cache_key)
    if cached is not None:
        logger.info(f"Meta prediction cache hit: {gene_name}/{meta_model_name}")
        base_pred, meta_pred, annotations_df = cached
    else:
        logger.info(f"Meta prediction cache miss: {gene_name}/{meta_model_name}")
        resources = get_model_resources(base_model_name)
        build, annotation_source = resources.build, resources.annotation_source
        annotations_dir = resources.get_annotations_dir(create=True)

        annotations_result = await loop.run_in_executor(
            None, lambda: prepare_splice_site_annotations(
                output_dir=str(annotations_dir), genes=[gene_name],
                build=build, annotation_source=annotation_source, verbosity=0),
        )
        annotations_df = annotations_result['splice_sites_df']

        genes_df = await loop.run_in_executor(
            None, lambda: prepare_gene_data(
                genes=[gene_name], build=build,
                annotation_source=annotation_source, verbosity=0),
        )
        if genes_df.height == 0:
            raise HTTPException(
                status_code=404,
                detail=f"Gene '{gene_name}' not found in {build}/{annotation_source}",
            )
        row = genes_df.row(0, named=True)
        try:
            base_pred, meta_pred = await loop.run_in_executor(
                None, lambda: build_overlay_predictions(
                    meta_model_name, row['gene_id'], gene_name,
                    row['chrom'], row['strand'], row['start'], row['end']),
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        _meta_cache_put(cache_key, (base_pred, meta_pred, annotations_df))

    filtered_annot = filter_annotations_by_transcript(
        annotations_df, mode='canonical', verbosity=0,
    )
    base_eval, meta_eval = await asyncio.gather(
        loop.run_in_executor(None, lambda: evaluate_splice_site_predictions(
            predictions=base_pred, annotations_df=filtered_annot, threshold=threshold,
            consensus_window=2, collect_tn=False, verbosity=0, return_pr_metrics=False)),
        loop.run_in_executor(None, lambda: evaluate_splice_site_predictions(
            predictions=meta_pred, annotations_df=filtered_annot, threshold=meta_threshold,
            consensus_window=2, collect_tn=False, verbosity=0, return_pr_metrics=False)),
    )
    return _build_overlay_response(
        gene_name, base_model_name, meta_model_name, base_pred, meta_pred,
        filtered_annot, base_eval[1], meta_eval[1], threshold, meta_threshold,
        truth=truth,
    )


@app.get("/api/genome/{gene_name}/annotation-tracks")
async def genome_annotation_tracks(
    gene_name: str,
    chrom: str = Query(..., description="Chromosome, with or without the chr prefix"),
):
    """Stacked annotation tracks for one gene, plus the Ensembl \\ MANE delta set.

    Separate from ``/predict`` on purpose: this is additive context for the
    chart, so a missing or unbuilt annotation degrades the tracks without
    touching the prediction or its counts.
    """
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, lambda: gene_annotation_tracks(gene_name, chrom)
        )
    except Exception as e:
        logger.exception(f"Annotation tracks failed for {gene_name}")
        raise HTTPException(status_code=500, detail=str(e))


def resolve_truth(model: str, meta: str | None, requested: str | None) -> str:
    """Which annotation TP/FP/FN is scored against.

    Auto-resolves to the **meta model's training annotation** when one is
    overlaid. The genome view historically took its truth from the base model's
    resources, which was right when the page served base models only — but it
    means M2-S, built to find sites MANE omits, was scored on MANE, where every
    such site it found counted as a false positive. Measured on TARDBP at 0.9:
    10/19/0 against MANE becomes **29/0/3** against Ensembl, on identical
    predictions.

    Validated against the tracks available **on the model's own genome build**.
    Reconciling builds here would be worse than refusing: the coordinates are
    simply different numbers, so a cross-build score reads as a confident zero
    rather than as an error.
    """
    res = get_model_resources(model)
    build = res.build
    offered = annotation_tracks.truth_sets_for_build(build)
    if requested:
        if requested not in offered:
            raise HTTPException(
                status_code=400,
                detail=f"Truth set '{requested}' is not available for {model} "
                       f"({build}). Expected one of {list(offered)}.",
            )
        return requested
    if meta:
        spec = get_meta_model_config(meta)
        train_annot = spec.get("train_annotation")
        if train_annot in offered:
            return train_annot
    return res.annotation_source


def _counts_at(positions_df) -> tuple[int, int, int]:
    t = positions_df["pred_type"].to_list() if positions_df.height > 0 else []
    return t.count("TP"), t.count("FP"), t.count("FN")


def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


@app.get("/api/genome/{gene_name}/threshold-sweep")
async def genome_threshold_sweep(
    gene_name: str,
    model: str = Query(..., description="Base model type"),
    meta: str | None = Query(None, description="Optional meta model to sweep alongside base"),
    steps: int = Query(19, ge=5, le=99, description="Linear grid points between 0.05 and 0.95"),
):
    """F1 vs threshold for this gene, and the F1-optimal point for each model.

    Exists because 0.5 is not a meaningful operating point under this class
    imbalance, and the right threshold differs per model: base and meta have
    very different score distributions, so a single shared cutoff flatters
    whichever one happens to match it.

    **This is a per-gene, post-hoc optimum on the data being displayed**, not a
    held-out operating point and not the model's published threshold. It is a
    navigation aid for the slider. The response also carries ``held_out``, the
    per-model optimum measured across every held-out gene, which is the number
    to quote.

    Unlike the stored held-out sweep, this one needs no prevalence correction:
    it scores every position of the gene, with nothing subsampled.
    """
    if model not in servable_models():
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")
    if meta:
        meta = resolve_meta_model_name(meta)
        if meta not in list_available_meta_models():
            raise HTTPException(status_code=400, detail=f"Unknown meta model: {meta}")

    loop = asyncio.get_event_loop()
    # A linear grid stopping at 0.95 cannot locate a meta model's optimum: on the
    # held-out evaluation both promoted meta models peak at 0.99, and their F1 is
    # still climbing at 0.95. Reporting the last grid point as "optimal" would be
    # reporting the edge of the grid. Hence the tail.
    grid = [round(0.05 + i * (0.90 / (steps - 1)), 4) for i in range(steps)]
    grid += [0.96, 0.97, 0.98, 0.99, 0.995, 0.999]
    grid = sorted(set(grid))

    try:
        if meta:
            cached = _meta_cache_get((gene_name, meta))
            if cached is None:
                await _genome_predict_meta(gene_name, meta, 0.5, loop)
                cached = _meta_cache_get((gene_name, meta))
            base_pred, meta_pred, annotations_df = cached
            preds = {"base": base_pred, "meta": meta_pred}
        else:
            cached = _cache_get((gene_name, model))
            if cached is None:
                await genome_predict(gene_name, model=model, threshold=0.5, meta=None)
                cached = _cache_get((gene_name, model))
            predictions, annotations_df = cached
            preds = {"base": predictions}

        filtered_annot = filter_annotations_by_transcript(
            annotations_df, mode="canonical", verbosity=0,
        )

        def sweep(pred) -> list[dict]:
            out = []
            for thr in grid:
                _e, positions_df, _p = evaluate_splice_site_predictions(
                    predictions=pred, annotations_df=filtered_annot, threshold=thr,
                    consensus_window=2, collect_tn=False, verbosity=0,
                    return_pr_metrics=False,
                )
                tp, fp, fn = _counts_at(positions_df)
                out.append({"threshold": thr, "tp": tp, "fp": fp, "fn": fn,
                            "f1": round(_f1(tp, fp, fn), 4)})
            return out

        result = {}
        for label, pred in preds.items():
            curve = await loop.run_in_executor(None, lambda p=pred: sweep(p))
            best = max(curve, key=lambda r: r["f1"])
            result[label] = {"curve": curve, "best": best}

        return {
            "gene_name": gene_name, "model": model, "meta_model": meta,
            "note": "F1-optimal for THIS gene (post-hoc), not a held-out operating point.",
            "held_out": meta_metrics.held_out_operating_points(meta) if meta else None,
            # The per-gene sweep scores against the BASE model's annotation with
            # the canonical-transcript filter — MANE for openspliceai. Named in
            # the response so the panel can say so: an F1 computed on MANE alone
            # structurally penalises M2-S, which exists to find sites MANE omits.
            "eval_annotation": get_model_resources(model).annotation_source,
            "eval_scope": "canonical transcript only",
            **result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Threshold sweep failed for {gene_name}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/genome/{gene_name}/predict")
async def genome_predict(
    gene_name: str,
    model: str = Query(..., description="Base model type (e.g., openspliceai)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Base-model classification threshold"),
    meta: str | None = Query(None, description="Optional meta model (e.g. m1s.concat_fusion.cleanannot) for a base-vs-meta overlay"),
    meta_threshold: float | None = Query(
        None, ge=0.0, le=1.0,
        description="Meta-model threshold; defaults to `threshold` when omitted",
    ),
    truth: str | None = Query(
        None,
        description=(
            "Ground-truth annotation for TP/FP/FN: mane | ensembl | gencode. "
            "Omit for auto — the meta model's training annotation when one is "
            "overlaid, else the base model's."
        ),
    ),
):
    """Run on-demand splice site prediction for a single gene.

    Caches the expensive prediction + annotation steps per (gene, model).
    Only the lightweight evaluation is re-run when threshold changes.

    If ``meta`` is given, returns a base-vs-meta overlay instead: the base
    arrays carry the OpenSpliceAI scores the meta layer refines, and the
    ``meta_*`` fields carry the meta layer's prediction at the same positions.

    ``threshold`` and ``meta_threshold`` are separate because the two models
    score differently: on the held-out evaluation M2-S is F1-optimal at 0.99
    while its base model peaks at 0.25. Scoring both at one cutoff makes at
    least one of them look worse than it is. Omitting ``meta_threshold``
    keeps the single-cutoff behaviour.
    """
    available = servable_models()
    if model not in available:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    loop = asyncio.get_event_loop()

    # Meta-layer overlay path (base vs meta) when a meta model is requested.
    if meta:
        # Resolve retired keys (e.g. m1s_v4_cleanannot) so bookmarked URLs and
        # older notebooks keep working; the cache is keyed on the canonical name.
        meta = resolve_meta_model_name(meta)
        if meta not in list_available_meta_models():
            raise HTTPException(status_code=400, detail=f"Unknown meta model: {meta}")
        try:
            return await _genome_predict_meta(
                gene_name, meta, threshold, loop, meta_threshold=meta_threshold,
                truth=resolve_truth(model, meta, truth),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Meta prediction failed for {gene_name}/{meta}")
            raise HTTPException(status_code=500, detail=str(e))

    # Validate on the base-only path too — an unknown key must not fall through
    # to the default and report a yardstick the caller never asked for.
    truth = resolve_truth(model, None, truth)

    cache_key = (gene_name, model)

    try:
        # Check prediction cache (predictions + annotations are threshold-independent)
        cached = _cache_get(cache_key)
        if cached is not None:
            logger.info(f"Prediction cache hit: {gene_name}/{model}")
            predictions, annotations_df = cached
        else:
            logger.info(f"Prediction cache miss: {gene_name}/{model} — running pipeline")
            resources = get_model_resources(model)
            build = resources.build
            annotation_source = resources.annotation_source
            annotations_dir = resources.get_annotations_dir(create=True)

            # 1. Ground truth annotations (load from full genome-wide cache, filter in memory)
            annotations_result = await loop.run_in_executor(
                None,
                lambda: prepare_splice_site_annotations(
                    output_dir=str(annotations_dir),
                    genes=[gene_name],
                    build=build,
                    annotation_source=annotation_source,
                    verbosity=0,
                ),
            )
            annotations_df = annotations_result['splice_sites_df']

            # 2. Gene sequence data
            genes_df = await loop.run_in_executor(
                None,
                lambda: prepare_gene_data(
                    genes=[gene_name],
                    build=build,
                    annotation_source=annotation_source,
                    verbosity=0,
                ),
            )
            if genes_df.height == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Gene '{gene_name}' not found in {build}/{annotation_source}",
                )

            # 3-4. Load the model and predict (~3-10s per gene). Dispatch to
            # whichever loader serves this model — in-tree SpliceAI/OpenSpliceAI
            # or the predictor registry — see server/bio/base_inference.py.
            predictions = await base_predict_gene(
                model_name=model,
                gene_name=gene_name,
                genes_df=genes_df,
                loop=loop,
                context=10000,
            )

            if not predictions:
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction returned no results for '{gene_name}'",
                )

            # Cache predictions + annotations for future threshold changes
            _cache_put(cache_key, (predictions, annotations_df))

        # 5. Evaluate (cheap — only classification, re-run on every threshold)
        filtered_annot = filter_annotations_by_transcript(
            annotations_df, mode='canonical', verbosity=0,
        )
        eval_result = await loop.run_in_executor(
            None,
            lambda: evaluate_splice_site_predictions(
                predictions=predictions,
                annotations_df=filtered_annot,
                threshold=threshold,
                consensus_window=2,
                collect_tn=False,
                verbosity=0,
                return_pr_metrics=False,
            ),
        )
        _error_df, positions_df, _pr_metrics = eval_result

        # 6. Build response
        return _build_genome_response(
            gene_name, model, predictions, filtered_annot, positions_df, threshold,
            truth=truth,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction failed for {gene_name}")
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# Dev-only introspection (registered only when config.ENABLE_DEBUG_ENDPOINTS)
# =========================

def _lru_report(cache: OrderedDict, label: str, fields: tuple[str, ...]) -> dict:
    """One LRU's occupancy and eviction order.

    Order matters more than membership here. All three caches share a single
    capacity, so browsing during a session can silently evict a gene that was
    warmed for it, and the first entry listed is the one that goes next.
    """
    # strict=True: a key whose arity disagrees with `fields` means this report
    # is mislabelling the cache, which is worse than raising.
    entries = [
        dict(zip(fields, k, strict=True)) if isinstance(k, tuple) else {fields[0]: k}
        for k in cache.keys()
    ]
    return {
        "label": label,
        "size": len(cache),
        "capacity": config.MAX_CACHED_PREDICTIONS,
        "next_evicted": entries[0] if entries else None,
        "entries_oldest_first": entries,
    }


if config.ENABLE_DEBUG_ENDPOINTS:

    @app.get("/api/debug/cache")
    async def debug_cache() -> dict:
        """What the server currently holds in memory. **Dev/demo only.**

        Answers "will the next click be fast", which needs two things the logs
        do not put side by side: whether the *prediction* is cached, and
        whether the *model* that would produce it is loaded. A cache miss on a
        loaded model costs a second or two; a miss on an unloaded one costs a
        model load, which for SpliceAI is five TensorFlow models.

        Returns gene symbols and model names only — no filesystem paths, no
        request history. Disable with ``BIO_LAB_DEBUG=0``.
        """
        from .meta_model_cache import is_cached as is_meta_model_cached

        def safe(fn, name):
            try:
                return fn(name)
            except Exception:                      # a probe must never 500
                return None

        return {
            "dev_only": True,
            "note": ("Internal server state, for demo warm-up checks. "
                     "Disable with BIO_LAB_DEBUG=0."),
            "prediction_caches": [
                _lru_report(_prediction_cache, "base", ("gene", "model")),
                _lru_report(_meta_prediction_cache, "meta overlay", ("gene", "meta_model")),
                _lru_report(_m3_cache, "novel candidates (M3)", ("gene",)),
                {"label": "variant effect", **variant_inference.cache_stats()},
            ],
            "models_loaded": {
                "base": {m: safe(is_model_cached, m) for m in servable_models()},
                "meta": {m: safe(is_meta_model_cached, m)
                         for m in list_available_meta_models()},
            },
        }
