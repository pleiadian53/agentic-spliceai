"""AgenticSpliceAI Lab — FastAPI service for bioinformatics UI.

Gene browsing, metrics visualization, and splice site analysis.
Serves on port 8005 alongside chart_service (8003) and splice_service (8004).
"""

import asyncio
import json
import logging
import math
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
)
from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_splice_site_annotations,
    prepare_gene_data,
)
from agentic_spliceai.splice_engine.base_layer.prediction.core import (
    predict_splice_sites_for_genes,
)
from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import (
    evaluate_splice_site_predictions,
    filter_annotations_by_transcript,
)
from . import config
from .gene_cache import get_genes, get_gene_stats, get_chromosomes
from .model_cache import get_models as get_cached_models, is_cached as is_model_cached
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    logger.info("Starting AgenticSpliceAI Lab...")
    logger.info(f"Templates: {config.TEMPLATES_DIR}")
    logger.info(f"Cache dir: {config.CACHE_DIR}")

    models = list_available_models()
    logger.info(f"Available models: {models}")

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
    models = list_available_models()
    return templates.TemplateResponse("gene_browser.html", {
        "request": request,
        "models": models,
        "default_model": models[0] if models else None,
    })


# =========================
# API Routes — Models
# =========================

@app.get("/api/models", response_model=list[ModelInfo])
async def get_models():
    """List available models with build info."""
    result = []
    for name in list_available_models():
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

@app.get("/api/genes", response_model=GeneListResponse)
async def get_gene_list(
    model: str = Query(..., description="Model name"),
    chr: str | None = Query(None, description="Filter by chromosome"),
    search: str | None = Query(None, description="Search gene name or ID"),
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
    available = list_available_models()
    if model not in available:
        return GeneListResponse(
            genes=[], total=0, page=page, per_page=per_page, total_pages=0
        )

    df = get_genes(model)

    # Apply chromosome filter
    if chr:
        df = df.filter(pl.col("chrom") == chr)

    # Apply search filter (case-insensitive on gene_name and gene_id)
    if search:
        search_lower = search.lower()
        df = df.filter(
            pl.col("gene_name").str.to_lowercase().str.contains(search_lower, literal=True)
            | pl.col("gene_id").str.to_lowercase().str.contains(search_lower, literal=True)
        )

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
):
    """Summary statistics for a model's gene set."""
    available = list_available_models()
    if model not in available:
        return GeneStatsResponse(
            model=model, build="unknown", annotation_source="unknown",
            total_genes=0, per_chromosome={},
        )

    stats = get_gene_stats(model)
    return GeneStatsResponse(**stats)


@app.get("/api/genes/chromosomes")
async def get_chromosome_list(
    model: str = Query(..., description="Model name"),
):
    """Get sorted list of chromosomes for a model."""
    available = list_available_models()
    if model not in available:
        return []
    return get_chromosomes(model)


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
# Page Routes — Genome View
# =========================

@app.get("/genome/{gene_name}", response_class=HTMLResponse)
async def genome_view_page(request: Request, gene_name: str):
    """Genome view page for a specific gene."""
    models = list_available_models()
    return templates.TemplateResponse("genome_view.html", {
        "request": request,
        "gene_name": gene_name,
        "models": models,
        "default_model": models[0] if models else None,
        "meta_models": list_available_meta_models(),
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


def _build_genome_response(
    gene_name: str,
    model_name: str,
    predictions: dict,
    annotations_df: pl.DataFrame,
    positions_df: pl.DataFrame,
    threshold: float,
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
    threshold: float,
) -> dict:
    """Build a base-vs-meta overlay response (shared, peak-preserving downsample)."""
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
    # cap, but ALWAYS keep called sites (prob > threshold in either model) so no
    # TP/FP peak is dropped from the line.
    factor = max(1, n_total // MAX_PLOT_POINTS)
    if factor <= 1:
        idx = list(range(n_total))
    else:
        called = (bd > threshold) | (ba > threshold) | (md > threshold) | (ma > threshold)
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

    return GenomeResponse(
        gene_name=bp.get('gene_name', gene_name), gene_id=gene_id,
        chrom=bp.get('chrom', bp.get('seqname')), strand=bp['strand'],
        gene_start=bp['gene_start'], gene_end=bp['gene_end'],
        model=base_model_name, threshold=threshold,
        positions=[positions[i] for i in idx],
        donor_prob=[float(bd[i]) for i in idx],
        acceptor_prob=[float(ba[i]) for i in idx],
        gt_positions=gt_positions, gt_site_types=gt_site_types,
        markers=base_markers, n_tp=b_tp, n_fp=b_fp, n_fn=b_fn,
        downsample_factor=factor, total_positions=n_total,
        meta_model=meta_model_name,
        meta_donor_prob=[float(md[i]) for i in idx],
        meta_acceptor_prob=[float(ma[i]) for i in idx],
        meta_markers=meta_markers, meta_n_tp=m_tp, meta_n_fp=m_fp, meta_n_fn=m_fn,
    ).model_dump()


async def _genome_predict_meta(gene_name: str, meta_model_name: str,
                               threshold: float, loop) -> dict:
    """Genome prediction with a meta-layer overlay (base vs meta)."""
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
            predictions=meta_pred, annotations_df=filtered_annot, threshold=threshold,
            consensus_window=2, collect_tn=False, verbosity=0, return_pr_metrics=False)),
    )
    return _build_overlay_response(
        gene_name, base_model_name, meta_model_name, base_pred, meta_pred,
        filtered_annot, base_eval[1], meta_eval[1], threshold,
    )


@app.get("/api/genome/{gene_name}/predict")
async def genome_predict(
    gene_name: str,
    model: str = Query(..., description="Base model type (e.g., openspliceai)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Classification threshold"),
    meta: str | None = Query(None, description="Optional meta model (e.g. m1s_v3_neuronal) for a base-vs-meta overlay"),
):
    """Run on-demand splice site prediction for a single gene.

    Caches the expensive prediction + annotation steps per (gene, model).
    Only the lightweight evaluation is re-run when threshold changes.

    If ``meta`` is given, returns a base-vs-meta overlay instead: the base
    arrays carry the OpenSpliceAI scores the meta layer refines, and the
    ``meta_*`` fields carry the meta layer's prediction at the same positions.
    """
    available = list_available_models()
    if model not in available:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    loop = asyncio.get_event_loop()

    # Meta-layer overlay path (base vs meta) when a meta model is requested.
    if meta:
        if meta not in list_available_meta_models():
            raise HTTPException(status_code=400, detail=f"Unknown meta model: {meta}")
        try:
            return await _genome_predict_meta(gene_name, meta, threshold, loop)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Meta prediction failed for {gene_name}/{meta}")
            raise HTTPException(status_code=500, detail=str(e))

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

            # 3. Load ML models (cached after first call)
            models = await get_cached_models(model)

            # 4. Run prediction (~3-10s per gene)
            predictions = await loop.run_in_executor(
                None,
                lambda: predict_splice_sites_for_genes(
                    gene_df=genes_df,
                    models=models,
                    context=10000,
                    output_format='dict',
                    verbosity=0,
                ),
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
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction failed for {gene_name}")
        raise HTTPException(status_code=500, detail=str(e))
