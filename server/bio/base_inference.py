"""Base-model inference for the Lab UI.

The Lab UI offers a base model per gene view. Two different loaders can serve
one, and callers should not have to know which:

* **In-tree loader** — ``load_spliceai_models`` handles the SpliceAI (Keras)
  and OpenSpliceAI (PyTorch) weights that ship with the splice engine.
* **Predictor registry** — ``applications/base_layer`` resolves anything
  declared in ``src/agentic_spliceai/applications/base_layer/configs/predictors.yaml``,
  including foundation-model-derived checkpoints such as
  ``splicebert_classifier``.

Two config files, deliberately distinct
---------------------------------------
``src/agentic_spliceai/splice_engine/config/settings.yaml`` declares WHICH base
models exist (``base_models:``) and what each was trained on — this is what
``list_available_models`` reads, so it is what populates the menu. The
``predictors.yaml`` above declares HOW to construct a predictor object. A model
can appear in one and not the other, which is exactly the failure this module
exists to prevent.

The settings path is resolved package-relative in
``splice_engine/config/genomic_config.py`` (``Path(__file__).parent /
"settings.yaml"``, falling back to ``configs/genomic_resources.yaml`` or
``config/genomic_resources.yaml`` at the project root — neither present). Under
the editable install that always resolves inside ``src/``. Note that a stale
``build/lib/.../settings.yaml`` can exist from an old ``setup.py build``; it is
gitignored and never loaded, but it will show up in a grep — do not edit it.

The menu is built from ``settings.yaml`` while predictions used to be loaded
exclusively through the in-tree loader, so a registry-only predictor appeared in
the dropdown and then failed with "No .h5 model files found".
:func:`is_servable` and :func:`predict_gene` close that gap: the menu is
filtered by what can actually be loaded, and both loaders return the same
per-gene dict, so the response builder and evaluator stay unchanged.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import polars as pl

from agentic_spliceai.splice_engine.base_layer.prediction.core import (
    predict_splice_sites_for_genes,
)

from .model_cache import get_models as get_cached_models

logger = logging.getLogger(__name__)

# Base models served by the in-tree loader. Everything else is resolved through
# the predictor registry. Keeping this explicit (rather than probing) means the
# promoted demo path is never silently re-routed.
LEGACY_LOADER_MODELS = frozenset({"spliceai", "openspliceai"})

# Registry predictors are cached: constructing one validates its weights, and
# the first prediction loads the foundation model (seconds), so we keep it.
_predictor_cache: dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=1)


def _get_registry_predictor_sync(model_name: str):
    """Instantiate (and cache) a registry predictor. Cheap — loading is lazy."""
    if model_name in _predictor_cache:
        return _predictor_cache[model_name]

    from agentic_spliceai.applications.base_layer.registry import get_predictor

    predictor = get_predictor(model_name)
    _predictor_cache[model_name] = predictor
    logger.info(f"Registry predictor resolved: {model_name}")
    return predictor


def is_servable(model_name: str) -> bool:
    """Whether a base model can actually be loaded and run.

    Used to filter the model menu so the UI never offers a model it cannot
    serve. Registry predictors validate their weights at construction time,
    which is why an unavailable checkpoint is caught here rather than at the
    first click.
    """
    if model_name in LEGACY_LOADER_MODELS:
        return True
    try:
        _get_registry_predictor_sync(model_name)
        return True
    except Exception as exc:  # noqa: BLE001 — a missing optional dep is normal
        logger.info(f"Model {model_name!r} not servable, hiding from menu: {exc}")
        return False


def _positions_to_gene_dict(
    positions_df: pl.DataFrame,
    gene_row: dict[str, Any],
) -> dict[str, dict]:
    """Adapt a registry predictor's long-format output to the shared dict shape.

    The in-tree loader emits ``{gene_id: {positions, donor_prob, ...}}`` with
    positions ascending; registry predictors emit one row per position and, on
    the minus strand, in descending genomic order. Sorting here keeps both
    producers interchangeable for the response builder and the evaluator.
    """
    if positions_df is None or positions_df.height == 0:
        return {}

    df = positions_df.sort("position")
    gene_id = gene_row.get("gene_id") or gene_row.get("gene_name")

    neither = (
        df["neither_prob"].to_list()
        if "neither_prob" in df.columns
        else (1.0 - df["donor_prob"].to_numpy() - df["acceptor_prob"].to_numpy()).tolist()
    )

    return {
        str(gene_id): {
            "chrom": gene_row.get("chrom") or gene_row.get("seqname"),
            "gene_name": gene_row.get("gene_name", ""),
            "strand": gene_row.get("strand"),
            "gene_start": gene_row.get("start"),
            "gene_end": gene_row.get("end"),
            "positions": df["position"].to_list(),
            "donor_prob": df["donor_prob"].to_list(),
            "acceptor_prob": df["acceptor_prob"].to_list(),
            "neither_prob": neither,
        }
    }


async def predict_gene(
    model_name: str,
    gene_name: str,
    genes_df: pl.DataFrame,
    loop: asyncio.AbstractEventLoop | None = None,
    context: int = 10000,
) -> dict[str, dict]:
    """Run one gene through whichever loader serves ``model_name``.

    Returns the per-gene dict produced by ``predict_splice_sites_for_genes``
    (``output_format='dict'``), regardless of which loader ran.
    """
    loop = loop or asyncio.get_event_loop()

    if model_name in LEGACY_LOADER_MODELS:
        models = await get_cached_models(model_name)
        return await loop.run_in_executor(
            None,
            lambda: predict_splice_sites_for_genes(
                gene_df=genes_df,
                models=models,
                context=context,
                output_format="dict",
                verbosity=0,
            ),
        )

    predictor = await loop.run_in_executor(
        _executor, _get_registry_predictor_sync, model_name
    )
    result = await loop.run_in_executor(
        None, lambda: predictor.predict_genes(genes=[gene_name], verbosity=0)
    )
    if getattr(result, "error", None):
        raise RuntimeError(f"{model_name} failed on {gene_name}: {result.error}")

    gene_row = next(iter(genes_df.iter_rows(named=True)), {})
    return _positions_to_gene_dict(result.positions, gene_row)
