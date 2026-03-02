"""ML model loading cache for on-demand predictions.

Caches loaded SpliceAI/OpenSpliceAI models in memory so the ~5-10s model
loading cost is only paid once. Uses a thread pool executor to avoid
blocking FastAPI's async event loop during the synchronous load.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from agentic_spliceai.splice_engine.base_layer.prediction.core import load_spliceai_models
from agentic_spliceai.splice_engine.resources import get_model_resources

logger = logging.getLogger(__name__)

# In-memory cache: model_type -> loaded models list
_model_cache: Dict[str, List] = {}

# Single-worker executor to serialize model loading (prevents double-load race)
_executor = ThreadPoolExecutor(max_workers=1)


def _load_models_sync(model_type: str) -> List:
    """Load models synchronously (runs in thread pool)."""
    if model_type in _model_cache:
        return _model_cache[model_type]

    logger.info(f"Loading models: {model_type} (first load, will cache)")
    resources = get_model_resources(model_type)
    models = load_spliceai_models(
        model_type=model_type,
        build=resources.build,
        verbosity=1,
    )
    _model_cache[model_type] = models
    logger.info(f"Models cached: {model_type} ({len(models)} model(s))")
    return models


async def get_models(model_type: str) -> List:
    """Get loaded models, using cache when available.

    First call for a model_type loads from disk (~5-10s).
    Subsequent calls return instantly from memory.
    """
    if model_type in _model_cache:
        return _model_cache[model_type]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _load_models_sync, model_type)


def is_cached(model_type: str) -> bool:
    """Check if a model type is already loaded in memory."""
    return model_type in _model_cache


def clear_cache(model_type: str | None = None) -> None:
    """Clear model cache."""
    if model_type:
        _model_cache.pop(model_type, None)
    else:
        _model_cache.clear()
