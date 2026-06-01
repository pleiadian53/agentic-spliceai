"""In-memory cache for trained meta-layer models (M*-S).

Mirrors ``model_cache.py`` (base models): caches loaded meta-splice models so
the load cost is paid once, and serializes loads on a single-worker executor to
avoid a double-load race. Keyed by the meta-model name from the ``meta_models``
block of settings.yaml (e.g. ``m1s_v4_cleanannot``).
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import torch

from agentic_spliceai.splice_engine.meta_layer.models.loader import load_meta_model
from agentic_spliceai.splice_engine.resources import get_meta_model_config

logger = logging.getLogger(__name__)

# In-memory cache: meta-model name -> (model, config)
_meta_cache: Dict[str, Tuple[object, object]] = {}

# Single-worker executor to serialize loading (prevents double-load race)
_executor = ThreadPoolExecutor(max_workers=1)

# Meta inference runs on CPU in the Lab UI (one gene at a time; the v3 models
# are tiny — ~370K params — so CPU latency is sub-second once features exist).
_DEVICE = torch.device("cpu")


def _load_meta_sync(name: str) -> Tuple[object, object]:
    """Load a meta model synchronously (runs in thread pool)."""
    if name in _meta_cache:
        return _meta_cache[name]

    spec = get_meta_model_config(name)  # raises ValueError if unknown
    model_dir = Path(spec["dir"])
    if not (model_dir / "best.pt").exists():
        raise FileNotFoundError(
            f"Meta model '{name}' checkpoint not found at {model_dir}/best.pt"
        )
    logger.info(f"Loading meta model: {name} from {model_dir} (first load, will cache)")
    model, cfg = load_meta_model(model_dir, _DEVICE)
    _meta_cache[name] = (model, cfg)
    logger.info(f"Meta model cached: {name} (variant={getattr(cfg, 'variant', '?')})")
    return model, cfg


async def get_meta_model(name: str) -> Tuple[object, object]:
    """Get a loaded meta model + config, using cache when available."""
    if name in _meta_cache:
        return _meta_cache[name]
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _load_meta_sync, name)


def get_meta_model_sync(name: str) -> Tuple[object, object]:
    """Synchronous accessor (for use inside other run_in_executor callables)."""
    return _load_meta_sync(name)


def is_cached(name: str) -> bool:
    return name in _meta_cache


def clear_cache(name: str | None = None) -> None:
    if name:
        _meta_cache.pop(name, None)
    else:
        _meta_cache.clear()
