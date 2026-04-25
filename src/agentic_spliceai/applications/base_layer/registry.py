"""Plugin registry for base predictors.

Two registration paths:

1. **Built-in**: decorator-driven in-process registration
   (see ``predictors/spliceai.py``, ``predictors/openspliceai.py``).
2. **Manifest-driven**: YAML config at ``configs/predictors.yaml`` lists
   predictors by dotted import path. This is how foundation-model-derived
   classifiers are registered (no code change needed).

Lookup precedence: built-in first, then manifest-registered. Names must
be unique across both paths.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from .protocol import BasePredictor

logger = logging.getLogger(__name__)


# Name -> factory callable producing a BasePredictor instance.
_REGISTRY: Dict[str, Callable[..., BasePredictor]] = {}
_DESCRIPTIONS: Dict[str, Dict[str, object]] = {}
_MANIFEST_LOADED = False


def register_predictor(
    name: str,
    *,
    description: Optional[Dict[str, object]] = None,
) -> Callable[[Callable[..., BasePredictor]], Callable[..., BasePredictor]]:
    """Decorator registering a predictor factory under ``name``.

    The factory is called without arguments to instantiate the predictor
    when ``get_predictor(name)`` is invoked.

    Examples
    --------
    >>> @register_predictor("spliceai", description={"paper": "Jaganathan 2019"})
    ... def make_spliceai() -> BasePredictor:
    ...     return SpliceAIPredictor()
    """

    def decorator(factory: Callable[..., BasePredictor]) -> Callable[..., BasePredictor]:
        if name in _REGISTRY:
            raise ValueError(f"Predictor {name!r} already registered.")
        _REGISTRY[name] = factory
        if description is not None:
            _DESCRIPTIONS[name] = dict(description)
        logger.debug("Registered predictor %r via decorator", name)
        return factory

    return decorator


def get_predictor(name: str, **kwargs: object) -> BasePredictor:
    """Instantiate the registered predictor named ``name``.

    Loads built-ins on first call, then applies the YAML manifest.
    """
    _ensure_loaded()

    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Unknown predictor {name!r}. Registered: {available}. "
            f"Add to configs/predictors.yaml or use @register_predictor."
        )

    factory = _REGISTRY[name]
    predictor = factory(**kwargs) if kwargs else factory()

    if not isinstance(predictor, BasePredictor):
        raise TypeError(
            f"Factory for predictor {name!r} returned {type(predictor).__name__} "
            f"which does not satisfy the BasePredictor protocol."
        )

    return predictor


def list_predictors() -> List[Dict[str, object]]:
    """List all registered predictors with their descriptions.

    Returns a list of dicts suitable for pretty-printing on the CLI:
    ``[{"name": "spliceai", "training_build": "GRCh37", ...}, ...]``.
    """
    _ensure_loaded()

    rows: List[Dict[str, object]] = []
    for name in sorted(_REGISTRY.keys()):
        desc = dict(_DESCRIPTIONS.get(name, {}))
        desc.setdefault("name", name)
        rows.append(desc)
    return rows


def _ensure_loaded() -> None:
    """Load built-in predictors + manifest on first use."""
    global _MANIFEST_LOADED

    # Import the predictors package to run its decorators.
    importlib.import_module(f"{__package__}.predictors")

    if not _MANIFEST_LOADED:
        _load_manifest()
        _MANIFEST_LOADED = True


def _load_manifest() -> None:
    """Load predictors declared in ``configs/predictors.yaml``.

    Manifest entries have the form:

    .. code-block:: yaml

        predictors:
          my_custom_predictor:
            factory: "my_pkg.my_module:make_predictor"
            description:
              training_build: GRCh38
              annotation_source: mane
              notes: ...
    """
    manifest_path = Path(__file__).parent / "configs" / "predictors.yaml"
    if not manifest_path.exists():
        logger.debug("No predictors.yaml manifest at %s", manifest_path)
        return

    with open(manifest_path, "r") as fh:
        manifest = yaml.safe_load(fh) or {}

    predictors = manifest.get("predictors", {})
    for name, entry in predictors.items():
        if name in _REGISTRY:
            logger.debug("Manifest skipping %r: already built-in", name)
            continue

        factory_spec = entry.get("factory")
        if not factory_spec:
            logger.warning("Manifest entry %r has no 'factory' field — skipped", name)
            continue

        try:
            module_path, attr = factory_spec.split(":", 1)
            module = importlib.import_module(module_path)
            factory = getattr(module, attr)
        except Exception as exc:
            logger.warning(
                "Manifest entry %r failed to import factory %r: %s",
                name, factory_spec, exc,
            )
            continue

        _REGISTRY[name] = factory
        description = entry.get("description") or {}
        if description:
            _DESCRIPTIONS[name] = dict(description)
        logger.debug("Registered predictor %r from manifest", name)
