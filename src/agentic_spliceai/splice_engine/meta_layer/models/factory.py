"""Architecture registry + factory for sequence-level meta-splice models.

Dispatches between architecture variants (``v3`` dilated CNN, future
``v4_attn`` / ``v4_xattn`` / ``v5_transformer``) via a single ``arch`` name.
Used by the training driver and inference scripts so they don't have to
import each architecture's config/model directly.

Adding a new architecture:
    1. Implement the model + config in a new file in this package.
    2. Add a branch in :func:`build_model`.
    3. Append the key to :data:`ARCH_REGISTRY`.
    4. Add a tiny-config entry to ``tests/meta_layer/test_models_smoke.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


ARCH_REGISTRY: Tuple[str, ...] = ("v3",)


def _variant_for_mode(mode: str) -> str:
    return {"m1": "M1-S", "m2": "M2-S", "m3": "M3-S"}[mode]


def _num_classes_for_mode(mode: str) -> int:
    return 2 if mode == "m3" else 3


def build_model(
    arch: str,
    *,
    mode: str,
    hidden_dim: int,
    mm_channels: int,
    activation: str = "gelu",
    **extra: Any,
) -> Tuple[nn.Module, Any]:
    """Construct a meta-splice model + its config by architecture name.

    Parameters
    ----------
    arch : str
        Architecture key. See :data:`ARCH_REGISTRY` for available choices.
    mode : str
        Model variant ("m1", "m2", "m3"). Determines ``variant`` and
        ``num_classes`` on the config.
    hidden_dim : int
        Hidden dimension shared across streams.
    mm_channels : int
        Number of multimodal feature channels.
    activation : str
        Activation function for CNN/Transformer blocks.
    **extra :
        Architecture-specific overrides forwarded to the config dataclass.
        Unknown keys raise ``TypeError`` from the config — this is
        intentional fail-loud behavior.

    Returns
    -------
    (model, cfg)
        Instantiated model and its config. Config type depends on ``arch``.
    """
    if arch == "v3":
        from .meta_splice_model_v3 import MetaSpliceConfig, MetaSpliceModel
        cfg = MetaSpliceConfig(
            variant=_variant_for_mode(mode),
            hidden_dim=hidden_dim,
            mm_channels=mm_channels,
            num_classes=_num_classes_for_mode(mode),
            activation=activation,
            **extra,
        )
        return MetaSpliceModel(cfg), cfg

    raise ValueError(
        f"Unknown arch {arch!r}. Available: {ARCH_REGISTRY}. "
        f"Add new architectures by extending build_model() and ARCH_REGISTRY "
        f"in {__file__}."
    )
