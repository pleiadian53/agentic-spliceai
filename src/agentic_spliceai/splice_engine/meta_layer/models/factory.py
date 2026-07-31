"""Architecture registry + factory for sequence-level meta-splice models.

Dispatches between architectures via a single ``arch`` name. Used by the
training driver and inference scripts so they don't have to import each
architecture's config/model directly.

Architectures are named for the mechanism that distinguishes them, never
numbered. A bare ``vN`` is ambiguous in this project: the *data corpus* also
has generations, so "v4" could mean either the cross-attention architecture or
the clean-annotation corpus. See ``docs/meta_layer/methods/naming_convention.md``.

============================  ===================================  ============
``arch``                      Distinguishing mechanism             Module
============================  ===================================  ============
``concat_fusion``             3 streams concatenated → 1x1 conv    ``meta_splice_model_v3``
``xattn_fusion``              sequence attends to base+mm signal   ``meta_splice_v4_xattn``
============================  ===================================  ============

The module and class names still carry the old ordinals **and must not be
renamed**: ``config.pt`` pickles the fully-qualified class path, so every
trained checkpoint on disk resolves
``...models.meta_splice_model_v3.MetaSpliceConfig`` by literal string. Renaming
the module or the dataclass breaks loading of all existing checkpoints. The
``arch`` name is the identifier that is safe to change; the Python name is not.

Legacy ``arch`` keys (``v3``, ``v4_xattn``) still resolve via
:data:`ARCH_ALIASES` with a deprecation warning.

Adding a new architecture:
    1. Implement the model + config in a new file in this package.
    2. Add a branch in :func:`build_model`.
    3. Append a mechanism-descriptive key to :data:`ARCH_REGISTRY`.
    4. Extend the loader dispatch in :mod:`.loader` for the new config type.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


#: Canonical architecture names, in lineage order.
ARCH_REGISTRY: Tuple[str, ...] = ("concat_fusion", "xattn_fusion")

#: Retired ordinal names → canonical names. Accepted on input so existing
#: commands, shell scripts and pod job files keep working.
ARCH_ALIASES: dict[str, str] = {
    "v3": "concat_fusion",
    "v4_xattn": "xattn_fusion",
}


def resolve_arch(arch: str) -> str:
    """Normalize an architecture name, accepting retired ordinal aliases.

    Parameters
    ----------
    arch : str
        Canonical name from :data:`ARCH_REGISTRY`, or a legacy key from
        :data:`ARCH_ALIASES`.

    Returns
    -------
    str
        The canonical architecture name.

    Raises
    ------
    ValueError
        If ``arch`` is neither canonical nor a known alias.

    Examples
    --------
    >>> resolve_arch("concat_fusion")
    'concat_fusion'
    >>> resolve_arch("v3")
    'concat_fusion'
    """
    if arch in ARCH_REGISTRY:
        return arch
    if arch in ARCH_ALIASES:
        canonical = ARCH_ALIASES[arch]
        logger.warning(
            "arch %r is a retired ordinal name; use %r. Ordinal arch names are "
            "ambiguous with corpus generations (see naming_convention.md).",
            arch, canonical,
        )
        return canonical
    raise ValueError(
        f"Unknown arch {arch!r}. Available: {ARCH_REGISTRY} "
        f"(legacy aliases: {sorted(ARCH_ALIASES)}). "
        f"Add new architectures by extending build_model() and ARCH_REGISTRY "
        f"in {__file__}."
    )


def _variant_for_mode(mode: str) -> str:
    return {"m1": "M1-S", "m2": "M2-S", "m3": "M3-S"}[mode]


def _num_classes_for_mode(mode: str) -> int:
    # All variants are 3-class (donor/acceptor/neither). M3-S keeps the same
    # head as M1-S/M2-S; its only difference is the 255=ignore loss mask for
    # annotated sites (recognizer + post-filter framing). The earlier 2-class
    # M3-S stub was from a superseded per-site binary sketch.
    return 3


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
        Architecture name. See :data:`ARCH_REGISTRY` for available choices;
        retired ordinal names in :data:`ARCH_ALIASES` are also accepted.
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
    arch = resolve_arch(arch)

    if arch == "concat_fusion":
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

    if arch == "xattn_fusion":
        from .meta_splice_v4_xattn import MetaSpliceXAttnConfig, MetaSpliceXAttnModel
        cfg = MetaSpliceXAttnConfig(
            variant=_variant_for_mode(mode),
            hidden_dim=hidden_dim,
            mm_channels=mm_channels,
            num_classes=_num_classes_for_mode(mode),
            activation=activation,
            **extra,
        )
        return MetaSpliceXAttnModel(cfg), cfg

    # resolve_arch() guarantees a member of ARCH_REGISTRY, so reaching here
    # means a registry entry was added without a build_model() branch.
    raise ValueError(
        f"arch {arch!r} is in ARCH_REGISTRY but has no build_model() branch "
        f"in {__file__}."
    )
