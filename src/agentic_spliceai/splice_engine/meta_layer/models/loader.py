"""Canonical loader for trained sequence-level meta-splice models (M*-S).

A meta-splice checkpoint carries its architecture in ``config.pt``; the config
*type* determines which model class to instantiate:

  - ``MetaSpliceConfig``       → ``concat_fusion`` (:class:`MetaSpliceModel`)
  - ``MetaSpliceXAttnConfig``  → ``xattn_fusion``  (:class:`MetaSpliceXAttnModel`)

The config class is the *checkpoint's* record of its architecture, and it is
load-bearing: ``config.pt`` pickles the fully-qualified class path, so neither
the class nor its module may be renamed without breaking every checkpoint on
disk. The ordinals in those Python names (``meta_splice_model_v3``,
``meta_splice_v4_xattn``) are frozen history — the architecture *identifier*
is the ``arch`` name above. See ``docs/meta_layer/methods/naming_convention.md``.

This is the single source of truth for that dispatch so callers (the Bio Lab UI
meta-model cache, the UI-integration example scripts, eval drivers) never
re-implement it and never hardcode one architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

#: Config dataclass name → canonical ``arch`` name in
#: :data:`...models.factory.ARCH_REGISTRY`. Keyed by class *name* rather than
#: the class itself so this stays importable without pulling in torch.
CONFIG_TYPE_TO_ARCH: dict[str, str] = {
    "MetaSpliceConfig": "concat_fusion",
    "MetaSpliceXAttnConfig": "xattn_fusion",
}


def arch_of_config(cfg: object) -> str:
    """Return the canonical ``arch`` name a loaded config represents.

    The checkpoint's config class is the authoritative record of which
    architecture produced it — more reliable than a directory name or a
    registry entry, both of which are hand-maintained.

    Raises
    ------
    TypeError
        If the config class is not a known meta-splice architecture.
    """
    name = type(cfg).__name__
    if name not in CONFIG_TYPE_TO_ARCH:
        raise TypeError(
            f"Unknown meta-model config type {name!r}. Known: "
            f"{sorted(CONFIG_TYPE_TO_ARCH)}"
        )
    return CONFIG_TYPE_TO_ARCH[name]


def load_meta_model(model_dir: Path | str, device) -> Tuple[object, object]:
    """Load a trained meta-splice model, dispatching on its config type.

    Parameters
    ----------
    model_dir : Path or str
        Directory containing ``config.pt`` and ``best.pt``.
    device : torch.device or str
        Target device for the loaded model.

    Returns
    -------
    (model, config)
        ``model`` is in ``eval()`` mode on ``device``; ``config`` is the loaded
        config dataclass (carries ``window_size``, ``effective_context_padding``,
        ``variant``, etc.). Pass it to :func:`arch_of_config` for the
        architecture name.
    """
    import torch
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceConfig, MetaSpliceModel,
    )
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_v4_xattn import (
        MetaSpliceXAttnConfig, MetaSpliceXAttnModel,
    )

    model_dir = Path(model_dir)
    torch.serialization.add_safe_globals([MetaSpliceConfig, MetaSpliceXAttnConfig])
    cfg = torch.load(model_dir / "config.pt", map_location="cpu", weights_only=True)

    if isinstance(cfg, MetaSpliceXAttnConfig):
        model = MetaSpliceXAttnModel(cfg)
    elif isinstance(cfg, MetaSpliceConfig):
        model = MetaSpliceModel(cfg)
    else:
        raise TypeError(
            f"Unknown meta-model config type {type(cfg).__name__} in {model_dir}"
        )

    model.load_state_dict(
        torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model, cfg
