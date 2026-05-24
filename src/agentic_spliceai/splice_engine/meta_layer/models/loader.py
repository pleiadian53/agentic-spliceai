"""Canonical loader for trained sequence-level meta-splice models (M*-S).

A meta-splice checkpoint carries its architecture in ``config.pt``; the config
*type* determines which model class to instantiate:

  - ``MetaSpliceConfig``       → v3 dilated-CNN (:class:`MetaSpliceModel`)
  - ``MetaSpliceXAttnConfig``  → v4 cross-attention (:class:`MetaSpliceXAttnModel`)

This is the single source of truth for that dispatch so callers (the Bio Lab UI
meta-model cache, the UI-integration example scripts, eval drivers) never
re-implement it and never hardcode one architecture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple


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
        ``variant``, etc.).
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
