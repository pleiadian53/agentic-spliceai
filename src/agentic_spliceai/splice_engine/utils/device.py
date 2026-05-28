"""Torch device resolution.

One canonical rule so scripts "do the right thing" on every machine without a
flag: prefer CUDA when present, else CPU. **MPS is never auto-selected** — torch
2.5.x has BatchNorm1d-backward autograd bugs on the Apple MPS backend, and CPU is
fast enough for the small meta-layer models. This is why a GPU pod uses the GPU
automatically while an M1 laptop falls back to CPU (not MPS).
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a CLI ``--device`` spec to a ``torch.device``.

    Parameters
    ----------
    spec :
        ``"auto"`` (default) → ``cuda`` if available, else ``cpu`` — **never MPS
        automatically**. Any explicit value (``"cuda"``, ``"cpu"``, ``"mps"``,
        ``"cuda:1"``, …) is honored as-is so MPS can still be forced deliberately.

    Returns
    -------
    torch.device
    """
    if spec == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")  # deliberately skip MPS (see module docstring)
        log.info("Device auto-resolved to '%s' (cuda_available=%s)", dev, torch.cuda.is_available())
        return dev
    if spec == "mps":
        log.warning("Device 'mps' explicitly requested — known torch 2.5.x BatchNorm1d "
                    "autograd bugs; prefer 'auto' (cuda/cpu).")
    return torch.device(spec)
