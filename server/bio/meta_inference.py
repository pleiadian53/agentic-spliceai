"""Meta-layer inference for the Bio Lab genome view (Phase C).

Given a gene and a meta-model name, this assembles the model's input
(``gene_data`` = sequence + dense base scores + multimodal features from the
Phase-A ``.npz`` cache), runs sliding-window inference, and returns BOTH the
base and the meta predictions as **base-format prediction dicts** — the same
shape ``predict_splice_sites_for_genes(..., output_format='dict')`` produces.

Returning the base-format dict means the existing evaluation + response
machinery (`evaluate_splice_site_predictions`, the genome-response builder) is
reused unchanged for the meta overlay. Both base and meta lines are sourced from
the SAME ``.npz`` frame (the base scores are exactly the OpenSpliceAI scores the
meta model was trained to refine), so the overlay is apples-to-apples and
position-aligned by construction.

The ``.npz`` is the Phase-A feature cache (``output/meta_layer/ui_cache/``,
warmed by ``02_build_showcase_feature_cache.py`` / the Phase-E warmer). If it is
missing, we raise ``FileNotFoundError`` with guidance rather than silently
streaming bigWigs on the request path.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene
from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
    _load_gene_npz,
)

from .meta_model_cache import get_meta_model_sync

logger = logging.getLogger(__name__)

UI_CACHE_DIR = Path("output/meta_layer/ui_cache/gene_cache")
_DEVICE = torch.device("cpu")


def _pred_dict(
    gene_id: str, gene_name: str, chrom: str, strand: str,
    start: int, end: int, scores: np.ndarray,
) -> dict:
    """Wrap a dense ``[L,3]`` score array as a base-format predictions dict.

    Channel order is the meta convention ``[donor, acceptor, neither]`` (matching
    the cached ``base_scores`` and the model output), so ``donor_prob`` = col 0.
    """
    L = scores.shape[0]
    positions = list(range(int(start), int(start) + L))
    return {
        gene_id: {
            "chrom": chrom,
            "gene_name": gene_name,
            "strand": strand,
            "gene_start": int(start),
            "gene_end": int(end),
            "positions": positions,
            "donor_prob": scores[:, 0].astype(float).tolist(),
            "acceptor_prob": scores[:, 1].astype(float).tolist(),
            "neither_prob": scores[:, 2].astype(float).tolist(),
        }
    }


def build_overlay_predictions(
    meta_model_name: str,
    gene_id: str,
    gene_name: str,
    chrom: str,
    strand: str,
    start: int,
    end: int,
) -> Tuple[dict, dict]:
    """Return ``(base_predictions, meta_predictions)`` as base-format dicts.

    Both share the same positions (the gene span ``[start, start+L)``), so the
    caller can evaluate and overlay them directly.

    Raises
    ------
    FileNotFoundError
        If the Phase-A feature ``.npz`` for ``gene_id`` is not cached.
    """
    model, cfg = get_meta_model_sync(meta_model_name)

    npz = UI_CACHE_DIR / f"{gene_id}.npz"
    if not npz.exists():
        raise FileNotFoundError(
            f"No feature cache for {gene_id} at {npz}. Warm it first: "
            f"python examples/UI_integration/02_build_showcase_feature_cache.py "
            f"--genes {gene_name}"
        )

    data = _load_gene_npz(npz)
    base_scores = np.asarray(data["base_scores"], dtype=np.float32)  # [L, 3]
    L = base_scores.shape[0]
    span = int(end) - int(start)
    if L != span:
        # The cache was built on the gene span; trust L and log the mismatch so a
        # silent off-by-N in the overlay positions never goes unnoticed.
        logger.warning(
            "%s: cache length %d != gene span %d (%s:%d-%d); using cache length",
            gene_id, L, span, chrom, start, end,
        )

    meta_probs = infer_full_gene(
        model, data,
        window_size=cfg.window_size,
        context_padding=cfg.effective_context_padding,
        device=_DEVICE,
    )  # [L, 3], meta order [donor, acceptor, neither]

    base_pred = _pred_dict(gene_id, gene_name, chrom, strand, start, end, base_scores)
    meta_pred = _pred_dict(gene_id, gene_name, chrom, strand, start, end, meta_probs)
    return base_pred, meta_pred
