#!/usr/bin/env python
"""Phase A (M1-S/M2-S → Bio Lab UI): build per-gene dense feature cache + verify M1-S.

This is the foundation for the UI integration (see
``dev/meta_layer/UI_integration/m1s_m2s_ui_integration_plan.md``).  It
**reuses the training gene-cache builder** (``build_gene_cache`` in
``meta_layer/data/sequence_level_dataset.py``) rather than writing a
parallel feature provider, so the dense ``[L, 9]`` features fed to the
model at demo time are byte-identical to what the model trained on
(same ``DenseFeatureExtractor``, same channel order, same fills).

What it does
------------
1. Resolves FASTA / splice sites / base scores via the resource manager
   (OpenSpliceAI, MANE GRCh38) — no hardcoded paths.
2. For each showcase gene, builds a dense-feature ``.npz`` at a flat
   UI cache dir (default ``output/meta_layer/ui_cache/gene_cache``).
   Conservation / epigenetic / chromatin channels stream from remote
   bigWigs (one range query per track per gene); junction + RBP come
   from local parquets.  Existing ``.npz`` files are reused (resume-safe).
3. Loads the trained M1-S checkpoint (architecture auto-detected from
   ``config.pt``) and runs ``infer_full_gene`` end-to-end, reporting how
   the meta layer changes scores at the gene's true splice sites
   (the "recover the base model's misses" story).

Usage
-----
    # Default: BRCA1 + ALS showcase genes, build + verify with M1-S
    python 12_build_showcase_feature_cache.py

    # Specific genes only, no verification (just warm the cache)
    python 12_build_showcase_feature_cache.py --genes BRCA1 TP53 --no-verify

    # Point at a local bigWig cache to avoid slow remote streaming
    python 12_build_showcase_feature_cache.py --bigwig-cache data/cache/bigwig
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment

setup_example_environment()

log = logging.getLogger(__name__)

# Default showcase set: BRCA1 (familiar cancer gene) + ALS panel.  All are
# in MANE GRCh38 and have precomputed OpenSpliceAI base scores.
DEFAULT_GENES = ["BRCA1", "STMN2", "UNC13A", "SOD1", "TARDBP", "FUS", "C9orf72"]

DEFAULT_MODEL_DIR = Path("output/meta_layer/m1s_v4_cleanannot")
DEFAULT_CACHE_DIR = Path("output/meta_layer/ui_cache/gene_cache")

# The 6 dense bigWig-derived channels (conservation + epigenetic + chromatin).
# These are ~60-100% non-zero across any multi-kb gene — an all-zero column
# means the remote bigWig stream failed (e.g. transient UCSC SSL error) and the
# extractor silently zero-filled it.  Junction (6,7) and RBP (8) are legitimately
# sparse/zero and excluded from this check.  Index order matches CHANNEL_NAMES.
DENSE_BIGWIG_CHANNELS = (0, 1, 2, 3, 4, 5)
HEALTH_MIN_NONZERO_FRAC = 0.01

# RBP (eCLIP) is resolved internally by DenseFeatureExtractor like every other
# modality — it loads the full all-cell-line union by default (no flag, no
# zeroing). The v3+ M*-S checkpoints train on that same channel, so the cache is
# byte-identical to training with zero special-casing here.


def degraded_channels(mm_features: np.ndarray) -> List[int]:
    """Return indices of dense bigWig channels that came back (near-)all-zero.

    A degraded channel signals a failed remote stream, not real biology —
    the cache would no longer be byte-identical to training.
    """
    bad = []
    for idx in DENSE_BIGWIG_CHANNELS:
        if idx >= mm_features.shape[1]:
            continue
        if float((mm_features[:, idx] != 0).mean()) < HEALTH_MIN_NONZERO_FRAC:
            bad.append(idx)
    return bad


def load_meta_model(model_dir: Path, device):
    """Load a trained meta-splice model, auto-detecting its architecture.

    M1-S and M2-S checkpoints carry their config in ``config.pt``.  The
    config *type* determines the architecture: ``MetaSpliceConfig`` → v3
    dilated-CNN; ``MetaSpliceXAttnConfig`` → v4 cross-attention.  This
    mirrors the dispatch the Phase B in-memory model cache will use.
    """
    import torch
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceConfig,
        MetaSpliceModel,
    )
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_v4_xattn import (
        MetaSpliceXAttnConfig,
        MetaSpliceXAttnModel,
    )

    torch.serialization.add_safe_globals([MetaSpliceConfig, MetaSpliceXAttnConfig])
    cfg = torch.load(model_dir / "config.pt", map_location="cpu", weights_only=True)

    if isinstance(cfg, MetaSpliceXAttnConfig):
        model = MetaSpliceXAttnModel(cfg)
    elif isinstance(cfg, MetaSpliceConfig):
        model = MetaSpliceModel(cfg)
    else:
        raise TypeError(f"Unknown config type {type(cfg).__name__} in {model_dir}")

    model.load_state_dict(
        torch.load(model_dir / "best.pt", map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model, cfg


def report_gene(
    gene_id: str,
    probs: np.ndarray,
    base: np.ndarray,
    labels: np.ndarray,
    mm_features: np.ndarray,
    elapsed: float,
) -> dict:
    """Print and return a base-vs-meta comparison at the gene's true sites."""
    rowsum = probs.sum(axis=1)
    bad = degraded_channels(mm_features)
    health = "OK" if not bad else f"DEGRADED channels {bad}"
    print(
        f"\n### {gene_id}  len={len(labels):,}  infer={elapsed:.1f}s  "
        f"rowsum[{rowsum.min():.3f},{rowsum.max():.3f}]  features={health}"
    )
    summary = {"gene_id": gene_id, "length": int(len(labels)), "features_ok": not bad}
    for cls, nm in [(0, "donor"), (1, "acceptor")]:
        pos = np.where(labels == cls)[0]
        if len(pos) == 0:
            continue
        base_hit = int((base[pos, cls] >= 0.5).sum())
        meta_hit = int((probs[pos, cls] >= 0.5).sum())
        recovered = int(((base[pos, cls] < 0.5) & (probs[pos, cls] >= 0.5)).sum())
        print(
            f"   {nm:8s} {len(pos):3d} true sites | "
            f"base mean={base[pos, cls].mean():.3f} hit@0.5={base_hit:3d} | "
            f"meta mean={probs[pos, cls].mean():.3f} hit@0.5={meta_hit:3d} | "
            f"recovered={recovered}"
        )
        summary[nm] = {
            "n_true": int(len(pos)),
            "base_mean": float(base[pos, cls].mean()),
            "meta_mean": float(probs[pos, cls].mean()),
            "base_hits_at_0.5": base_hit,
            "meta_hits_at_0.5": meta_hit,
            "recovered_misses": recovered,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase A: build dense feature cache for UI showcase genes + verify M1-S",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES,
                        help="Gene symbols or IDs (default: BRCA1 + ALS panel)")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="Flat dense-feature cache dir (one .npz per gene)")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help="Trained M1-S dir (config.pt + best.pt)")
    parser.add_argument("--bigwig-cache", type=Path, default=None,
                        help="Local bigWig cache dir (avoids slow remote streaming)")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Rebuild attempts for genes with failed-stream channels")
    parser.add_argument("--no-verify", action="store_true",
                        help="Build the cache only; skip M1-S inference")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import pandas as pd
    import polars as pl

    from agentic_spliceai.splice_engine.resources import get_model_resources
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
        DenseFeatureExtractor,
        DenseFeatureConfig,
    )
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        build_gene_cache,
    )

    # ── Resolve resources (OpenSpliceAI base model, MANE annotations) ────
    resources = get_model_resources("openspliceai")
    registry = resources.get_registry()
    fasta_path = str(resources.get_fasta_path())
    splice_sites_path = Path(registry.stash) / "splice_sites_enhanced.tsv"
    base_scores_dir = registry.get_base_model_eval_dir("openspliceai") / "precomputed"
    gtf_path = str(registry.get_gtf_path())

    print(f"  FASTA:        {fasta_path}")
    print(f"  Splice sites: {splice_sites_path}")
    print(f"  Base scores:  {base_scores_dir}")
    print(f"  Cache dir:    {args.cache_dir}")

    splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")
    gene_annotations = extract_gene_annotations(gtf_path, verbosity=0)

    # ── Resolve gene symbols → canonical gene_id (matches training cache) ─
    resolved: List[str] = []
    for g in args.genes:
        row = gene_annotations.filter(pl.col("gene_id") == g)
        if row.height == 0:
            row = gene_annotations.filter(pl.col("gene_name") == g)
        if row.height == 0:
            log.warning("Gene %s not found in MANE annotations; skipping", g)
            continue
        resolved.append(row.row(0, named=True)["gene_id"])
    if not resolved:
        print("ERROR: no requested genes resolved")
        return 1
    print(f"  Showcase genes ({len(resolved)}): {', '.join(resolved)}")

    # ── Build dense feature cache (REUSE training path) ──────────────────
    # build_gene_cache streams one bigWig range query per track per gene and
    # is resume-safe (skips existing .npz).  Transient UCSC SSL errors can
    # silently zero-fill a channel, so after each pass we health-check the
    # dense channels and rebuild any degraded gene (deleting its .npz first,
    # since the builder would otherwise skip it).  Directly de-risks the plan's
    # #1 risk: "bigWig streaming slow/flaky".
    print(f"\n  Building dense feature cache (9 channels)...")
    t0 = time.time()
    pending = list(resolved)
    for attempt in range(1, args.max_retries + 1):
        feat_config = DenseFeatureConfig(build="GRCh38", bigwig_cache_dir=args.bigwig_cache)
        extractor = DenseFeatureExtractor(feat_config)
        build_gene_cache(
            pending, splice_sites_df, fasta_path,
            base_scores_dir, extractor, gene_annotations,
            cache_dir=args.cache_dir,
        )
        extractor.close()

        # Health check: find genes with a degraded (failed-stream) channel
        from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
            CHANNEL_NAMES,
        )
        degraded = []
        for gid in resolved:
            npz = args.cache_dir / f"{gid}.npz"
            if not npz.exists():
                continue
            mm = np.load(npz, allow_pickle=True)["mm_features"]
            bad = degraded_channels(mm)
            if bad:
                degraded.append((gid, bad))
        if not degraded:
            break
        bad_desc = ", ".join(
            f"{g} [{','.join(CHANNEL_NAMES[i] for i in idxs)}]" for g, idxs in degraded
        )
        if attempt < args.max_retries:
            log.warning("Attempt %d: degraded channels in %s — deleting + rebuilding",
                        attempt, bad_desc)
            for gid, _ in degraded:
                (args.cache_dir / f"{gid}.npz").unlink(missing_ok=True)
            pending = [g for g, _ in degraded]
        else:
            log.error("Still degraded after %d attempts: %s. These genes have "
                      "zero-filled channels (NOT byte-identical to training).",
                      args.max_retries, bad_desc)
    print(f"  Cached {sum((args.cache_dir / f'{g}.npz').exists() for g in resolved)} "
          f"genes in {time.time() - t0:.1f}s")

    if args.no_verify:
        return 0

    # ── Verify: run M1-S end-to-end on each cached gene ──────────────────
    import torch
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene

    device = torch.device(args.device)
    model, cfg = load_meta_model(args.model_dir, device)
    ctx = cfg.effective_context_padding
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'=' * 70}")
    print(f"Verify: {cfg.variant} ({type(model).__name__}), {n_params:,} params, "
          f"blend={cfg.blend_mode}, RF={cfg.receptive_field}, ctx_pad={ctx}")
    print(f"  Model dir: {args.model_dir}")
    print('=' * 70)

    summaries = []
    for gene_id in resolved:
        npz = args.cache_dir / f"{gene_id}.npz"
        if not npz.exists():
            log.warning("No cache for %s; skipping verification", gene_id)
            continue
        data = _load_gene_npz(npz)
        t0 = time.time()
        probs = infer_full_gene(
            model, data, window_size=cfg.window_size,
            context_padding=ctx, device=device,
        )
        summaries.append(
            report_gene(gene_id, probs, data["base_scores"], data["labels"],
                        data["mm_features"], time.time() - t0)
        )

    # ── Roll-up ──────────────────────────────────────────────────────────
    total_recovered = sum(
        s.get(k, {}).get("recovered_misses", 0)
        for s in summaries for k in ("donor", "acceptor")
    )
    print(f"\n{'=' * 70}")
    print(f"Phase A verification: {len(summaries)} genes, "
          f"{total_recovered} base-model misses recovered by M1-S @ 0.5")
    print('=' * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
