#!/usr/bin/env python
"""Experiment: can the meta layer (M1-S / M2-S) detect ALS cryptic splice sites
that the base model misses?

The TDP-43-repressed cryptic sites in STMN2 / UNC13A are a true out-of-distribution
probe: they are absent from every training annotation AND from the GTEx junction
feature channel (zero junction evidence). So this asks whether the meta layer's
sequence + conservation pathway can flag splice sites that have no junction
support and weren't in training — the defective isoforms ALS pathology induces.

For each gene it runs three models on the cached gene window:
  - **base** (OpenSpliceAI scores, read from the cache),
  - **M1-S** (canonical meta model, v3),
  - **M2-S** (alternative-site meta model, v3),
and reports P(splice)=donor+acceptor at:
  - the **cryptic** splice sites (from als_cryptic_sites.py), and
  - the **canonical** splice sites (npz labels) as a positive control.

Coordinate mapping (verified): npz index = genomic_1based - annotation_start.
Probes take the max over a small ±window to be robust to the exact
last-exonic/first-intronic site convention.

Usage:
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/03_cryptic_site_detection.py
    # writes a browser-viz TSV of base-vs-meta tracks at/around the cryptic loci
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment

setup_example_environment()

from als_cryptic_sites import EVENTS  # noqa: E402  (local module, after path setup)

CACHE_DIR = Path("output/meta_layer/ui_cache/gene_cache")
MODELS = {
    "M1-S": Path("output/meta_layer/m1s_v2_logit_blend"),
    "M2-S": Path("output/meta_layer/m2s_v3_baseline_repro"),
}
PROBE_HALF_WINDOW = 3  # ±bp around a site (site-convention slack)


def load_model(model_dir: Path, device):
    import torch
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceConfig, MetaSpliceModel,
    )
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_v4_xattn import (
        MetaSpliceXAttnConfig, MetaSpliceXAttnModel,
    )
    torch.serialization.add_safe_globals([MetaSpliceConfig, MetaSpliceXAttnConfig])
    cfg = torch.load(model_dir / "config.pt", map_location="cpu", weights_only=True)
    Model = MetaSpliceXAttnModel if isinstance(cfg, MetaSpliceXAttnConfig) else MetaSpliceModel
    model = Model(cfg)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device).eval()
    return model, cfg


def p_splice_window(probs: np.ndarray, idx: int, half: int) -> Tuple[float, float, float]:
    """Max donor, max acceptor, max P(splice)=donor+acc over [idx-half, idx+half]."""
    lo, hi = max(0, idx - half), min(len(probs), idx + half + 1)
    w = probs[lo:hi]
    return float(w[:, 0].max()), float(w[:, 1].max()), float((w[:, 0] + w[:, 1]).max())


def main() -> int:
    import torch
    from agentic_spliceai.splice_engine.resources import get_model_resources
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene
    import polars as pl

    parser = argparse.ArgumentParser(description="Probe base/M1-S/M2-S at ALS cryptic sites")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("output/meta_layer/ui_cache/als_cryptic"))
    args = parser.parse_args()
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_model_resources("openspliceai").get_registry()
    ga = extract_gene_annotations(str(reg.get_gtf_path()), verbosity=0)

    print("Loading meta models...")
    models = {name: load_model(d, device) for name, d in MODELS.items()}

    track_rows: List[dict] = []  # for browser-viz TSV

    for ev in EVENTS:
        gid = f"gene-{ev.gene}"
        npz_path = CACHE_DIR / f"{gid}.npz"
        if not npz_path.exists():
            print(f"  [skip] no cache for {gid} (run 02_build_showcase_feature_cache.py)")
            continue
        row = ga.filter(pl.col("gene_id") == gid).row(0, named=True)
        gstart = int(row["start"])              # annotation start; index = pos_1based - gstart
        data = _load_gene_npz(npz_path)
        base = data["base_scores"]              # [L,3] OpenSpliceAI
        labels = data["labels"]
        L = len(labels)

        # Run meta models
        preds = {"base": base}
        for name, (model, cfg) in models.items():
            preds[name] = infer_full_gene(
                model, data, window_size=cfg.window_size,
                context_padding=cfg.effective_context_padding, device=device,
            )

        # Positive control: canonical sites (from labels) — mean P(splice)
        canon_idx = np.where(labels != 2)[0]
        print(f"\n{'='*78}\n{ev.gene} ({ev.chrom}{ev.strand})  {ev.mechanism}\n{'='*78}")
        print(f"  Positive control — {len(canon_idx)} canonical splice sites, mean P(splice):")
        for name in ("base", *MODELS):
            ps = (preds[name][canon_idx, 0] + preds[name][canon_idx, 1])
            print(f"     {name:5s}: mean={ps.mean():.3f}  min={ps.min():.3f}")

        # Cryptic sites
        print(f"  CRYPTIC sites (absent from training + GTEx junctions):")
        for s in ev.sites:
            idx = s.pos - gstart
            if not (0 <= idx < L):
                print(f"     {s.kind}@{s.pos:,}: index {idx} out of range — skip")
                continue
            print(f"     {s.kind} @ {ev.chrom}:{s.pos:,} (idx {idx}) — {s.note}")
            for name in ("base", *MODELS):
                d, a, ps = p_splice_window(preds[name], idx, PROBE_HALF_WINDOW)
                detect = "DETECTS" if ps >= 0.5 else ("weak" if ps >= 0.1 else "misses")
                print(f"        {name:5s}: donor={d:.3f} acc={a:.3f} Pspl={ps:.3f} -> {detect}")
                track_rows.append({
                    "gene": ev.gene, "chrom": ev.chrom, "pos": s.pos, "site_kind": s.kind,
                    "model": name, "donor": round(d, 4), "acceptor": round(a, 4),
                    "p_splice": round(ps, 4),
                })

    # ── Browser-viz TSV (base vs meta at cryptic loci) ───────────────────────
    if track_rows:
        out = args.out_dir / "cryptic_site_scores.tsv"
        pl.DataFrame(track_rows).write_csv(out, separator="\t")
        print(f"\nWrote {out}  ({len(track_rows)} rows: gene × site × model)")

    print(f"\n{'='*78}")
    print("Read: base 'misses' a cryptic site (low P) where a meta model 'DETECTS' it")
    print("(high P) = the meta layer recovered an ALS cryptic site from sequence alone.")
    print("If all three miss the cryptic sites, that motivates M3 (novel-site discovery).")
    print('='*78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
