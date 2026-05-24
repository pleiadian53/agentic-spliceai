#!/usr/bin/env python
"""M4 prototype: TDP-43 (TARDBP) ablation counterfactual on ALS cryptic sites.

The M4 idea: TDP-43 binding *represses* the STMN2/UNC13A cryptic exons; they
appear when TDP-43 is LOST (ALS/FTD). So model the disease as a counterfactual —
take the multimodal features WITH neuronal TDP-43 binding (WT), then ABLATE the
TDP-43 signal (simulate knockout), and ask whether the model now predicts the
cryptic sites become active.

Ablation is done at the DATA level (no architecture change): the RBP eCLIP parquet
is rbp-resolved, so KO = the augmented parquet with `rbp == "TARDBP"` rows removed.
Only the `rbp_n_bound` channel differs between WT and KO; all other channels and
the base scores are identical.

Interpretation (updated 2026-05-23): the v3_neuronal M1-S/M2-S models train on
the FULL RBP union INCLUDING genome-wide neuronal TDP-43 (SH-SY5Y/H9), so TDP-43
binding at these cryptic loci is now IN-distribution. That makes this M4 test
valid: if the model learned that TDP-43 binding REPRESSES the STMN2/UNC13A
cryptic exons, ablating it (KO) should RAISE P(splice) at the cryptic sites
(Δ(KO−WT) > 0). The earlier ENCODE-only models saw RBP≈0 at these neuronal genes
(K562/HepG2 barely express them), so neuronal TDP-43 was out-of-distribution and
the delta wasn't biologically meaningful. Read the per-site Δ below for the
verdict with the retrained models.

Usage:
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/04_tdp43_ablation_counterfactual.py
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment

setup_example_environment()

from als_cryptic_sites import EVENTS  # noqa: E402

CACHE_DIR = Path("output/meta_layer/ui_cache/gene_cache")
MODELS = {
    "M1-S": Path("output/meta_layer/m1s_v3_neuronal"),
    "M2-S": Path("output/meta_layer/m2s_v3_neuronal"),
}
PROBE_HALF_WINDOW = 3


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


def main() -> int:
    import torch
    import polars as pl
    from agentic_spliceai.splice_engine.resources import get_model_resources
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
        DenseFeatureExtractor, DenseFeatureConfig, CHANNEL_NAMES,
    )
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import _load_gene_npz
    from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene

    parser = argparse.ArgumentParser(description="M4 TDP-43 ablation counterfactual")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    rbp_ch = CHANNEL_NAMES.index("rbp_n_bound")

    # WT = the full all-cell-line RBP union the models trained on (resolved
    # internally by the extractor — no flag, exactly like every other modality).
    # KO = that union with TARDBP removed (simulate TDP-43 knockout). Only the
    # rbp_n_bound channel differs between WT and KO.
    ext_wt = DenseFeatureExtractor(DenseFeatureConfig(build="GRCh38"))
    peaks = ext_wt._get_eclip_peaks()
    if peaks is None:
        print("ERROR: no RBP eCLIP parquet resolved — cannot run TDP-43 ablation.")
        return 1
    ko_path = Path(tempfile.gettempdir()) / "eclip_ko_tardbp.parquet"
    peaks.filter(pl.col("rbp") != "TARDBP").write_parquet(ko_path)
    ext_ko = DenseFeatureExtractor(DenseFeatureConfig(build="GRCh38", eclip_parquet=ko_path))

    reg = get_model_resources("openspliceai").get_registry()
    ga = extract_gene_annotations(str(reg.get_gtf_path()), verbosity=0)
    models = {name: load_model(d, device) for name, d in MODELS.items()}

    for ev in EVENTS:
        gid = f"gene-{ev.gene}"
        npz = CACHE_DIR / f"{gid}.npz"
        if not npz.exists():
            print(f"  [skip] no cache for {gid}")
            continue
        row = ga.filter(pl.col("gene_id") == gid).row(0, named=True)
        gstart, gend = int(row["start"]), int(row["end"])
        data = _load_gene_npz(npz)
        L = len(data["labels"])

        # Recompute the RBP channel WT (TDP-43 present) vs KO (removed) over the gene span
        rbp_wt = ext_wt._query_rbp(ev.chrom, gstart, gend)
        rbp_ko = ext_ko._query_rbp(ev.chrom, gstart, gend)

        def with_rbp(vec):
            mm = data["mm_features"].copy()
            mm[:, rbp_ch] = vec[:L]
            return {**data, "mm_features": mm}

        data_wt, data_ko = with_rbp(rbp_wt), with_rbp(rbp_ko)

        print(f"\n{'='*80}\n{ev.gene} ({ev.chrom}{ev.strand}) — TDP-43 ablation\n{'='*80}")
        for s in ev.sites:
            idx = s.pos - gstart
            if not (0 <= idx < L):
                continue
            lo, hi = max(0, idx - PROBE_HALF_WINDOW), min(L, idx + PROBE_HALF_WINDOW + 1)
            wt_rbp = float(rbp_wt[lo:hi].max())
            ko_rbp = float(rbp_ko[lo:hi].max())
            print(f"  {s.kind} @ {ev.chrom}:{s.pos:,}  |  TDP-43 feature "
                  f"WT={wt_rbp:.0f} → KO={ko_rbp:.0f} (Δ={wt_rbp - ko_rbp:+.0f})")
            for name, (model, cfg) in models.items():
                pw = infer_full_gene(model, data_wt, window_size=cfg.window_size,
                                     context_padding=cfg.effective_context_padding, device=device)
                pk = infer_full_gene(model, data_ko, window_size=cfg.window_size,
                                     context_padding=cfg.effective_context_padding, device=device)
                ps_wt = float((pw[lo:hi, 0] + pw[lo:hi, 1]).max())
                ps_ko = float((pk[lo:hi, 0] + pk[lo:hi, 1]).max())
                print(f"        {name}: P(splice) WT={ps_wt:.3f}  KO={ps_ko:.3f}  "
                      f"Δ(KO−WT)={ps_ko - ps_wt:+.3f}")

    ext_wt.close(); ext_ko.close()
    print(f"\n{'='*80}")
    print("Models trained on the full RBP union INCLUDING neuronal TDP-43 (SH-SY5Y/H9),")
    print("so TDP-43 binding at these cryptic loci is in-distribution. If the model learned")
    print("TDP-43 REPRESSION, KO (TDP-43 removed) should RAISE P(splice) at the cryptic sites")
    print("(Δ(KO−WT) > 0). Read the per-site Δ above for the verdict.")
    print('='*80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
