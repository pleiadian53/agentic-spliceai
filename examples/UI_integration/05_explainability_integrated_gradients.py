#!/usr/bin/env python
"""Integrated Gradients explainability for M2-S at an ALS cryptic splice site.

Question (NEXT STEPS #4): *why* does the meta layer assign splice probability to
the UNC13A cryptic donor that the base model misses?  Hypothesis: it is the
intrinsic GT–AG splice strength carried by the DNA sequence channel, not a
TDP-43-specific signal in the RBP channel (which at neuronal genes is weak —
see 04_tdp43_ablation_counterfactual.py).

Method
------
Build one inference window centred on the cryptic site and run captum's
Integrated Gradients on the model's P(splice) = P(donor)+P(acceptor) at that
position, attributing w.r.t. (a) the one-hot DNA sequence and (b) the 9
multimodal channels.  Base scores are held FIXED as the IG ``additional_forward_args``
so the attribution isolates what the *meta* layer adds on top of the base prior.

Outputs (organised under output/meta_layer/ig_<gene>_<kind>/)
-------------------------------------------------------------
- ``ig_summary.json``     : P(splice), per-channel multimodal attribution, top
                            sequence positions, GT/AG dinucleotide check, IG
                            convergence delta.
- ``ig_attribution.npz``  : raw per-base sequence attribution + per-channel,
                            per-position multimodal attribution arrays.
- ``ig_figure.png``       : (top) sequence saliency track around the site,
                            (bottom) 9-channel multimodal attribution bar.

Usage (pod uses the container python; locally the agentic-spliceai env):
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/05_explainability_integrated_gradients.py \
        --gene UNC13A --kind donor \
        --model-dir output/meta_layer/m2s_v3_neuronal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment

setup_example_environment()

from als_cryptic_sites import EVENTS  # noqa: E402

DEFAULT_MODEL_DIR = Path("output/meta_layer/m2s_v3_neuronal")
DEFAULT_CACHE_DIR = Path("output/meta_layer/ui_cache/gene_cache")
LOGO_HALF_WINDOW = 40  # bp each side of the site for the sequence saliency track


def load_meta_model(model_dir: Path, device):
    """Load a trained meta-splice model, dispatching on its config type."""
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
    from captum.attr import IntegratedGradients

    from agentic_spliceai.splice_engine.resources import get_model_resources
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    from agentic_spliceai.splice_engine.features.dense_feature_extractor import CHANNEL_NAMES
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz, _one_hot_encode,
    )

    parser = argparse.ArgumentParser(description="Integrated Gradients at an ALS cryptic site")
    parser.add_argument("--gene", default="UNC13A", help="Gene symbol (must be in als_cryptic_sites)")
    parser.add_argument("--kind", default="donor", choices=["donor", "acceptor"],
                        help="Which cryptic site to explain")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-steps", type=int, default=64, help="IG interpolation steps")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    # ── Resolve the cryptic site ─────────────────────────────────────────
    ev = next((e for e in EVENTS if e.gene == args.gene), None)
    if ev is None:
        print(f"ERROR: {args.gene} not in als_cryptic_sites.EVENTS")
        return 1
    site = next((s for s in ev.sites if s.kind == args.kind), None)
    if site is None:
        print(f"ERROR: no {args.kind} site for {args.gene}")
        return 1
    out_dir = args.output_dir or Path(f"output/meta_layer/ig_{args.gene}_{args.kind}".lower())
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + cached features for the gene ────────────────────────
    model, cfg = load_meta_model(args.model_dir, device)
    gid = f"gene-{ev.gene}"
    npz = args.cache_dir / f"{gid}.npz"
    if not npz.exists():
        print(f"ERROR: no feature cache for {gid} at {npz} — run 02 first.")
        return 1
    data = _load_gene_npz(npz)
    sequence = data["sequence"]
    base_scores = data["base_scores"]          # [L, 3]
    mm_features = data["mm_features"]          # [L, C]
    gene_len = len(sequence)

    reg = get_model_resources("openspliceai").get_registry()
    ga = extract_gene_annotations(str(reg.get_gtf_path()), verbosity=0)
    gstart = int(ga.filter(pl.col("gene_id") == gid).row(0, named=True)["start"])
    donor_idx = site.pos - gstart              # index into the + strand gene span
    if not (0 <= donor_idx < gene_len):
        print(f"ERROR: site {site.pos} maps to idx {donor_idx} outside gene [0,{gene_len})")
        return 1

    # ── Build ONE window centred on the cryptic site (mirrors infer_full_gene) ─
    W = cfg.window_size
    ctx = cfg.effective_context_padding
    out_start = min(max(0, donor_idx - W // 2), max(0, gene_len - W))
    center_idx = donor_idx - out_start                       # output-frame index
    seq_start = max(0, out_start - ctx // 2)
    seq_end = min(gene_len, (out_start + W) + (ctx - ctx // 2))
    seq_oh = _one_hot_encode(sequence[seq_start:seq_end])     # [4, sub_len]
    total_len = W + ctx
    pad_off = 0
    if seq_oh.shape[1] < total_len:
        pad_off = (total_len - seq_oh.shape[1]) // 2
        padded = np.zeros((4, total_len), dtype=np.float32)
        padded[:, pad_off:pad_off + seq_oh.shape[1]] = seq_oh
        seq_oh = padded
    seq_center = (donor_idx - seq_start) + pad_off            # sequence-frame index of the site

    base_win = base_scores[out_start:out_start + W].astype(np.float32)        # [W, 3]
    mm_win = mm_features[out_start:out_start + W].T.astype(np.float32).copy()  # [C, W]

    seq_t = torch.from_numpy(seq_oh).unsqueeze(0).to(device)
    base_t = torch.from_numpy(base_win).unsqueeze(0).to(device)
    mm_t = torch.from_numpy(mm_win).unsqueeze(0).to(device)

    # ── Forward that returns the scalar P(splice) at the site ────────────
    def fwd(seq, mm, base):
        probs = model(seq, base, mm, return_logits=False)     # [1, W, 3]
        return probs[:, center_idx, 0] + probs[:, center_idx, 1]  # [1]

    with torch.no_grad():
        p_splice = float(fwd(seq_t, mm_t, base_t).item())
        base_p = float(base_scores[donor_idx, 0] + base_scores[donor_idx, 1])

    # ── Integrated Gradients on (sequence, multimodal); base held fixed ──
    ig = IntegratedGradients(fwd)
    (attr_seq, attr_mm), delta = ig.attribute(
        inputs=(seq_t, mm_t),
        baselines=(torch.zeros_like(seq_t), torch.zeros_like(mm_t)),
        additional_forward_args=(base_t,),
        n_steps=args.n_steps,
        return_convergence_delta=True,
    )
    attr_seq = attr_seq.squeeze(0).detach().cpu().numpy()     # [4, total_len]
    attr_mm = attr_mm.squeeze(0).detach().cpu().numpy()       # [C, W]

    # Per-position attribution of the *present* base (signed) → saliency track
    seq_present = (attr_seq * seq_oh).sum(axis=0)             # [total_len]
    lo = max(0, seq_center - LOGO_HALF_WINDOW)
    hi = min(total_len, seq_center + LOGO_HALF_WINDOW + 1)
    track = seq_present[lo:hi]
    bases = np.array(list("ACGT"))
    letters = bases[seq_oh[:, lo:hi].argmax(axis=0)]
    rel_pos = np.arange(lo, hi) - seq_center                  # 0 == the splice site

    # Per-channel multimodal attribution
    ch_signed = attr_mm.sum(axis=1)                           # [C] signed total over window
    ch_abs = np.abs(attr_mm).sum(axis=1)                      # [C] magnitude over window
    ch_at_site = attr_mm[:, center_idx]                       # [C] at the site position

    seq_total = float(np.abs(attr_seq).sum())
    mm_total = float(np.abs(attr_mm).sum())
    seq_frac = seq_total / (seq_total + mm_total + 1e-12)

    # GT/AG dinucleotide sanity at the site (+ strand frame)
    s0 = donor_idx - seq_start + pad_off
    dinuc = "".join(bases[seq_oh[:, s0:s0 + 2].argmax(axis=0)]) if s0 + 2 <= total_len else "?"

    # ── Report ───────────────────────────────────────────────────────────
    order = np.argsort(ch_abs)[::-1]
    print(f"\n{'='*72}")
    print(f"Integrated Gradients — {ev.gene} cryptic {site.kind} @ {ev.chrom}:{site.pos:,} "
          f"({ev.strand})")
    print(f"  model={args.model_dir.name} ({cfg.variant})  n_steps={args.n_steps}")
    print(f"  P(splice) meta={p_splice:.3f}  base={base_p:.3f}  (+strand dinuc @site='{dinuc}')")
    print(f"  attribution mass: sequence={seq_frac*100:.1f}%  multimodal={(1-seq_frac)*100:.1f}%")
    print(f"  IG convergence delta={float(delta.item()):.2e}")
    print(f"  per-channel multimodal attribution (|sum| over window, signed):")
    for c in order:
        print(f"    {CHANNEL_NAMES[c]:16s}  |{ch_abs[c]:.4f}|  signed={ch_signed[c]:+.4f}  "
              f"@site={ch_at_site[c]:+.4f}")
    print('='*72)

    summary = {
        "gene": ev.gene, "chrom": ev.chrom, "strand": ev.strand,
        "site_kind": site.kind, "site_pos_1based": site.pos, "site_note": site.note,
        "model_dir": str(args.model_dir), "variant": cfg.variant,
        "p_splice_meta": p_splice, "p_splice_base": base_p,
        "plus_strand_dinucleotide_at_site": dinuc,
        "attribution_mass": {
            "sequence": seq_total, "multimodal": mm_total,
            "sequence_fraction": seq_frac,
        },
        "ig_convergence_delta": float(delta.item()),
        "multimodal_channels": {
            CHANNEL_NAMES[c]: {
                "abs_sum": float(ch_abs[c]),
                "signed_sum": float(ch_signed[c]),
                "at_site": float(ch_at_site[c]),
            } for c in range(len(CHANNEL_NAMES))
        },
        "sequence_track": {
            "rel_pos": rel_pos.tolist(),
            "letters": letters.tolist(),
            "attribution": track.tolist(),
        },
        "n_steps": args.n_steps,
    }
    (out_dir / "ig_summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(
        out_dir / "ig_attribution.npz",
        attr_seq=attr_seq, attr_mm=attr_mm, seq_present=seq_present,
        seq_onehot=seq_oh, base_win=base_win, mm_win=mm_win,
        center_idx=center_idx, seq_center=seq_center,
    )
    print(f"\nWrote {out_dir/'ig_summary.json'} and ig_attribution.npz")

    # ── Figure (best-effort; skips cleanly if matplotlib missing) ────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 7),
                                       gridspec_kw={"height_ratios": [2, 1]})
        colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in track]
        ax0.bar(rel_pos, track, color=colors, width=0.9)
        ax0.axvline(0, color="#3572a5", lw=1.2, ls="--")
        ax0.set_title(f"{ev.gene} cryptic {site.kind} @ {ev.chrom}:{site.pos:,} — "
                      f"sequence saliency (IG); meta P(splice)={p_splice:.3f}, base={base_p:.3f}")
        ax0.set_xlabel("position relative to cryptic site (bp)")
        ax0.set_ylabel("IG attribution\n(present base)")
        if len(rel_pos) <= 90:
            ax0.set_xticks(rel_pos[::2])
            ax0.set_xticklabels([f"{l}\n{p:+d}" for l, p in zip(letters[::2], rel_pos[::2])],
                                fontsize=6, family="monospace")
        order_plot = np.argsort(ch_abs)[::-1]
        ax1.bar(range(len(CHANNEL_NAMES)),
                [ch_signed[c] for c in order_plot],
                color=["#e74c3c" if ch_signed[c] < 0 else "#27ae60" for c in order_plot])
        ax1.set_xticks(range(len(CHANNEL_NAMES)))
        ax1.set_xticklabels([CHANNEL_NAMES[c] for c in order_plot], rotation=45,
                            ha="right", fontsize=8)
        ax1.set_ylabel("IG attribution\n(signed, over window)")
        ax1.set_title("multimodal channel attribution")
        fig.tight_layout()
        fig.savefig(out_dir / "ig_figure.png", dpi=130)
        print(f"Wrote {out_dir/'ig_figure.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"(figure skipped: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
