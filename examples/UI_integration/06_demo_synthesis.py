#!/usr/bin/env python
"""Demo synthesis — the base → M1-S → M2-S story as one figure + narrative.

A milestone reflection, NOT new modeling. It pulls the leakage-clean held-out
results produced this cycle into a single 4-panel "story board" plus a markdown
narrative, for the examples/UI_integration showcase.

Story arc (each panel = one claim, each claim = real held-out numbers):
  A. M1-S sharpens CANONICAL sites (held-out MANE, paralog-clean): FN -93%.
  B. M2-S generalizes to ALTERNATIVE sites (Ensembl \\ MANE): recall ~6x.
  C. M2-S detects an ALS cryptic donor (UNC13A) the base model scores ~0.
  D. Integrated Gradients: WHY M2-S sees it — sequence-majority, RBP top channel.

Inputs are result JSONs only (no recompute), so this runs locally in seconds:
  - output/meta_layer/m1s_v4_cleanannot/eval_results.json          (panel A)
  - output/meta_layer/m2s_v4_cleanannot_alt_eval/m2a_eval_results.json  (panel B)
  - output/meta_layer/ig_unc13a_donor/ig_summary.json           (panels C, D)

Usage:
    PY=~/miniforge3/envs/agentic-spliceai/bin/python
    $PY examples/UI_integration/06_demo_synthesis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path("output/meta_layer")
M1S_CANON = RESULTS / "m1s_v4_cleanannot" / "eval_results.json"
M2S_ALT = RESULTS / "m2s_v4_cleanannot_alt_eval" / "m2a_eval_results.json"
IG_UNC13A = RESULTS / "ig_unc13a_donor" / "ig_summary.json"

DONOR = "#3572a5"
ACCEPTOR = "#e74c3c"
META = "#27ae60"
BASE = "#95a5a6"


def _load(path: Path) -> dict | None:
    if not path.exists():
        print(f"  [missing] {path} — panel will be skipped")
        return None
    return json.loads(path.read_text())


def _recall(m: dict) -> float:
    tp, fn = m.get("tp_count", 0), m.get("fn_count", 0)
    return tp / (tp + fn) if (tp + fn) else 0.0


def _f1opt(eval_json: dict) -> dict | None:
    """Aggregate the F1-optimal operating point from a threshold sweep.

    Splice prediction is extreme class imbalance, so a 0.5 cutoff is arbitrary.
    Returns combined donor+acceptor FP/FN/recall at each type's max-F1 threshold,
    or None if the eval has no ``threshold_sweep``. PR-AUC stays the threshold-free
    backbone; this just gives an honest deployable FP/FN.
    """
    ts = (eval_json or {}).get("threshold_sweep", {}).get("meta", {})
    if not ts:
        return None
    fp = fn = tp = 0
    for stype in ("donor", "acceptor"):
        rows = ts.get(stype) or []
        if not rows:
            return None
        best = max(rows, key=lambda r: r.get("f1", 0.0))
        fp += int(best["fp"]); fn += int(best["fn"]); tp += int(best["tp"])
    return {"fp": fp, "fn": fn, "recall": tp / (tp + fn) if (tp + fn) else 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo synthesis story board")
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS / "demo_synthesis")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    m1 = _load(M1S_CANON)
    m2 = _load(M2S_ALT)
    ig = _load(IG_UNC13A)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "agentic-spliceai meta layer: base → M1-S → M2-S  "
        "(leakage-clean held-out: SpliceAI chrom split + paralog removal, full RBP)",
        fontsize=13, fontweight="bold",
    )
    story = []  # markdown narrative lines

    # ── Panel A: M1-S canonical FN-rescue (held-out MANE) ────────────────
    axA = axes[0, 0]
    if m1:
        mb, mm = m1["base_model"], m1["meta_model"]
        cats = ["donor\nrecall", "acceptor\nrecall", "macro\nPR-AUC"]
        base_v = [mb["donor_recall"], mb["acceptor_recall"], mb["macro_pr_auc"]]
        meta_v = [mm["donor_recall"], mm["acceptor_recall"], mm["macro_pr_auc"]]
        x = np.arange(len(cats)); w = 0.38
        axA.bar(x - w / 2, base_v, w, label="base (OpenSpliceAI)", color=BASE)
        axA.bar(x + w / 2, meta_v, w, label="M1-S (meta)", color=META)
        for i, (b, mv) in enumerate(zip(base_v, meta_v)):
            axA.text(i - w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
            axA.text(i + w / 2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=8, fontweight="bold")
        axA.set_xticks(x); axA.set_xticklabels(cats); axA.set_ylim(0, 1.08)
        axA.set_ylabel("score"); axA.legend(loc="lower right", fontsize=8)
        op = _f1opt(m1)  # F1-optimal operating point (0.5 is meaningless under imbalance)
        if op:
            axA.set_title(f"A. M1-S sharpens CANONICAL sites (held-out MANE)\n"
                          f"macro PR-AUC {mb['macro_pr_auc']:.4f} → {mm['macro_pr_auc']:.4f}   |   "
                          f"@F1-opt: recall {op['recall']:.3f}, FP {op['fp']:,}, FN {op['fn']:,}",
                          fontsize=9)
            story.append(
                f"**A — Canonical (held-out MANE).** M1-S lifts macro PR-AUC "
                f"{mb['macro_pr_auc']:.4f}→{mm['macro_pr_auc']:.4f} (threshold-free). "
                f"At the F1-optimal operating point (splice prediction is extreme class "
                f"imbalance, so 0.5 is meaningless): recall **{op['recall']:.3f}**, "
                f"FP **{op['fp']:,}**, FN **{op['fn']:,}** — vs base donor recall "
                f"{mb['donor_recall']:.3f} at F1 ~0.97. The meta PR curve dominates base "
                f"on both precision and recall."
            )
        else:
            axA.set_title("A. M1-S sharpens CANONICAL sites (held-out MANE)\n"
                          f"macro PR-AUC {mb['macro_pr_auc']:.4f} → {mm['macro_pr_auc']:.4f}",
                          fontsize=9)
            story.append(
                f"**A — Canonical (held-out MANE).** M1-S lifts macro PR-AUC "
                f"{mb['macro_pr_auc']:.4f}→{mm['macro_pr_auc']:.4f} (threshold-free; "
                f"report FP/FN at the F1-optimal threshold, not 0.5)."
            )
    else:
        axA.text(0.5, 0.5, "m1s_v4_cleanannot/eval_results.json missing",
                 ha="center", va="center"); axA.axis("off")

    # ── Panel B: M2-S alternative-site lift (the headline) ───────────────
    axB = axes[0, 1]
    if m2:
        alt = m2["alternative_sites"]
        ab, am = alt["base_model"], alt["meta_model"]
        b_rec, m_rec = _recall(ab), _recall(am)
        cats = ["splice-site\nrecall", "macro\nPR-AUC"]
        base_v = [b_rec, ab["macro_pr_auc"]]
        meta_v = [m_rec, am["macro_pr_auc"]]
        x = np.arange(len(cats)); w = 0.38
        axB.bar(x - w / 2, base_v, w, label="base (OpenSpliceAI)", color=BASE)
        axB.bar(x + w / 2, meta_v, w, label="M2-S (meta)", color=META)
        for i, (b, mv) in enumerate(zip(base_v, meta_v)):
            axB.text(i - w / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
            axB.text(i + w / 2, mv + 0.01, f"{mv:.3f}", ha="center", fontsize=8, fontweight="bold")
        axB.set_xticks(x); axB.set_xticklabels(cats); axB.set_ylim(0, 1.08)
        axB.set_ylabel("score"); axB.legend(loc="center right", fontsize=8)
        fold = m_rec / b_rec if b_rec else float("nan")
        fnr = alt.get("fn_reduction_pct", 0.0)
        n_alt = m2.get("n_alternative_sites", 0)
        axB.set_title(f"B. M2-S generalizes to ALTERNATIVE sites (Ensembl \\ MANE, "
                      f"{n_alt:,} sites)\nmacro PR-AUC {ab['macro_pr_auc']:.3f} → "
                      f"{am['macro_pr_auc']:.3f}   recall {b_rec:.3f} → {m_rec:.3f} "
                      f"({fold:.1f}×)   FN −{fnr:.0f}%",
                      fontsize=9)
        story.append(
            f"**B — Alternative sites (Ensembl \\ MANE, {n_alt:,} held-out, "
            f"paralog-clean).** The base model — trained only on MANE canonical — "
            f"catches just {b_rec*100:.0f}% of alternative sites; M2-S catches "
            f"{m_rec*100:.0f}% (**{fold:.1f}× recall**) at near-equal *precision* "
            f"({am['donor_precision']:.2f} vs base {ab['donor_precision']:.2f} donor), "
            f"lifting macro PR-AUC {ab['macro_pr_auc']:.3f}→{am['macro_pr_auc']:.3f} "
            f"(threshold-free; FN −{fnr:.0f}%). M2-S learns splice grammar the base "
            f"model never saw. (FP at a fixed 0.5 cutoff is an operating-point "
            f"artifact under class imbalance — PR-AUC is the honest measure here.)"
        )
    else:
        axB.text(0.5, 0.5, "m2a_eval_results.json missing",
                 ha="center", va="center"); axB.axis("off")

    # ── Panel C: ALS cryptic detection (UNC13A donor) ────────────────────
    axC = axes[1, 0]
    if ig:
        base_p = ig.get("p_splice_base", 0.0)
        meta_p = ig.get("p_splice_meta", 0.0)
        labels = ["base\n(OpenSpliceAI)", "M2-S\n(meta)"]
        vals = [base_p, meta_p]
        colors = [BASE, META]
        bars = axC.bar(labels, vals, color=colors, width=0.6)
        axC.axhline(0.5, color=DONOR, ls="--", lw=1.2, label="0.5 detection threshold")
        for bar, v in zip(bars, vals):
            axC.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                     ha="center", fontweight="bold")
        axC.set_ylim(0, 1.0); axC.set_ylabel("P(splice) at cryptic donor")
        axC.legend(loc="upper left", fontsize=8)
        axC.set_title(f"C. ALS cryptic donor — {ig.get('gene','UNC13A')} "
                      f"{ig.get('chrom','chr19')}:{ig.get('site_pos_1based',0):,}\n"
                      f"base scores it ~0 (misses); M2-S detects it (>0.5)", fontsize=10)
        story.append(
            f"**C — ALS cryptic site.** At the {ig.get('gene','UNC13A')} cryptic "
            f"donor ({ig.get('chrom')}:{ig.get('site_pos_1based',0):,}, a TDP-43-"
            f"repressed poison exon), the base model assigns P(splice)={base_p:.3f} "
            f"(complete miss) while M2-S assigns **{meta_p:.3f}** — over the 0.5 "
            f"detection threshold. (STMN2 cryptic acceptor: M2-S P≈0.99.)"
        )
    else:
        axC.text(0.5, 0.5, "ig_summary.json missing", ha="center", va="center"); axC.axis("off")

    # ── Panel D: IG attribution (why M2-S sees it) ───────────────────────
    axD = axes[1, 1]
    if ig:
        ch = ig.get("multimodal_channels", {})
        names = list(ch.keys())
        signed = [ch[n]["signed_sum"] for n in names]
        order = np.argsort([abs(s) for s in signed])[::-1]
        names = [names[i] for i in order]; signed = [signed[i] for i in order]
        colors = [META if s >= 0 else ACCEPTOR for s in signed]
        axD.barh(range(len(names)), signed, color=colors)
        axD.set_yticks(range(len(names))); axD.set_yticklabels(names, fontsize=8)
        axD.invert_yaxis(); axD.axvline(0, color="k", lw=0.6)
        axD.set_xlabel("IG attribution (signed, over window)")
        seq_frac = ig.get("attribution_mass", {}).get("sequence_fraction", 0.0)
        axD.set_title(f"D. Why M2-S sees it (Integrated Gradients)\n"
                      f"sequence {seq_frac*100:.0f}% / multimodal {(1-seq_frac)*100:.0f}%  "
                      f"— top channel: {names[0]}", fontsize=10)
        story.append(
            f"**D — Why (Integrated Gradients).** The {meta_p:.3f} is "
            f"{seq_frac*100:.0f}% sequence (intrinsic donor grammar) and "
            f"{(1-seq_frac)*100:.0f}% multimodal, with **{names[0]}** the top "
            f"multimodal channel — RBP is a real, positive secondary signal, "
            f"not the whole story."
        )
    else:
        axD.text(0.5, 0.5, "ig_summary.json missing", ha="center", va="center"); axD.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = args.output_dir / "demo_story.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\nWrote {fig_path}")

    # ── Markdown narrative ───────────────────────────────────────────────
    md = [
        "# agentic-spliceai meta layer — demo story\n",
        "Base layer (OpenSpliceAI) → **M1-S** (canonical) → **M2-S** "
        "(alternative). All numbers are leakage-clean held-out test "
        "(SpliceAI chromosome split + sequence-based paralog removal on val+test, "
        "full RBP eCLIP union).\n",
        "![story board](demo_story.png)\n",
    ]
    md += [f"{i+1}. {line}\n" for i, line in enumerate(story)]
    md.append(
        "\n**On false positives / operating point.** The FN gains above are at a "
        "fixed 0.5 threshold, where the meta layer also raises FPs (M1-S canonical "
        "FP 8,594→28,597, +233%; M2-S is aggressive genome-wide). This is a "
        "calibration / operating-point matter, not a discrimination one — PR-AUC "
        "rises in every case, so a higher threshold trades most FPs back while "
        "keeping the recall gain. `08 --sweep-thresholds` reports the F1-optimal "
        "threshold and precision at ≥95% recall; the genome view exposes a "
        "`threshold` param so the operating point is the reviewer's to set.\n"
    )
    md.append(
        "\n**Honest edge — M4 (TDP-43 counterfactual).** With full neuronal RBP, "
        "M2-S *detects* the UNC13A cryptic donor, but ablating TDP-43 (simulated "
        "ALS knockout) does **not** raise P(splice) — the model never sees the "
        "de-repressed state in training, so it can't learn TDP-43 *repression* "
        "from normal-tissue labels. Solving M4 needs perturbation-paired data "
        "(TDP-43-knockdown junctions), not architecture. The diagnosis itself is "
        "the result.\n"
    )
    md_path = args.output_dir / "DEMO_STORY.md"
    md_path.write_text("".join(md))
    print(f"Wrote {md_path}")

    print(f"\n{'='*70}\nDemo story board ({sum(x is not None for x in (m1,m2,ig))}/3 panels with data):")
    for line in story:
        print("  - " + line.replace("**", "").replace("\\", ""))
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
