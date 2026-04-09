#!/usr/bin/env python
"""Predict splice consequences from a variant's delta scores.

Extends Phase 1A (delta prediction) with Phase 1B (consequence
interpretation): maps delta events to exon-intron boundaries, pairs
loss + gain events into junction changes, classifies the consequence
(exon skipping, intron retention, donor/acceptor shift, cryptic
activation), and optionally analyzes reading frame impact.

Usage:
    # Single variant
    python 01b_splice_consequences.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --chrom chr11 --pos 47333193 --ref C --alt A \
        --gene MYBPC3 --strand -

    # Batch from config file
    python 01b_splice_consequences.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --config examples/variant_analysis/test_variants.yaml

    # With delta plot
    python 01b_splice_consequences.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --config examples/variant_analysis/test_variants.yaml \
        --plot output/variant_plots/consequence.png
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


def load_variants_from_config(config_path: Path) -> list:
    """Load variant definitions from a YAML config file."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    variants = cfg.get("variants", [])
    if not variants:
        print(f"ERROR: No variants found in {config_path}")
        sys.exit(1)

    for v in variants:
        if not all(k in v for k in ("chrom", "pos", "ref", "alt")):
            print(f"ERROR: Each variant needs chrom, pos, ref, alt. Got: {v}")
            sys.exit(1)

    return variants


def plot_delta_with_consequences(result, consequence, output_path: Path = None) -> None:
    """Plot delta scores annotated with splice consequence events."""
    import matplotlib.pyplot as plt

    delta = result.delta
    L = len(delta)
    positions = range(result.window_start, result.window_start + L)
    variant_pos = result.position - 1

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # Donor channel
    axes[0].fill_between(positions, delta[:, 0], 0, alpha=0.3, color="#3572a5")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(variant_pos, color="red", linewidth=1, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Δ Donor")
    axes[0].set_title(
        f"{result.chrom}:{result.position} {result.ref}>{result.alt}"
        + (f" ({result.gene})" if result.gene else "")
        + f" — {consequence.consequence_type}"
    )

    # Acceptor channel
    axes[1].fill_between(positions, delta[:, 1], 0, alpha=0.3, color="#e74c3c")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(variant_pos, color="red", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Δ Acceptor")
    axes[1].set_xlabel("Genomic position")

    # Annotate top events with consequence labels
    for event in result.events[:5]:
        ax = axes[0] if event.splice_type == "donor" else axes[1]
        color = "#27ae60" if event.is_gain else "#e74c3c"
        label = "gain" if event.is_gain else "loss"
        ax.annotate(
            f"{label}\nΔ={event.delta:+.3f}",
            xy=(event.position, event.delta),
            fontsize=7, color=color, ha="center",
        )

    # Add consequence summary as text box
    textstr = consequence.summary
    fig.text(
        0.02, 0.01, textstr, fontsize=8, fontfamily="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Predict splice consequences from variant delta scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to meta-layer checkpoint (best.pt)")
    parser.add_argument("--fasta", type=Path, required=True,
                        help="Reference genome FASTA")
    parser.add_argument("--gtf", type=Path, default=None,
                        help="GTF annotation file for gene structure. "
                             "Default: MANE GTF via resource manager.")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config with variant(s)")
    parser.add_argument("--chrom", default=None)
    parser.add_argument("--pos", type=int, default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--alt", default=None)
    parser.add_argument("--gene", default=None)
    parser.add_argument("--strand", default="+", choices=["+", "-"])
    parser.add_argument("--base-model", default="openspliceai")
    parser.add_argument("--no-multimodal", action="store_true",
                        help="Skip multimodal features (faster)")
    parser.add_argument("--bigwig-cache", type=Path, default=None)
    parser.add_argument("--event-threshold", type=float, default=0.1,
                        help="Minimum |delta| to report (default: 0.1)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save delta+consequence plot (PNG/PDF)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=None,
                        help="Save structured results as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── Resolve variant(s) ──────────────────────────────────────────
    if args.config:
        variants = load_variants_from_config(args.config)
    elif args.chrom and args.pos and args.ref and args.alt:
        variants = [{"chrom": args.chrom, "pos": args.pos,
                      "ref": args.ref, "alt": args.alt,
                      "gene": args.gene, "strand": args.strand}]
    else:
        parser.error("Provide either --config or all of --chrom --pos --ref --alt")

    # ── Resolve GTF for gene structure ──────────────────────────────
    if args.gtf:
        gtf_path = str(args.gtf)
    else:
        from agentic_spliceai.splice_engine.resources import get_model_resources
        resources = get_model_resources("openspliceai")
        gtf_path = str(resources.get_gtf_path())

    # ── Initialize runner + detector ────────────────────────────────
    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )
    from agentic_spliceai.splice_engine.meta_layer.inference.splice_event_detector import (
        SpliceEventDetector,
    )

    runner = VariantRunner(
        meta_checkpoint=args.checkpoint,
        fasta_path=args.fasta,
        bigwig_cache_dir=args.bigwig_cache,
        base_model=args.base_model,
        device=args.device,
        event_threshold=args.event_threshold,
    )
    detector = SpliceEventDetector(gtf_path=gtf_path)

    # ── Process each variant ────────────────────────────────────────
    all_results = []

    for i, v in enumerate(variants):
        gene = v.get("gene", "")
        strand = v.get("strand", "+")

        print(f"\n{'='*70}")
        print(f"Variant {i+1}/{len(variants)}: "
              f"{v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}"
              + (f" ({gene})" if gene else ""))
        print(f"{'='*70}")

        # Phase 1A: delta computation
        delta_result = runner.run(
            chrom=v["chrom"],
            position=v["pos"],
            ref=v["ref"],
            alt=v["alt"],
            gene=gene,
            strand=strand,
            use_multimodal=not args.no_multimodal,
        )

        # Phase 1B: consequence analysis
        consequence = detector.analyze(delta_result, gene=gene)

        # Print results
        print(f"\n{consequence.report()}")
        print(f"\nDelta summary:")
        print(f"  DS_DG: {delta_result.max_donor_gain:+.4f}")
        print(f"  DS_DL: {-delta_result.max_donor_loss:+.4f}")
        print(f"  DS_AG: {delta_result.max_acceptor_gain:+.4f}")
        print(f"  DS_AL: {-delta_result.max_acceptor_loss:+.4f}")

        # Plot
        if args.plot:
            if len(variants) == 1:
                plot_path = args.plot
            else:
                stem = args.plot.stem
                plot_path = args.plot.with_name(f"{stem}_{i+1}{args.plot.suffix}")
            plot_delta_with_consequences(delta_result, consequence, plot_path)

        all_results.append({
            "variant": f"{v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}",
            "gene": gene,
            "strand": strand,
            "consequence_type": consequence.consequence_type,
            "confidence": consequence.confidence,
            "affected_exons": consequence.affected_exons,
            "frame_preserved": consequence.frame_preserved,
            "summary": consequence.summary,
            "ds_dg": delta_result.max_donor_gain,
            "ds_dl": delta_result.max_donor_loss,
            "ds_ag": delta_result.max_acceptor_gain,
            "ds_al": delta_result.max_acceptor_loss,
            "n_events": len(delta_result.events),
        })

    # ── Save JSON results ───────────────────────────────────────────
    if args.json:
        import json
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nJSON results saved to {args.json}")

    runner.close()
    print(f"\nDone — {len(variants)} variant(s) processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
