#!/usr/bin/env python
"""Predict splice site changes caused by a single nucleotide variant.

Runs the M1-S meta-splice model on reference and alternate sequences,
computes per-position delta scores, and reports donor/acceptor gain/loss
events following the SpliceAI convention (DS_DG, DS_DL, DS_AG, DS_AL).

This is the entry point for M4 variant effect prediction — Phase 1A.

Variants can be specified on the command line or via a YAML config file
for convenience and batch reproducibility.

Usage:
    # Single variant via CLI args
    python 01_single_variant_delta.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --fasta data/mane/GRCh38/hg38.fa \\
        --chrom chr17 --pos 43094464 --ref A --alt C \\
        --gene BRCA1

    # Single or multiple variants via config file
    python 01_single_variant_delta.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --fasta data/mane/GRCh38/hg38.fa \\
        --config variants.yaml

    # Quick mode (no multimodal features, faster)
    python 01_single_variant_delta.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --fasta data/mane/GRCh38/hg38.fa \\
        --config variants.yaml --no-multimodal

Config file format (YAML):
    variants:
      - chrom: chr17
        pos: 43094464
        ref: A
        alt: C
        gene: BRCA1       # optional
      - chrom: chr7
        pos: 117559590
        ref: T
        alt: G
        gene: CFTR
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


def plot_delta(result, output_path: Path = None) -> None:
    """Plot per-position delta scores for donor and acceptor channels."""
    import matplotlib.pyplot as plt

    delta = result.delta
    L = len(delta)
    positions = range(result.window_start, result.window_start + L)
    variant_pos = result.position - 1  # 0-based for plotting

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Donor channel
    axes[0].fill_between(positions, delta[:, 0], 0, alpha=0.4, color="#3572a5")
    axes[0].axhline(0, color="gray", linewidth=0.5)
    axes[0].axvline(variant_pos, color="red", linewidth=1, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Δ Donor")
    axes[0].set_title(
        f"{result.chrom}:{result.position} {result.ref}>{result.alt}"
        + (f" ({result.gene})" if result.gene else "")
    )

    # Acceptor channel
    axes[1].fill_between(positions, delta[:, 1], 0, alpha=0.4, color="#e74c3c")
    axes[1].axhline(0, color="gray", linewidth=0.5)
    axes[1].axvline(variant_pos, color="red", linewidth=1, linestyle="--", alpha=0.7)
    axes[1].set_ylabel("Δ Acceptor")
    axes[1].set_xlabel("Genomic position")

    # Mark detected events
    for event in result.events[:5]:
        ax = axes[0] if event.splice_type == "donor" else axes[1]
        color = "#27ae60" if event.is_gain else "#e74c3c"
        ax.annotate(
            f"{event.event_type}\nΔ={event.delta:+.3f}",
            xy=(event.position, event.delta),
            fontsize=7, color=color, ha="center",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def load_variants_from_config(config_path: Path) -> list:
    """Load variant definitions from a YAML config file.

    Expected format::

        variants:
          - chrom: chr17
            pos: 43094464
            ref: A
            alt: C
            gene: BRCA1
          - chrom: chr7
            pos: 117559590
            ref: T
            alt: G
            gene: CFTR
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Predict splice site changes from a single variant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to M1-S checkpoint (best.pt)")
    parser.add_argument("--fasta", type=Path, required=True,
                        help="Reference genome FASTA")
    parser.add_argument("--config", type=Path, default=None,
                        help="YAML config with variant(s). Overrides --chrom/pos/ref/alt.")
    parser.add_argument("--chrom", default=None,
                        help="Chromosome (e.g., chr17)")
    parser.add_argument("--pos", type=int, default=None,
                        help="Variant position (1-based)")
    parser.add_argument("--ref", default=None,
                        help="Reference allele")
    parser.add_argument("--alt", default=None,
                        help="Alternate allele")
    parser.add_argument("--gene", default=None,
                        help="Gene name (for display)")
    parser.add_argument("--strand", default="+", choices=["+", "-"],
                        help="Gene strand (default: +). Required for correct "
                             "base model scoring on minus-strand genes.")
    parser.add_argument("--base-model", default="openspliceai",
                        help="Base model for ref/alt scoring (default: openspliceai). "
                             "Use 'none' to skip base model (uniform 1/3 prior).")
    parser.add_argument("--no-multimodal", action="store_true",
                        help="Skip multimodal features (faster, less accurate)")
    parser.add_argument("--bigwig-cache", type=Path, default=None,
                        help="Local BigWig cache directory")
    parser.add_argument("--event-threshold", type=float, default=0.1,
                        help="Minimum |delta| to report as splice event (default: 0.1)")
    parser.add_argument("--plot", type=Path, default=None,
                        help="Save delta plot to this path (PNG/PDF). "
                             "For multiple variants, appends variant index.")
    parser.add_argument("--device", default="cpu",
                        help="Inference device (default: cpu)")
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

    # ── Initialize runner (once, reused across variants) ────────────
    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )

    runner = VariantRunner(
        meta_checkpoint=args.checkpoint,
        fasta_path=args.fasta,
        bigwig_cache_dir=args.bigwig_cache,
        base_model=args.base_model,
        device=args.device,
        event_threshold=args.event_threshold,
    )

    # ── Run each variant ────────────────────────────────────────────
    for i, v in enumerate(variants):
        print(f"\n{'='*60}")
        print(f"Variant {i+1}/{len(variants)}: "
              f"{v['chrom']}:{v['pos']} {v['ref']}>{v['alt']}"
              + (f" ({v.get('gene', '')})" if v.get("gene") else ""))
        print(f"{'='*60}")

        result = runner.run(
            chrom=v["chrom"],
            position=v["pos"],
            ref=v["ref"],
            alt=v["alt"],
            gene=v.get("gene"),
            strand=v.get("strand", "+"),
            use_multimodal=not args.no_multimodal,
        )

        print(result.summary())

        if args.plot:
            if len(variants) == 1:
                plot_path = args.plot
            else:
                stem = args.plot.stem
                plot_path = args.plot.with_name(f"{stem}_{i+1}{args.plot.suffix}")
            plot_delta(result, plot_path)

    runner.close()
    print(f"\nDone — {len(variants)} variant(s) processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
