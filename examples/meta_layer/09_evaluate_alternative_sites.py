#!/usr/bin/env python
"""Evaluate M1-S at alternative splice sites (M2a/M2b).

Tests whether the meta-layer generalizes beyond MANE training annotations
by evaluating at splice sites present in a richer annotation (Ensembl or
GENCODE) but absent from MANE.

Reports two sets of metrics:
  1. **Overall**: all sites in the evaluation annotation
  2. **Alternative-only**: sites in (eval_annotation \\ MANE) — the M2a metric

For M2b with GENCODE, also reports tiered breakdown:
  - Tier 1: GENCODE ∩ Ensembl, not MANE (well-supported alternatives)
  - Tier 2: GENCODE-only (rare isoforms, computational predictions)

Uses streaming evaluation — only one gene's arrays in memory at a time.

Usage:
    # M2a: Ensembl \\ MANE (requires Ensembl gene cache)
    python 09_evaluate_alternative_sites.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --annotation-source ensembl \\
        --build-cache \\
        --base-scores-dir data/ensembl/GRCh38/openspliceai_eval/precomputed

    # From existing cache
    python 09_evaluate_alternative_sites.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --annotation-source ensembl \\
        --cache-dir /path/to/ensembl_gene_cache

    # M2b: GENCODE \\ MANE (requires GENCODE gene cache)
    python 09_evaluate_alternative_sites.py \\
        --checkpoint output/meta_layer/m1s/best.pt \\
        --annotation-source gencode \\
        --gtf data/gencode/GRCh38/gencode.v47.annotation.gtf \\
        --build-cache
"""

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MANE reference site set
# ---------------------------------------------------------------------------


def build_mane_site_set() -> Set[Tuple[str, int, str]]:
    """Load MANE splice sites as a set of (chrom_bare, position, splice_type).

    Chromosome names are normalized to bare format (no 'chr' prefix) for
    cross-annotation-source comparison.
    """
    import pandas as pd
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources("openspliceai")
    mane_path = Path(resources.get_registry().stash) / "splice_sites_enhanced.tsv"
    mane_df = pd.read_csv(mane_path, sep="\t", usecols=["chrom", "position", "splice_type"])
    mane_df["chrom"] = mane_df["chrom"].str.replace("chr", "", regex=False)
    return set(zip(mane_df["chrom"], mane_df["position"], mane_df["splice_type"]))


def build_alternative_site_mask(
    gene_id: str,
    labels: np.ndarray,
    gene_annotations: "polars.DataFrame",
    mane_sites: Set[Tuple[str, int, str]],
) -> np.ndarray:
    """Build a boolean mask identifying alternative (eval_annotation \\ MANE) sites.

    Parameters
    ----------
    gene_id : str
        Gene identifier.
    labels : np.ndarray
        Gene-relative label array [L] (0=donor, 1=acceptor, 2=neither).
    gene_annotations : polars.DataFrame
        Gene annotations with gene_id, chrom, start columns.
    mane_sites : set
        Set of (chrom_bare, absolute_position, splice_type) from MANE.

    Returns
    -------
    np.ndarray
        Boolean mask [L] — True at alternative splice sites (in eval
        annotation but NOT in MANE).
    """
    import polars as pl

    row = gene_annotations.filter(pl.col("gene_id") == gene_id)
    if row.height == 0:
        row = gene_annotations.filter(pl.col("gene_name") == gene_id)
    if row.height == 0:
        return np.zeros(len(labels), dtype=bool)

    gene_start = int(row[0, "start"])
    chrom = str(row[0, "chrom"]).replace("chr", "")

    label_to_type = {0: "donor", 1: "acceptor"}
    mask = np.zeros(len(labels), dtype=bool)

    # Only check splice site positions (skip the vast majority of "neither")
    splice_positions = np.where(labels < 2)[0]
    for pos in splice_positions:
        splice_type = label_to_type[int(labels[pos])]
        abs_pos = gene_start + pos
        if (chrom, abs_pos, splice_type) not in mane_sites:
            mask[pos] = True

    return mask


from agentic_spliceai.splice_engine.eval.sequence_inference import infer_full_gene


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="M2a/M2b: Evaluate meta model at alternative splice sites",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to trained model checkpoint (best.pt)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to config.pt (default: same dir as checkpoint)",
    )
    parser.add_argument(
        "--annotation-source", choices=["ensembl", "gencode"], default="ensembl",
        help="Evaluation annotation source (default: ensembl for M2a)",
    )
    parser.add_argument(
        "--gtf", type=Path, default=None,
        help="Custom GTF path (required for gencode, optional for ensembl)",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Directory with gene cache (.npz files). "
             "Default: <checkpoint-dir>/gene_cache/<annotation_source>",
    )
    parser.add_argument(
        "--build-cache", action="store_true",
        help="Build gene cache before evaluating.",
    )
    parser.add_argument(
        "--base-scores-dir", type=Path, default=None,
        help="Base model predictions directory for cache building.",
    )
    parser.add_argument(
        "--bigwig-cache", type=Path, default=None,
        help="Local directory with cached conservation bigWig files.",
    )
    parser.add_argument(
        "--test-chroms", nargs="+", default=None,
        help="Evaluate on genes from these chromosomes "
             "(default: SpliceAI test set chr1,3,5,7,9)",
    )
    parser.add_argument(
        "--max-genes", type=int, default=None,
        help="Limit number of genes for quick testing",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory for results (default: checkpoint dir)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device for inference (default: cpu)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import torch

    # ── Load model ───────────────────────────────────────────────────
    config_path = args.config or args.checkpoint.parent / "config.pt"
    if not config_path.exists():
        print(f"ERROR: config not found at {config_path}")
        return 1

    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceModel, MetaSpliceConfig,
    )

    device = torch.device(args.device)
    torch.serialization.add_safe_globals([MetaSpliceConfig])
    cfg = torch.load(config_path, map_location="cpu", weights_only=True)
    model = MetaSpliceModel(cfg).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {cfg.variant}, {n_params:,} params")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")

    # ── Resolve evaluation annotation ────────────────────────────────
    from agentic_spliceai.splice_engine.resources import get_model_resources, get_genomic_registry
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import extract_gene_annotations
    from agentic_spliceai.splice_engine.eval.splitting import (
        build_gene_split, gene_chromosomes_from_dataframe,
    )

    resources = get_model_resources("openspliceai")
    ann_src = args.annotation_source

    if args.gtf:
        # Custom GTF (e.g., GENCODE comprehensive)
        gtf_path = str(args.gtf)
        print(f"  Annotation: {ann_src} (custom GTF: {args.gtf.name})")
    elif ann_src == "ensembl":
        ann_registry = get_genomic_registry(build="GRCh38", release="112")
        gtf_path = str(ann_registry.get_gtf_path())
        print(f"  Annotation: Ensembl GRCh38.112")
    else:
        print(f"ERROR: --gtf required for annotation-source={ann_src}")
        return 1

    gene_annotations = extract_gene_annotations(gtf_path, verbosity=0)

    # ── Resolve gene set ─────────────────────────────────────────────
    gene_chroms = gene_chromosomes_from_dataframe(gene_annotations)

    if args.test_chroms:
        # Normalize to match annotation's chromosome naming convention
        raw_chroms = args.test_chroms
        # Build both forms for matching (some annotations use "chr1", others "1")
        test_chroms_set = set()
        for c in raw_chroms:
            bare = c.replace("chr", "")
            test_chroms_set.add(bare)
            test_chroms_set.add(f"chr{bare}")
        test_genes = sorted(g for g, c in gene_chroms.items() if c in test_chroms_set)
        test_chroms = sorted(set(gene_chroms[g] for g in test_genes))
        print(f"  Eval genes: {len(test_genes)} on {test_chroms}")
    else:
        gene_split = build_gene_split(gene_chroms, preset="spliceai", val_fraction=0.0)
        test_genes = sorted(gene_split.test_genes)
        test_chroms = sorted(set(
            gene_chroms.get(g, "unknown") for g in test_genes if g in gene_chroms
        ))
        print(f"  Eval genes: {len(test_genes)} (SpliceAI test: {test_chroms})")

    if args.max_genes:
        test_genes = test_genes[:args.max_genes]
        print(f"  Limited to {len(test_genes)} genes")

    # ── Cache directory ──────────────────────────────────────────────
    cache_dir = args.cache_dir or args.checkpoint.parent / "gene_cache" / ann_src

    # ── Build cache if requested ─────────────────────────────────────
    if args.build_cache:
        import pandas as pd
        from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
            build_gene_cache,
        )
        from agentic_spliceai.splice_engine.features.dense_feature_extractor import (
            DenseFeatureExtractor, DenseFeatureConfig,
        )
        from agentic_spliceai.splice_engine.eval.streaming_metrics import (
            preflight_check,
        )

        fasta_path = str(resources.get_fasta_path())

        # Splice sites from the evaluation annotation
        if args.gtf:
            # For custom GTF, generate splice sites on the fly
            from agentic_spliceai.splice_engine.base_layer.data.preparation import (
                prepare_splice_site_annotations,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            ss_result = prepare_splice_site_annotations(
                output_dir=cache_dir,
                gtf_path=gtf_path,
                build="GRCh38",
                annotation_source=ann_src,
                verbosity=1,
            )
            splice_sites_path = Path(ss_result["splice_sites_file"])
        elif ann_src == "ensembl":
            splice_sites_path = Path(ann_registry.stash) / "splice_sites_enhanced.tsv"
        else:
            print(f"ERROR: Cannot resolve splice sites for {ann_src}")
            return 1

        if args.base_scores_dir:
            base_scores_dir = args.base_scores_dir
        else:
            base_scores_dir = resources.get_registry().get_base_model_eval_dir(
                "openspliceai"
            ) / "precomputed"

        # Fail fast if dependencies or data are missing
        preflight_check(
            needs_bigwig=True,
            needs_pyfaidx=True,
            fasta_path=fasta_path,
            base_scores_dir=base_scores_dir,
        )

        print(f"\n  Building gene cache ({len(test_genes)} genes)...")
        print(f"    Cache dir:    {cache_dir}")
        print(f"    Splice sites: {splice_sites_path}")
        print(f"    Base scores:  {base_scores_dir}")

        splice_sites_df = pd.read_csv(splice_sites_path, sep="\t")
        feat_config = DenseFeatureConfig(
            build="GRCh38",
            bigwig_cache_dir=args.bigwig_cache,
        )
        extractor = DenseFeatureExtractor(feat_config)

        t_cache = time.time()
        build_gene_cache(
            test_genes, splice_sites_df, fasta_path,
            base_scores_dir, extractor, gene_annotations,
            cache_dir=cache_dir,
        )
        extractor.close()
        print(f"    Cache built in {time.time() - t_cache:.1f}s\n")

    # ── Load MANE reference sites ────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"M2 Alternative Site Evaluation ({ann_src} \\ MANE)")
    print(f"{'='*70}\n")

    mane_sites = build_mane_site_set()
    print(f"  MANE reference: {len(mane_sites):,} unique sites")

    # ── Streaming evaluation ─────────────────────────────────────────
    from agentic_spliceai.splice_engine.meta_layer.data.sequence_level_dataset import (
        _load_gene_npz,
    )
    from agentic_spliceai.splice_engine.eval.streaming_metrics import (
        StreamingEvaluator, print_comparison_report,
    )

    min_length = 5001 + 400
    overall_eval = StreamingEvaluator()  # all sites
    alt_eval = StreamingEvaluator()      # alternative sites only

    n_skipped = 0
    n_alt_sites_total = 0
    n_shared_sites_total = 0

    t0 = time.time()
    for i, gene_id in enumerate(test_genes):
        npz_path = cache_dir / f"{gene_id}.npz"
        if not npz_path.exists():
            n_skipped += 1
            continue

        data = _load_gene_npz(npz_path)
        if len(data["sequence"]) < min_length:
            n_skipped += 1
            del data
            continue

        # Inference
        meta_probs = infer_full_gene(model, data, device=device)
        base_probs = data["base_scores"]
        labels = data["labels"]

        # Overall metrics (all sites)
        overall_eval.update(meta_probs, base_probs, labels, gene_id)

        # Alternative site mask
        alt_mask = build_alternative_site_mask(
            gene_id, labels, gene_annotations, mane_sites,
        )
        n_alt = int(alt_mask.sum())
        n_splice = int((labels < 2).sum())
        n_alt_sites_total += n_alt
        n_shared_sites_total += n_splice - n_alt

        # Update alt evaluator with only alternative positions
        if n_alt > 0:
            alt_eval.update(
                meta_probs[alt_mask],
                base_probs[alt_mask],
                labels[alt_mask],
                gene_id,
            )

        del data, meta_probs, base_probs, labels, alt_mask

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(test_genes)} genes "
                  f"(alt sites: {n_alt_sites_total:,})...")

    elapsed = time.time() - t0
    overall_eval.n_skipped = n_skipped
    print(f"\n  Inference complete: {overall_eval.n_genes} genes in {elapsed:.1f}s "
          f"({n_skipped} skipped)")
    print(f"  Shared sites (MANE ∩ {ann_src}): {n_shared_sites_total:,}")
    print(f"  Alternative sites ({ann_src} \\ MANE): {n_alt_sites_total:,}")
    print(f"  Accumulator memory: {overall_eval.memory_usage_mb() + alt_eval.memory_usage_mb():.1f} MB")

    if overall_eval.n_genes == 0:
        print("ERROR: No genes evaluated. Check --cache-dir path.")
        return 1

    # ── Overall metrics ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Overall Metrics (all sites)")
    print(f"{'='*70}")

    overall_results = overall_eval.compute()
    print_comparison_report(overall_results)

    # ── Alternative site metrics ─────────────────────────────────────
    if alt_eval.n_genes > 0:
        print(f"\n{'='*70}")
        print(f"Alternative Site Metrics ({ann_src} \\ MANE)")
        print(f"{'='*70}")

        alt_results = alt_eval.compute()
        print_comparison_report(alt_results)
    else:
        print(f"\n  No alternative sites found in evaluated genes.")
        alt_results = None

    # ── Save results ─────────────────────────────────────────────────
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": cfg.variant,
        "checkpoint": str(args.checkpoint),
        "eval_annotation": ann_src,
        "n_genes": overall_eval.n_genes,
        "n_skipped": n_skipped,
        "n_shared_sites": n_shared_sites_total,
        "n_alternative_sites": n_alt_sites_total,
        "test_chromosomes": test_chroms,
        "overall": overall_results,
    }
    if alt_results:
        results["alternative_sites"] = alt_results

    suffix = "m2a" if ann_src == "ensembl" else "m2b"
    results_path = output_dir / f"{suffix}_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Summary
    overall_meta = overall_results["meta_model"]
    overall_base = overall_results["base_model"]
    print(f"\n{'='*70}")
    print(f"Summary")
    print(f"  Overall — Base PR-AUC: {overall_base['macro_pr_auc']:.4f}, "
          f"Meta: {overall_meta['macro_pr_auc']:.4f}")
    if alt_results:
        alt_meta = alt_results["meta_model"]
        alt_base = alt_results["base_model"]
        print(f"  Alt sites — Base PR-AUC: {alt_base['macro_pr_auc']:.4f}, "
              f"Meta: {alt_meta['macro_pr_auc']:.4f}")
        print(f"  Alt FN reduction: {alt_results['fn_reduction_pct']:+.1f}%")
    print(f"{'='*70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
