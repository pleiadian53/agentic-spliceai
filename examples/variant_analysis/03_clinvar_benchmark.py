#!/usr/bin/env python
"""Benchmark splice variant prediction on ClinVar pathogenic vs benign.

Batch-scores ClinVar splice-relevant variants with the meta-layer,
computes ROC/PR curves comparing meta-layer delta scores against
base model delta scores, and outputs a benchmark summary.

Prerequisites:
    - ClinVar filtered Parquet from 02_clinvar_download.py
    - OR raw ClinVar VCF (will filter on-the-fly)
    - Meta-layer checkpoint (M1-S or M2-S)
    - Reference FASTA

Usage:
    # From filtered Parquet
    python 03_clinvar_benchmark.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --clinvar data/clinvar/clinvar_splice_snvs.parquet \
        --output-dir examples/variant_analysis/results/clinvar_analysis/

    # From raw VCF (filters to splice-relevant)
    python 03_clinvar_benchmark.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --clinvar-vcf data/clinvar/clinvar.vcf.gz \
        --output-dir examples/variant_analysis/results/clinvar_analysis/

    # Quick test (first N variants)
    python 03_clinvar_benchmark.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --clinvar data/clinvar/clinvar_splice_snvs.parquet \
        --max-variants 50 --device cpu
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)


def _base_max_within_radius(base_delta: np.ndarray, radius: int) -> float:
    """Max |Δ| on base model over donor/acceptor within ±radius of window center.

    Mirrors DeltaResult.max_delta_within_radius() for the base_delta array,
    so base and meta are compared on the same scoring window.
    """
    L = len(base_delta)
    center = L // 2
    lo = max(0, center - radius)
    hi = min(L, center + radius + 1)
    sl = base_delta[lo:hi]
    if sl.size == 0:
        return 0.0
    return float(max(
        sl[:, 0].max(), -sl[:, 0].min(),
        sl[:, 1].max(), -sl[:, 1].min(),
    ))


def _load_gene_strands(gtf_path: str) -> Dict[str, str]:
    """Load gene name → strand mapping from GTF."""
    from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
        extract_gene_annotations,
    )
    ann = extract_gene_annotations(gtf_path, verbosity=0)
    strands = {}
    for row in ann.iter_rows(named=True):
        name = row.get("gene_name", "")
        if name:
            strands[name] = row.get("strand", "+")
    return strands


def _load_variants_from_parquet(parquet_path: Path) -> list:
    """Load pre-filtered variants from Parquet."""
    import polars as pl
    df = pl.read_parquet(str(parquet_path))
    variants = []
    for row in df.iter_rows(named=True):
        variants.append(row)
    return variants


def _load_variants_from_vcf(
    vcf_path: Path, min_stars: int = 1,
) -> list:
    """Load and filter variants from raw ClinVar VCF."""
    from agentic_spliceai.splice_engine.meta_layer.data.clinvar_loader import (
        ClinVarLoader,
    )
    loader = ClinVarLoader(vcf_path)
    records = loader.get_splice_relevant(min_stars=min_stars)
    return [
        {
            "clinvar_id": r.clinvar_id,
            "chrom": r.chrom,
            "position": r.position,
            "ref_allele": r.ref_allele,
            "alt_allele": r.alt_allele,
            "gene": r.gene,
            "classification": r.classification,
            "review_stars": r.review_stars,
            "molecular_consequence": r.molecular_consequence,
        }
        for r in records
    ]


def run_benchmark(
    variants: list,
    checkpoint: Path,
    fasta_path: Path,
    gene_strands: Dict[str, str],
    device: str = "cpu",
    base_model: str = "openspliceai",
    use_multimodal: bool = False,
    bigwig_cache_dir: Optional[Path] = None,
    score_radius: int = 50,
) -> List[Dict]:
    """Score each variant and collect delta scores.

    Returns list of dicts with variant info + delta scores.
    """
    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )

    runner = VariantRunner(
        meta_checkpoint=checkpoint,
        fasta_path=fasta_path,
        base_model=base_model,
        device=device,
        event_threshold=0.1,
        bigwig_cache_dir=bigwig_cache_dir,
    )

    results = []
    n_errors = 0
    t0 = time.time()

    for i, v in enumerate(variants):
        chrom = v["chrom"]
        pos = v["position"]
        ref = v["ref_allele"]
        alt = v["alt_allele"]
        gene = v.get("gene", "")
        strand = gene_strands.get(gene, "+")

        try:
            r = runner.run(
                chrom, pos, ref, alt,
                gene=gene, strand=strand,
                use_multimodal=use_multimodal,
            )

            results.append({
                "clinvar_id": v.get("clinvar_id", ""),
                "chrom": chrom,
                "position": pos,
                "ref": ref,
                "alt": alt,
                "gene": gene,
                "strand": strand,
                "classification": v.get("classification", ""),
                "review_stars": v.get("review_stars", 0),
                "molecular_consequence": v.get("molecular_consequence", ""),
                # Meta-layer delta scores
                "meta_ds_dg": float(r.max_donor_gain),
                "meta_ds_dl": float(r.max_donor_loss),
                "meta_ds_ag": float(r.max_acceptor_gain),
                "meta_ds_al": float(r.max_acceptor_loss),
                "meta_max_delta": float(r.max_delta_within_radius(score_radius)),
                "meta_max_delta_fullwindow": float(r.max_delta),
                # Base model delta scores — slice to the same ±radius as meta
                "base_ds_dg": float(r.base_delta[:, 0].max()),
                "base_ds_dl": float(-r.base_delta[:, 0].min()),
                "base_ds_ag": float(r.base_delta[:, 1].max()),
                "base_ds_al": float(-r.base_delta[:, 1].min()),
                "base_max_delta": float(_base_max_within_radius(r.base_delta, score_radius)),
                "base_max_delta_fullwindow": float(max(
                    r.base_delta[:, 0].max(), -r.base_delta[:, 0].min(),
                    r.base_delta[:, 1].max(), -r.base_delta[:, 1].min(),
                )),
                "n_events": len(r.events),
            })

        except Exception as e:
            n_errors += 1
            log.warning("Error scoring %s:%d %s>%s (%s): %s",
                        chrom, pos, ref, alt, gene, e)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Scored {i+1}/{len(variants)} variants "
                  f"({rate:.1f}/s, {n_errors} errors)")

    runner.close()

    elapsed = time.time() - t0
    print(f"\n  Scoring complete: {len(results)} scored, "
          f"{n_errors} errors, {elapsed:.0f}s")
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute ROC-AUC and PR-AUC for pathogenic vs benign."""
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Separate pathogenic vs benign
    labels = []
    meta_scores = []
    base_scores = []

    for r in results:
        if r["classification"] == "Pathogenic":
            labels.append(1)
        elif r["classification"] == "Benign":
            labels.append(0)
        else:
            continue
        meta_scores.append(r["meta_max_delta"])
        base_scores.append(r["base_max_delta"])

    labels = np.array(labels)
    meta_scores = np.array(meta_scores)
    base_scores = np.array(base_scores)

    if len(labels) < 10 or labels.sum() < 5 or (1 - labels).sum() < 5:
        print("  WARNING: Too few variants for meaningful ROC/PR analysis")
        return {
            "n_pathogenic": int(labels.sum()),
            "n_benign": int((1 - labels).sum()),
            "warning": "insufficient_data",
        }

    metrics = {
        "n_pathogenic": int(labels.sum()),
        "n_benign": int((1 - labels).sum()),
        "n_total": len(labels),
        "meta_roc_auc": float(roc_auc_score(labels, meta_scores)),
        "meta_pr_auc": float(average_precision_score(labels, meta_scores)),
        "base_roc_auc": float(roc_auc_score(labels, base_scores)),
        "base_pr_auc": float(average_precision_score(labels, base_scores)),
    }

    # Threshold analysis
    for threshold in [0.1, 0.2, 0.5, 0.8]:
        meta_pred = (meta_scores >= threshold).astype(int)
        base_pred = (base_scores >= threshold).astype(int)
        metrics[f"meta_sensitivity_t{threshold}"] = float(
            meta_pred[labels == 1].mean()) if labels.sum() > 0 else 0
        metrics[f"meta_specificity_t{threshold}"] = float(
            1 - meta_pred[labels == 0].mean()) if (1 - labels).sum() > 0 else 0
        metrics[f"base_sensitivity_t{threshold}"] = float(
            base_pred[labels == 1].mean()) if labels.sum() > 0 else 0
        metrics[f"base_specificity_t{threshold}"] = float(
            1 - base_pred[labels == 0].mean()) if (1 - labels).sum() > 0 else 0

    return metrics


def generate_plots(results: List[Dict], output_dir: Path) -> None:
    """Generate ROC curve, PR curve, and delta distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            roc_curve, precision_recall_curve,
            roc_auc_score, average_precision_score,
        )
    except ImportError:
        print("  Skipping plots (matplotlib not available)")
        return

    labels = []
    meta_scores = []
    base_scores = []
    for r in results:
        if r["classification"] == "Pathogenic":
            labels.append(1)
        elif r["classification"] == "Benign":
            labels.append(0)
        else:
            continue
        meta_scores.append(r["meta_max_delta"])
        base_scores.append(r["base_max_delta"])

    labels = np.array(labels)
    meta_scores = np.array(meta_scores)
    base_scores = np.array(base_scores)

    if len(labels) < 10:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for scores, label, color in [
        (meta_scores, "Meta-layer", "#3572a5"),
        (base_scores, "Base model", "#e74c3c"),
    ]:
        fpr, tpr, _ = roc_curve(labels, scores)
        auc = float(roc_auc_score(labels, scores))
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ClinVar Splice Variant Classification: ROC")
    ax.legend()
    fig.savefig(output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'roc_curve.png'}")

    # PR curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for scores, label, color in [
        (meta_scores, "Meta-layer", "#3572a5"),
        (base_scores, "Base model", "#e74c3c"),
    ]:
        prec, rec, _ = precision_recall_curve(labels, scores)
        ap = float(average_precision_score(labels, scores))
        ax.plot(rec, prec, color=color, label=f"{label} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("ClinVar Splice Variant Classification: PR")
    ax.legend()
    fig.savefig(output_dir / "pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'pr_curve.png'}")

    # Delta distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scores, title in [
        (axes[0], meta_scores, "Meta-layer max |Δ|"),
        (axes[1], base_scores, "Base model max |Δ|"),
    ]:
        ax.hist(scores[labels == 1], bins=30, alpha=0.6, color="#e74c3c",
                label=f"Pathogenic (n={int(labels.sum())})")
        ax.hist(scores[labels == 0], bins=30, alpha=0.6, color="#3572a5",
                label=f"Benign (n={int((1-labels).sum())})")
        ax.set_xlabel("Max |Δ| score")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend()
    fig.savefig(output_dir / "delta_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_dir / 'delta_distributions.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark splice variant prediction on ClinVar",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--clinvar", type=Path, default=None,
                        help="Filtered ClinVar Parquet from 02_clinvar_download.py")
    parser.add_argument("--clinvar-vcf", type=Path, default=None,
                        help="Raw ClinVar VCF (will filter on-the-fly)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("examples/variant_analysis/results/clinvar_analysis/"))
    parser.add_argument("--max-variants", type=int, default=None,
                        help="Limit number of variants (for quick testing)")
    parser.add_argument("--min-stars", type=int, default=1)
    parser.add_argument("--base-model", default="openspliceai")
    parser.add_argument("--no-multimodal", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bigwig-cache", type=Path, default=None,
                        help="Local BigWig cache dir for conservation/epigenetic/chromatin features")
    parser.add_argument("--score-radius", type=int, default=50,
                        help="Radius (bp) around the variant for max_delta reduction. "
                             "Matches OpenSpliceAI's dist_var=50. Set to 2500 to approximate "
                             "the legacy full-window behavior.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    if not args.clinvar and not args.clinvar_vcf:
        parser.error("Provide either --clinvar (Parquet) or --clinvar-vcf (VCF)")

    print("=" * 60)
    print("ClinVar Splice Variant Benchmark")
    print("=" * 60)

    # Load variants
    if args.clinvar:
        print(f"\n  Loading from Parquet: {args.clinvar}")
        variants = _load_variants_from_parquet(args.clinvar)
    else:
        print(f"\n  Loading from VCF: {args.clinvar_vcf}")
        variants = _load_variants_from_vcf(args.clinvar_vcf, args.min_stars)

    if args.max_variants:
        variants = variants[:args.max_variants]

    n_path = sum(1 for v in variants if v.get("classification") == "Pathogenic")
    n_ben = sum(1 for v in variants if v.get("classification") == "Benign")
    print(f"  Variants: {len(variants)} ({n_path} pathogenic, {n_ben} benign)")

    # Load gene strands
    from agentic_spliceai.splice_engine.resources import get_model_resources
    resources = get_model_resources("openspliceai")
    gtf_path = str(resources.get_gtf_path())
    print(f"  Loading gene strands from GTF...")
    gene_strands = _load_gene_strands(gtf_path)
    print(f"  {len(gene_strands)} genes with strand info")

    # Score variants
    print(f"\n  Scoring with checkpoint: {args.checkpoint.name}")
    print(f"  Device: {args.device}")
    results = run_benchmark(
        variants, args.checkpoint, args.fasta,
        gene_strands, args.device, args.base_model,
        use_multimodal=not args.no_multimodal,
        bigwig_cache_dir=args.bigwig_cache,
        score_radius=args.score_radius,
    )

    if not results:
        print("  ERROR: No variants scored successfully")
        return 1

    # Compute metrics
    print(f"\n  Computing metrics...")
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Variants scored:   {len(results)}")
    print(f"  Pathogenic:        {metrics.get('n_pathogenic', 0)}")
    print(f"  Benign:            {metrics.get('n_benign', 0)}")
    if "meta_roc_auc" in metrics:
        print(f"  Meta ROC-AUC:      {metrics['meta_roc_auc']:.4f}")
        print(f"  Base ROC-AUC:      {metrics['base_roc_auc']:.4f}")
        print(f"  Meta PR-AUC:       {metrics['meta_pr_auc']:.4f}")
        print(f"  Base PR-AUC:       {metrics['base_pr_auc']:.4f}")
        print(f"  Meta improvement:  {metrics['meta_roc_auc'] - metrics['base_roc_auc']:+.4f} ROC-AUC")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = args.output_dir / "delta_scores.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Scores: {scores_path}")

    metrics_path = args.output_dir / "benchmark_metrics.json"
    metrics["checkpoint"] = str(args.checkpoint)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Generate plots
    if not args.no_plots:
        print(f"\n  Generating plots...")
        generate_plots(results, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
