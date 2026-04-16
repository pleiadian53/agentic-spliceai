#!/usr/bin/env python
"""Benchmark against MutSpliceDB RNA-seq validated splice variants.

Validates both delta score magnitude AND predicted consequence type
against experimentally observed splicing effects from TCGA RNA-seq.

MutSpliceDB provides the ground truth: each variant has a documented
splicing effect (intron retention, exon skipping) confirmed by RNA-seq
junction analysis.  This script checks:
  1. Does the model detect a splice-altering event (delta > threshold)?
  2. Does the predicted consequence match the observed effect?

Data preparation (one-time):
    python scripts/data/parse_mutsplicedb.py
    # Parses raw CSV export → data/mutsplicedb/splice_sites_induced.tsv

Usage:
    python 04_mutsplicedb_benchmark.py \
        --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --output-dir examples/variant_analysis/results/mutsplicedb_analysis/

    # Quick test
    python 04_mutsplicedb_benchmark.py \
        --checkpoint output/meta_layer/m2c/best.pt \
        --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
        --max-variants 20 --device cpu
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

# MutSpliceDB effect → expected SpliceEventDetector consequence mapping
_EFFECT_TO_CONSEQUENCE = {
    "intron_retention": {"intron_retention", "donor_destruction", "donor_shift",
                         "acceptor_destruction", "acceptor_shift"},
    "exon_skipping": {"exon_skipping"},
}


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


def _base_max_within_radius(base_delta: np.ndarray, radius: int) -> float:
    """Max |Δ| on base model over donor/acceptor within ±radius of window center."""
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


def run_benchmark(
    variants: list,
    checkpoint: Path,
    fasta_path: Path,
    gtf_path: str,
    gene_strands: Dict[str, str],
    device: str = "cpu",
    delta_threshold: float = 0.1,
    use_multimodal: bool = False,
    bigwig_cache_dir: Optional[Path] = None,
    score_radius: int = 50,
) -> List[Dict]:
    """Score each MutSpliceDB variant and compare with expected effect."""
    from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import (
        VariantRunner,
    )
    from agentic_spliceai.splice_engine.meta_layer.inference.splice_event_detector import (
        SpliceEventDetector,
    )

    runner = VariantRunner(
        meta_checkpoint=checkpoint,
        fasta_path=fasta_path,
        base_model="openspliceai",
        device=device,
        event_threshold=delta_threshold,
        bigwig_cache_dir=bigwig_cache_dir,
    )
    detector = SpliceEventDetector(gtf_path=gtf_path)

    results = []
    n_errors = 0
    t0 = time.time()

    for i, v in enumerate(variants):
        chrom = v.chrom
        pos = v.position
        ref = v.ref_allele
        alt = v.alt_allele
        gene = v.gene
        strand = gene_strands.get(gene, v.strand)

        try:
            r = runner.run(
                chrom, pos, ref, alt,
                gene=gene, strand=strand,
                use_multimodal=use_multimodal,
            )
            consequence = detector.analyze(r, gene=gene)

            # Check if predicted consequence matches expected effect
            expected_consequences = _EFFECT_TO_CONSEQUENCE.get(v.effect_type, set())
            consequence_match = consequence.consequence_type in expected_consequences

            results.append({
                "gene": gene,
                "chrom": chrom,
                "position": pos,
                "ref": ref,
                "alt": alt,
                "hgvs": v.hgvs,
                "strand": strand,
                "expected_effect": v.effect_type,
                "predicted_consequence": consequence.consequence_type,
                "consequence_match": consequence_match,
                "confidence": consequence.confidence,
                "affected_exons": consequence.affected_exons,
                "meta_max_delta": float(r.max_delta_within_radius(score_radius)),
                "meta_max_delta_fullwindow": float(r.max_delta),
                "base_max_delta": float(_base_max_within_radius(r.base_delta, score_radius)),
                "base_max_delta_fullwindow": float(max(
                    r.base_delta[:, 0].max(), -r.base_delta[:, 0].min(),
                    r.base_delta[:, 1].max(), -r.base_delta[:, 1].min(),
                )),
                "meta_ds_dg": float(r.max_donor_gain),
                "meta_ds_dl": float(r.max_donor_loss),
                "meta_ds_ag": float(r.max_acceptor_gain),
                "meta_ds_al": float(r.max_acceptor_loss),
                "n_events": len(r.events),
                "evidence_source": v.evidence_source,
            })

        except Exception as e:
            n_errors += 1
            log.warning("Error scoring %s %s:%d: %s", gene, chrom, pos, e)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Scored {i+1}/{len(variants)} variants ({rate:.1f}/s)")

    runner.close()

    elapsed = time.time() - t0
    print(f"\n  Complete: {len(results)} scored, {n_errors} errors, {elapsed:.0f}s")
    return results


def compute_metrics(results: List[Dict], thresholds: list = None) -> Dict:
    """Compute detection rate and consequence concordance."""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.5, 0.8]

    meta_deltas = np.array([r["meta_max_delta"] for r in results])
    base_deltas = np.array([r["base_max_delta"] for r in results])

    metrics = {
        "n_variants": len(results),
        "n_consequence_match": sum(1 for r in results if r["consequence_match"]),
        "consequence_concordance": sum(1 for r in results if r["consequence_match"]) / max(len(results), 1),
        "meta_mean_delta": float(meta_deltas.mean()),
        "meta_median_delta": float(np.median(meta_deltas)),
        "base_mean_delta": float(base_deltas.mean()),
        "base_median_delta": float(np.median(base_deltas)),
    }

    # Detection rate at various thresholds
    for t in thresholds:
        metrics[f"meta_detection_rate_t{t}"] = float((meta_deltas >= t).mean())
        metrics[f"base_detection_rate_t{t}"] = float((base_deltas >= t).mean())

    # Confidence distribution
    for conf in ["HIGH", "MODERATE", "LOW"]:
        metrics[f"confidence_{conf.lower()}"] = sum(
            1 for r in results if r["confidence"] == conf
        )

    # Effect type breakdown
    by_effect = {}
    for r in results:
        et = r["expected_effect"]
        if et not in by_effect:
            by_effect[et] = {"n": 0, "detected_t02": 0, "consequence_match": 0}
        by_effect[et]["n"] += 1
        if r["meta_max_delta"] >= 0.2:
            by_effect[et]["detected_t02"] += 1
        if r["consequence_match"]:
            by_effect[et]["consequence_match"] += 1
    metrics["by_effect_type"] = by_effect

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark against MutSpliceDB RNA-seq validated variants",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument("--mutsplicedb", type=Path,
                        default=Path("data/mutsplicedb/splice_sites_induced.tsv"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("examples/variant_analysis/results/mutsplicedb_analysis/"))
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--delta-threshold", type=float, default=0.1)
    parser.add_argument("--no-multimodal", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bigwig-cache", type=Path, default=None,
                        help="Local BigWig cache dir for conservation/epigenetic/chromatin features")
    parser.add_argument("--score-radius", type=int, default=50,
                        help="Radius (bp) around the variant for max_delta reduction. "
                             "Matches OpenSpliceAI's dist_var=50.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")

    print("=" * 60)
    print("MutSpliceDB Benchmark (RNA-seq Validated Variants)")
    print("=" * 60)

    # Load MutSpliceDB
    from agentic_spliceai.splice_engine.meta_layer.data.mutsplicedb_loader import (
        MutSpliceDBLoader,
    )
    loader = MutSpliceDBLoader(args.mutsplicedb)
    variants = list(loader.iter_variants())

    if args.max_variants:
        variants = variants[:args.max_variants]

    stats = loader.get_statistics()
    print(f"\n  MutSpliceDB: {stats['total']} variants, {stats['n_genes']} genes")
    print(f"  Effect types: {stats['effect_types']}")
    print(f"  Scoring: {len(variants)} variants")

    # Load gene strands
    from agentic_spliceai.splice_engine.resources import get_model_resources
    resources = get_model_resources("openspliceai")
    gtf_path = str(resources.get_gtf_path())
    gene_strands = _load_gene_strands(gtf_path)

    # Run benchmark
    print(f"\n  Checkpoint: {args.checkpoint.name}")
    results = run_benchmark(
        variants, args.checkpoint, args.fasta, gtf_path,
        gene_strands, args.device, args.delta_threshold,
        use_multimodal=not args.no_multimodal,
        bigwig_cache_dir=args.bigwig_cache,
        score_radius=args.score_radius,
    )

    if not results:
        print("  ERROR: No variants scored")
        return 1

    # Compute metrics
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print(f"  Results")
    print(f"{'='*60}")
    print(f"  Variants scored:          {metrics['n_variants']}")
    print(f"  Consequence concordance:  {metrics['consequence_concordance']:.1%}")
    print(f"  Meta mean |Δ|:            {metrics['meta_mean_delta']:.3f}")
    print(f"  Base mean |Δ|:            {metrics['base_mean_delta']:.3f}")
    print(f"\n  Detection rates (variant is splice-altering if |Δ| ≥ threshold):")
    for t in [0.1, 0.2, 0.5]:
        meta = metrics[f"meta_detection_rate_t{t}"]
        base = metrics[f"base_detection_rate_t{t}"]
        print(f"    t={t}: meta={meta:.1%}, base={base:.1%}")
    print(f"\n  By effect type:")
    for et, info in metrics.get("by_effect_type", {}).items():
        det_rate = info["detected_t02"] / max(info["n"], 1) * 100
        conc_rate = info["consequence_match"] / max(info["n"], 1) * 100
        print(f"    {et}: {info['n']} variants, "
              f"detected={det_rate:.0f}%, consequence_match={conc_rate:.0f}%")

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = args.output_dir / "delta_scores.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Scores: {scores_path}")

    metrics["checkpoint"] = str(args.checkpoint)
    metrics_path = args.output_dir / "benchmark_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
