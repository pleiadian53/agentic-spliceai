#!/usr/bin/env python
"""Verify data statistics and evaluation metrics across annotation sources and models.

Reproduces and cross-checks the numbers documented in:
  examples/meta_layer/results/alternative_site_evaluation_results.md

Sections:
  1. Annotation statistics: gene counts, splice site counts (row vs unique)
  2. Set differences: Ensembl \\ MANE, GENCODE \\ MANE
  3. Model evaluation metrics from result JSONs
  4. Cross-model comparison tables

Usage:
    python examples/meta_layer/10_verify_evaluation_stats.py
    python examples/meta_layer/10_verify_evaluation_stats.py --section data
    python examples/meta_layer/10_verify_evaluation_stats.py --section metrics
    python examples/meta_layer/10_verify_evaluation_stats.py --section all
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Data statistics
# ---------------------------------------------------------------------------

def _resolve_splice_sites_path(source: str) -> Path:
    """Resolve splice sites path via resource manager.

    Uses the genomic registry for MANE and Ensembl.  GENCODE is not
    yet registered, so it falls back to the conventional path.
    """
    from agentic_spliceai.splice_engine.resources import (
        get_model_resources,
        get_genomic_registry,
    )

    if source == "MANE":
        registry = get_model_resources("openspliceai").get_registry()
        path = registry.resolve("splice_sites")
        if path:
            return Path(path)
    elif source == "Ensembl":
        registry = get_genomic_registry(build="GRCh38", release="112")
        path = registry.resolve("splice_sites")
        if path:
            return Path(path)
    elif source == "GENCODE":
        # GENCODE not yet in registry — use conventional path
        return Path("data/gencode/GRCh38/splice_sites_enhanced.tsv")

    # Fallback to conventional path
    source_dir = source.lower()
    return Path(f"data/{source_dir}/GRCh38/splice_sites_enhanced.tsv")


def verify_annotation_stats() -> None:
    """Verify gene and splice site counts across MANE, Ensembl, GENCODE."""
    import polars as pl

    print("=" * 70)
    print("Section 1: Annotation Statistics")
    print("=" * 70)

    sources = {
        name: _resolve_splice_sites_path(name)
        for name in ["MANE", "Ensembl", "GENCODE"]
    }

    stats = {}
    for name, path in sources.items():
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND ({path})")
            continue

        df = pl.read_csv(str(path), separator="\t")
        n_rows = len(df)
        n_genes = df["gene_id"].n_unique()
        n_gene_names = df["gene_name"].n_unique() if "gene_name" in df.columns else 0

        # Unique (chrom, position) — the true unique site count
        unique_sites = df.select("chrom", "position").unique()
        n_unique = len(unique_sites)

        # Donor/acceptor breakdown
        n_donors = len(df.filter(pl.col("splice_type") == "donor"))
        n_acceptors = len(df.filter(pl.col("splice_type") == "acceptor"))

        stats[name] = {
            "rows": n_rows,
            "genes": n_genes,
            "gene_names": n_gene_names,
            "unique_sites": n_unique,
            "donors": n_donors,
            "acceptors": n_acceptors,
        }

        print(f"\n  {name} ({path.name}):")
        print(f"    Rows (per-transcript):    {n_rows:>10,}")
        print(f"    Unique (chrom, position): {n_unique:>10,}")
        print(f"    Genes (gene_id):          {n_genes:>10,}")
        print(f"    Gene names:               {n_gene_names:>10,}")
        print(f"    Donors:                   {n_donors:>10,}")
        print(f"    Acceptors:                {n_acceptors:>10,}")

    # Set differences
    if "MANE" in stats and "Ensembl" in stats:
        print(f"\n  --- Set Differences ---")
        for name in ["Ensembl", "GENCODE"]:
            if name not in stats:
                continue
            diff = stats[name]["unique_sites"] - stats["MANE"]["unique_sites"]
            print(f"    {name} \\ MANE (approx): ~{diff:,} unique sites")

    # Documented values check
    print(f"\n  --- Cross-check with documented values ---")
    documented = {
        "MANE": {"genes": 18200, "unique_sites": 367000},
        "Ensembl": {"genes": 39291, "unique_sites": 738000},
        "GENCODE": {"genes": 54117, "unique_sites": 928000},
    }
    for name, expected in documented.items():
        if name not in stats:
            continue
        actual_genes = stats[name]["genes"]
        actual_sites = stats[name]["unique_sites"]
        gene_match = abs(actual_genes - expected["genes"]) < 500
        site_match = abs(actual_sites - expected["unique_sites"]) < 5000
        print(
            f"    {name:8} genes: {actual_genes:>6,} "
            f"(doc: ~{expected['genes']:,}) {'OK' if gene_match else 'MISMATCH'}"
            f"  |  sites: {actual_sites:>7,} "
            f"(doc: ~{expected['unique_sites']:,}) {'OK' if site_match else 'MISMATCH'}"
        )


# ---------------------------------------------------------------------------
# Model evaluation metrics
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_alt_metrics(data: dict) -> Optional[dict]:
    """Extract alternative site metrics from eval result JSON."""
    alt = data.get("alternative_sites")
    if alt is None:
        return None
    meta = alt.get("meta_model", {})
    base = alt.get("base_model", {})
    return {
        "meta_pr_auc": meta.get("macro_pr_auc"),
        "base_pr_auc": base.get("macro_pr_auc"),
        "meta_tp": meta.get("tp_count"),
        "meta_fn": meta.get("fn_count"),
        "meta_fp": meta.get("fp_count"),
        "base_tp": base.get("tp_count"),
        "base_fn": base.get("fn_count"),
        "base_fp": base.get("fp_count"),
        "fn_reduction": data.get("fn_reduction_pct", alt.get("fn_reduction_pct")),
        "n_genes": data.get("n_genes"),
        "n_skipped": data.get("n_skipped"),
        "n_alt_sites": data.get("n_alternative_sites"),
    }


def _extract_canonical_metrics(data: dict) -> Optional[dict]:
    """Extract canonical (Eval-MANE) metrics."""
    meta = data.get("meta_model", {})
    base = data.get("base_model", {})
    return {
        "meta_pr_auc": meta.get("macro_pr_auc"),
        "base_pr_auc": base.get("macro_pr_auc"),
        "meta_tp": meta.get("tp_count"),
        "meta_fn": meta.get("fn_count"),
        "meta_fp": meta.get("fp_count"),
        "base_tp": base.get("tp_count"),
        "base_fn": base.get("fn_count"),
        "base_fp": base.get("fp_count"),
        "fn_reduction": data.get("fn_reduction_pct"),
    }


def verify_evaluation_metrics() -> None:
    """Load and display all evaluation result JSONs."""
    print("\n" + "=" * 70)
    print("Section 2: Model Evaluation Metrics")
    print("=" * 70)

    result_dir = Path("output/meta_layer")

    # Define all result files with their model/protocol labels
    results = [
        # (label, json_path, extract_fn)
        ("M1-S v2 on Eval-MANE",
         result_dir / "m1s_v2_logit_blend/eval_results.json",
         _extract_canonical_metrics),
        ("M2-S on Eval-MANE",
         result_dir / "m1_m2c_eval_results.json",
         _extract_canonical_metrics),

        ("M1-S v1 on Eval-Ensembl-Alt",
         result_dir / "m2a_eval/m2a_eval_results.json",
         _extract_alt_metrics),
        ("M1-S v2 on Eval-Ensembl-Alt",
         result_dir / "m2a_v2_eval_results.json",
         _extract_alt_metrics),
        ("M2-S on Eval-Ensembl-Alt (with min_length skip)",
         result_dir / "m2a_m2c_eval_results.json",
         _extract_alt_metrics),
        ("M2-S on Eval-Ensembl-Alt (all genes)",
         result_dir / "m2a_m2c_eval_v2_results.json",
         _extract_alt_metrics),

        ("M1-S v2 on Eval-GENCODE-Alt",
         result_dir / "m2b_v2_eval_results.json",
         _extract_alt_metrics),
        ("M2-S on Eval-GENCODE-Alt (all genes)",
         result_dir / "m2b_m2c_eval_v2_results.json",
         _extract_alt_metrics),
    ]

    for label, path, extract_fn in results:
        data = _load_json(path)
        if data is None:
            print(f"\n  {label}: FILE NOT FOUND ({path})")
            continue

        metrics = extract_fn(data)
        if metrics is None:
            print(f"\n  {label}: no metrics extracted")
            continue

        meta_recall = None
        if metrics["meta_tp"] and metrics["meta_fn"]:
            total = metrics["meta_tp"] + metrics["meta_fn"]
            meta_recall = metrics["meta_tp"] / total * 100 if total > 0 else 0

        print(f"\n  {label}")
        print(f"    File: {path}")
        print(f"    Meta PR-AUC:  {metrics['meta_pr_auc']:.4f}")
        print(f"    Base PR-AUC:  {metrics['base_pr_auc']:.4f}")
        print(f"    Meta TPs:     {metrics['meta_tp']:>10,}")
        print(f"    Meta FNs:     {metrics['meta_fn']:>10,}")
        print(f"    Meta FPs:     {metrics['meta_fp']:>10,}")
        print(f"    Base TPs:     {metrics['base_tp']:>10,}")
        print(f"    Base FNs:     {metrics['base_fn']:>10,}")
        print(f"    Base FPs:     {metrics['base_fp']:>10,}")
        if meta_recall is not None:
            print(f"    Meta Recall:  {meta_recall:.1f}%")
        if metrics.get("fn_reduction") is not None:
            print(f"    FN reduction: {metrics['fn_reduction']:+.1f}%")
        if metrics.get("n_genes") is not None:
            print(f"    Genes eval'd: {metrics['n_genes']:,}"
                  f" (skipped: {metrics.get('n_skipped', '?')})")


def verify_comparison_table() -> None:
    """Print the key comparison table matching the results document."""
    print("\n" + "=" * 70)
    print("Section 3: Cross-Model Comparison")
    print("=" * 70)

    result_dir = Path("output/meta_layer")

    # Eval-Ensembl-Alt comparison
    print("\n  --- Eval-Ensembl-Alt: Alternative Sites (Ensembl \\ MANE) ---\n")
    print(f"  {'Model':<35} {'PR-AUC':>8} {'Recall':>8} {'TPs':>10} {'FNs':>10} {'FPs':>10}")
    print(f"  {'-'*83}")

    eval_ensembl = [
        ("Base (OpenSpliceAI)",
         result_dir / "m2a_v2_eval_results.json", "base"),
        ("M1-S v2 (MANE-trained)",
         result_dir / "m2a_v2_eval_results.json", "meta"),
        ("M2-S (Ensembl-trained, all genes)",
         result_dir / "m2a_m2c_eval_v2_results.json", "meta"),
    ]

    for label, path, model_key in eval_ensembl:
        data = _load_json(path)
        if data is None:
            print(f"  {label:<35} FILE NOT FOUND")
            continue
        alt = data.get("alternative_sites", {})
        m = alt.get(f"{model_key}_model", {})
        tp = m.get("tp_count", 0)
        fn = m.get("fn_count", 0)
        fp = m.get("fp_count", 0)
        pr_auc = m.get("macro_pr_auc", 0)
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        print(f"  {label:<35} {pr_auc:>8.4f} {recall:>7.1f}% {tp:>10,} {fn:>10,} {fp:>10,}")

    # Eval-GENCODE-Alt comparison
    print(f"\n  --- Eval-GENCODE-Alt: Alternative Sites (GENCODE \\ MANE) ---\n")
    print(f"  {'Model':<35} {'PR-AUC':>8} {'Recall':>8} {'TPs':>10} {'FNs':>10} {'FPs':>10}")
    print(f"  {'-'*83}")

    eval_gencode = [
        ("Base (OpenSpliceAI)",
         result_dir / "m2b_m2c_eval_v2_results.json", "base"),
        ("M2-S (Ensembl-trained, all genes)",
         result_dir / "m2b_m2c_eval_v2_results.json", "meta"),
    ]

    for label, path, model_key in eval_gencode:
        data = _load_json(path)
        if data is None:
            print(f"  {label:<35} FILE NOT FOUND")
            continue
        alt = data.get("alternative_sites", {})
        m = alt.get(f"{model_key}_model", {})
        tp = m.get("tp_count", 0)
        fn = m.get("fn_count", 0)
        fp = m.get("fp_count", 0)
        pr_auc = m.get("macro_pr_auc", 0)
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        print(f"  {label:<35} {pr_auc:>8.4f} {recall:>7.1f}% {tp:>10,} {fn:>10,} {fp:>10,}")

    # Eval-MANE comparison
    print(f"\n  --- Eval-MANE: Canonical Sites ---\n")
    print(f"  {'Model':<35} {'PR-AUC':>8} {'FNs':>10} {'FPs':>12} {'FN Red':>8}")
    print(f"  {'-'*78}")

    eval_mane = [
        ("M1-S v2 (MANE-trained)",
         result_dir / "m1s_v2_logit_blend/eval_results.json"),
        ("M2-S (Ensembl-trained)",
         result_dir / "m1_m2c_eval_results.json"),
    ]

    for label, path in eval_mane:
        data = _load_json(path)
        if data is None:
            print(f"  {label:<35} FILE NOT FOUND")
            continue
        meta = data.get("meta_model", {})
        pr_auc = meta.get("macro_pr_auc", 0)
        fn = meta.get("fn_count", 0)
        fp = meta.get("fp_count", 0)
        fn_red = data.get("fn_reduction_pct", 0)
        print(f"  {label:<35} {pr_auc:>8.4f} {fn:>10,} {fp:>12,} {fn_red:>+7.1f}%")

    # Training metrics
    print(f"\n  --- Training Best Metrics ---\n")
    print(f"  {'Model':<25} {'Epoch':>6} {'Val PR-AUC':>12} {'Val Acc':>10}")
    print(f"  {'-'*58}")

    train_metrics = [
        ("M1-S v2", result_dir / "m1s_v2_logit_blend/best_metrics.json"),
        ("M2-S", result_dir / "m2c/best_metrics.json"),
    ]

    for label, path in train_metrics:
        data = _load_json(path)
        if data is None:
            print(f"  {label:<25} FILE NOT FOUND")
            continue
        print(f"  {label:<25} {data.get('epoch', '?'):>6} "
              f"{data.get('macro_pr_auc', 0):>12.4f} "
              f"{data.get('accuracy', 0):>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify data statistics and evaluation metrics",
    )
    parser.add_argument(
        "--section", choices=["data", "metrics", "comparison", "all"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    if args.section in ("data", "all"):
        verify_annotation_stats()

    if args.section in ("metrics", "all"):
        verify_evaluation_metrics()

    if args.section in ("comparison", "all"):
        verify_comparison_table()

    print(f"\n{'='*70}")
    print("Verification complete.")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
