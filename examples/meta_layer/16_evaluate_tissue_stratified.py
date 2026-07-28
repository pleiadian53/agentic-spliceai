#!/usr/bin/env python
"""Tissue-stratified evaluation of alternative-site recall (M2-S).

Stratifies alternative-site (Ensembl ∖ MANE) recall by the **GTEx tissue whose
junctions support each site**, using the per-tissue GTEx v8 junction table
(``junctions_gtex_v8_by_tissue.parquet``, 54 tissues).

This is an *evidence-stratified* view, not a tissue-conditioned model. M2-S is
tissue-agnostic — it emits one prediction per site regardless of tissue — so the
per-tissue numbers reflect **which alternative sites carry tissue-specific junction
support and how detectable they are**, not tissue-specific prediction. Alternative
sites with no GTEx junction support in the selected tissues fall out of every bucket
(a known, reported bias).

Base-model recall runs fully locally from the precomputed base scores. Meta (M2-S)
recall needs per-site outcomes from the neural eval (dense multimodal features →
pod); supply them with ``--meta-outcomes`` to fill the meta column (see the pod
runner ``ops_eval_tissue_pod.sh``).

Example (base, local)::

    python examples/meta_layer/16_evaluate_tissue_stratified.py \\
        --tissues dnase5 --models base

Output: ``<output-dir>/tissue_stratified.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _example_utils import get_project_root  # noqa: E402  (marker-based root finding)

logger = logging.getLogger("tissue_stratified")

# GTEx SMTSD names for the 5 tissues that match the chromatin-accessibility DNase set.
DNASE5_TISSUES: dict[str, str] = {
    "Brain - Cortex": "Brain cortex",
    "Heart - Left Ventricle": "Heart",
    "Lung": "Lung",
    "Muscle - Skeletal": "Muscle",
    "Liver": "Liver",
}

DEFAULT_TEST_CHROMS = ["chr1", "chr3", "chr5", "chr7", "chr9"]


def _norm_chrom(col: pl.Expr) -> pl.Expr:
    """Normalize a chromosome column to ``chr``-prefixed form (chr1, chrX, ...)."""
    s = col.cast(pl.Utf8)
    return pl.when(s.str.starts_with("chr")).then(s).otherwise(pl.concat_str([pl.lit("chr"), s]))


def load_site_set(path: Path, chroms: list[str]) -> pl.DataFrame:
    """Load a ``splice_sites_enhanced.tsv`` as unique (chrom, position, splice_type) rows."""
    lf = (
        pl.scan_csv(path, separator="\t")
        .select(["chrom", "position", "splice_type"])
        .with_columns(_norm_chrom(pl.col("chrom")).alias("chrom"))
        .filter(pl.col("chrom").is_in(chroms))
        .unique()
    )
    return lf.collect()


def alternative_sites(ensembl: pl.DataFrame, mane: pl.DataFrame) -> pl.DataFrame:
    """Alternative sites = Ensembl site set minus the MANE site set."""
    return ensembl.join(mane, on=["chrom", "position", "splice_type"], how="anti")


def build_tissue_index(parquet: Path, tissues: dict[str, str], chroms: list[str],
                       min_reads: int) -> pl.DataFrame:
    """Map each GTEx junction to its two exon-boundary positions and the tissues
    that support it. Returns (chrom, position, tissue) rows.

    Junction boundaries follow the junction modality's convention:
    ``donor_pos = start - 1`` and ``acceptor_pos = end + 1``.
    """
    df = (
        pl.scan_parquet(parquet)
        .filter(pl.col("tissue").is_in(list(tissues)) & (pl.col("total_reads") >= min_reads))
        .with_columns(_norm_chrom(pl.col("chrom")).alias("chrom"))
        .filter(pl.col("chrom").is_in(chroms))
        .select(["chrom", "start", "end", "tissue"])
        .collect()
    )
    donor = df.select(["chrom", "tissue", (pl.col("start") - 1).alias("position")])
    acceptor = df.select(["chrom", "tissue", (pl.col("end") + 1).alias("position")])
    return pl.concat([donor, acceptor]).unique()


def base_outcomes(pred_path: Path, sites: pl.DataFrame, chroms: list[str]) -> pl.DataFrame:
    """Attach a base-model ``detected`` flag to each site via argmax over the three
    per-position class probabilities from the precomputed base scores."""
    preds = (
        pl.scan_csv(pred_path, separator="\t")
        .with_columns(_norm_chrom(pl.col("chrom")).alias("chrom"))
        .filter(pl.col("chrom").is_in(chroms))
        .select(["chrom", "position", "donor_prob", "acceptor_prob", "neither_prob"])
        .collect()
    )
    joined = sites.join(preds, on=["chrom", "position"], how="left")
    # argmax over (donor, acceptor, neither); a site is "detected" when its own class wins.
    detected = (
        pl.when(pl.col("splice_type") == "donor")
        .then((pl.col("donor_prob") >= pl.col("acceptor_prob")) & (pl.col("donor_prob") >= pl.col("neither_prob")))
        .otherwise((pl.col("acceptor_prob") >= pl.col("donor_prob")) & (pl.col("acceptor_prob") >= pl.col("neither_prob")))
    )
    # Sites with no base score in the given chroms stay null (excluded from recall),
    # never counted as a miss.
    return joined.with_columns(
        pl.when(pl.col("donor_prob").is_null()).then(None).otherwise(detected).alias("base_detected")
    )


def _recall(frame: pl.DataFrame, col: str) -> tuple[float | None, int]:
    """(mean of a boolean detection column ignoring nulls, count of non-null)."""
    if col not in frame.columns:
        return None, 0
    n = int(frame[col].is_not_null().sum())
    return (float(frame[col].mean()) if n else None), n


def stratify(sites: pl.DataFrame, tissue_index: pl.DataFrame, tissues: dict[str, str]) -> dict:
    """Per-tissue base (and meta, if present) recall over alternative sites.

    ``sites`` must already carry a boolean ``base_detected`` column and optionally
    ``meta_detected`` (nulls = not scored)."""
    per = sites.join(tissue_index, on=["chrom", "position"], how="inner")
    has_meta = "meta_detected" in sites.columns and int(sites["meta_detected"].is_not_null().sum()) > 0

    n_alt_total = sites.height
    n_alt_supported = sites.join(
        tissue_index.select(["chrom", "position"]).unique(), on=["chrom", "position"], how="inner"
    ).height

    rows = []
    for smtsd, display in tissues.items():
        sub = per.filter(pl.col("tissue") == smtsd)
        base_recall, n_scored = _recall(sub, "base_detected")
        meta_recall, _ = _recall(sub, "meta_detected") if has_meta else (None, 0)
        rows.append({
            "tissue": smtsd,
            "display": display,
            "n_alt_sites": sub.height,
            "n_scored": n_scored,
            "base_recall": base_recall,
            "meta_recall": meta_recall,
        })
    rows.sort(key=lambda r: r["n_alt_sites"], reverse=True)

    overall_base, n_scored_total = _recall(sites, "base_detected")
    return {
        "operating_point": "argmax",
        "n_alt_sites_total": n_alt_total,
        "n_alt_sites_supported": n_alt_supported,
        "n_alt_sites_scored": n_scored_total,
        "overall_base_recall": overall_base,
        "has_meta": has_meta,
        "tissues": rows,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    root = get_project_root()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-annotation", default="ensembl", help="Annotation whose extra sites define 'alternative' (default: ensembl).")
    ap.add_argument("--test-chroms", nargs="+", default=DEFAULT_TEST_CHROMS, help="Held-out chromosomes to evaluate.")
    ap.add_argument("--tissues", choices=["dnase5", "all"], default="dnase5", help="Tissue set (default: dnase5).")
    ap.add_argument("--models", default="base", help="Comma-separated: 'base' and/or 'meta'. Meta needs --meta-outcomes.")
    ap.add_argument("--min-reads", type=int, default=1, help="Min GTEx total_reads for a tissue to count as supporting a site.")
    ap.add_argument("--mane-sites", type=Path, default=None)
    ap.add_argument("--eval-sites", type=Path, default=None)
    ap.add_argument("--base-scores", type=Path, default=None, help="Base predictions .tsv (chrom, position, *_prob).")
    ap.add_argument("--tissue-parquet", type=Path, default=None)
    ap.add_argument("--meta-outcomes", type=Path, default=None, help="Parquet (chrom, position, splice_type, meta_detected) from the pod neural eval.")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    chroms = [c if c.startswith("chr") else f"chr{c}" for c in args.test_chroms]
    tissues = DNASE5_TISSUES if args.tissues == "dnase5" else None  # 'all' resolved below

    mane_path = args.mane_sites or root / "data" / "mane" / "GRCh38" / "splice_sites_enhanced.tsv"
    eval_path = args.eval_sites or root / "data" / args.eval_annotation / "GRCh38" / "splice_sites_enhanced.tsv"
    base_path = args.base_scores or root / "data" / args.eval_annotation / "GRCh38" / "openspliceai_eval" / "precomputed" / "predictions.tsv"
    tissue_parquet = args.tissue_parquet or root / "data" / "GRCh38" / "junction_data" / "junctions_gtex_v8_by_tissue.parquet"
    out_dir = args.output_dir or root / "output" / "meta_layer" / "m2s_v4_cleanannot_alt_eval"

    for p in (mane_path, eval_path, tissue_parquet):
        if not p.exists():
            raise FileNotFoundError(f"Required input not found: {p}")
    if args.meta_outcomes is None and not base_path.exists():
        raise FileNotFoundError(f"Base scores not found: {base_path} (needed without --meta-outcomes)")

    logger.info("Deriving alternative sites (%s ∖ MANE) on %s ...", args.eval_annotation, ",".join(chroms))
    mane = load_site_set(mane_path, chroms)
    ev = load_site_set(eval_path, chroms)
    alt = alternative_sites(ev, mane)
    logger.info("  MANE sites: %d | %s sites: %d | alternative: %d", mane.height, args.eval_annotation, ev.height, alt.height)

    if tissues is None:  # --tissues all
        all_t = (
            pl.scan_parquet(tissue_parquet).select("tissue").unique().collect()["tissue"].to_list()
        )
        tissues = {t: t for t in sorted(all_t)}

    logger.info("Building tissue-support index (%d tissues, min_reads=%d) ...", len(tissues), args.min_reads)
    tissue_index = build_tissue_index(tissue_parquet, tissues, chroms, args.min_reads)

    models = {m.strip() for m in args.models.split(",")}
    outcomes = None
    if args.meta_outcomes:
        if not args.meta_outcomes.exists():
            raise SystemExit(f"--meta-outcomes not found: {args.meta_outcomes}")
        outcomes = pl.read_parquet(args.meta_outcomes).with_columns(
            _norm_chrom(pl.col("chrom")).alias("chrom")
        )
    elif "meta" in models:
        raise SystemExit(
            "Meta recall requested but no --meta-outcomes parquet. Meta per-site outcomes come from "
            "the neural eval (09 --dump-site-outcomes on a pod); see ops_eval_tissue_pod.sh."
        )

    if outcomes is not None:
        # Both base_detected and meta_detected come from the neural eval dump (09),
        # which used the real held-out base scores + M2-S inference.
        logger.info("Attaching per-site outcomes from %s ...", args.meta_outcomes.name)
        keep = ["chrom", "position", "splice_type"] + [
            c for c in ("base_detected", "meta_detected") if c in outcomes.columns
        ]
        alt = alt.join(outcomes.select(keep), on=["chrom", "position", "splice_type"], how="left")
        if "base_detected" not in alt.columns:
            alt = alt.with_columns(pl.lit(None, dtype=pl.Boolean).alias("base_detected"))
    else:
        logger.info("Scoring base-model detection at alternative sites (local base scores) ...")
        alt = base_outcomes(base_path, alt, chroms)

    result = stratify(alt, tissue_index, tissues)
    result["eval_annotation"] = args.eval_annotation
    result["test_chromosomes"] = chroms
    result["min_reads"] = args.min_reads
    result["caveat"] = (
        "Evidence-stratified, not tissue-conditioned: M2-S is tissue-agnostic, so per-tissue "
        "differences reflect which alternative sites carry junction support in each tissue and how "
        "detectable they are, not tissue-specific prediction. Alternative sites without GTEx junction "
        "support in these tissues are excluded from every bucket. GTEx v8 = bulk adult tissue."
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tissue_stratified.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("\nWrote %s", out_path)
    obr = result["overall_base_recall"]
    logger.info("Overall: %d alt sites | %d supported in %d tissues | base recall %s (scored locally: %d)",
                result["n_alt_sites_total"], result["n_alt_sites_supported"], len(tissues),
                f"{obr:.3f}" if obr is not None else "n/a — base scores not local for these chroms",
                result["n_alt_sites_scored"])
    for r in result["tissues"]:
        br = f"{r['base_recall']:.3f}" if r["base_recall"] is not None else "  -- "
        logger.info("  %-24s n=%-7d base_recall=%s", r["display"], r["n_alt_sites"], br)


if __name__ == "__main__":
    main()
