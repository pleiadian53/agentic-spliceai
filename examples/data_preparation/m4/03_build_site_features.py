#!/usr/bin/env python
"""Build the M4 site × feature table: the multimodal vector of every regulated site.

``kd_events.parquet`` labels A3SS/A5SS events (signed long-form ΔPSI under 185
HepG2 knockdowns). Each event has three sites: the long and short alternative
sites and the shared junction partner on the flanking exon. M4 needs the
meta-layer's tabular feature vector (base scores + conservation + epigenetic +
junction + RBP + chromatin) for every one of them. Labels are per (event,
regulator); these features are per site, joined to events by coordinate later.

Every site's features are **computed** from contiguous per-nucleotide base
predictions, so that the base-score context features see true neighbours:

* ``computed_mane`` — the site lies in a MANE gene: a ±``--window`` nt window is
  pulled from the stored predictions (``precomputed/``), run through
  ``FeaturePipeline`` with the ``full_stack.yaml`` modality configs that built
  the genome-wide parquets, and the centre row is kept.
* ``computed_ensembl`` / ``computed_span`` — the site's gene is not in MANE
  (lncRNA, pseudogene, retired ID). The base model needs only sequence, so it is
  run on demand (``BaseModelRunner.predict_spans``) over the Ensembl gene span,
  or over site ± ``--span-flank`` nt when no Ensembl gene contains it.

**No site is dropped.** Joins are keyed on coordinates ``(chrom, position,
strand)``, never gene symbol: ~1K "non-MANE" sites are MANE genes under renamed
symbols (MARS → MARS1, SEPT7 → SEPTIN7, TAZ → TAFAZZIN). ``gene_set`` tags each
site ``mane`` / ``ensembl_only`` / ``unannotated`` so results can be stratified:
OpenSpliceAI was trained on protein-coding MANE genes, so non-MANE sites are out
of distribution for the base model.

Why not reuse the genome-wide feature parquets (``analysis_sequences/``)? They
cover ~65% of these sites, but they were sampled *before* feature engineering
(score > 0.01 + 0.5% background), and the base-score context features use
``shift().over("gene_id")``, i.e. the neighbouring *row*. In those parquets the
neighbouring row is the next *sampled* position, not the adjacent nucleotide
(``context_score_m1`` equals the previous sampled row's score 99.98% of the time
on chr21). They are therefore used only as a **reference check**: every site
present there is compared column by column with the computed vector. Non-base-
score modalities must agree; base-score context columns are expected to differ.

Output: ``data/mane/GRCh38/m4_labels/site_features.parquet`` (one row per
unique site; bare chrom, 1-based exonic-boundary position, as in kd_events) +
``output/meta_layer/m4_labels/site_features_report.json``.

Usage:
    # one chromosome, local modalities only
    python examples/data_preparation/m4/03_build_site_features.py \\
        --chromosomes 21 --skip-remote-modalities

    # full build on a pod (all modalities; restartable)
    python examples/data_preparation/m4/03_build_site_features.py \\
        --bigwig-cache /runpod-volume/bigwig_cache --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _example_utils import setup_example_environment  # noqa: E402

setup_example_environment()

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "examples/features"))
from config_loader import load_workflow_config  # noqa: E402

import agentic_spliceai.splice_engine.features.modalities  # noqa: E402,F401  (registers modalities)
from agentic_spliceai.splice_engine.features import FeaturePipeline  # noqa: E402
from agentic_spliceai.splice_engine.features.track_cache import default_cache_dir  # noqa: E402

M4_LABELS = REPO / "data/mane/GRCh38/m4_labels"
EVENTS = M4_LABELS / "kd_events.parquet"
EVAL_DIR = REPO / "data/mane/GRCh38/openspliceai_eval"
FEATURE_DIR = EVAL_DIR / "analysis_sequences"
PRED_DIR = EVAL_DIR / "precomputed"
OUT = M4_LABELS / "site_features.parquet"
REPORT = REPO / "output/meta_layer/m4_labels/site_features_report.json"

KEY = ["chrom", "position", "strand"]
PRED_COLS = ["gene_id", "gene_name", "chrom", "position", "strand",
             "gene_start", "gene_end", "donor_prob", "acceptor_prob", "neither_prob"]
REMOTE_MODALITIES = ("epigenetic", "chrom_access")  # bigWigs not in the local cache


def _norm_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Bare chrom + Int64 position + string strand (the m3/m4 label convention)."""
    return df.with_columns(
        pl.col("chrom").cast(pl.String).str.replace(r"^chr", ""),
        pl.col("position").cast(pl.Int64),
        pl.col("strand").cast(pl.String),
    )


def load_sites(chromosomes: list[str] | None) -> pl.DataFrame:
    """Unique event sites (long, short alternative sites + shared partner).

    A site can be claimed by events in more than one gene (68 sites genome-wide), so
    each site carries the sorted lists of every candidate ``ensg`` and ``gene_symbol``;
    keeping an arbitrary one made gene coverage depend on row order.
    """
    lf = pl.scan_parquet(EVENTS)
    if chromosomes:
        lf = lf.filter(pl.col("chrom").is_in(chromosomes))
    lf = lf.select("chrom", "strand", "gene_id", "gene_symbol",
                   "long_site", "short_site", "partner_site").unique()
    return (
        pl.concat([
            lf.select("chrom", pl.col(f"{r}_site").alias("position"), "strand",
                      "gene_id", "gene_symbol")
            for r in ("long", "short", "partner")
        ])
        .with_columns(pl.col("gene_id").str.replace(r"\..*$", "").alias("ensg"))
        .group_by(KEY)
        .agg(pl.col("ensg").unique().sort(), pl.col("gene_symbol").unique().sort())
        .collect()
        .sort(KEY)
    )


def reference_rows(chrom: str, sites: pl.DataFrame) -> pl.DataFrame:
    """Rows of the genome-wide (score-thresholded) feature parquet that are regulated sites.

    Empty when that store is absent or is not the full feature table (a pod volume can
    hold a stripped copy); the reference check is a diagnostic and is then skipped.
    """
    path = FEATURE_DIR / f"analysis_sequences_chr{chrom}.parquet"
    if not path.exists():
        return pl.DataFrame()
    if not {*KEY, "donor_prob"} <= set(pl.scan_parquet(path).collect_schema().names()):
        return pl.DataFrame()
    feat = _norm_keys(
        pl.scan_parquet(path)
        .filter(pl.col("position").is_in(sites["position"].cast(pl.Float64).unique().to_list()))
        .collect()
    )
    return (
        feat.join(sites.select(KEY), on=KEY, how="semi")
        .sort([*KEY, "gene_id"]).unique(KEY, keep="first", maintain_order=True)
    )


def _windows(sites: pl.DataFrame, w: int) -> pl.DataFrame:
    """(position, strand) pairs within ±w of each site."""
    return (
        sites.select("position", "strand")
        .with_columns(pl.int_ranges(pl.col("position") - w, pl.col("position") + w + 1)
                      .alias("position"))
        .explode("position").unique()
    )


def mane_windows(chrom: str, sites: pl.DataFrame, w: int) -> pl.DataFrame:
    """Stored per-nucleotide predictions around ``sites`` (all overlapping MANE genes)."""
    path = PRED_DIR / f"predictions_chr{chrom}.parquet"
    if sites.height == 0 or not path.exists():
        return pl.DataFrame()
    win = _windows(sites, w)
    rows = (
        pl.scan_parquet(path)
        .filter(pl.col("position").is_in(win["position"].unique().to_list()))
        .select(PRED_COLS)
        .collect()
    )
    return rows.join(win, on=["position", "strand"], how="semi")


def _ensembl_genes(runner_state: dict) -> pl.DataFrame:
    """Ensembl gene coordinates (bare chrom), parsed from the registry GTF once per run."""
    if "ensembl_genes" not in runner_state:
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            load_gene_annotations,
        )
        from agentic_spliceai.splice_engine.resources import get_genomic_registry

        gtf = get_genomic_registry(build="GRCh38").get_gtf_path(validate=True)
        genes = load_gene_annotations(gtf_path=gtf, verbosity=0)
        if "chrom" not in genes.columns:
            genes = genes.rename({"seqname": "chrom"})
        runner_state["ensembl_genes"] = genes.drop("seqname", strict=False).with_columns(
            pl.col("chrom").cast(pl.String).str.replace(r"^chr", ""))
    return runner_state["ensembl_genes"]


def on_demand_windows(
    chrom: str, sites: pl.DataFrame, w: int, flank: int, runner_state: dict,
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Run the base model for sites with no stored prediction.

    Returns the window rows and a {site_key: gene_set} map
    (``ensembl_only`` if an Ensembl gene span was used, ``unannotated`` otherwise).
    """
    from agentic_spliceai.splice_engine.base_layer.data import filter_by_genes
    from agentic_spliceai.splice_engine.base_layer.models.runner import BaseModelRunner
    from agentic_spliceai.splice_engine.base_layer.prediction.core import load_spliceai_models

    if sites.height == 0:
        return pl.DataFrame(), {}
    if "models" not in runner_state:
        runner_state["runner"] = BaseModelRunner()
        runner_state["models"] = load_spliceai_models(
            model_type=runner_state["model"], build="GRCh38", verbosity=0)
    runner, models = runner_state["runner"], runner_state["models"]

    # Gene coordinates only: predict_spans extracts sequence from the model's FASTA
    # (the same GRCh38 primary assembly), so the Ensembl GTF is parsed once, not per chrom.
    genes = filter_by_genes(_ensembl_genes(runner_state),
                            sites["ensg"].explode().unique().to_list(),
                            gene_column="gene_id", verbosity=0)
    genes = genes.filter(pl.col("chrom") == chrom)
    # A site is covered by its Ensembl gene only if it lies inside it on the same strand.
    covered = (
        sites.select(*KEY, "ensg").explode("ensg")
        .join(genes.select(pl.col("gene_id").alias("ensg"), "start", "end",
                                pl.col("strand").alias("g_strand")), on="ensg", how="inner")
        .filter((pl.col("position") >= pl.col("start")) & (pl.col("position") <= pl.col("end"))
                & (pl.col("strand") == pl.col("g_strand")))
        .select(KEY).unique()
    )
    uncovered = sites.join(covered, on=KEY, how="anti")
    gene_set = {f"{r['chrom']}:{r['position']}:{r['strand']}": "ensembl_only"
                for r in covered.iter_rows(named=True)}

    frames = []
    used = genes.filter(pl.col("gene_id").is_in(
        sites.join(covered, on=KEY, how="semi")["ensg"].explode().unique().to_list()))
    if used.height:
        frames.append(runner.predict_spans(runner_state["model"], used, models=models, verbosity=0))
    if uncovered.height:
        spans = uncovered.select(
            pl.format("span:{}:{}:{}", "chrom", "position", "strand").alias("gene_id"),
            "chrom",
            (pl.col("position") - flank).clip(lower_bound=1).alias("start"),
            (pl.col("position") + flank).alias("end"),
            "strand",
        )
        frames.append(runner.predict_spans(runner_state["model"], spans, models=models, verbosity=0))
        gene_set.update({f"{r['chrom']}:{r['position']}:{r['strand']}": "unannotated"
                         for r in uncovered.iter_rows(named=True)})
    preds = pl.concat(frames, how="vertical_relaxed").with_columns(
        pl.format("chr{}", pl.col("chrom").cast(pl.String).str.replace(r"^chr", "")).alias("chrom"))
    return preds.join(_windows(sites, w), on=["position", "strand"], how="semi"), gene_set


def compute_features(windows: pl.DataFrame, centres: pl.DataFrame,
                     pipeline: FeaturePipeline) -> pl.DataFrame:
    """Run the feature pipeline on contiguous windows; keep one centre row per site.

    A position inside two overlapping same-strand genes is predicted once per gene, and
    the probabilities differ with the gene span (e.g. chr22:19847389 scores 0.0014 in
    GNB1L, 0.094 in RTL10). A row from a gene the site's events were called in
    (``centres.gene_symbol``, a list) wins; ``gene_id`` order breaks remaining ties.
    """
    if windows.height == 0:
        return pl.DataFrame()
    df = windows.unique(["gene_id", "position", "strand"]).sort(["gene_id", "position"])
    out = _norm_keys(pipeline.transform(df))
    return (
        out.join(centres.select(*KEY, "gene_symbol"), on=KEY, how="inner")
        .with_columns((~pl.col("gene_symbol").list.contains(pl.col("gene_name")))
                      .alias("_other_gene"))
        .sort([*KEY, "_other_gene", "gene_id"])
        .unique(KEY, keep="first", maintain_order=True)
        .drop("_other_gene", "gene_symbol")
    )


def reference_check(reference: pl.DataFrame, computed: pl.DataFrame,
                    base_score_cols: set[str], metadata_cols: set[str],
                    tol: float = 1e-5) -> dict:
    """Compare computed vectors with the genome-wide parquet rows for the same sites.

    Returns the columns whose max relative |Δ| exceeds ``tol``, split into base-score
    columns (expected: the parquets used sampled neighbours) and everything else
    (must be empty for the compute path to be trusted). ``metadata_cols``
    (annotation/sequence outputs, not model inputs) are skipped.
    """
    j = reference.join(computed, on=KEY, suffix="__c")
    over: dict[str, float] = {}
    n_cols = 0
    for c in computed.columns:
        if (c in KEY or c in metadata_cols or f"{c}__c" not in j.columns
                or not computed[c].dtype.is_numeric()):
            continue
        n_cols += 1
        ref, new = j[c].cast(pl.Float64), j[f"{c}__c"].cast(pl.Float64)
        rel = ((ref - new).abs() / ref.abs().clip(lower_bound=1.0)).max()  # float32-safe
        if rel is not None and rel > tol:
            over[c] = float(rel)
    return {
        "n_sites": j.height,
        "n_columns_compared": n_cols,
        "base_score_cols_differ": sorted(c for c in over if c in base_score_cols),
        "other_cols_differ": {c: v for c, v in over.items() if c not in base_score_cols},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chromosomes", nargs="+", default=None,
                    help="bare chrom names (default: all in kd_events)")
    ap.add_argument("--window", type=int, default=10,
                    help="±nt of contiguous predictions per site (base_scores context needs ≥2)")
    ap.add_argument("--span-flank", type=int, default=5000,
                    help="±nt around a site with no Ensembl gene (SpliceAI-10k sees ±5 kb)")
    ap.add_argument("--model", default="openspliceai")
    ap.add_argument("--config", default="full_stack", help="examples/features/configs/<name>.yaml")
    ap.add_argument("--skip-remote-modalities", action="store_true",
                    help=f"omit {REMOTE_MODALITIES} (bigWigs stream from ENCODE when uncached)")
    ap.add_argument("--bigwig-cache", type=Path, default=None,
                    help="directory of cached bigWigs (default: settings.yaml cache.bigwig_dir; "
                         "on a pod: /runpod-volume/bigwig_cache)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished per-chromosome parts in <out>.parts/")
    ap.add_argument("--parts-dir", type=Path, default=None,
                    help="per-chromosome checkpoint dir (default: <out>.parts). Point extra "
                         "workers (--chromosomes subset, own --out/--report) at the same dir to "
                         "parallelise; a final --resume run assembles the full table.")
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    pipe_cfg, _, _ = load_workflow_config(config_name=args.config)
    if args.skip_remote_modalities:
        pipe_cfg.modalities = [m for m in pipe_cfg.modalities if m not in REMOTE_MODALITIES]
    # Point bigWig modalities at the cache so cached tracks are read locally;
    # uncached ones still fall back to remote.
    bw_cache = args.bigwig_cache or default_cache_dir()
    print(f"bigWig cache: {bw_cache} ({'present' if bw_cache.exists() else 'MISSING'})")
    for cfg in pipe_cfg.modality_configs.values():
        if hasattr(cfg, "cache_dir") and cfg.cache_dir is None and bw_cache.exists():
            cfg.cache_dir = bw_cache
    pipeline = FeaturePipeline(pipe_cfg)
    print(f"Pipeline: {pipe_cfg.modalities}")

    sites_all = load_sites(args.chromosomes)
    chroms = sorted(sites_all["chrom"].unique().to_list(),
                    key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    print(f"{sites_all.height:,} unique regulated sites on {len(chroms)} chromosomes")

    runner_state: dict = {"model": args.model}
    schema = pipeline.get_output_schema()
    base_score_cols = set(schema.get("base_scores", []))
    metadata_cols = set(schema.get("annotation", [])) | set(schema.get("sequence", [])) | {
        "transcript_id", "transcript_count", "gene_start", "gene_end"}
    parts, report = [], {"per_chrom": {}, "reference_check": {}, "config": vars(args) | {
        "out": str(args.out), "bigwig_cache": str(bw_cache), "modalities": pipe_cfg.modalities}}
    parts_dir = args.parts_dir or args.out.with_suffix(".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for chrom in chroms:
        tc = time.time()
        sites = sites_all.filter(pl.col("chrom") == chrom)
        part_path, part_meta = parts_dir / f"chr{chrom}.parquet", parts_dir / f"chr{chrom}.json"
        if args.resume and part_path.exists() and part_meta.exists():
            parts.append(pl.read_parquet(part_path))
            meta = json.loads(part_meta.read_text())
            report["per_chrom"][chrom] = meta["per_chrom"]
            if meta.get("reference_check"):
                report["reference_check"][chrom] = meta["reference_check"]
            print(f"  chr{chrom}: resumed from {part_path.name}")
            continue

        mane_win = mane_windows(chrom, sites, args.window)
        mane_found = (_norm_keys(mane_win).join(sites.select(KEY), on=KEY, how="semi")
                      .select(KEY).unique() if mane_win.height else sites.select(KEY).head(0))
        rest = sites.join(mane_found, on=KEY, how="anti")
        od_win, od_set = on_demand_windows(chrom, rest, args.window, args.span_flank, runner_state)

        centres_mane = sites.join(mane_found, on=KEY, how="semi")
        tagged = [compute_features(mane_win, centres_mane, pipeline).with_columns(
            feature_source=pl.lit("computed_mane"), gene_set=pl.lit("mane"))]
        tier_od = compute_features(od_win, rest, pipeline)
        if tier_od.height:
            site_set = pl.DataFrame({"_k": list(od_set), "gene_set": list(od_set.values())},
                                    schema={"_k": pl.String, "gene_set": pl.String})
            tagged.append(
                tier_od.with_columns(pl.format("{}:{}:{}", *KEY).alias("_k"))
                .join(site_set, on="_k", how="left").drop("_k")
                .with_columns(feature_source=pl.when(pl.col("gene_set") == "ensembl_only")
                              .then(pl.lit("computed_ensembl")).otherwise(pl.lit("computed_span")))
            )
        chrom_df = pl.concat([t for t in tagged if t.height], how="diagonal_relaxed")
        unresolved = sites.join(chrom_df.select(KEY), on=KEY, how="anti").height
        parts.append(chrom_df)

        reference = reference_rows(chrom, sites)
        if reference.height:
            report["reference_check"][chrom] = reference_check(
                reference, chrom_df, base_score_cols, metadata_cols)
        rc = report["reference_check"].get(chrom, {})

        counts = chrom_df.group_by("feature_source").len().sort("feature_source")
        report["per_chrom"][chrom] = {"sites": sites.height, "unresolved": unresolved,
                                      **dict(counts.iter_rows())}
        # Checkpoint, written atomically (tmp + rename) so a concurrent worker or a
        # resuming run never reads a half-written part; the .json lands last.
        tmp = part_path.with_suffix(f".{os.getpid()}.tmp")
        chrom_df.write_parquet(tmp)
        os.replace(tmp, part_path)
        tmp_meta = part_meta.with_suffix(f".{os.getpid()}.tmp")
        tmp_meta.write_text(json.dumps({"per_chrom": report["per_chrom"][chrom],
                                        "reference_check": rc}, indent=2, default=str))
        os.replace(tmp_meta, part_meta)
        print(f"  chr{chrom}: {sites.height:,} sites → {dict(counts.iter_rows())} "
              f"unresolved={unresolved} | ref-check {rc.get('n_sites', 0)} sites: "
              f"{len(rc.get('base_score_cols_differ', []))} base-score cols differ (expected), "
              f"other differ={rc.get('other_cols_differ', {})} ({time.time() - tc:.0f}s)")

    table = pl.concat(parts, how="diagonal_relaxed")
    totals = table.group_by("feature_source").len().sort("feature_source")
    report["totals"] = {"sites_in": sites_all.height, "sites_out": table.height,
                        **dict(totals.iter_rows()),
                        "gene_set": dict(table.group_by("gene_set").len().iter_rows())}
    report["runtime_seconds"] = round(time.time() - t0, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.write_parquet(args.out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{table.height:,}/{sites_all.height:,} sites, {table.width} cols → {args.out}")
    print(f"  by source: {dict(totals.iter_rows())}")
    print(f"  by gene_set: {report['totals']['gene_set']}")
    print(f"Report → {args.report}")
    return 0 if table.height == sites_all.height else 1


if __name__ == "__main__":
    sys.exit(main())
