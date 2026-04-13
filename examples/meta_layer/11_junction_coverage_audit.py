#!/usr/bin/env python3
"""Junction coverage audit: does GTEx junction data substantially cover
Ensembl-annotated splice sites?

This script is the automated counterpart to
``notebooks/meta_layer/junction_coverage_audit.ipynb``. It runs
end-to-end on the SpliceAI test chromosomes (by default) and writes
structured outputs that answer three research questions:

  Q1. **Annotated-site coverage.** For each Ensembl splice site on the
      chosen chromosomes, does GTEx see any junction supporting it?
      At what read depth and tissue breadth?
  Q2. **Canonical vs alt-only breakdown.** Is coverage worse for
      Ensembl-alt-only sites (i.e., sites in Ensembl but not MANE)?
  Q3. **Junctions outside annotations.** How many junction coordinates
      fall on positions *not* in any Ensembl splice site?

Output layout (default ``output/meta_layer/junction_coverage_audit/``):

    q1_annotated_site_coverage.parquet    per-site coverage table
    q1_coverage_by_type.csv               summary by (splice_type, strand)
    q2_canonical_vs_alt.csv               MANE vs alt-only breakdown
    q3_novel_junction_sides.parquet       junction sides not in Ensembl
    q3_novel_summary.csv                  counts + depth/breadth stats
    depth_breadth_hist.png                covered-site histograms
    findings.md                           auto-generated report

Usage:

    python examples/meta_layer/11_junction_coverage_audit.py
    python examples/meta_layer/11_junction_coverage_audit.py \\
        --chroms all \\
        --output-dir output/meta_layer/junction_coverage_audit_fullgenome
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — writes PNGs without a display
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# SpliceAI test split — same chromosomes our PR-AUCs were computed on.
SPLICEAI_TEST_CHROMS = ["1", "3", "5", "7", "9"]


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------

def normalize_junctions(junction_parquet: Path, strand_lookup: pl.DataFrame) -> pl.LazyFrame:
    """Load the GTEx junction parquet, normalize chromosome names, and emit
    one row per (junction, side) with the derived exonic splice-site position.

    Junctions are stored as intron coordinates (start/end inclusive, 1-based).
    On + strand: donor = start-1, acceptor = end+1.
    On - strand: donor = end+1, acceptor = start-1.

    GTEx doesn't carry strand explicitly, so we recover it via a gene_id join
    against the Ensembl splice-sites table (each gene has a single strand).
    """
    junc = (
        pl.scan_parquet(junction_parquet)
        # Strip chr prefix and gene_id version suffix for clean joining
        .with_columns(pl.col("chrom").str.replace("^chr", "").alias("chrom"))
        .with_columns(pl.col("gene_id").str.replace(r"\..*$", "").alias("gene_id"))
    )
    junc_with_strand = junc.join(strand_lookup.lazy(), on="gene_id", how="inner")

    # Emit one row per (junction, donor-side) and per (junction, acceptor-side)
    donor_side = junc_with_strand.with_columns(
        [
            pl.when(pl.col("strand") == "+")
            .then(pl.col("start") - 1)
            .otherwise(pl.col("end") + 1)
            .alias("position"),
            pl.lit("donor").alias("splice_type"),
        ]
    )
    acceptor_side = junc_with_strand.with_columns(
        [
            pl.when(pl.col("strand") == "+")
            .then(pl.col("end") + 1)
            .otherwise(pl.col("start") - 1)
            .alias("position"),
            pl.lit("acceptor").alias("splice_type"),
        ]
    )
    cols = [
        "chrom",
        "strand",
        "position",
        "splice_type",
        "gene_id",
        "max_reads",
        "sum_reads",
        "n_tissues",
        "tissue_breadth",
    ]
    return pl.concat([donor_side.select(cols), acceptor_side.select(cols)])


# ---------------------------------------------------------------------------
# Q1: Annotated-site coverage
# ---------------------------------------------------------------------------

def compute_q1(
    ensembl_sites: pl.LazyFrame, junction_sides: pl.LazyFrame
) -> pl.DataFrame:
    """Left-join Ensembl sites against junction sides. A site is 'covered'
    if at least one junction lands on its (chrom, position, strand, splice_type).
    Depth and tissue breadth are aggregated with max() across matching junctions.
    """
    ens_uniq = ensembl_sites.select(
        ["chrom", "position", "strand", "splice_type", "gene_id"]
    ).unique()

    return (
        ens_uniq.join(
            junction_sides.select(
                [
                    "chrom",
                    "position",
                    "strand",
                    "splice_type",
                    "sum_reads",
                    "n_tissues",
                    "tissue_breadth",
                ]
            ),
            on=["chrom", "position", "strand", "splice_type"],
            how="left",
        )
        .group_by(["chrom", "position", "strand", "splice_type", "gene_id"])
        .agg(
            [
                pl.col("sum_reads").max().alias("max_sum_reads"),
                pl.col("n_tissues").max().alias("max_n_tissues"),
                pl.col("tissue_breadth").max().alias("max_tissue_breadth"),
            ]
        )
        .with_columns(pl.col("max_sum_reads").is_not_null().alias("covered"))
    ).collect()


def summarize_q1_by_type(coverage: pl.DataFrame) -> pl.DataFrame:
    """Aggregate Q1 coverage by (splice_type, strand)."""
    return (
        coverage.group_by(["splice_type", "strand"])
        .agg(
            [
                pl.len().alias("n_sites"),
                pl.col("covered").mean().alias("coverage_rate"),
                pl.col("max_sum_reads")
                .filter(pl.col("covered"))
                .median()
                .alias("median_reads_when_covered"),
                pl.col("max_n_tissues")
                .filter(pl.col("covered"))
                .median()
                .alias("median_tissues_when_covered"),
            ]
        )
        .sort(["splice_type", "strand"])
    )


# ---------------------------------------------------------------------------
# Q2: MANE vs alt-only breakdown
# ---------------------------------------------------------------------------

def compute_q2(
    coverage: pl.DataFrame, mane_sites: pl.LazyFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split coverage into MANE-canonical vs Ensembl-alt-only by structural
    coordinate tuple (not gene_id — MANE uses RefSeq gene symbols, Ensembl
    uses ENSG, so gene_id-based joins would return zero).

    Returns two frames:
      - ``canonical_vs_alt`` (2 rows): MANE vs alt-only summary.
      - ``canonical_vs_alt_by_strand`` (4 rows): further broken down by strand.
        Useful for detecting strand-asymmetric annotation artifacts.
    """
    mane_keys = (
        mane_sites.select(["chrom", "position", "strand", "splice_type"])
        .unique()
        .collect()
    )
    coverage_with_flag = coverage.join(
        mane_keys.with_columns(pl.lit(True).alias("in_mane")),
        on=["chrom", "position", "strand", "splice_type"],
        how="left",
    ).with_columns(pl.col("in_mane").fill_null(False))

    def _agg(df: pl.DataFrame, by_cols: list[str]) -> pl.DataFrame:
        return df.group_by(by_cols).agg(
            [
                pl.len().alias("n_sites"),
                pl.col("covered").mean().alias("coverage_rate"),
                pl.col("max_sum_reads")
                .filter(pl.col("covered"))
                .median()
                .alias("median_reads_when_covered"),
                pl.col("max_n_tissues")
                .filter(pl.col("covered"))
                .median()
                .alias("median_tissues_when_covered"),
            ]
        )

    canonical_vs_alt = _agg(coverage_with_flag, ["in_mane"]).sort("in_mane", descending=True)
    by_strand = _agg(coverage_with_flag, ["strand", "in_mane"]).sort(
        ["strand", "in_mane"], descending=[False, True]
    )
    return canonical_vs_alt, by_strand


# ---------------------------------------------------------------------------
# Q3: Novel junctions (outside Ensembl annotation)
# ---------------------------------------------------------------------------

def compute_q3(
    junction_sides: pl.LazyFrame, ensembl_sites: pl.LazyFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (novel_sides, summary). A junction side is 'novel' if its
    (chrom, position, strand, splice_type) does not match any Ensembl site."""
    ens_keys = ensembl_sites.select(
        ["chrom", "position", "strand", "splice_type"]
    ).unique().with_columns(pl.lit(True).alias("_in_ensembl"))

    novel = (
        junction_sides.join(
            ens_keys,
            on=["chrom", "position", "strand", "splice_type"],
            how="left",
        )
        .filter(pl.col("_in_ensembl").is_null())
        .drop("_in_ensembl")
    ).collect()

    summary = (
        novel.group_by("splice_type")
        .agg(
            [
                pl.len().alias("n_novel_sides"),
                pl.col("sum_reads").median().alias("median_reads"),
                pl.col("n_tissues").median().alias("median_tissues"),
                pl.col("tissue_breadth").median().alias("median_tissue_breadth"),
            ]
        )
        .sort("splice_type")
    )
    return novel, summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_depth_breadth(coverage: pl.DataFrame, output_path: Path) -> None:
    covered = coverage.filter(pl.col("covered"))
    if covered.is_empty():
        logger.warning("No covered sites to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(
        covered["max_n_tissues"].to_numpy(),
        bins=54,
        edgecolor="black",
        alpha=0.8,
    )
    axes[0].set_xlabel("# tissues supporting site (out of 54)")
    axes[0].set_ylabel("# covered Ensembl sites")
    axes[0].set_title("Tissue breadth at covered sites")
    axes[0].grid(True, alpha=0.3)

    depth_clipped = covered["max_sum_reads"].clip(upper_bound=200).to_numpy()
    axes[1].hist(depth_clipped, bins=50, edgecolor="black", alpha=0.8, color="steelblue")
    axes[1].set_xlabel("sum_reads across 54 tissues (clipped at 200)")
    axes[1].set_ylabel("# covered Ensembl sites")
    axes[1].set_title("Read depth at covered sites")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Auto-generated findings.md
# ---------------------------------------------------------------------------

def write_findings(
    output_dir: Path,
    chrom_list: list[str],
    coverage: pl.DataFrame,
    by_type: pl.DataFrame,
    canonical_vs_alt: pl.DataFrame,
    canonical_vs_alt_by_strand: pl.DataFrame,
    novel: pl.DataFrame,
    novel_summary: pl.DataFrame,
    n_ensembl: int,
    n_mane: int,
    n_junction: int,
) -> None:
    """Fill in the findings template with actual numbers."""
    overall_rate = float(coverage["covered"].mean())
    covered_only = coverage.filter(pl.col("covered"))
    median_tissues = (
        float(covered_only["max_n_tissues"].median())
        if not covered_only.is_empty()
        else 0.0
    )
    median_reads = (
        float(covered_only["max_sum_reads"].median())
        if not covered_only.is_empty()
        else 0.0
    )

    # MANE vs alt
    mane_row = canonical_vs_alt.filter(pl.col("in_mane")).row(0, named=True) if (
        canonical_vs_alt.filter(pl.col("in_mane")).height
    ) else {"coverage_rate": None, "n_sites": 0}
    alt_row = canonical_vs_alt.filter(~pl.col("in_mane")).row(0, named=True) if (
        canonical_vs_alt.filter(~pl.col("in_mane")).height
    ) else {"coverage_rate": None, "n_sites": 0}

    mane_rate = mane_row["coverage_rate"]
    alt_rate = alt_row["coverage_rate"]
    gap_pts = (
        (mane_rate - alt_rate) * 100
        if mane_rate is not None and alt_rate is not None
        else None
    )

    n_junction_sides = int(novel_summary["n_novel_sides"].sum()) if not novel_summary.is_empty() else 0
    n_novel = len(novel)

    lines: list[str] = []
    lines.append("# Junction Coverage Audit — Findings\n\n")
    if chrom_list == ["all"]:
        scope_desc = "genome-wide (all canonical chromosomes)"
    else:
        scope_desc = (f"{', '.join(chrom_list)} "
                      f"({len(chrom_list)} of 24 canonical chromosomes)")
    lines.append(f"**Chromosomes analyzed:** {scope_desc}\n\n")
    lines.append("## Input sizes\n\n")
    lines.append(f"- Ensembl splice sites (filtered chroms): **{n_ensembl:,}**\n")
    lines.append(f"- MANE splice sites (filtered chroms): **{n_mane:,}**\n")
    lines.append(f"- GTEx junction sides (filtered chroms, 2 per unique junction): "
                 f"**{n_junction:,}**\n\n")
    lines.append("---\n\n")

    lines.append("## Q1. Annotated-site coverage\n\n")
    lines.append(f"**Overall coverage rate: {overall_rate:.1%}** of Ensembl splice sites "
                 f"have at least one matching GTEx junction.\n\n")
    lines.append(f"- Median tissues supporting a covered site: **{median_tissues:.0f} / 54**\n")
    lines.append(f"- Median summed reads at a covered site: **{median_reads:,.0f}**\n\n")
    lines.append("### Coverage by splice type and strand\n\n")
    lines.append(by_type.to_pandas().to_markdown(index=False) + "\n\n")
    lines.append("---\n\n")

    lines.append("## Q2. Canonical (MANE) vs alt-only (Ensembl \\ MANE)\n\n")
    if mane_rate is not None:
        lines.append(f"- **MANE canonical:** {mane_row['n_sites']:,} sites, "
                     f"coverage {mane_rate:.1%}\n")
    if alt_rate is not None:
        lines.append(f"- **Alt-only (Ensembl \\ MANE):** {alt_row['n_sites']:,} sites, "
                     f"coverage {alt_rate:.1%}\n")
    if gap_pts is not None:
        lines.append(f"- **Gap:** {gap_pts:+.1f} percentage points "
                     f"{'(canonical > alt-only)' if gap_pts > 0 else '(alt-only ≥ canonical)'}\n\n")
    lines.append(canonical_vs_alt.to_pandas().to_markdown(index=False) + "\n\n")

    lines.append("### Strand × MANE breakdown (artifact check)\n\n")
    lines.append(canonical_vs_alt_by_strand.to_pandas().to_markdown(index=False) + "\n\n")
    # Detect and flag any strand asymmetry concentrated in alt-only
    try:
        plus_alt = canonical_vs_alt_by_strand.filter(
            (pl.col("strand") == "+") & (~pl.col("in_mane"))
        ).row(0, named=True)["coverage_rate"]
        minus_alt = canonical_vs_alt_by_strand.filter(
            (pl.col("strand") == "-") & (~pl.col("in_mane"))
        ).row(0, named=True)["coverage_rate"]
        alt_strand_gap = (plus_alt - minus_alt) * 100
        if abs(alt_strand_gap) >= 20:
            direction = "+ > −" if alt_strand_gap > 0 else "− > +"
            lines.append(
                f"> ⚠️  **Strand asymmetry detected in alt-only coverage:** "
                f"{alt_strand_gap:+.1f} pts ({direction}). MANE sites are balanced "
                f"across strands, so this is unlikely to be junction-data bias — "
                f"suggests a strand-asymmetric artifact in the Ensembl splice-sites "
                f"enumeration. Worth auditing the TSV generation before drawing "
                f"strong biological conclusions from the alt-only coverage rate.\n\n"
            )
    except Exception:
        pass

    lines.append("### Interpretation\n\n")
    if alt_rate is not None and mane_rate is not None:
        if alt_rate < 0.5 and mane_rate >= 0.8:
            lines.append(
                "Coverage is strong for canonical transcripts but thin for alt-only "
                "sites. This ceilings what GTEx-only junction features can contribute "
                "to alt-site detection — the model's alt-site recall gains from "
                "junction features come from a *minority* of alt sites, the "
                "well-expressed ones.\n\n"
            )
        elif alt_rate >= 0.8 and mane_rate >= 0.8:
            lines.append(
                "Both canonical and alt-only sites are broadly covered. Junction "
                "evidence is broadly informative and M3 has real raw material to "
                "draw on.\n\n"
            )
        elif alt_rate < 0.6 and mane_rate < 0.8:
            lines.append(
                "Junction data is fundamentally sparse — neither canonical nor "
                "alt-only coverage is high. M2-S/M3's junction channel would "
                "benefit from richer sources (recount3, ENCODE long-read, etc.).\n\n"
            )
    lines.append("---\n\n")

    lines.append("## Q3. Junctions outside Ensembl annotation\n\n")
    lines.append(f"Junction sides that do **not** match any Ensembl splice site "
                 f"are the upper bound on what GTEx could surface as novel.\n\n")
    lines.append(f"- Total junction sides on selected chroms: **{n_junction:,}**\n")
    lines.append(f"- Junction sides NOT in Ensembl (novel candidates): "
                 f"**{n_novel:,}** ({n_novel / max(1, n_junction):.1%})\n\n")
    lines.append("### Per splice type\n\n")
    lines.append(novel_summary.to_pandas().to_markdown(index=False) + "\n\n")
    lines.append("### Interpretation for M3\n\n")
    lines.append(
        "M3's goal is to detect *novel* splice sites beyond annotations. "
        f"The {n_novel:,} novel junction sides above are the absolute ceiling on "
        "how many GTEx can help with. If median depth and breadth there are "
        "lower than at annotated sites, those novel candidates will be harder "
        "to distinguish from noise — a fundamental data issue architecture "
        "can't fix.\n\n"
    )

    lines.append("---\n\n")
    lines.append("## Artifacts\n\n")
    lines.append("| File | Contents |\n|---|---|\n")
    lines.append("| `q1_annotated_site_coverage.parquet` | Per-site coverage table |\n")
    lines.append("| `q1_coverage_by_type.csv` | Summary by (splice_type, strand) |\n")
    lines.append("| `q2_canonical_vs_alt.csv` | MANE vs alt-only breakdown |\n")
    lines.append("| `q3_novel_junction_sides.parquet` | Novel junction sides (per-row) |\n")
    lines.append("| `q3_novel_summary.csv` | Per-splice-type novel summary |\n")
    lines.append("| `depth_breadth_hist.png` | Histograms at covered sites |\n")

    (output_dir / "findings.md").write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--junction-parquet",
        type=Path,
        default=Path("data/GRCh38/junction_data/junctions_gtex_v8.parquet"),
    )
    p.add_argument(
        "--ensembl-splice-sites",
        type=Path,
        default=Path("data/ensembl/GRCh38/splice_sites_enhanced.tsv"),
    )
    p.add_argument(
        "--mane-splice-sites",
        type=Path,
        default=Path("data/mane/GRCh38/splice_sites_enhanced.tsv"),
    )
    p.add_argument(
        "--chroms",
        nargs="+",
        default=SPLICEAI_TEST_CHROMS,
        help=(
            "Chromosomes to analyze (Ensembl naming — bare, no chr prefix). "
            "Default: SpliceAI test split (1, 3, 5, 7, 9). Use 'all' for genome-wide."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/meta_layer/junction_coverage_audit"),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for p in (args.junction_parquet, args.ensembl_splice_sites, args.mane_splice_sites):
        if not p.exists():
            logger.error("Missing input: %s", p)
            return 1

    chrom_filter = None if args.chroms == ["all"] else args.chroms
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Ensembl splice sites: %s", args.ensembl_splice_sites)
    ensembl = pl.scan_csv(
        args.ensembl_splice_sites,
        separator="\t",
        schema_overrides={"chrom": pl.Utf8},
    )
    if chrom_filter is not None:
        ensembl = ensembl.filter(pl.col("chrom").is_in(chrom_filter))

    logger.info("Loading MANE splice sites: %s", args.mane_splice_sites)
    mane = pl.scan_csv(
        args.mane_splice_sites,
        separator="\t",
        schema_overrides={"chrom": pl.Utf8},
    ).with_columns(pl.col("chrom").str.replace("^chr", "").alias("chrom"))
    if chrom_filter is not None:
        mane = mane.filter(pl.col("chrom").is_in(chrom_filter))

    # Recover strand per junction-gene
    gene_strand = ensembl.select(["gene_id", "strand"]).unique().collect()
    logger.info("Gene→strand lookup: %d genes", gene_strand.height)

    logger.info("Loading + normalizing GTEx junctions: %s", args.junction_parquet)
    junc_sides = normalize_junctions(args.junction_parquet, gene_strand)
    if chrom_filter is not None:
        junc_sides = junc_sides.filter(pl.col("chrom").is_in(chrom_filter))

    # Counts
    n_ensembl = ensembl.select(pl.len()).collect().item()
    n_mane = mane.select(pl.len()).collect().item()
    n_junction_sides = junc_sides.select(pl.len()).collect().item()
    logger.info("Ensembl sites: %d | MANE sites: %d | Junction sides: %d",
                n_ensembl, n_mane, n_junction_sides)

    # Q1
    logger.info("Computing Q1: annotated-site coverage")
    coverage = compute_q1(ensembl, junc_sides)
    by_type = summarize_q1_by_type(coverage)
    coverage.write_parquet(args.output_dir / "q1_annotated_site_coverage.parquet")
    by_type.write_csv(args.output_dir / "q1_coverage_by_type.csv")
    logger.info("Q1 overall coverage: %.1f%%", 100 * float(coverage["covered"].mean()))

    # Q2
    logger.info("Computing Q2: canonical vs alt-only")
    canonical_vs_alt, canonical_vs_alt_by_strand = compute_q2(coverage, mane)
    canonical_vs_alt.write_csv(args.output_dir / "q2_canonical_vs_alt.csv")
    canonical_vs_alt_by_strand.write_csv(args.output_dir / "q2_canonical_vs_alt_by_strand.csv")

    # Q3
    logger.info("Computing Q3: novel junction coordinates")
    novel, novel_summary = compute_q3(junc_sides, ensembl)
    novel.write_parquet(args.output_dir / "q3_novel_junction_sides.parquet")
    novel_summary.write_csv(args.output_dir / "q3_novel_summary.csv")
    logger.info("Q3 novel junction sides: %d (%.1f%% of total)",
                len(novel),
                100 * len(novel) / max(1, n_junction_sides))

    # Plots + findings
    plot_depth_breadth(coverage, args.output_dir / "depth_breadth_hist.png")
    chrom_list = chrom_filter if chrom_filter is not None else ["all"]
    write_findings(
        args.output_dir,
        chrom_list,
        coverage,
        by_type,
        canonical_vs_alt,
        canonical_vs_alt_by_strand,
        novel,
        novel_summary,
        n_ensembl,
        n_mane,
        n_junction_sides,
    )

    logger.info("All artifacts written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
