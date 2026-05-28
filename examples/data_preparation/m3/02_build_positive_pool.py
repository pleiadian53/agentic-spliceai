"""Phase B1 — assemble the M3 positive label pool (GTEx-novel arm).

Takes the A1 cross-annotation survivors (junction sides absent from
Ensembl ∪ GENCODE ∪ RefSeq-curated) and turns them into M3 positive labels:

  1. Apply the depth / tissue-breadth minimum filter.
  2. Annotate each site with its intronic splice dinucleotide (GT for donors,
     AG for acceptors, in transcript orientation) via the genome FASTA.
  3. Tag provenance (``gtex_novel``) and write the positive pool.

Tier-1 disease catalogs (TDP-43, SF3B1, U2AF1, SMN2, ...) are folded in by a
separate B1b step with their own provenance tags; this script produces the
GTEx-novel arm only.

Coordinate convention (from ``examples/meta_layer/11_junction_coverage_audit.py::normalize_junctions``):
junctions are stored as 1-based inclusive intron coordinates; ``position`` is
the *exonic* splice-site base — donor = last exonic base, acceptor = first
exonic base. The canonical intronic dinucleotides therefore sit just inside
the intron, on the transcript strand:

  + donor    : genomic [p+1, p+2]            expect GT
  - donor    : revcomp(genomic [p-2, p-1])   expect GT
  + acceptor : genomic [p-2, p-1]            expect AG
  - acceptor : revcomp(genomic [p+1, p+2])   expect AG

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/02_build_positive_pool.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]

SURVIVORS = REPO / "output/meta_layer/m3_label_audit/A1_survivors_final.parquet"
GTEX_JUNCTIONS = REPO / "data/GRCh38/junction_data/junctions_gtex_v8.parquet"
ENSEMBL_SITES = REPO / "data/ensembl/GRCh38/splice_sites_enhanced.tsv"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT_DIR = REPO / "data/mane/GRCh38/m3_labels"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def intronic_dinucleotide(
    fasta: pyfaidx.Fasta, chrom: str, position: int, strand: str, splice_type: str
) -> str:
    """Return the transcript-oriented intronic dinucleotide for a splice site.

    ``position`` is 1-based exonic. Returns uppercase 2-mer, or "" if the
    window runs off the contig.
    """
    if chrom not in fasta:
        return ""
    p = position  # 1-based
    seq = fasta[chrom]
    try:
        if strand == "+":
            if splice_type == "donor":
                d = str(seq[p : p + 2])          # genomic [p+1, p+2]
            else:                                 # acceptor
                d = str(seq[p - 3 : p - 1])      # genomic [p-2, p-1]
        else:  # minus strand → reverse-complement the genomic window
            if splice_type == "donor":
                d = revcomp(str(seq[p - 3 : p - 1]))  # genomic [p-2, p-1]
            else:                                      # acceptor
                d = revcomp(str(seq[p : p + 2]))       # genomic [p+1, p+2]
    except Exception:
        return ""
    return d.upper()


def annotate_dinucleotides(df: pl.DataFrame, fasta: pyfaidx.Fasta) -> pl.DataFrame:
    """Add a ``dinuc`` column and a boolean ``canonical_dinuc`` column."""
    dinucs = [
        intronic_dinucleotide(fasta, c, p, s, t)
        for c, p, s, t in zip(df["chrom"], df["position"], df["strand"], df["splice_type"])
    ]
    out = df.with_columns(pl.Series("dinuc", dinucs))
    canonical = (
        ((pl.col("splice_type") == "donor") & (pl.col("dinuc") == "GT"))
        | ((pl.col("splice_type") == "acceptor") & (pl.col("dinuc") == "AG"))
    )
    return out.with_columns(canonical.alias("canonical_dinuc"))


def self_check(fasta: pyfaidx.Fasta, n: int = 20000) -> float:
    """Validate the dinucleotide offset logic on GTEx-junction-derived sites.

    The survivors are derived with the *junction* coordinate convention
    (intron start/end ± 1; see ``examples/meta_layer/11_junction_coverage_audit.py``). We validate
    the offset against that same convention: derive donor/acceptor positions
    from real GTEx junctions and confirm the canonical GT/AG rate is ~98%.

    (Note: the MANE ``splice_sites_enhanced.tsv`` is NOT a valid reference here
    — its minus-strand positions carry an off-by-one quirk that drags the
    apparent rate to ~0.90. The junction convention has no such asymmetry.)
    """
    strand_lookup = (
        pl.scan_csv(ENSEMBL_SITES, separator="\t")
        .select(["gene_id", "strand"]).unique().collect()
    )
    j = (
        pl.scan_parquet(GTEX_JUNCTIONS)
        .with_columns(
            pl.col("chrom").str.replace("^chr", "").alias("chrom"),
            pl.col("gene_id").str.replace(r"\..*$", "").alias("gene_id"),
        )
        .collect()
        .join(strand_lookup, on="gene_id", how="inner")
    )
    donor = j.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("start") - 1).otherwise(pl.col("end") + 1).alias("position"),
        pl.lit("donor").alias("splice_type"),
    )
    acceptor = j.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("end") + 1).otherwise(pl.col("start") - 1).alias("position"),
        pl.lit("acceptor").alias("splice_type"),
    )
    sides = pl.concat([
        donor.select(["chrom", "position", "strand", "splice_type"]),
        acceptor.select(["chrom", "position", "strand", "splice_type"]),
    ]).unique()
    sides = sides.sample(min(n, sides.height), seed=1)
    checked = annotate_dinucleotides(sides, fasta)
    rate = checked["canonical_dinuc"].mean()
    print(f"[self-check] GTEx junction sides (junction convention) GT/AG rate: "
          f"{rate:.4f} (n={checked.height:,}; expect ~0.97-0.98)")
    if rate is not None and rate < 0.95:
        raise SystemExit(
            f"[self-check FAILED] GT/AG rate {rate:.3f} < 0.95 — dinucleotide "
            "offset logic is wrong; aborting before writing labels."
        )
    return rate


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the M3 GTEx-novel positive pool (Phase B1).")
    ap.add_argument("--min-reads", type=int, default=100, help="Minimum sum_reads (default 100).")
    ap.add_argument("--min-tissues", type=int, default=5, help="Minimum n_tissues (default 5).")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    print(f"Loading survivors: {SURVIVORS}")
    surv = pl.read_parquet(SURVIVORS).with_columns(
        pl.col("chrom").cast(pl.String).str.replace_all(r"^chr", "")
    )
    print(f"  {surv.height:,} A1 survivor sites")

    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)

    if not args.skip_self_check:
        self_check(fasta)

    # Depth / tissue filter
    filt = surv.filter(
        (pl.col("sum_reads") >= args.min_reads) & (pl.col("n_tissues") >= args.min_tissues)
    )
    print(f"  after depth>={args.min_reads} & tissues>={args.min_tissues}: "
          f"{filt.height:,} ({100*filt.height/surv.height:.1f}%)")

    # Dinucleotide annotation
    filt = annotate_dinucleotides(filt, fasta)
    gtag = filt["canonical_dinuc"].sum()
    print(f"  GT/AG canonical dinucleotide: {gtag:,} / {filt.height:,} "
          f"({100*gtag/filt.height:.1f}%)")
    print("  dinucleotide distribution (top 8):")
    print(filt.group_by("dinuc").len().sort("len", descending=True).head(8))

    # Provenance + label schema
    cols = ["chrom", "position", "strand", "splice_type", "gene_id", "source", "label",
            "dinuc", "canonical_dinuc", "sum_reads", "max_reads", "n_tissues", "tissue_breadth"]
    pool_all = filt.with_columns(
        pl.lit("gtex_novel").alias("source"),
        pl.lit(1).cast(pl.Int8).alias("label"),
    ).select(cols)
    pool_canonical = pool_all.filter(pl.col("canonical_dinuc"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Primary positive pool = canonical GT/AG only (purity over quantity; we have headroom).
    out_primary = OUT_DIR / "positives_gtex_novel.parquet"
    pool_canonical.write_parquet(out_primary)
    # Full filtered set incl. non-canonical, for sensitivity analysis.
    out_all = OUT_DIR / "positives_gtex_novel_all_dinuc.parquet"
    pool_all.write_parquet(out_all)

    print(f"\nWrote PRIMARY (canonical GT/AG) {out_primary} ({pool_canonical.height:,} rows)")
    print(f"Wrote FULL (all dinuc)         {out_all} ({pool_all.height:,} rows)")
    print("\nPrimary pool by splice_type:")
    print(pool_canonical.group_by("splice_type").len().sort("splice_type"))


if __name__ == "__main__":
    main()
