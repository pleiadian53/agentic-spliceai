"""Phase B1b-merge — assemble the M3 training positive pool.

Merges the bulk novel-site sources into one annotation-clean training pool:

  positives_pooled.parquet = (GTEx-novel) ∪ (SpliceVault-novel)

Both arms are required to be ABSENT from every major annotation (Ensembl ∪
GENCODE ∪ RefSeq-curated) so that:
  (a) the "novel" semantics of M3 hold, and
  (b) no coordinate can be both a positive here and a decoy *negative* in B2.

Disease catalogs (TDP-43, SF3B1, ENCODE-KD) are NOT merged here — they are
reserved as held-out eval anchors (Phase D2) by a separate builder, and are
anti-joined out of this pool to prevent train/eval contamination.

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/04_merge_positives.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "data/mane/GRCh38/m3_labels"
JOIN = ["chrom", "position", "strand", "splice_type"]


def _strip(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.String).str.replace_all(r"^chr", "")


def annotation_union() -> pl.DataFrame:
    """Distinct annotated splice sites: GENCODE v47 ∪ RefSeq-curated.

    GENCODE v47 is the same human annotation as Ensembl (Ensembl/Havana merge),
    so it subsumes the Ensembl set; we use GENCODE alone for that arm. Both TSVs
    are regenerated from GTFs with the strand-correct splice-site extractor (the
    old `splice_sites_enhanced.tsv` had a minus-strand position bug — see
    genomic_extraction.extract_splice_sites_from_exons).
    """
    def keys(path: Path, transcript_prefix: str | None = None) -> pl.DataFrame:
        lf = pl.scan_csv(path, separator="\t").with_columns(_strip("chrom").alias("chrom"))
        if transcript_prefix:
            lf = lf.filter(pl.col("transcript_id").str.contains(transcript_prefix))
        return lf.select(JOIN).unique().collect()
    gen = keys(REPO / "data/gencode/GRCh38/splice_sites_enhanced.tsv")
    ref = keys(REPO / "data/refseq/GRCh38/splice_sites_enhanced.tsv",
               transcript_prefix=r"NM_|NR_")  # curated RefSeq only
    return pl.concat([gen, ref]).unique()


def main() -> None:
    ann = annotation_union()
    print(f"annotation union: {ann.height:,} distinct sites")

    # --- GTEx-novel arm (already annotation-filtered + canonical GT/AG) ---
    gtex = (
        pl.read_parquet(LABELS / "positives_gtex_novel.parquet")
        .with_columns(_strip("chrom").alias("chrom"))
        .select(JOIN + ["sum_reads", "n_tissues", "dinuc"])
        .rename({"sum_reads": "gtex_sum_reads", "n_tissues": "gtex_n_tissues"})
    )
    print(f"GTEx-novel: {gtex.height:,}")

    # --- SpliceVault arm (filter to novel + canonical) ---
    sv = (
        pl.read_parquet(LABELS / "positives_splicevault.parquet")
        .with_columns(_strip("chrom").alias("chrom"))
        .filter(pl.col("canonical_dinuc"))
        .join(ann, on=JOIN, how="anti")  # keep only novel
        .select(JOIN + ["support", "dinuc"])
        .rename({"support": "sv_freq_pct"})
    )
    print(f"SpliceVault-novel: {sv.height:,}")

    # --- Outer-join the two arms on the splice-site coordinate ---
    merged = gtex.join(sv, on=JOIN, how="full", coalesce=True)
    # dinuc: coalesce (both arms agree on canonical sites)
    merged = merged.with_columns(
        pl.coalesce(["dinuc", "dinuc_right"]).alias("dinuc"),
    ).drop("dinuc_right")

    merged = merged.with_columns(
        pl.col("gtex_sum_reads").is_not_null().alias("in_gtex_novel"),
        pl.col("sv_freq_pct").is_not_null().alias("in_splicevault"),
    )
    merged = merged.with_columns(
        pl.when(pl.col("in_gtex_novel") & pl.col("in_splicevault")).then(pl.lit("gtex_novel+splicevault"))
        .when(pl.col("in_gtex_novel")).then(pl.lit("gtex_novel"))
        .otherwise(pl.lit("splicevault")).alias("sources"),
        pl.lit(1).cast(pl.Int8).alias("label"),
    )

    # Safety: re-apply annotation anti-join to the whole pool (GTEx arm already
    # clean, but this guarantees the invariant for the merged set).
    before = merged.height
    merged = merged.join(ann, on=JOIN, how="anti")
    print(f"pool after annotation anti-join: {merged.height:,} (removed {before - merged.height:,})")

    merged = merged.select(
        JOIN + ["label", "sources", "in_gtex_novel", "in_splicevault",
                "gtex_sum_reads", "gtex_n_tissues", "sv_freq_pct", "dinuc"]
    ).sort(["chrom", "position"])

    out = LABELS / "positives_pooled.parquet"
    merged.write_parquet(out)
    print(f"\nWrote {out} ({merged.height:,} rows)")
    print("\nBy source provenance:")
    print(merged.group_by("sources").len().sort("len", descending=True))
    print("\nBy splice_type:")
    print(merged.group_by("splice_type").len().sort("splice_type"))
    print(f"\nIn both sources: {merged.filter(pl.col('in_gtex_novel') & pl.col('in_splicevault')).height:,}")


if __name__ == "__main__":
    main()
