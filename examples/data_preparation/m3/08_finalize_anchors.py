"""Phase B1b-finalize — combine M3 held-out disease anchors + decontaminate pool.

1. Harmonize the per-source anchor parquets (TDP-43, SF3B1, ENCODE-KD) into one
   `disease_anchors.parquet` with a unified schema and union of mechanisms when
   a site appears in multiple catalogs.
2. Anti-join the anchor coordinates out of `positives_pooled.parquet` so no
   held-out eval site can leak into M3 training.
3. Report how many anchors are "novel" (absent from Ensembl ∪ GENCODE ∪
   RefSeq-curated) — M3's actual target — vs already annotated.

Run:
    python examples/data_preparation/m3/08_finalize_anchors.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "data/mane/GRCh38/m3_labels"
ANCH = LABELS / "anchors"
JOIN = ["chrom", "position", "strand", "splice_type"]
COMMON = JOIN + ["source", "mechanism", "support", "support_kind", "build_origin", "dinuc", "detail"]


def _strip(c):
    return pl.col(c).cast(pl.String).str.replace_all(r"^chr", "")


def annotation_union() -> pl.DataFrame:
    """GENCODE v47 ∪ RefSeq-curated (strand-correct TSVs; GENCODE subsumes Ensembl)."""
    def keys(path, transcript_prefix=None):
        lf = pl.scan_csv(path, separator="\t").with_columns(_strip("chrom").alias("chrom"))
        if transcript_prefix:
            lf = lf.filter(pl.col("transcript_id").str.contains(transcript_prefix))
        return lf.select(JOIN).unique().collect()
    gen = keys(REPO / "data/gencode/GRCh38/splice_sites_enhanced.tsv")
    ref = keys(REPO / "data/refseq/GRCh38/splice_sites_enhanced.tsv", transcript_prefix=r"NM_|NR_")
    return pl.concat([gen, ref]).unique()


def load_anchor(name: str, detail_col: str) -> pl.DataFrame:
    df = pl.read_parquet(ANCH / name).with_columns(_strip("chrom").alias("chrom"))
    if detail_col in df.columns:
        df = df.with_columns(pl.col(detail_col).cast(pl.String).alias("detail"))
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.String).alias("detail"))
    if "support" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("support"))
    return df.select(COMMON)


def main() -> None:
    parts = [
        load_anchor("anchors_tdp43.parquet", "gene_name"),
        load_anchor("anchors_sf3b1.parquet", "gene_name"),
        load_anchor("anchors_encode_kd.parquet", "rbps"),
    ]
    anchors = pl.concat(parts)
    print("Per-source anchor counts:")
    print(anchors.group_by("source").len().sort("len", descending=True))

    # Dedup across catalogs: a site in multiple catalogs → union of sources/mechanisms.
    anchors = (
        anchors.group_by(JOIN)
        .agg(
            pl.col("source").unique().sort().str.join("+").alias("source"),
            pl.col("mechanism").unique().sort().str.join("+").alias("mechanism"),
            pl.col("support").max(),
            pl.col("support_kind").first(),
            pl.col("build_origin").first(),
            pl.col("dinuc").first(),
            pl.col("detail").drop_nulls().unique().str.join("+").alias("detail"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    )
    print(f"\nCombined unique anchor sites: {anchors.height:,}")

    # Novelty: how many anchors are absent from all major annotations (M3 target)?
    ann = annotation_union()
    novel = anchors.join(ann, on=JOIN, how="anti")
    print(f"  novel (not in Ensembl∪GENCODE∪RefSeq): {novel.height:,} "
          f"({100*novel.height/anchors.height:.1f}%)")
    print(f"  already annotated:                      {anchors.height - novel.height:,}")
    anchors = anchors.with_columns(
        anchors.join(ann, on=JOIN, how="anti").select(JOIN)
        .with_columns(pl.lit(True).alias("is_novel"))
        .pipe(lambda d: anchors.join(d, on=JOIN, how="left"))["is_novel"].fill_null(False)
    )

    out_anchor = LABELS / "disease_anchors.parquet"
    anchors.write_parquet(out_anchor)
    print(f"Wrote {out_anchor} ({anchors.height:,} rows)")

    # Decontaminate the training pool.
    pool = pl.read_parquet(LABELS / "positives_pooled.parquet").with_columns(_strip("chrom").alias("chrom"))
    before = pool.height
    pool_clean = pool.join(anchors.select(JOIN), on=JOIN, how="anti")
    removed = before - pool_clean.height
    pool_clean.write_parquet(LABELS / "positives_pooled.parquet")
    print(f"\nTraining pool decontamination: removed {removed:,} anchor sites "
          f"→ {pool_clean.height:,} training positives")


if __name__ == "__main__":
    main()
