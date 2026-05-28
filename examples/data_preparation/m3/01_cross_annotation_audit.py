"""Phase A1 — cross-annotation audit of the GTEx novel-junction pool.

Takes the novel junction sides surfaced by the junction-coverage audit (step 0,
`examples/meta_layer/11_junction_coverage_audit.py` → q3_novel_junction_sides)
and re-checks them against the STRAND-CORRECT annotation union (GENCODE v47 ∪
RefSeq-curated ∪ Ensembl GRCh38). Sites absent from all of them are the
genuinely-novel GTEx candidate pool that feeds step 02.

History: the original inline A1 used the minus-strand-buggy annotation TSVs and
reported ~65K "novel" survivors; ~52K of those were minus-strand ANNOTATED
sites that leaked through the bug (see memory `feedback-splice-site-coordinate-
validation`). With the corrected annotation, the genuinely-novel GTEx pool is
much smaller (~hundreds) — SpliceVault (step 03) carries the real bulk.

Outputs (output/meta_layer/m3_label_audit/):
  A1_survivors_final.parquet   GTEx novel sides absent from all annotations
  A1_cross_annotation.md       (the prose findings; maintained separately)

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/01_cross_annotation_audit.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[3]
NOVEL = REPO / "output/meta_layer/junction_coverage_audit_fullgenome/q3_novel_junction_sides.parquet"
OUT_DIR = REPO / "output/meta_layer/m3_label_audit"
JOIN = ["chrom", "position", "strand", "splice_type"]


def _strip(c):
    return pl.col(c).cast(pl.String).str.replace_all(r"^chr", "")


def annotation_union() -> pl.DataFrame:
    """GENCODE v47 ∪ RefSeq-curated (strand-corrected TSVs).

    GENCODE v47 is the same human annotation as Ensembl (Ensembl/Havana merge),
    so it subsumes the Ensembl set — we use GENCODE for that arm. Identical to
    the union used in steps 04/08/09 so the pipeline is internally consistent.
    """
    def keys(path, prefix=None):
        lf = pl.scan_csv(path, separator="\t").with_columns(_strip("chrom").alias("chrom"))
        if prefix:
            lf = lf.filter(pl.col("transcript_id").str.contains(prefix))
        return lf.select(JOIN).unique().collect()
    gen = keys(REPO / "data/gencode/GRCh38/splice_sites_enhanced.tsv")
    ref = keys(REPO / "data/refseq/GRCh38/splice_sites_enhanced.tsv", r"NM_|NR_")
    return pl.concat([gen, ref]).unique()


def main() -> None:
    novel = pl.read_parquet(NOVEL).with_columns(_strip("chrom").alias("chrom"))
    print(f"GTEx novel junction sides (from junction audit Q3): {novel.height:,}")

    ann = annotation_union()
    print(f"annotation union (GENCODE ∪ RefSeq-curated ∪ Ensembl): {ann.height:,} sites")

    survivors = novel.join(ann, on=JOIN, how="anti")
    removed = novel.height - survivors.height
    print(f"absent from all annotations (genuinely novel): {survivors.height:,} "
          f"(removed {removed:,} now-annotated, {100*removed/novel.height:.1f}%)")
    print(survivors.group_by("splice_type").len().sort("splice_type"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "A1_survivors_final.parquet"
    survivors.write_parquet(out)
    print(f"\nWrote {out} ({survivors.height:,} rows) → consumed by step 02_build_positive_pool")


if __name__ == "__main__":
    main()
