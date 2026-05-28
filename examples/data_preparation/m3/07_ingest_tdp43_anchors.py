"""Phase B1b — ingest TDP-43 (ALS) cryptic splice sites as held-out anchors.

The STMN2 / UNC13A cryptic splice sites activated by TDP-43 nuclear loss in
ALS/FTD — the cleanest, nanopore-confirmed disease cryptic catalog. Tiny (3
sites) but gold-standard; used as held-out generalization sentinels (Phase D2),
NOT training positives.

Source (already GRCh38, splice-site-aligned, curated):
  output/meta_layer/ui_cache/als_cryptic/als_cryptic_splice_sites_grch38.tsv
  (single source of truth: examples/UI_integration/als_cryptic_sites.py)

Output: data/mane/GRCh38/m3_labels/anchors/anchors_tdp43.parquet

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/07_ingest_tdp43_anchors.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "output/meta_layer/ui_cache/als_cryptic/als_cryptic_splice_sites_grch38.tsv"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT = REPO / "data/mane/GRCh38/m3_labels/anchors/anchors_tdp43.parquet"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def dinuc(fasta, chrom, pos, strand, splice_type) -> str:
    c = chrom.replace("chr", "")
    if c not in fasta:
        return ""
    s = fasta[c]
    if strand == "+":
        return str(s[pos : pos + 2]) if splice_type == "donor" else str(s[pos - 3 : pos - 1])
    return revcomp(str(s[pos - 3 : pos - 1])) if splice_type == "donor" else revcomp(str(s[pos : pos + 2]))


def main() -> None:
    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)
    src = pl.read_csv(SRC, separator="\t")

    rows = []
    for r in src.iter_rows(named=True):
        c = r["chrom"].replace("chr", "")
        p, strand, stype = int(r["splice_site_pos_1based"]), r["strand"], r["kind"]
        d = dinuc(fasta, c, p, strand, stype)
        exp = "GT" if stype == "donor" else "AG"
        rows.append((c, p, strand, stype, d, d == exp, r["gene"]))

    anchor = pl.DataFrame(
        rows,
        schema=["chrom", "position", "strand", "splice_type", "dinuc", "canonical_dinuc", "gene_name"],
        orient="row",
    ).with_columns(
        pl.lit("tdp43_cryptic").alias("source"),
        pl.lit(1).cast(pl.Int8).alias("label"),
        pl.lit("ALS_TDP43_loss").alias("mechanism"),
        pl.lit(None, dtype=pl.Float64).alias("support"),
        pl.lit("curated_nanopore").alias("support_kind"),
        pl.lit("GRCh38_native").alias("build_origin"),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_parquet(OUT)
    print(f"Wrote {OUT} ({anchor.height} sites)")
    print(anchor.select(["gene_name", "chrom", "position", "strand", "splice_type", "dinuc", "canonical_dinuc"]))
    rate = anchor["canonical_dinuc"].mean()
    print(f"canonical GT/AG rate: {rate:.3f} (expect 1.0)")


if __name__ == "__main__":
    main()
