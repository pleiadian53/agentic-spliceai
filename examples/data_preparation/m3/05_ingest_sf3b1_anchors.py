"""Phase B1b — ingest DeBoever 2015 SF3B1 cryptic 3'SS as held-out anchors.

SF3B1 hotspot mutations (K700E etc.) activate cryptic 3' splice sites ~10-30 nt
upstream of the canonical 3'SS in MDS/CLL/uveal melanoma/breast cancer. DeBoever
et al. 2015 (PLOS Comput Biol) catalogued 619 such cryptic 3'SS. We use them as
held-out disease anchors (Phase D2), NOT training positives.

Source: Figshare fileset 10.6084/m9.figshare.1120663, `file_s03` splicing table
(hg19). The `3'SS` column is the cryptic acceptor. We liftover hg19→GRCh38 and
auto-detect the column's coordinate convention by finding the single global
offset that maximises the AG rate in the project's acceptor convention.

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/05_ingest_sf3b1_anchors.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pyfaidx
from pyliftover import LiftOver

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "data/sf3b1_deboever/file_s03_splicing_table.tsv"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT = REPO / "data/mane/GRCh38/m3_labels/anchors/anchors_sf3b1.parquet"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def acc_dinuc(fasta, chrom: str, pos: int, strand: str) -> str:
    """Acceptor dinucleotide (expect AG) in the project exonic-base convention."""
    c = chrom.replace("chr", "")
    if c not in fasta:
        return ""
    s = fasta[c]
    return str(s[pos - 3 : pos - 1]) if strand == "+" else revcomp(str(s[pos : pos + 2]))


def main() -> None:
    df = pl.read_csv(SRC, separator="\t").select(["chrom", "strand", "3'SS", "gene"])
    print(f"SF3B1 cryptic 3'SS (hg19): {df.height}")

    lo = LiftOver("hg19", "hg38")
    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)

    lifted = []
    unmapped = 0
    for chrom, strand, ss3, gene in df.iter_rows():
        res = lo.convert_coordinate(chrom, int(ss3))  # 0-based input? pyliftover is 0-based
        if not res:
            unmapped += 1
            continue
        new_chrom, new_pos, _, _ = res[0]
        lifted.append((new_chrom.replace("chr", ""), new_pos, strand, gene))
    print(f"lifted to GRCh38: {len(lifted)} (unmapped {unmapped})")

    # Auto-detect the convention offset PER STRAND (the 0/1-based + exon-base
    # convention shift is strand-dependent: empirically +1 for '+', -1 for '-').
    best_off = {}
    for strand in ("+", "-"):
        sub = [(c, p) for c, p, st, _ in lifted if st == strand]
        bo, br = 0, -1.0
        for off in range(-3, 4):
            rate = sum(acc_dinuc(fasta, c, p + off, strand) == "AG" for c, p in sub) / len(sub)
            if rate > br:
                br, bo = rate, off
        best_off[strand] = bo
        print(f"strand {strand}: best offset {bo:+d} → AG rate {br:.4f} (n={len(sub)})")

    rows = []
    for c, p, st, gene in lifted:
        pos = p + best_off[st]
        d = acc_dinuc(fasta, c, pos, st)
        rows.append((c, pos, st, "acceptor", d, d == "AG", gene))

    overall = sum(r[5] for r in rows) / len(rows)
    print(f"overall AG rate (strand-specific offsets): {overall:.4f}")
    if overall < 0.90:
        raise SystemExit(f"[FAILED] AG rate {overall:.3f} < 0.90 — liftover/convention issue.")

    anchor = pl.DataFrame(
        rows,
        schema=["chrom", "position", "strand", "splice_type", "dinuc", "canonical_dinuc", "gene_name"],
        orient="row",
    ).filter(pl.col("canonical_dinuc")).with_columns(
        pl.lit("sf3b1_cryptic").alias("source"),
        pl.lit(1).cast(pl.Int8).alias("label"),
        pl.lit("SF3B1_mutant_cryptic_3ss").alias("mechanism"),
        pl.lit(None, dtype=pl.Float64).alias("support"),
        pl.lit("deboever2015_curated").alias("support_kind"),
        pl.lit("lifted_hg19").alias("build_origin"),
    ).unique(subset=["chrom", "position", "strand", "splice_type"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_parquet(OUT)
    print(f"\nWrote {OUT} ({anchor.height} unique canonical cryptic acceptors)")


if __name__ == "__main__":
    main()
