"""Phase B1b — ingest ENCODE RBP-knockdown cryptic sites as held-out anchors.

SpliceTools (Flemington lab) provides rMATS JCEC differential-splicing tables
for ~186 ENCODE RBP knockdowns (K562/HepG2). When an RBP is knocked down, the
splice sites whose usage INCREASES are de-repressed/activated by that factor —
mechanism-attributed cryptic sites. We extract the KD-gained alt-3'SS (cryptic
acceptor) and alt-5'SS (cryptic donor) coordinates as held-out disease anchors
(Phase D2), tagged by RBP.

rMATS A3SS/A5SS semantics:
  * Long vs short exon share one boundary and differ at the regulated splice
    site. A3SS differ at the ACCEPTOR; A5SS differ at the DONOR.
  * KD-gained form = sign of IncLevelDifference (test - control): >0 → long
    form gained, <0 → short form gained. The gained form's regulated boundary
    is the de-repressed cryptic site.
  * Boundary → splice-site coordinate (then per-strand offset scan vs GT/AG
    absorbs rMATS 0-based / convention shifts; build auto-detected by the oracle).

Run (after `mamba activate agentic-spliceai`, and after files are in
data/encode_kd_splicetools/1_RBP_kd/):
    python examples/data_preparation/m3/06_ingest_encode_kd_anchors.py --fdr 0.01 --min-dpsi 0.1
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]
KD_DIR = REPO / "data/encode_kd_splicetools/1_RBP_kd"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT = REPO / "data/mane/GRCh38/m3_labels/anchors/anchors_encode_kd.parquet"

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


def gained_boundary(row: dict, event: str) -> int:
    """Return the genomic boundary coordinate of the KD-gained regulated site."""
    gained_long = row["IncLevelDifference"] > 0
    s, st = row["strand"], row["strand"]
    if event == "A3SS":  # acceptor differs
        if st == "+":  # acceptor = exon start
            return row["longExonStart_0base"] if gained_long else row["shortES"]
        else:           # acceptor = exon end
            return row["longExonEnd"] if gained_long else row["shortEE"]
    else:  # A5SS, donor differs
        if st == "+":  # donor = exon end
            return row["longExonEnd"] if gained_long else row["shortEE"]
        else:           # donor = exon start
            return row["longExonStart_0base"] if gained_long else row["shortES"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest ENCODE RBP-KD cryptic anchors.")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--min-dpsi", type=float, default=0.1)
    args = ap.parse_args()

    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)
    raw = []  # (chrom, boundary, strand, splice_type, rbp, dpsi)
    n_files = 0
    for event, stype in (("A3SS", "acceptor"), ("A5SS", "donor")):
        for fp in sorted(glob.glob(str(KD_DIR / f"*_{event}.MATS.JCEC.txt"))):
            rbp = Path(fp).name.split("_test_cntl_")[0]
            try:
                df = pl.read_csv(fp, separator="\t", quote_char='"')
            except Exception:
                continue
            n_files += 1
            df = df.filter(
                (pl.col("FDR") < args.fdr) & (pl.col("IncLevelDifference").abs() >= args.min_dpsi)
            )
            for r in df.iter_rows(named=True):
                b = gained_boundary(r, event)
                raw.append((r["chr"], int(b), r["strand"], stype, rbp, float(r["IncLevelDifference"])))
    print(f"parsed {n_files} files → {len(raw):,} significant KD-gained events "
          f"(FDR<{args.fdr}, |dPSI|>={args.min_dpsi})")
    if not raw:
        raise SystemExit("No events — is the download complete?")

    df = pl.DataFrame(raw, schema=["chrom", "boundary", "strand", "splice_type", "rbp", "dpsi"], orient="row")

    # Per (strand, splice_type) offset scan vs canonical dinucleotide (absorbs
    # rMATS 0-based + build convention; if best rate is low for all, build is wrong).
    best = {}
    for st in ("+", "-"):
        for ty in ("donor", "acceptor"):
            sub = df.filter((pl.col("strand") == st) & (pl.col("splice_type") == ty))
            if sub.height == 0:
                continue
            exp = "GT" if ty == "donor" else "AG"
            bo, br = 0, -1.0
            for off in range(-4, 5):
                rate = sum(dinuc(fasta, c, p + off, st, ty) == exp
                           for c, p in zip(sub["chrom"], sub["boundary"])) / sub.height
                if rate > br:
                    br, bo = rate, off
            best[(st, ty)] = bo
            print(f"  {st} {ty:8s}: best offset {bo:+d} → canonical rate {br:.4f} (n={sub.height})")

    rows = []
    for c, b, st, ty, rbp, dpsi in raw:
        pos = b + best.get((st, ty), 0)
        d = dinuc(fasta, c, pos, st, ty)
        exp = "GT" if ty == "donor" else "AG"
        rows.append((c.replace("chr", ""), pos, st, ty, d, d == exp, rbp, dpsi))

    out = pl.DataFrame(
        rows,
        schema=["chrom", "position", "strand", "splice_type", "dinuc", "canonical_dinuc", "rbp", "dpsi"],
        orient="row",
    )
    overall = out["canonical_dinuc"].mean()
    print(f"overall canonical rate: {overall:.4f}")
    if overall < 0.85:
        raise SystemExit(f"[FAILED] canonical rate {overall:.3f} < 0.85 — build/convention issue.")

    # Keep canonical; dedup to unique site, keep max |dPSI| and a representative RBP.
    out = out.filter(pl.col("canonical_dinuc"))
    out = (
        out.with_columns(pl.col("dpsi").abs().alias("abs_dpsi"))
        .sort("abs_dpsi", descending=True)
        .group_by(["chrom", "position", "strand", "splice_type"])
        .agg(
            pl.col("rbp").unique().str.concat("+").alias("rbps"),
            pl.col("abs_dpsi").max().alias("max_abs_dpsi"),
            pl.col("dinuc").first(),
        )
        .with_columns(
            pl.lit("encode_kd_cryptic").alias("source"),
            pl.lit(1).cast(pl.Int8).alias("label"),
            pl.lit("RBP_knockdown_derepressed").alias("mechanism"),
            pl.col("max_abs_dpsi").alias("support"),
            pl.lit("rmats_abs_dpsi").alias("support_kind"),
            pl.lit("splicetools_native").alias("build_origin"),
            pl.lit(True).alias("canonical_dinuc"),
        )
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT)
    print(f"\nWrote {OUT} ({out.height:,} unique canonical KD-gained cryptic sites)")
    print(out.group_by("splice_type").len().sort("splice_type"))


if __name__ == "__main__":
    main()
