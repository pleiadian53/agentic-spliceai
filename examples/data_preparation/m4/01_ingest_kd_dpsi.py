"""Phase 1 — build the M4 perturbation-paired ΔPSI label corpus.

M4 is the conditional arm of the meta layer: given a regulator perturbation
(knock down RBP R), predict the resulting splicing change (signed ΔPSI at splice
sites). Its missing training ingredient is perturbation-paired labels — WT vs
knockdown ΔPSI across many regulators. SpliceTools (Flemington lab) provides
rMATS JCEC differential-splicing tables for ~185 ENCODE RBP knockdowns
(K562/HepG2), which supply exactly that signal.

rMATS A3SS/A5SS semantics:
  * The long and short isoforms share one splice-site boundary and differ at the
    regulated one; their usage is complementary. A3SS differ at the ACCEPTOR;
    A5SS differ at the DONOR.
  * IncLevel1 / IncLevel2 = per-replicate PSI of the long (inclusion) isoform in
    the KD (SAMPLE_1) vs control (SAMPLE_2); IncLevelDifference = PSI_KD − PSI_ctrl.

Label representation — Option B (per-regulated-site, signed). Each event emits
TWO rows per RBP: the long-form alternative site with dpsi = +IncLevelDifference
and the short-form alternative site with dpsi = −IncLevelDifference. This matches
M4's per-splice-site output grain (like M1/M2/M3), makes "does KD raise or lower
usage of THIS donor/acceptor" a direct label, and lets every emitted site be
GT/AG-validated by strand. Regulator identity (rbp) is the conditioning variable
and is NEVER collapsed; both ΔPSI signs (repression and de-repression) are kept;
no FDR/|dPSI| prefilter — the full distribution is retained and thresholded
downstream. Rows carry event_uid / form / paired_position so a consumer can split
leakage-safely by event and recover the pair.

Coordinate handling reuses the M3 anchor ingester's proven machinery
(examples/data_preparation/m3/06_ingest_encode_kd_anchors.py): a per-(strand,
splice_type) GT/AG offset scan absorbs the rMATS 0-based convention and validates
the mapping; overall canonical rate < 0.85 hard-fails.

Run (after `mamba activate agentic-spliceai`, files in
data/encode_kd_splicetools/1_RBP_kd/):
    python examples/data_preparation/m4/01_ingest_kd_dpsi.py
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
OUT = REPO / "data/mane/GRCh38/m4_labels/kd_dpsi.parquet"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# rMATS A3SS/A5SS boundary geometry: for each (event, strand), which exon-boundary
# column carries the long-form vs short-form regulated splice site. Verified against
# the data (the complementary boundary is shared with frac 1.00 within a group).
_BOUNDARY = {
    # event: (strand: (long_col, short_col))
    "A3SS": {"+": ("longExonStart_0base", "shortES"), "-": ("longExonEnd", "shortEE")},
    "A5SS": {"+": ("longExonEnd", "shortEE"), "-": ("longExonStart_0base", "shortES")},
}


def revcomp(s: str) -> str:
    """Reverse-complement a nucleotide string."""
    return s.translate(_COMP)[::-1]


def dinuc(fasta: pyfaidx.Fasta, chrom: str, pos: int, strand: str, splice_type: str) -> str:
    """Canonical dinucleotide at a splice site (GT donor / AG acceptor), strand-aware.

    Mirrors the M3 ingester: `pos` is the exonic-base position; the 2-mer is read
    intronic-side and reverse-complemented on the minus strand.
    """
    c = chrom.replace("chr", "")
    if c not in fasta:
        return ""
    s = fasta[c]
    if strand == "+":
        return str(s[pos : pos + 2]) if splice_type == "donor" else str(s[pos - 3 : pos - 1])
    return (
        revcomp(str(s[pos - 3 : pos - 1]))
        if splice_type == "donor"
        else revcomp(str(s[pos : pos + 2]))
    )


def _mean_psi(col: str) -> pl.Expr:
    """Mean of a comma-separated replicate-PSI string, skipping `NA` tokens.

    Cast to Utf8 first: a knockdown with a single replicate has no comma, so polars
    infers a Float64 column that `str.split` cannot consume (e.g. EXOSC9's control).
    """
    return (
        pl.col(col)
        .cast(pl.Utf8)
        .str.split(",")
        .list.eval(pl.element().cast(pl.Float64, strict=False))
        .list.mean()
    )


def parse_file(fp: str, event: str, splice_type: str) -> pl.DataFrame:
    """Parse one rMATS JCEC table into signed per-regulated-site rows (both forms).

    Args:
        fp: Path to a `*_{event}.MATS.JCEC.txt` table.
        event: "A3SS" or "A5SS".
        splice_type: "acceptor" (A3SS) or "donor" (A5SS).

    Returns:
        A frame with one row per (event, form) carrying the raw regulated-site
        `boundary` (offset applied later), signed `dpsi`, and event metadata.
    """
    rbp = Path(fp).name.split("_test_cntl_")[0]
    df = pl.read_csv(fp, separator="\t", quote_char='"')

    plus_long, plus_short = _BOUNDARY[event]["+"]
    minus_long, minus_short = _BOUNDARY[event]["-"]
    long_b = pl.when(pl.col("strand") == "+").then(pl.col(plus_long)).otherwise(pl.col(minus_long))
    short_b = (
        pl.when(pl.col("strand") == "+").then(pl.col(plus_short)).otherwise(pl.col(minus_short))
    )

    base = df.select(
        pl.col("chr").str.replace(r"^chr", "").alias("chrom"),
        pl.col("strand"),
        pl.lit(splice_type).alias("splice_type"),
        pl.lit(event).alias("event_type"),
        pl.lit(rbp).alias("rbp"),
        pl.col("IncLevelDifference").cast(pl.Float64, strict=False).alias("inclvl_diff"),
        _mean_psi("IncLevel1").alias("psi_kd_long"),
        _mean_psi("IncLevel2").alias("psi_ctrl_long"),
        pl.col("FDR").cast(pl.Float64, strict=False).alias("fdr"),
        pl.col("PValue").cast(pl.Float64, strict=False).alias("pvalue"),
        pl.col("GeneID").cast(pl.Utf8).str.replace_all('"', "").alias("gene_id"),
        pl.col("geneSymbol").cast(pl.Utf8).str.replace_all('"', "").alias("gene_symbol"),
        pl.col("ID").cast(pl.Int64).alias("event_id"),
        long_b.cast(pl.Int64).alias("long_boundary"),
        short_b.cast(pl.Int64).alias("short_boundary"),
    )

    long_df = base.with_columns(
        pl.lit("long").alias("form"),
        pl.col("long_boundary").alias("boundary"),
        pl.col("inclvl_diff").alias("dpsi"),
        pl.col("psi_kd_long").alias("psi_kd"),
        pl.col("psi_ctrl_long").alias("psi_ctrl"),
    )
    short_df = base.with_columns(
        pl.lit("short").alias("form"),
        pl.col("short_boundary").alias("boundary"),
        (-pl.col("inclvl_diff")).alias("dpsi"),
        (1.0 - pl.col("psi_kd_long")).alias("psi_kd"),
        (1.0 - pl.col("psi_ctrl_long")).alias("psi_ctrl"),
    )
    cols = [
        "chrom",
        "strand",
        "splice_type",
        "event_type",
        "form",
        "rbp",
        "boundary",
        "dpsi",
        "inclvl_diff",
        "psi_kd",
        "psi_ctrl",
        "fdr",
        "pvalue",
        "gene_id",
        "gene_symbol",
        "event_id",
    ]
    return pl.concat([long_df.select(cols), short_df.select(cols)])


def find_offsets(
    df: pl.DataFrame, fasta: pyfaidx.Fasta, sample_n: int
) -> dict[tuple[str, str], int]:
    """Determine the boundary→exonic-position offset per (strand, splice_type).

    The offset is a fixed coordinate convention, so a per-group sample suffices to
    find the value that maximizes the canonical GT/AG rate.
    """
    best: dict[tuple[str, str], int] = {}
    for st in ("+", "-"):
        for ty in ("donor", "acceptor"):
            sub = df.filter((pl.col("strand") == st) & (pl.col("splice_type") == ty))
            if sub.height == 0:
                continue
            samp = sub if sub.height <= sample_n else sub.sample(sample_n, seed=0)
            exp = "GT" if ty == "donor" else "AG"
            chroms = samp["chrom"].to_list()
            bnds = samp["boundary"].to_list()
            bo, br = 0, -1.0
            for off in range(-4, 5):
                rate = sum(
                    dinuc(fasta, c, b + off, st, ty) == exp
                    for c, b in zip(chroms, bnds, strict=True)
                ) / len(chroms)
                if rate > br:
                    br, bo = rate, off
            best[(st, ty)] = bo
            print(
                f"  {st} {ty:8s}: best offset {bo:+d} → canonical rate {br:.4f} "
                f"(scan n={len(chroms)}, group n={sub.height:,})"
            )
    return best


def apply_offset_and_validate(
    df: pl.DataFrame, fasta: pyfaidx.Fasta, best: dict[tuple[str, str], int]
) -> pl.DataFrame:
    """Add `position` (offset-corrected), `dinuc`, and `canonical_dinuc` to every row.

    Dinucleotides are read chromosome-by-chromosome (each contig loaded once) so the
    full corpus is validated without per-position file seeks.
    """
    off_expr = pl.lit(0)
    for (st, ty), o in best.items():
        off_expr = (
            pl.when((pl.col("strand") == st) & (pl.col("splice_type") == ty))
            .then(pl.lit(o))
            .otherwise(off_expr)
        )
    df = df.with_columns((pl.col("boundary") + off_expr).cast(pl.Int64).alias("position"))
    df = df.with_row_index("_ri")

    out = [""] * df.height
    for chrom in df["chrom"].unique().to_list():
        c = chrom.replace("chr", "")
        if c not in fasta:
            continue
        seq = fasta[c][:].seq
        sub = df.filter(pl.col("chrom") == chrom).select("_ri", "position", "strand", "splice_type")
        for ri, pos, st, ty in zip(
            sub["_ri"], sub["position"], sub["strand"], sub["splice_type"], strict=True
        ):
            if st == "+":
                d = seq[pos : pos + 2] if ty == "donor" else seq[pos - 3 : pos - 1]
            else:
                d = (
                    revcomp(seq[pos - 3 : pos - 1])
                    if ty == "donor"
                    else revcomp(seq[pos : pos + 2])
                )
            out[ri] = d
        del seq

    df = df.drop("_ri").with_columns(pl.Series("dinuc", out))
    df = df.with_columns(
        (
            ((pl.col("splice_type") == "donor") & (pl.col("dinuc") == "GT"))
            | ((pl.col("splice_type") == "acceptor") & (pl.col("dinuc") == "AG"))
        ).alias("canonical_dinuc")
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the M4 perturbation-paired ΔPSI label corpus.")
    ap.add_argument(
        "--limit-files",
        type=int,
        default=0,
        help="debug: parse only the first N files per event type (0 = all)",
    )
    ap.add_argument(
        "--sample-n",
        type=int,
        default=8000,
        help="rows sampled per (strand, splice_type) group for the offset scan",
    )
    args = ap.parse_args()

    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)

    frames: list[pl.DataFrame] = []
    n_files = 0
    for event, stype in (("A3SS", "acceptor"), ("A5SS", "donor")):
        files = sorted(glob.glob(str(KD_DIR / f"*_{event}.MATS.JCEC.txt")))
        if args.limit_files:
            files = files[: args.limit_files]
        for fp in files:
            try:
                frames.append(parse_file(fp, event, stype))
                n_files += 1
            except Exception as e:  # noqa: BLE001 — skip an unreadable table, report it
                print(f"  [skip] {Path(fp).name}: {e}")
    if not frames:
        raise SystemExit(f"No rMATS tables parsed under {KD_DIR} — is the download present?")

    df = pl.concat(frames)
    print(
        f"parsed {n_files} files → {df.height:,} site-rows ({df.height // 2:,} events, "
        f"{df['rbp'].n_unique()} RBPs)"
    )

    print("offset scan (boundary → exonic splice-site position):")
    best = find_offsets(df, fasta, args.sample_n)

    df = apply_offset_and_validate(df, fasta, best)
    overall = df["canonical_dinuc"].mean()
    print(f"overall canonical rate: {overall:.4f}")
    if overall < 0.85:
        raise SystemExit(f"[FAILED] canonical rate {overall:.3f} < 0.85 — build/convention issue.")

    df = df.with_columns(
        (pl.col("rbp") + ":" + pl.col("event_type") + ":" + pl.col("event_id").cast(pl.Utf8)).alias(
            "event_uid"
        )
    )
    long_pos = df.filter(pl.col("form") == "long").select(
        "event_uid", pl.col("position").alias("_long_pos")
    )
    short_pos = df.filter(pl.col("form") == "short").select(
        "event_uid", pl.col("position").alias("_short_pos")
    )
    df = (
        df.join(long_pos, on="event_uid", how="left")
        .join(short_pos, on="event_uid", how="left")
        .with_columns(
            pl.when(pl.col("form") == "long")
            .then(pl.col("_short_pos"))
            .otherwise(pl.col("_long_pos"))
            .alias("paired_position")
        )
        .drop("_long_pos", "_short_pos", "boundary")
        .with_columns(
            pl.lit("encode_kd_splicetools").alias("source"),
            pl.lit("splicetools_native").alias("build_origin"),
        )
    )

    final_cols = [
        "chrom",
        "position",
        "strand",
        "splice_type",
        "event_type",
        "form",
        "dinuc",
        "canonical_dinuc",
        "rbp",
        "dpsi",
        "inclvl_diff",
        "psi_kd",
        "psi_ctrl",
        "fdr",
        "pvalue",
        "gene_id",
        "gene_symbol",
        "event_id",
        "event_uid",
        "paired_position",
        "source",
        "build_origin",
    ]
    df = df.select(final_cols)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT)
    print(
        f"\nWrote {OUT} ({df.height:,} rows, {df['rbp'].n_unique()} RBPs, "
        f"{df['event_uid'].n_unique():,} events)"
    )
    print(df.group_by(["event_type", "form"]).len().sort(["event_type", "form"]))
    print("canonical rate by strand:")
    print(
        df.group_by("strand")
        .agg(pl.col("canonical_dinuc").mean().alias("canonical_rate"), pl.len().alias("n"))
        .sort("strand")
    )


if __name__ == "__main__":
    main()
