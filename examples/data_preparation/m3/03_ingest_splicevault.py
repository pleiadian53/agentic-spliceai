"""Phase B1b — ingest SpliceVault 300K cryptic splice sites into M3 positives.

SpliceVault (Dawes 2023, Nat Genet) records, for each annotated splice site,
the empirically observed cryptic-donor (CD) / cryptic-acceptor (CA) events used
when that site is disrupted, across 335K RNA-seq samples. We turn the CD/CA
events into novel-splice-site positive labels.

Source file: the Ensembl VEP-plugin bundle `SpliceVault_data_GRCh38.tsv.gz`
(GRCh38-native). It is variant-keyed (one row per splice-disrupting variant),
so the splice-site info is massively redundant — we stream-dedup on
`(transcript_id, splicevault_site_loc)` to stay memory-bounded.

Coordinate handling (validated empirically with the GT/AG oracle):
  * `splicevault_site_loc` is the INTRON-ADJACENT base (G of GT for donors,
    G of AG for acceptors), NOT the exonic base. Annotated sites validate at
    99.6% GT/AG under this convention.
  * Each event's offset is transcript-relative: cryptic_loc = site_loc + offset
    (+ strand) or site_loc - offset (- strand).
  * The offset carries a ±1 convention ambiguity, so we SNAP to the nearest
    canonical GT/AG within ±2 (tie toward 0). nomatch (~0.1%) is dropped.
  * The snapped intron-adjacent loc is then converted to the project's EXONIC
    base convention (donor: -1 on +strand / +1 on -strand; acceptor mirror),
    matching `02_build_positive_pool.py`, and re-validated with the standard
    dinucleotide function.

Run (after `mamba activate agentic-spliceai`):
    python examples/data_preparation/m3/03_ingest_splicevault.py --min-freq 1.0
"""
from __future__ import annotations

import argparse
import gzip
import re
from pathlib import Path

import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]
SV_GZ = REPO / "data/splicevault/GRCh38/SpliceVault_data_GRCh38.tsv.gz"
ENSEMBL_SITES = REPO / "data/ensembl/GRCh38/splice_sites_enhanced.tsv"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
OUT_DIR = REPO / "data/mane/GRCh38/m3_labels"

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
_EV_RE = re.compile(r"Top\d+:(CD|CA|ES);([+-]?\d+);([\d.]+)%")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def sv_dinuc(seq, loc: int, strand: str, splice_type: str) -> str:
    """Transcript-oriented dinucleotide at a SpliceVault intron-adjacent loc."""
    if strand == "+":
        return str(seq[loc - 1 : loc + 1]) if splice_type == "donor" else str(seq[loc - 2 : loc])
    return revcomp(str(seq[loc - 2 : loc])) if splice_type == "donor" else revcomp(str(seq[loc - 1 : loc + 1]))


def my_dinuc(seq, pos: int, strand: str, splice_type: str) -> str:
    """Transcript-oriented dinucleotide at a project exonic-base position
    (matches 02_build_positive_pool.py)."""
    if strand == "+":
        return str(seq[pos : pos + 2]) if splice_type == "donor" else str(seq[pos - 3 : pos - 1])
    return revcomp(str(seq[pos - 3 : pos - 1])) if splice_type == "donor" else revcomp(str(seq[pos : pos + 2]))


def sv_loc_to_exonic(loc: int, strand: str, splice_type: str) -> int:
    """Convert SpliceVault intron-adjacent loc → project exonic-base position."""
    if splice_type == "donor":
        return loc - 1 if strand == "+" else loc + 1
    return loc + 1 if strand == "+" else loc - 1  # acceptor


def load_strand_lookup() -> dict[str, str]:
    df = (
        pl.scan_csv(ENSEMBL_SITES, separator="\t")
        .select(["transcript_id", "strand"])
        .unique()
        .collect()
    )
    return dict(df.iter_rows())


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest SpliceVault cryptic sites (Phase B1b).")
    ap.add_argument("--min-freq", type=float, default=1.0,
                    help="Minimum cryptic-event frequency %% (default 1.0).")
    ap.add_argument("--snap-window", type=int, default=2,
                    help="Snap reconstructed loc to nearest GT/AG within ±this (default 2).")
    ap.add_argument("--max-lines", type=int, default=0, help="Debug: cap input lines (0=all).")
    args = ap.parse_args()

    strand = load_strand_lookup()
    print(f"strand lookup: {len(strand):,} transcripts")
    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)

    # Stream-dedup on (transcript_id, site_loc) → keep the top4 events string.
    seen: set[tuple[str, str]] = set()
    records: list[tuple[str, int, str, str, int, float]] = []  # chrom,spos,strand,site_type,off,freq
    n = 0
    with gzip.open(SV_GZ, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            n += 1
            if args.max_lines and n > args.max_lines:
                break
            p = line.rstrip("\n").split("\t")
            tx = p[idx["transcript_id"]]
            loc = p[idx["splicevault_site_loc"]]
            if (tx, loc) in seen or ":" not in loc:
                continue
            seen.add((tx, loc))
            st = strand.get(tx.split(".")[0]) or strand.get(tx)
            if st is None:
                continue
            chrom, spos = loc.split(":")
            spos = int(spos)
            chrom = chrom.replace("chr", "")
            for m in _EV_RE.finditer(p[idx["splicevault_top4_events"]]):
                et, off, fr = m.group(1), int(m.group(2)), float(m.group(3))
                if et == "ES" or fr < args.min_freq:
                    continue
                ctype = "donor" if et == "CD" else "acceptor"
                records.append((chrom, spos, st, ctype, off, fr))
            if n % 5_000_000 == 0:
                print(f"  ...{n:,} lines, {len(seen):,} unique sites, {len(records):,} events kept")

    print(f"Scanned {n:,} lines → {len(seen):,} unique annotated sites → "
          f"{len(records):,} CD/CA events (freq ≥ {args.min_freq}%)")

    # Reconstruct + snap + convert, per event.
    snap_order = sorted(range(-args.snap_window, args.snap_window + 1), key=abs)
    out_rows = []
    nomatch = 0
    for chrom, spos, st, ctype, off, fr in records:
        if chrom not in fasta:
            continue
        seq = fasta[chrom]
        cp = spos + off if st == "+" else spos - off
        exp = "GT" if ctype == "donor" else "AG"
        snapped = None
        for d in snap_order:
            if sv_dinuc(seq, cp + d, st, ctype) == exp:
                snapped = cp + d
                break
        if snapped is None:
            nomatch += 1
            continue
        my_pos = sv_loc_to_exonic(snapped, st, ctype)
        out_rows.append((chrom, my_pos, st, ctype, fr))

    print(f"Reconstructed {len(out_rows):,} cryptic sites; nomatch dropped: {nomatch:,} "
          f"({100*nomatch/max(1,len(records)):.2f}%)")

    df = pl.DataFrame(
        out_rows, schema=["chrom", "position", "strand", "splice_type", "freq"], orient="row"
    )
    # Dedup cryptic site coordinate; keep max freq across transcripts.
    df = df.group_by(["chrom", "position", "strand", "splice_type"]).agg(pl.col("freq").max())

    # Re-validate with the project exonic-base dinucleotide function.
    def _mydin(c, p, s, t):
        return my_dinuc(fasta[c], p, s, t) if c in fasta else ""
    dinucs = [_mydin(c, p, s, t) for c, p, s, t in
              zip(df["chrom"], df["position"], df["strand"], df["splice_type"])]
    df = df.with_columns(pl.Series("dinuc", dinucs))
    canonical = (
        ((pl.col("splice_type") == "donor") & (pl.col("dinuc") == "GT"))
        | ((pl.col("splice_type") == "acceptor") & (pl.col("dinuc") == "AG"))
    )
    df = df.with_columns(canonical.alias("canonical_dinuc"))
    rate = df["canonical_dinuc"].mean()
    print(f"Re-validation (exonic-base convention) GT/AG: {rate:.4f} (n={df.height:,})")
    if rate < 0.97:
        raise SystemExit(f"[FAILED] post-conversion GT/AG {rate:.3f} < 0.97 — convention bug.")

    pool = df.filter(pl.col("canonical_dinuc")).with_columns(
        pl.lit("splicevault").alias("source"),
        pl.lit(1).cast(pl.Int8).alias("label"),
        pl.col("freq").alias("support"),
        pl.lit("max_cryptic_freq_pct").alias("support_kind"),
        pl.lit(None, dtype=pl.String).alias("gene_id"),
        pl.lit("GRCh38_native").alias("build_origin"),
    ).select(
        ["chrom", "position", "strand", "splice_type", "source", "label",
         "dinuc", "canonical_dinuc", "support", "support_kind", "gene_id", "build_origin"]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "positives_splicevault.parquet"
    pool.write_parquet(out)
    print(f"\nWrote {out} ({pool.height:,} rows)")
    print("By splice_type:")
    print(pool.group_by("splice_type").len().sort("splice_type"))
    print("Support (cryptic freq %) quantiles:")
    for q in (0.1, 0.5, 0.9):
        print(f"  Q{int(q*100)}: {pool['support'].quantile(q):.2f}%")


if __name__ == "__main__":
    main()
