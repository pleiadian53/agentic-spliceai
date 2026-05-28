"""Phase B3 — build the ENCODE4 long-read held-out truth set for M3.

Independent, functional validation of the M3 label pool. ENCODE4 long-read
RNA-seq transcript models (TALON, GRCh38) span full transcripts, so a splice
site appearing in them is confirmed to participate in a coherent isoform — not
an alignment artifact. We use them two ways:

  1. **Functional validation** — what fraction of the M3 positive pool
     (`positives_pooled.parquet`) is confirmed by long reads?
  2. **Anti-circular eval truth set (Phase D1)** — long-read splice sites that
     are ALSO absent from annotation = long-read-confirmed NOVEL sites; M3 is
     scored against these (not against "absent from annotation", which would be
     circular).

Input: per-biosample long-read transcriptome GTFs in
`data/encode_longread/GRCh38/gtf/*.gtf.gz` (downloaded by the B3 step; one per
ENCODE biosample, manifest in `../manifest.tsv`).

Splice-site extraction uses the strand-correct intron/junction convention
(matches `genomic_extraction` after the minus-strand fix); the GT/AG-by-strand
oracle is the acceptance gate (≥0.95 both strands).

Run (after: mamba activate agentic-spliceai):
    python examples/data_preparation/m3/10_build_longread_truth.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]
GTF_DIR = REPO / "data/encode_longread/GRCh38/gtf"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
LABELS = REPO / "data/mane/GRCh38/m3_labels"
OUT_DIR = REPO / "data/encode_longread/GRCh38"
JOIN = ["chrom", "position", "strand", "splice_type"]
_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _strip(c):
    return pl.col(c).cast(pl.String).str.replace_all(r"^chr", "")


def extract_sites(gtf_gz: Path) -> pl.DataFrame:
    """Strand-correct splice sites from a (gzipped) long-read transcript GTF.

    Intron/junction convention (consecutive exons sorted by genomic start):
      + : donor = e_i.end,        acceptor = e_{i+1}.start
      - : donor = e_{i+1}.start,  acceptor = e_i.end
    """
    exons = (
        pl.scan_csv(
            gtf_gz, separator="\t", has_header=False, comment_prefix="#",
            new_columns=["chrom", "src", "feat", "start", "end", "score", "strand", "frame", "attr"],
            schema_overrides={"start": pl.Int64, "end": pl.Int64},
        )
        .filter(pl.col("feat") == "exon")
        .with_columns(
            _strip("chrom").alias("chrom"),
            pl.col("attr").str.extract(r'transcript_id "([^"]+)"', 1).alias("tx"),
        )
        .select(["chrom", "start", "end", "strand", "tx"])
        .collect()
    )
    if exons.height == 0:
        return pl.DataFrame(schema={"chrom": pl.String, "position": pl.Int64, "strand": pl.String, "splice_type": pl.String})
    exons = exons.sort(["tx", "start"]).with_columns(
        pl.col("start").shift(-1).over("tx").alias("next_start"),
        (pl.col("tx") == pl.col("tx").shift(-1)).alias("has_next"),
    )
    pairs = exons.filter(pl.col("has_next"))
    donor = pairs.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("end")).otherwise(pl.col("next_start")).alias("position"),
        pl.lit("donor").alias("splice_type"),
    ).select(JOIN)
    acceptor = pairs.with_columns(
        pl.when(pl.col("strand") == "+").then(pl.col("next_start")).otherwise(pl.col("end")).alias("position"),
        pl.lit("acceptor").alias("splice_type"),
    ).select(JOIN)
    return pl.concat([donor, acceptor]).unique()


def gtag_by_strand(df: pl.DataFrame, fasta, n: int = 40000) -> dict:
    d = df.head(n)
    out = {}
    for st in ("+", "-"):
        sub = d.filter(pl.col("strand") == st)
        if sub.height == 0:
            out[st] = None
            continue
        hits = 0
        for c, p, t in zip(sub["chrom"], sub["position"], sub["splice_type"]):
            cc = c.replace("chr", "")
            if cc not in fasta:
                continue
            s = fasta[cc]
            d2 = (str(s[p:p+2]) if t == "donor" else str(s[p-3:p-1])) if st == "+" \
                else (revcomp(str(s[p-3:p-1])) if t == "donor" else revcomp(str(s[p:p+2])))
            hits += d2 == ("GT" if t == "donor" else "AG")
        out[st] = hits / sub.height
    return out


def annotation_union() -> pl.DataFrame:
    def keys(path, prefix=None):
        lf = pl.scan_csv(path, separator="\t").with_columns(_strip("chrom").alias("chrom"))
        if prefix:
            lf = lf.filter(pl.col("transcript_id").str.contains(prefix))
        return lf.select(JOIN).unique().collect()
    gen = keys(REPO / "data/gencode/GRCh38/splice_sites_enhanced.tsv")
    ref = keys(REPO / "data/refseq/GRCh38/splice_sites_enhanced.tsv", r"NM_|NR_")
    return pl.concat([gen, ref]).unique()


def main() -> None:
    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)
    gtfs = sorted(glob.glob(str(GTF_DIR / "*.gtf.gz")))
    print(f"long-read GTFs: {len(gtfs)}")

    # Per-GTF (biosample) site sets → pool with tissue-support count.
    per = []
    for i, g in enumerate(gtfs, 1):
        sites = extract_sites(Path(g)).with_columns(pl.lit(Path(g).stem).alias("biosample"))
        per.append(sites)
        if i % 10 == 0:
            print(f"  extracted {i}/{len(gtfs)}")
    allsites = pl.concat(per)
    lr = (
        allsites.group_by(JOIN)
        .agg(pl.col("biosample").n_unique().alias("n_biosamples"))
    )
    print(f"long-read splice sites (distinct): {lr.height:,}")

    rate = gtag_by_strand(lr, fasta)
    print(f"GT/AG-by-strand (acceptance gate): {rate}")
    if any(v is not None and v < 0.95 for v in rate.values()):
        raise SystemExit(f"[FAILED] long-read GT/AG {rate} < 0.95 — extraction/convention issue.")

    ann = annotation_union()
    lr_novel = lr.join(ann, on=JOIN, how="anti")
    print(f"long-read NOVEL sites (not in GENCODE∪RefSeq): {lr_novel.height:,} "
          f"({100*lr_novel.height/lr.height:.1f}%)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lr.with_columns(
        lr.join(ann, on=JOIN, how="anti").select(JOIN).with_columns(pl.lit(True).alias("is_novel"))
        .pipe(lambda d: lr.join(d, on=JOIN, how="left"))["is_novel"].fill_null(False)
    ).write_parquet(OUT_DIR / "longread_splice_sites.parquet")
    lr_novel.write_parquet(OUT_DIR / "longread_truth_novel.parquet")

    # --- Functional validation: confirmation of the M3 positive pool ---
    pos = pl.read_parquet(LABELS / "positives_pooled.parquet").with_columns(_strip("chrom").alias("chrom"))
    conf = pos.join(lr.select(JOIN), on=JOIN, how="semi")
    print("\n=== M3 positive-pool functional confirmation (long-read) ===")
    print(f"positives total:            {pos.height:,}")
    print(f"confirmed by long reads:    {conf.height:,} ({100*conf.height/pos.height:.1f}%)")
    for src in ["splicevault", "gtex_novel", "gtex_novel+splicevault"]:
        sub = pos.filter(pl.col("sources") == src)
        if sub.height:
            c = sub.join(lr.select(JOIN), on=JOIN, how="semi").height
            print(f"  {src:24s}: {c:,}/{sub.height:,} ({100*c/sub.height:.1f}%)")

    # anchors confirmation (bonus)
    anch = pl.read_parquet(LABELS / "disease_anchors.parquet").with_columns(_strip("chrom").alias("chrom"))
    ac = anch.join(lr.select(JOIN), on=JOIN, how="semi").height
    print(f"disease anchors confirmed:  {ac:,}/{anch.height:,} ({100*ac/anch.height:.1f}%)")

    print(f"\nWrote {OUT_DIR}/longread_splice_sites.parquet ({lr.height:,})")
    print(f"Wrote {OUT_DIR}/longread_truth_novel.parquet ({lr_novel.height:,}) — D1 eval truth set")


if __name__ == "__main__":
    main()
