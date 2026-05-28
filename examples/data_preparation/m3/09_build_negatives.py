"""Phase B2 — M3 negative sampling (recognizer + post-filter framing).

Design decision (2026-05-26): M3 is a splice-site RECOGNIZER. A novel cryptic
donor and an annotated canonical donor are sequence-identical, so we do NOT use
annotated sites as decoy negatives (that would be a contradictory label). Instead:
  - Positives  = the novel pooled sites (donor/acceptor)              [from B1]
  - Negatives  = true NON-sites:
      * hard : positions carrying a canonical GT/AG dinucleotide (look like a
               site) but are neither annotated nor novel — forces the model
               past the bare dinucleotide.
      * easy : random gene-body positions without a canonical dinucleotide.
  - Annotated sites = MASKED (ignore-index) in the loss — neither + nor -.
    Novelty is applied at INFERENCE by exact set-subtraction, not learned.

Outputs (data/mane/GRCh38/m3_labels/):
  negatives.parquet        chrom,position,strand,splice_type,category(hard|easy),dinuc,label(=0/neither)
  annotation_mask.parquet  chrom,position,strand,splice_type   (loss ignore-index)

Negatives are excluded if they coincide with any real site (annotated ∪ novel)
on the same (chrom, position, strand, splice_type). Local + fast (FASTA point
queries only; no bigWig). The multimodal feature extraction for these positions
is a separate (pod) step in Phase C.

Run:
    python examples/data_preparation/m3/09_build_negatives.py --neg-per-class 154000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import pyfaidx

REPO = Path(__file__).resolve().parents[3]
LABELS = REPO / "data/mane/GRCh38/m3_labels"
GENE_FEATURES = REPO / "data/ensembl/GRCh38/gene_features.parquet"
FASTA = REPO / "data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
JOIN = ["chrom", "position", "strand", "splice_type"]
_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def dinuc(seq, pos: int, strand: str, splice_type: str) -> str:
    """Transcript-oriented intronic dinucleotide at an exonic-base position."""
    if strand == "+":
        return str(seq[pos : pos + 2]) if splice_type == "donor" else str(seq[pos - 3 : pos - 1])
    return revcomp(str(seq[pos - 3 : pos - 1])) if splice_type == "donor" else revcomp(str(seq[pos : pos + 2]))


def _strip(c):
    return pl.col(c).cast(pl.String).str.replace_all(r"^chr", "")


def annotation_union() -> pl.DataFrame:
    """GENCODE v47 ∪ RefSeq-curated (the corrected, strand-fixed TSVs)."""
    def keys(path, prefix=None):
        lf = pl.scan_csv(path, separator="\t").with_columns(_strip("chrom").alias("chrom"))
        if prefix:
            lf = lf.filter(pl.col("transcript_id").str.contains(prefix))
        return lf.select(JOIN).unique().collect()
    gen = keys(REPO / "data/gencode/GRCh38/splice_sites_enhanced.tsv")
    ref = keys(REPO / "data/refseq/GRCh38/splice_sites_enhanced.tsv", r"NM_|NR_")
    return pl.concat([gen, ref]).unique()


def main() -> None:
    ap = argparse.ArgumentParser(description="Build M3 negatives + annotation mask (Phase B2).")
    ap.add_argument("--neg-per-class", type=int, default=154000,
                    help="Target count for EACH of hard/easy negatives (default ~= n_positives).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    fasta = pyfaidx.Fasta(str(FASTA), sequence_always_upper=True, rebuild=False)

    # --- real-site exclusion set = annotated ∪ novel positives ---
    ann = annotation_union()
    ann.write_parquet(LABELS / "annotation_mask.parquet")
    print(f"annotation mask (GENCODE ∪ RefSeq-curated): {ann.height:,} sites → annotation_mask.parquet")

    pos = pl.read_parquet(LABELS / "positives_pooled.parquet").with_columns(_strip("chrom").alias("chrom"))
    real = pl.concat([ann, pos.select(JOIN)]).unique()
    real_keys = set(zip(real["chrom"], real["position"], real["strand"], real["splice_type"]))
    print(f"real-site exclusion set (annotated ∪ novel): {len(real_keys):,}")

    # --- gene bodies for sampling (canonical chroms, protein_coding + lncRNA) ---
    genes = (
        pl.read_parquet(GENE_FEATURES)
        .with_columns(_strip("chrom").alias("chrom"))
        .filter(pl.col("chrom").is_in([str(i) for i in range(1, 23)] + ["X", "Y"]))
        .filter((pl.col("end") - pl.col("start")) > 400)
    )
    g_chrom = genes["chrom"].to_list()
    g_start = genes["start"].to_numpy()
    g_end = genes["end"].to_numpy()
    g_strand = genes["strand"].to_list()
    g_len = (g_end - g_start).astype(float)
    g_prob = g_len / g_len.sum()
    n_genes = genes.height
    print(f"sampling from {n_genes:,} gene bodies")

    target = args.neg_per_class
    hard, easy = [], []
    seen = set()
    batch = 200_000
    iters = 0
    while (len(hard) < target or len(easy) < target) and iters < 60:
        iters += 1
        gi = rng.choice(n_genes, size=batch, p=g_prob)
        offs = rng.random(batch)
        for k in range(batch):
            j = gi[k]
            chrom = g_chrom[j]
            if chrom not in fasta:
                continue
            strand = g_strand[j]
            # avoid the outer 100 bp (near TSS/TES) to keep negatives clean intronic/internal
            p = int(g_start[j] + 100 + offs[k] * max(1, (g_end[j] - g_start[j] - 200)))
            seq = fasta[chrom]
            d_don = dinuc(seq, p, strand, "donor")
            d_acc = dinuc(seq, p, strand, "acceptor")
            if d_don == "GT":
                stype, dn, canon = "donor", d_don, True
            elif d_acc == "AG":
                stype, dn, canon = "acceptor", d_acc, True
            else:
                # easy: assign a type at random, record the (non-canonical) donor-frame dinuc
                stype = "donor" if (k & 1) else "acceptor"
                dn = d_don if stype == "donor" else d_acc
                canon = False
            key = (chrom, p, strand, stype)
            if key in seen or key in real_keys:
                continue
            seen.add(key)
            if canon and len(hard) < target:
                hard.append((chrom, p, strand, stype, "hard", dn))
            elif not canon and len(easy) < target:
                easy.append((chrom, p, strand, stype, "easy", dn))
        if iters % 5 == 0:
            print(f"  iter {iters}: hard={len(hard):,} easy={len(easy):,}")

    neg = pl.DataFrame(
        hard + easy,
        schema=["chrom", "position", "strand", "splice_type", "category", "dinuc"],
        orient="row",
    ).with_columns(pl.lit(0).cast(pl.Int8).alias("label"))  # 0 = neither

    neg.write_parquet(LABELS / "negatives.parquet")
    print(f"\nWrote negatives.parquet ({neg.height:,} rows)")
    print(neg.group_by("category").len().sort("category"))
    print("by splice_type:")
    print(neg.group_by(["category", "splice_type"]).len().sort(["category", "splice_type"]))
    # sanity: hard negatives must be 100% canonical dinuc, easy 0%
    hr = neg.filter(pl.col("category") == "hard")
    canon_rate = ((hr["splice_type"] == "donor") & (hr["dinuc"] == "GT")).sum() + \
                 ((hr["splice_type"] == "acceptor") & (hr["dinuc"] == "AG")).sum()
    print(f"hard-negative canonical-dinuc rate: {canon_rate/hr.height:.4f} (expect 1.0)")


if __name__ == "__main__":
    main()
