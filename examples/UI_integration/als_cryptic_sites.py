#!/usr/bin/env python
"""Curated, splice-site-aligned GRCh38 coordinates for ALS/FTD TDP-43 cryptic exons.

Single source of truth for the STMN2 and UNC13A cryptic splicing events used by
the UI-integration experiments.  Coordinates are GRCh38, verified 2026-05-21 to
match the Ensembl GRCh38.112 annotation the meta models trained on (canonical
exons exact; canonical junctions present in GTEx; cryptic junctions absent from
GTEx — TDP-43-repressed in normal tissue).  See
``dev/meta_layer/UI_integration/`` and the project memory for the audit.

Two distinct mechanisms:
- **STMN2** (chr8, + strand): a cryptic exon in intron 1 carrying a premature
  polyadenylation site + stop → the transcript truncates after exon 1 + cryptic
  exon (exons 2–5 dropped).  The single novel splice site is the cryptic
  ACCEPTOR; its 3′ end is a polyA site, not a splice donor.
  (Klim 2019; Melamed 2019, Nat Neurosci.)
- **UNC13A** (chr19, − strand): a 128-bp internal "poison" cassette exon between
  transcript exons 20–21 → frameshift/PTC → NMD.  Two novel splice sites
  (a donor and an acceptor) flank the cassette; the rest of the transcript is
  intact.  ALS/FTD risk SNP rs12608932 lies in this intron.
  (Brown 2022; Ma 2022, Nature.)

NOTE on the older `data/ensembl/ALS/` BEDs: the STMN2 cryptic exon start there
(79,616,821 1-based) was 1 bp upstream of the canonical AG acceptor; the correct
first exonic base is 79,616,822.  UNC13A was already correct.

All positions here are **1-based** (Ensembl/GTF convention); BED output converts
to 0-based half-open.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class CrypticSpliceSite:
    """One TDP-43-repressed cryptic splice site (the novel donor/acceptor)."""
    pos: int                              # 1-based genomic position of the splice site
    kind: Literal["donor", "acceptor"]    # biological (strand-aware) splice-site type
    note: str = ""


@dataclass
class CrypticEvent:
    gene: str
    transcript: str                       # Ensembl canonical transcript (matches M2-S annotation)
    chrom: str                            # UCSC-style ("chr8"); bare form derived as needed
    strand: Literal["+", "-"]
    cryptic_exon_start: int               # 1-based, first exonic base (splice-site aligned)
    cryptic_exon_end: int                 # 1-based, last exonic base (polyA for STMN2)
    mechanism: str                        # short human description
    sites: List[CrypticSpliceSite] = field(default_factory=list)


# ── Curated events (GRCh38, splice-site aligned) ─────────────────────────────
EVENTS: List[CrypticEvent] = [
    CrypticEvent(
        gene="STMN2", transcript="ENST00000220876", chrom="chr8", strand="+",
        cryptic_exon_start=79_616_822, cryptic_exon_end=79_617_048,
        mechanism="cryptic exon in intron 1 with premature polyA → truncated, "
                  "non-functional STMN2 (exons 2–5 dropped)",
        sites=[
            CrypticSpliceSite(79_616_822, "acceptor",
                              "cryptic 3' acceptor (AG at 79,616,820–21); "
                              "splices from exon-1 canonical donor"),
        ],
    ),
    CrypticEvent(
        gene="UNC13A", transcript="ENST00000519716", chrom="chr19", strand="-",
        cryptic_exon_start=17_642_414, cryptic_exon_end=17_642_541,
        mechanism="128-bp poison cassette between transcript exons 20–21 → "
                  "frameshift/PTC → NMD → reduced UNC13A (rest of transcript intact)",
        sites=[
            # - strand: donor (5'SS) at low genomic coord, acceptor (3'SS) at high.
            CrypticSpliceSite(17_642_414, "donor",    "cryptic 5' donor (toward exon 20)"),
            CrypticSpliceSite(17_642_541, "acceptor", "cryptic 3' acceptor (toward exon 21)"),
        ],
    ),
]


def write_bed_and_tsv(out_dir: Path) -> None:
    """Emit corrected cryptic-exon BED + cryptic splice-site TSV (GRCh38)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bed = out_dir / "als_cryptic_exons_grch38.bed"
    tsv = out_dir / "als_cryptic_splice_sites_grch38.tsv"

    with open(bed, "w") as f:
        f.write("# GRCh38 cryptic exons (splice-site aligned). BED 0-based half-open.\n")
        for ev in EVENTS:
            f.write(f"{ev.chrom}\t{ev.cryptic_exon_start - 1}\t{ev.cryptic_exon_end}\t"
                    f"{ev.gene}_cryptic_exon\t0\t{ev.strand}\n")

    with open(tsv, "w") as f:
        f.write("gene\ttranscript\tchrom\tstrand\tsplice_site_pos_1based\tkind\tnote\n")
        for ev in EVENTS:
            for s in ev.sites:
                f.write(f"{ev.gene}\t{ev.transcript}\t{ev.chrom}\t{ev.strand}\t"
                        f"{s.pos}\t{s.kind}\t{s.note}\n")

    print(f"Wrote {bed}")
    print(f"Wrote {tsv}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Emit corrected GRCh38 ALS cryptic-exon BED/TSV")
    p.add_argument("--out-dir", type=Path,
                   default=Path("output/meta_layer/ui_cache/als_cryptic"))
    args = p.parse_args()
    write_bed_and_tsv(args.out_dir)
    print("\nCurated cryptic events:")
    for ev in EVENTS:
        print(f"  {ev.gene} ({ev.chrom}{ev.strand}) cryptic exon "
              f"{ev.cryptic_exon_start:,}-{ev.cryptic_exon_end:,}: "
              f"{', '.join(f'{s.kind}@{s.pos:,}' for s in ev.sites)}")
