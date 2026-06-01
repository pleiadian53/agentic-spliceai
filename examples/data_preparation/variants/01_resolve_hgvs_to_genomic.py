#!/usr/bin/env python
"""Resolve HGVS coding-notation strings to exact genomic coordinates.

Use case: cleaning variant catalogs (MutSpliceDB, SpliceVault, ClinVar
derivations) that ship transcript-relative HGVS but only an approximate
locus-center genomic ``position``. The variant-runner uses ``position``
as the mutation site — if it's off from the actual variant base, we
mutate the wrong base and benchmark numbers are biased low.

This script:
  1. Loads a variant catalog TSV (default: MutSpliceDB).
  2. For each row, parses the HGVS via :class:`HgvsResolver` and resolves
     to genomic ``(chrom, position, ref, alt, strand)``.
  3. Audits resolved coords against the catalog's recorded ``position``
     and against the FASTA (does the resolved ref base match FASTA?).
  4. Optionally writes a corrected TSV with the resolver's genomic coords
     (column ``position_resolved``, ``strand_resolved``, ``ref_genomic``,
     ``alt_genomic``).

Default mode is **audit-only** (no writes) — pass ``--write-output PATH``
to materialise the corrected TSV.

Usage
-----
::

    # Audit MutSpliceDB
    python examples/data_preparation/variants/01_resolve_hgvs_to_genomic.py

    # Audit + write corrected TSV
    python examples/data_preparation/variants/01_resolve_hgvs_to_genomic.py \\
        --write-output data/mutsplicedb/splice_sites_induced_resolved.tsv

    # A different catalog with the same schema
    python examples/data_preparation/variants/01_resolve_hgvs_to_genomic.py \\
        --input data/my_catalog.tsv --hgvs-column my_hgvs_field
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import polars as pl
import pyfaidx

# Allow running as a script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from agentic_spliceai.splice_engine.utils.hgvs_resolver import (
    GenomicVariant,
    HgvsResolutionError,
    HgvsResolver,
)

logger = logging.getLogger(__name__)


def _normalize_chrom_for_fasta(chrom: str, fasta_keys: set) -> Optional[str]:
    """Strip or add chr-prefix to match the FASTA's naming."""
    if chrom in fasta_keys:
        return chrom
    bare = chrom[3:] if chrom.startswith("chr") else chrom
    if bare in fasta_keys:
        return bare
    chr_prefixed = f"chr{chrom}" if not chrom.startswith("chr") else chrom
    if chr_prefixed in fasta_keys:
        return chr_prefixed
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve HGVS strings to genomic coordinates via a gffutils DB."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/mutsplicedb/splice_sites_induced.tsv"),
        help="Input variant catalog (TSV). Default: MutSpliceDB.",
    )
    parser.add_argument(
        "--hgvs-column",
        default="inducing_variant",
        help="Column containing the HGVS coding-notation string.",
    )
    parser.add_argument(
        "--position-column",
        default="position",
        help="Column with the catalog's recorded (often approximate) genomic position. "
             "Used only for audit comparison; the resolver does not consume it.",
    )
    parser.add_argument(
        "--gffutils-db",
        type=Path,
        default=Path("data/mane/GRCh38/annotations.db"),
        help="Path to gffutils FeatureDB built from a RefSeq/MANE GFF/GTF.",
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=Path("data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"),
        help="Reference FASTA for the FASTA-ref-match audit.",
    )
    parser.add_argument(
        "--write-output",
        type=Path,
        default=None,
        help="If given, write the corrected TSV here. Adds columns "
             "`position_resolved`, `strand_resolved`, `ref_genomic`, `alt_genomic`, "
             "and `resolution_status` ('ok' / 'failed' / 'unparseable_hgvs').",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process only the first N rows (smoke).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        return 2
    if not args.gffutils_db.exists():
        logger.error("gffutils DB not found: %s", args.gffutils_db)
        return 2
    if not args.fasta.exists():
        logger.error("FASTA not found: %s", args.fasta)
        return 2

    logger.info("Loading inputs ...")
    resolver = HgvsResolver(args.gffutils_db)
    fasta = pyfaidx.Fasta(str(args.fasta))
    fasta_keys = set(fasta.keys())
    df = pl.read_csv(str(args.input), separator="\t")
    if args.max_rows:
        df = df.head(args.max_rows)
    logger.info("Processing %d rows from %s", len(df), args.input.name)

    rows = df.to_dicts()

    resolved_pos: list[Optional[int]] = []
    resolved_strand: list[Optional[str]] = []
    resolved_g_ref: list[Optional[str]] = []
    resolved_g_alt: list[Optional[str]] = []
    resolution_status: list[str] = []
    fasta_match: list[Optional[bool]] = []
    position_delta: list[Optional[int]] = []  # resolved - catalog
    strand_disagreement: list[Optional[bool]] = []

    for row in rows:
        hgvs = row.get(args.hgvs_column, "") or ""
        try:
            v: GenomicVariant = resolver.resolve(hgvs)
        except HgvsResolutionError as e:
            resolved_pos.append(None)
            resolved_strand.append(None)
            resolved_g_ref.append(None)
            resolved_g_alt.append(None)
            status = "unparseable_hgvs" if "Unparseable" in str(e) else "resolution_failed"
            resolution_status.append(status)
            fasta_match.append(None)
            position_delta.append(None)
            strand_disagreement.append(None)
            continue

        resolved_pos.append(v.position)
        resolved_strand.append(v.strand)
        resolved_g_ref.append(v.ref)
        resolved_g_alt.append(v.alt)
        resolution_status.append("ok")

        # FASTA validation
        fa_key = _normalize_chrom_for_fasta(v.chrom, fasta_keys)
        if fa_key is None:
            fasta_match.append(None)
        else:
            fasta_base = str(fasta[fa_key][v.position - 1]).upper()
            fasta_match.append(fasta_base == v.ref)

        # Delta vs catalog position
        try:
            cat_pos = int(row[args.position_column])
            position_delta.append(v.position - cat_pos)
        except (KeyError, TypeError, ValueError):
            position_delta.append(None)

        # Strand disagreement vs catalog
        cat_strand = row.get("strand")
        if cat_strand in ("+", "-"):
            strand_disagreement.append(v.strand != cat_strand)
        else:
            strand_disagreement.append(None)

    # ── Audit summary ────────────────────────────────────────────
    n = len(rows)
    status_ctr = Counter(resolution_status)
    n_resolved = status_ctr["ok"]
    n_fasta_ok = sum(1 for x in fasta_match if x is True)
    n_fasta_bad = sum(1 for x in fasta_match if x is False)
    n_strand_flipped = sum(1 for x in strand_disagreement if x is True)
    deltas = [d for d in position_delta if d is not None]
    n_pos_exact = sum(1 for d in deltas if d == 0)
    n_pos_offset = sum(1 for d in deltas if d != 0)

    print()
    print("=" * 64)
    print(f"HGVS Resolution Audit — {args.input.name}")
    print("=" * 64)
    print(f"Total rows:                 {n}")
    print(f"Resolution status:")
    for k, v in status_ctr.most_common():
        print(f"  {k:24s}  {v:4d}  ({100 * v / n:5.1f}%)")
    print()
    print(f"FASTA ref-base match (of resolved):")
    print(f"  match:    {n_fasta_ok}/{n_resolved}  ({100 * n_fasta_ok / n_resolved:.1f}%)")
    print(f"  mismatch: {n_fasta_bad}/{n_resolved}  ({100 * n_fasta_bad / n_resolved:.1f}%)")
    print()
    print(f"Position vs catalog `{args.position_column}` (of resolved):")
    print(f"  exact match: {n_pos_exact}/{n_resolved}  ({100 * n_pos_exact / n_resolved:.1f}%)")
    print(f"  off by ≥1bp: {n_pos_offset}/{n_resolved}  ({100 * n_pos_offset / n_resolved:.1f}%)")
    if deltas:
        abs_d = sorted(abs(d) for d in deltas if d != 0)
        if abs_d:
            print(
                f"  offset distribution (|bp|): min={min(abs_d)} "
                f"median={abs_d[len(abs_d)//2]} mean={sum(abs_d)/len(abs_d):.1f} "
                f"max={max(abs_d)}"
            )
    print()
    print(f"Strand vs catalog `strand` (of resolved):")
    n_strand_compared = sum(1 for x in strand_disagreement if x is not None)
    if n_strand_compared:
        print(
            f"  agree:    {n_strand_compared - n_strand_flipped}/{n_strand_compared}  "
            f"({100 * (n_strand_compared - n_strand_flipped) / n_strand_compared:.1f}%)"
        )
        print(
            f"  disagree: {n_strand_flipped}/{n_strand_compared}  "
            f"({100 * n_strand_flipped / n_strand_compared:.1f}%)  "
            "← catalog strand is wrong for these"
        )
    print("=" * 64)

    if args.write_output:
        out = df.with_columns(
            pl.Series("position_resolved", resolved_pos, dtype=pl.Int64),
            pl.Series("strand_resolved", resolved_strand),
            pl.Series("ref_genomic", resolved_g_ref),
            pl.Series("alt_genomic", resolved_g_alt),
            pl.Series("resolution_status", resolution_status),
        )
        args.write_output.parent.mkdir(parents=True, exist_ok=True)
        out.write_csv(str(args.write_output), separator="\t")
        print(f"\nWrote: {args.write_output}  ({len(out)} rows, +5 columns)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
