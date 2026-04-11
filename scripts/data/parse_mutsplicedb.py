#!/usr/bin/env python
"""Parse MutSpliceDB CSV export into splice_sites_induced.tsv.

Converts raw MutSpliceDB CSV (downloaded manually from
https://brb.nci.nih.gov/splicing/) into a standardized TSV with
genomic coordinates, parsed effect types, and gene annotations.

Input:  data/mutsplicedb/MutSpliceDB_BRP_2025-12-18.csv
Output: data/mutsplicedb/splice_sites_induced.tsv

The Locus field (e.g., "chr13:95166057-95166257") provides a 200bp
genomic window around the variant.  We use the center as the
approximate variant position.

Usage:
    python scripts/data/parse_mutsplicedb.py
    python scripts/data/parse_mutsplicedb.py \
        --input data/mutsplicedb/MutSpliceDB_BRP_2025-12-18.csv \
        --output data/mutsplicedb/splice_sites_induced.tsv
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Regex patterns
_LOCUS_PATTERN = re.compile(r"(chr[\dXYMT]+):(\d+)-(\d+)")
_HGVS_PATTERN = re.compile(r"(NM_\S+):c\.\S+")

# Effect type normalization
_EFFECT_MAP = {
    "intron inclusion": "intron_retention",
    "intron retention": "intron_retention",
    "exon skip": "exon_skipping",
    "exon skipping": "exon_skipping",
    "alternative 5": "alternative_5ss",
    "alternative 3": "alternative_3ss",
    "cryptic exon": "cryptic_exon",
}


def _normalize_effect(raw_effect: str) -> tuple[str, str]:
    """Normalize splicing effect description to a standardized type.

    Returns (effect_type, site_type).
    """
    lower = raw_effect.lower()
    for key, effect_type in _EFFECT_MAP.items():
        if key in lower:
            site_type = f"{effect_type}_region"
            return effect_type, site_type
    return "unknown", "unknown"


def _parse_locus(locus: str) -> tuple[str, int, int]:
    """Parse locus string like 'chr13:95166057-95166257'.

    Returns (chrom, start, end).
    """
    m = _LOCUS_PATTERN.match(locus)
    if not m:
        return "", 0, 0
    return m.group(1), int(m.group(2)), int(m.group(3))


def _infer_strand(gene: str, gene_strands: dict) -> str:
    """Look up gene strand from pre-loaded annotations."""
    return gene_strands.get(gene, "+")


def parse_mutsplicedb(
    input_path: Path,
    output_path: Path,
    gtf_path: str | Path | None = None,
) -> int:
    """Parse MutSpliceDB CSV into standardized TSV.

    Parameters
    ----------
    input_path : Path
        Raw MutSpliceDB CSV export.
    output_path : Path
        Output TSV path.
    gtf_path : Path, optional
        GTF for strand lookup. If None, all strands default to "+".

    Returns
    -------
    int
        Number of records written.
    """
    # Load gene strands from GTF if available
    gene_strands: dict = {}
    if gtf_path:
        try:
            from agentic_spliceai.splice_engine.base_layer.data.genomic_extraction import (
                extract_gene_annotations,
            )
            ann = extract_gene_annotations(str(gtf_path), verbosity=0)
            for row in ann.iter_rows(named=True):
                name = row.get("gene_name", "")
                if name:
                    gene_strands[name] = row.get("strand", "+")
            log.info("Loaded strand info for %d genes from GTF", len(gene_strands))
        except Exception as e:
            log.warning("Could not load GTF for strand lookup: %s", e)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0

    with open(input_path) as fin, open(output_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.writer(fout, delimiter="\t")

        # Write header
        writer.writerow([
            "chrom", "position", "strand", "site_type", "inducing_variant",
            "effect_type", "gene", "transcript_id", "evidence_source",
            "evidence_samples", "confidence", "notes",
        ])

        for row in reader:
            gene = row.get("Gene Symbol", "").strip()
            mutation = row.get("Mutation", "").strip()
            effect_raw = row.get("Splicing effect", "").strip()
            sample = row.get("Sample", "").strip()
            source = row.get("Source", "").strip()
            genome = row.get("Genome version", "").strip()
            locus = row.get("Locus", "").strip()

            # Parse locus
            chrom, start, end = _parse_locus(locus)
            if not chrom:
                n_skipped += 1
                log.debug("Skipping: no locus for %s %s", gene, mutation)
                continue

            position = (start + end) // 2

            # Parse transcript ID from HGVS
            hgvs_match = _HGVS_PATTERN.match(mutation)
            transcript_id = hgvs_match.group(1) if hgvs_match else ""

            # Normalize effect type
            effect_type, site_type = _normalize_effect(effect_raw)

            # Strand from GTF
            strand = _infer_strand(gene, gene_strands)

            # Confidence heuristic
            confidence = "medium"  # all MutSpliceDB entries have RNA-seq evidence

            notes = f"From locus {locus}, genome {genome}"

            writer.writerow([
                chrom, position, strand, site_type, mutation,
                effect_type, gene, transcript_id, source,
                sample, confidence, notes,
            ])
            n_written += 1

    log.info("Written %d records to %s (%d skipped)", n_written, output_path, n_skipped)
    return n_written


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse MutSpliceDB CSV export into splice_sites_induced.tsv",
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("data/mutsplicedb/MutSpliceDB_BRP_2025-12-18.csv"),
        help="Raw MutSpliceDB CSV export",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/mutsplicedb/splice_sites_induced.tsv"),
        help="Output TSV path",
    )
    parser.add_argument(
        "--gtf", type=Path, default=None,
        help="GTF annotation for strand lookup (optional)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        print("Download from https://brb.nci.nih.gov/splicing/")
        return 1

    n = parse_mutsplicedb(args.input, args.output, args.gtf)
    print(f"\nDone: {n} variants parsed → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
