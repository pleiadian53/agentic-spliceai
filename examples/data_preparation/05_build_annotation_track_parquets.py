#!/usr/bin/env python
"""Build slim, track-ready splice-site parquets for each annotation source.

The Bio Lab UI renders annotation *tracks* (MANE / Ensembl / GENCODE, plus the
``Ensembl \\ MANE`` delta that M2-S is built to find). Serving those from the
``splice_sites_enhanced.tsv`` files is too slow to do per request — a per-gene
scan costs 0.2 s (MANE, 54 MB) to 1.4 s (GENCODE, 487 MB). This converts each
one, once, into a deduplicated parquet keyed for per-gene lookup.

Two things are normalized at build time so no consumer can get them wrong:

1. **The ``chr`` prefix.** MANE and GENCODE use ``chr1``; **Ensembl uses ``1``**.
   Comparing them unnormalized yields *zero* overlap and therefore a delta set
   that silently marks every Ensembl site as "alternative". This failure is
   plausible-looking rather than loud, so it is fixed here rather than in each
   caller. Output ``chrom`` is always bare (``1``, ``X``).
2. **Transcript multiplicity.** A site supported by 12 transcripts appears 12
   times in the TSV. Rows are collapsed to one per distinct site, with the
   transcript count kept as ``n_transcripts`` (useful as a support signal).

Output (per source, next to its TSV)::

    data/<source>/GRCh38/splice_sites_track.parquet
        chrom str (bare) · position i64 · strand str · splice_type str
        gene_name str · n_transcripts u32

Usage::

    python examples/data_preparation/05_build_annotation_track_parquets.py
    python examples/data_preparation/05_build_annotation_track_parquets.py --sources ensembl
    python examples/data_preparation/05_build_annotation_track_parquets.py --force

Idempotent: skips a source whose parquet is newer than its TSV unless --force.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

logger = logging.getLogger("build_annotation_tracks")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Annotation sources are declared in settings.yaml; the derived TSV lives beside
# each source's data directory under the same filename by project convention.
SOURCES: dict[str, Path] = {
    "mane": PROJECT_ROOT / "data" / "mane" / "GRCh38",
    "ensembl": PROJECT_ROOT / "data" / "ensembl" / "GRCh38",
    "gencode": PROJECT_ROOT / "data" / "gencode" / "GRCh38",
}
TSV_NAME = "splice_sites_enhanced.tsv"
OUT_NAME = "splice_sites_track.parquet"

KEEP = ["chrom", "position", "strand", "splice_type", "gene_name", "transcript_id"]


def build_one(source: str, src_dir: Path, force: bool = False) -> dict | None:
    """Convert one annotation's splice-site TSV into a track parquet."""
    tsv = src_dir / TSV_NAME
    out = src_dir / OUT_NAME

    if not tsv.exists():
        logger.error("%s: missing %s", source, tsv)
        return None

    if out.exists() and not force and out.stat().st_mtime >= tsv.stat().st_mtime:
        logger.info("%s: up to date, skipping (--force to rebuild)", source)
        return {"source": source, "skipped": True, "path": out}

    t0 = time.time()
    logger.info("%s: reading %s (%.0f MB)", source, tsv.name, tsv.stat().st_size / 1e6)

    df = (
        pl.scan_csv(tsv, separator="\t", infer_schema_length=5000)
        .select(KEEP)
        # Bare chromosome: Ensembl ships "1", MANE/GENCODE ship "chr1".
        .with_columns(
            pl.col("chrom").cast(pl.Utf8).str.replace(r"^chr", "").alias("chrom")
        )
        .group_by(["chrom", "position", "strand", "splice_type", "gene_name"])
        .agg(pl.col("transcript_id").n_unique().cast(pl.UInt32).alias("n_transcripts"))
        .sort(["gene_name", "chrom", "position"])
        .collect()
    )

    df.write_parquet(out, compression="zstd")
    elapsed = time.time() - t0

    stats = {
        "source": source,
        "skipped": False,
        "path": out,
        "sites": df.height,
        "genes": df["gene_name"].n_unique(),
        "mb": out.stat().st_size / 1e6,
        "seconds": elapsed,
    }
    logger.info(
        "%s: %s sites over %s genes -> %s (%.1f MB, %.0fs)",
        source, f"{stats['sites']:,}", f"{stats['genes']:,}",
        out.name, stats["mb"], elapsed,
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--sources", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES),
        help="Annotation sources to build (default: all).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if the parquet is newer than its TSV.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    results = [
        r for s in args.sources
        if (r := build_one(s, SOURCES[s], force=args.force)) is not None
    ]
    if len(results) != len(args.sources):
        return 1

    built = [r for r in results if not r["skipped"]]
    if built:
        print("\nBuilt:")
        for r in built:
            print(
                f"  {r['source']:9} {r['sites']:>9,} sites  "
                f"{r['genes']:>7,} genes  {r['mb']:>6.1f} MB"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
