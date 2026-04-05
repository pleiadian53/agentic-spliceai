#!/usr/bin/env python3
"""Aggregate GTEx v8 STAR junction read counts by tissue.

Reads the GTEx v8 STAR junctions GCT file (357K junctions × 17K samples)
and aggregates read counts per tissue using sample metadata. Produces a
compact per-tissue junction summary suitable for the JunctionModality.

Input files (expected in --raw-dir):
  - GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz  (4.25 GB)
  - GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt         (11 MB)

Output:
  - junctions_gtex_v8.parquet  — per-junction, per-tissue summary

Coordinate convention: start/end use STAR's intron boundaries (1-based).
  - start = first intronic base = donor exon boundary + 1
  - end   = last intronic base  = acceptor exon boundary - 1
The JunctionModality converts these to exon boundary positions when loading.

Usage:
    python scripts/aggregate_gtex_junctions.py \\
        --raw-dir data/mane/GRCh38/junction_data/raw \\
        --output data/mane/GRCh38/junction_data/junctions_gtex_v8.parquet
"""

from __future__ import annotations

import argparse
import gzip
import logging
import time
from pathlib import Path

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

GCT_FILENAME = "GTEx_Analysis_2017-06-05_v8_STARv2.5.3a_junctions.gct.gz"
META_FILENAME = "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"


def build_sample_to_tissue(meta_path: Path) -> dict[str, str]:
    """Map sample IDs to tissue names from GTEx metadata."""
    meta = pl.read_csv(
        meta_path, separator="\t", infer_schema_length=10000
    )
    # SAMPID = full sample ID, SMTSD = tissue detail name
    mapping = {}
    for row in meta.select("SAMPID", "SMTSD").iter_rows():
        mapping[row[0]] = row[1]
    logger.info("Sample→tissue mapping: %d samples, %d tissues",
                len(mapping), len(set(mapping.values())))
    return mapping


def parse_junction_id(junction_id: str) -> tuple[str, int, int]:
    """Parse 'chr1_12058_12178' → ('chr1', 12058, 12178)."""
    parts = junction_id.split("_")
    # Handle chroms like chr1, chrX, chrM — rejoin if chrom has underscores
    # Standard format: chr{N}_{start}_{end}
    end = int(parts[-1])
    start = int(parts[-2])
    chrom = "_".join(parts[:-2])
    return chrom, start, end


def aggregate_junctions(
    gct_path: Path,
    sample_to_tissue: dict[str, str],
    chromosomes: set[str] | None = None,
) -> pl.DataFrame:
    """Stream-process the GCT file, aggregating counts per tissue per junction.

    For each junction, computes per-tissue:
      - total_reads: sum of read counts across all samples in the tissue
      - n_samples: number of samples with ≥1 read

    Parameters
    ----------
    gct_path : Path
        Path to the gzipped GCT file.
    sample_to_tissue : dict
        Sample ID → tissue name mapping.
    chromosomes : set of str, optional
        If provided, only process junctions on these chromosomes.

    Returns
    -------
    pl.DataFrame
        Columns: chrom, start, end, gene_id, tissue, total_reads, n_samples
    """
    t0 = time.time()

    with gzip.open(gct_path, "rt") as f:
        # Line 1: version
        version = f.readline().strip()
        logger.info("GCT version: %s", version)

        # Line 2: dimensions
        dims = f.readline().strip().split("\t")
        n_junctions, n_samples = int(dims[0]), int(dims[1])
        logger.info("Dimensions: %d junctions × %d samples", n_junctions, n_samples)

        # Line 3: header (Name, Description, sample1, sample2, ...)
        header = f.readline().strip().split("\t")
        sample_ids = header[2:]  # skip Name, Description
        assert len(sample_ids) == n_samples, (
            f"Expected {n_samples} samples, got {len(sample_ids)}"
        )

        # Build tissue index: for each tissue, which column indices belong to it
        tissue_names = sorted(set(sample_to_tissue.values()))
        tissue_to_idx: dict[str, list[int]] = {t: [] for t in tissue_names}
        matched = 0
        for i, sid in enumerate(sample_ids):
            tissue = sample_to_tissue.get(sid)
            if tissue is not None:
                tissue_to_idx[tissue].append(i)
                matched += 1

        logger.info(
            "Matched %d / %d GCT samples to tissues (%d tissues)",
            matched, n_samples, len(tissue_names),
        )

        # Pre-compute numpy index arrays per tissue for fast slicing
        tissue_indices = {
            t: np.array(idxs, dtype=np.int64)
            for t, idxs in tissue_to_idx.items()
            if len(idxs) > 0
        }
        active_tissues = sorted(tissue_indices.keys())

        # Process junctions row by row
        rows: list[dict] = []
        processed = 0
        skipped = 0

        for line in f:
            parts = line.rstrip("\n").split("\t")
            junction_id = parts[0]
            gene_id = parts[1]

            # Parse junction coordinates
            chrom, start, end = parse_junction_id(junction_id)

            # Filter chromosomes if requested
            if chromosomes is not None and chrom not in chromosomes:
                skipped += 1
                continue

            # Parse counts as numpy array for fast tissue aggregation
            counts = np.array(parts[2:], dtype=np.int32)

            for tissue in active_tissues:
                idxs = tissue_indices[tissue]
                tissue_counts = counts[idxs]
                total = int(tissue_counts.sum())
                n_supporting = int((tissue_counts > 0).sum())

                if total > 0:  # only store non-zero tissues
                    rows.append({
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "gene_id": gene_id,
                        "tissue": tissue,
                        "total_reads": total,
                        "n_samples": n_supporting,
                    })

            processed += 1
            if processed % 50_000 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (n_junctions - processed) / rate if rate > 0 else 0
                logger.info(
                    "Processed %d / %d junctions (%.0f/s, ETA %.0fmin) — %d tissue-rows so far",
                    processed, n_junctions, rate, eta / 60, len(rows),
                )

    elapsed = time.time() - t0
    logger.info(
        "Done: %d junctions processed, %d skipped (chrom filter), "
        "%d tissue-rows, %.1fs",
        processed, skipped, len(rows), elapsed,
    )

    df = pl.DataFrame(rows)
    return df


def compute_junction_summary(tissue_df: pl.DataFrame) -> pl.DataFrame:
    """Compute per-junction summary stats across tissues.

    From the tissue-level data, produces a single row per junction with:
      - max_reads: highest total_reads in any tissue
      - sum_reads: total reads across all tissues
      - n_tissues: number of tissues with support
      - tissue_breadth: n_tissues / total_tissues
      - max_tissue: tissue with highest read count
      - is_annotated: placeholder (to be filled from annotation later)

    Parameters
    ----------
    tissue_df : pl.DataFrame
        Output of aggregate_junctions().

    Returns
    -------
    pl.DataFrame
        One row per junction with summary columns.
    """
    n_total_tissues = tissue_df["tissue"].n_unique()

    summary = (
        tissue_df
        .group_by("chrom", "start", "end", "gene_id")
        .agg(
            pl.col("total_reads").max().alias("max_reads"),
            pl.col("total_reads").sum().alias("sum_reads"),
            pl.col("tissue").n_unique().alias("n_tissues"),
            # Tissue with max reads
            pl.col("tissue").sort_by("total_reads", descending=True).first().alias("max_tissue"),
            # Mean samples per tissue
            pl.col("n_samples").mean().alias("mean_samples_per_tissue"),
        )
        .with_columns(
            (pl.col("n_tissues") / n_total_tissues).alias("tissue_breadth"),
        )
        .sort("chrom", "start")
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/mane/GRCh38/junction_data/raw"),
        help="Directory with raw GTEx files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mane/GRCh38/junction_data"),
        help="Output directory for aggregated junction files",
    )
    parser.add_argument(
        "--chromosomes",
        nargs="+",
        default=None,
        help="Filter to specific chromosomes (e.g., chr19 chr21 chr22). Default: all.",
    )
    parser.add_argument(
        "--tissue-level",
        action="store_true",
        help="Also save per-tissue detail (larger file)",
    )
    args = parser.parse_args()

    gct_path = args.raw_dir / GCT_FILENAME
    meta_path = args.raw_dir / META_FILENAME

    if not gct_path.exists():
        raise FileNotFoundError(f"GCT file not found: {gct_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build sample→tissue mapping
    sample_to_tissue = build_sample_to_tissue(meta_path)

    # Parse chromosome filter
    chrom_filter = set(args.chromosomes) if args.chromosomes else None
    if chrom_filter:
        logger.info("Filtering to chromosomes: %s", sorted(chrom_filter))

    # Stream-aggregate junctions by tissue
    tissue_df = aggregate_junctions(gct_path, sample_to_tissue, chrom_filter)

    if tissue_df.height == 0:
        logger.warning("No junction data found. Check chromosome filter or input files.")
        return

    # Optionally save per-tissue detail
    if args.tissue_level:
        tissue_path = args.output_dir / "junctions_gtex_v8_by_tissue.parquet"
        tissue_df.write_parquet(tissue_path)
        logger.info("Tissue-level data: %s (%.1f MB)",
                     tissue_path, tissue_path.stat().st_size / 1e6)

    # Compute per-junction summary
    summary_df = compute_junction_summary(tissue_df)

    # Save summary
    summary_path = args.output_dir / "junctions_gtex_v8.parquet"
    summary_df.write_parquet(summary_path)
    logger.info("Junction summary: %s (%.1f MB)",
                 summary_path, summary_path.stat().st_size / 1e6)

    # Print stats
    print("\n" + "=" * 70)
    print("GTEx v8 Junction Aggregation — Complete")
    print("=" * 70)
    print(f"  Junctions:        {summary_df.height:,}")
    print(f"  Chromosomes:      {summary_df['chrom'].n_unique()}")
    print(f"  Tissues:          {tissue_df['tissue'].n_unique()}")
    n_annotated = summary_df.filter(pl.col("n_tissues") >= 10).height
    print(f"  Broad (≥10 tissues): {n_annotated:,} ({100*n_annotated/summary_df.height:.1f}%)")
    n_rare = summary_df.filter(pl.col("n_tissues") <= 2).height
    print(f"  Rare (≤2 tissues):   {n_rare:,} ({100*n_rare/summary_df.height:.1f}%)")
    print(f"\n  Output: {summary_path}")
    if args.tissue_level:
        print(f"  Tissue detail: {tissue_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
