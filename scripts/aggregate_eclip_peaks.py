#!/usr/bin/env python3
"""Download and aggregate ENCODE eCLIP narrowPeak data for the RBP modality.

Queries the ENCODE REST API for eCLIP experiments, downloads IDR-filtered
narrowPeak BED files, and consolidates them into a single parquet file
suitable for the RBPEclipModality.

Output schema:
    chrom: Utf8          — chromosome (e.g., 'chr1')
    start: Int64         — peak start (0-based)
    end: Int64           — peak end (exclusive)
    rbp: Utf8            — RBP target name (e.g., 'SRSF1')
    cell_line: Utf8      — biosample (e.g., 'K562')
    signal_value: Float64 — fold-enrichment over input control
    neg_log10_pvalue: Float64 — -log10(pValue)
    strand: Utf8         — peak strand (+, -, .)

Usage:
    # Download from ENCODE and aggregate
    python scripts/aggregate_eclip_peaks.py \\
        --output data/mane/GRCh38/rbp_data/

    # Use pre-downloaded BED files (skip ENCODE API)
    python scripts/aggregate_eclip_peaks.py \\
        --input-dir /path/to/downloaded/beds/ \\
        --output data/mane/GRCh38/rbp_data/

    # Dry run (query API but don't download)
    python scripts/aggregate_eclip_peaks.py --dry-run
"""

from __future__ import annotations

import argparse
import gzip
import io
import logging
import time
from pathlib import Path
from typing import Optional

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ENCODE_API_BASE = "https://www.encodeproject.org"

# narrowPeak BED format (10 columns)
NARROWPEAK_COLUMNS = [
    "chrom", "start", "end", "name", "score", "strand",
    "signal_value", "p_value", "q_value", "peak_offset",
]
NARROWPEAK_DTYPES = {
    "chrom": str, "start": int, "end": int, "name": str,
    "score": int, "strand": str, "signal_value": float,
    "p_value": float, "q_value": float, "peak_offset": int,
}

# Canonical chromosomes (GRCh38)
CANONICAL_CHROMS = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}

OUTPUT_FILENAME = "eclip_peaks.parquet"


def query_encode_eclip_experiments(
    cell_lines: tuple[str, ...] = ("K562", "HepG2"),
    assembly: str = "GRCh38",
    limit: int = 500,
) -> list[dict]:
    """Query ENCODE REST API for eCLIP experiments.

    Returns a list of experiment metadata dicts with fields:
    accession, target_label, biosample_term_name, files (list of file dicts).
    """
    search_url = f"{ENCODE_API_BASE}/search/"

    # ENCODE API: frame=embedded expands nested objects (target, files, etc.)
    # Do NOT use 'fields' parameter — it causes 404. Use frame instead.
    params = {
        "type": "Experiment",
        "assay_title": "eCLIP",
        "status": "released",
        "limit": limit,
        "format": "json",
        "frame": "embedded",
    }

    logger.info("Querying ENCODE for eCLIP experiments (assembly=%s)...", assembly)
    resp = requests.get(search_url, params=params, headers={"Accept": "application/json"})
    resp.raise_for_status()

    data = resp.json()
    experiments = data.get("@graph", [])
    logger.info("Found %d eCLIP experiments", len(experiments))

    # Filter by cell line if specified
    results = []
    for exp in experiments:
        biosample = exp.get("biosample_ontology", {}).get("term_name", "")
        if cell_lines and biosample not in cell_lines:
            continue

        target = exp.get("target", {}).get("label", "unknown")
        accession = exp.get("accession", "")

        # Find replicate-merged narrowPeak BED files for the target assembly.
        # ENCODE eCLIP experiments produce 3 BED files per experiment:
        #   - bio_reps=[1]: replicate 1 peaks
        #   - bio_reps=[2]: replicate 2 peaks
        #   - bio_reps=[1,2]: IDR-merged replicate-combined peaks (the one we want)
        # We select the merged file (len(bio_reps) > 1) for highest confidence.
        merged_files = []
        for f in exp.get("files", []):
            if not isinstance(f, dict):
                continue
            bio_reps = f.get("biological_replicates", [])
            if (
                f.get("file_format") == "bed"
                and f.get("file_format_type") == "narrowPeak"
                and f.get("assembly") == assembly
                and f.get("status") == "released"
                and len(bio_reps) > 1  # IDR-merged replicate-combined file
            ):
                merged_files.append({
                    "file_accession": f.get("accession", ""),
                    "href": f.get("href", ""),
                })

        if merged_files:
            results.append({
                "accession": accession,
                "target_label": target,
                "biosample": biosample,
                "files": merged_files,
            })

    logger.info(
        "After cell line filter: %d experiments with IDR-filtered peaks",
        len(results),
    )
    return results


def download_narrowpeak(
    file_href: str,
    file_accession: str,
    cache_dir: Optional[Path] = None,
) -> Optional[pl.DataFrame]:
    """Download and parse a single narrowPeak BED file from ENCODE.

    Parameters
    ----------
    file_href : str
        ENCODE file href path (e.g., '/files/ENCFFXXXXXX/@@download/...').
    file_accession : str
        File accession for caching.
    cache_dir : Path or None
        If set, cache downloaded files locally.

    Returns
    -------
    pl.DataFrame or None
        Parsed narrowPeak data, or None on failure.
    """
    # Check cache
    if cache_dir is not None:
        cached = cache_dir / f"{file_accession}.bed.gz"
        if cached.exists():
            logger.debug("Using cached: %s", cached)
            return _parse_narrowpeak(cached)

    url = f"{ENCODE_API_BASE}{file_href}"
    logger.debug("Downloading %s ...", url)

    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to download %s: %s", file_accession, e)
        return None

    content = resp.content

    # Cache if requested
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / f"{file_accession}.bed.gz"
        cached.write_bytes(content)

    # Parse
    return _parse_narrowpeak_bytes(content, file_accession)


def _parse_narrowpeak(path: Path) -> Optional[pl.DataFrame]:
    """Parse a narrowPeak BED file from a local path."""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt") as f:
                text = f.read()
        else:
            text = path.read_text()
        return _parse_narrowpeak_text(text, path.name)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", path, e)
        return None


def _parse_narrowpeak_bytes(content: bytes, name: str) -> Optional[pl.DataFrame]:
    """Parse narrowPeak BED from raw bytes (possibly gzipped)."""
    try:
        try:
            text = gzip.decompress(content).decode("utf-8")
        except gzip.BadGzipFile:
            text = content.decode("utf-8")
        return _parse_narrowpeak_text(text, name)
    except Exception as e:
        logger.warning("Failed to parse %s: %s", name, e)
        return None


def _parse_narrowpeak_text(text: str, name: str) -> Optional[pl.DataFrame]:
    """Parse narrowPeak BED from text content."""
    try:
        df = pl.read_csv(
            io.StringIO(text),
            separator="\t",
            has_header=False,
            new_columns=NARROWPEAK_COLUMNS,
            schema_overrides={
                "chrom": pl.Utf8,
                "start": pl.Int64,
                "end": pl.Int64,
                "name": pl.Utf8,
                "score": pl.Int64,
                "strand": pl.Utf8,
                "signal_value": pl.Float64,
                "p_value": pl.Float64,
                "q_value": pl.Float64,
                "peak_offset": pl.Int64,
            },
        )
        return df
    except Exception as e:
        logger.warning("Failed to parse narrowPeak text from %s: %s", name, e)
        return None


def load_local_beds(
    input_dir: Path,
    metadata_path: Optional[Path] = None,
) -> Optional[pl.DataFrame]:
    """Load pre-downloaded narrowPeak BED files from a directory.

    Expects filenames to contain the RBP name and cell line, or a
    metadata TSV mapping filenames to (rbp, cell_line).

    Parameters
    ----------
    input_dir : Path
        Directory containing .bed or .bed.gz files.
    metadata_path : Path or None
        Optional TSV with columns: filename, rbp, cell_line.
        If None, attempts to parse from ENCODE filename convention.
    """
    bed_files = sorted(input_dir.glob("*.bed*"))
    if not bed_files:
        logger.error("No .bed files found in %s", input_dir)
        return None

    # Load metadata if available
    file_meta: dict[str, tuple[str, str]] = {}
    if metadata_path is not None and metadata_path.exists():
        meta = pl.read_csv(metadata_path, separator="\t")
        for row in meta.iter_rows(named=True):
            file_meta[row["filename"]] = (row["rbp"], row["cell_line"])

    all_peaks: list[pl.DataFrame] = []
    for bed_path in bed_files:
        df = _parse_narrowpeak(bed_path)
        if df is None or df.height == 0:
            continue

        # Resolve RBP and cell line
        if bed_path.name in file_meta:
            rbp, cell_line = file_meta[bed_path.name]
        else:
            # Fall back to filename parsing (e.g., SRSF1_K562.bed.gz)
            stem = bed_path.stem.replace(".bed", "").replace(".gz", "")
            parts = stem.split("_")
            if len(parts) >= 2:
                rbp, cell_line = parts[0], parts[1]
            else:
                logger.warning(
                    "Cannot determine RBP/cell_line from %s. Skipping.",
                    bed_path.name,
                )
                continue

        df = df.with_columns([
            pl.lit(rbp).alias("rbp"),
            pl.lit(cell_line).alias("cell_line"),
        ])
        all_peaks.append(df)

    if not all_peaks:
        return None

    return pl.concat(all_peaks)


def aggregate_peaks(
    experiments: list[dict],
    cache_dir: Optional[Path] = None,
    rate_limit: float = 0.5,
) -> Optional[pl.DataFrame]:
    """Download and aggregate all eCLIP narrowPeak files.

    Parameters
    ----------
    experiments : list of dict
        Experiment metadata from query_encode_eclip_experiments().
    cache_dir : Path or None
        Cache downloaded files for re-runs.
    rate_limit : float
        Seconds between ENCODE API requests (be polite).
    """
    all_peaks: list[pl.DataFrame] = []
    total_files = sum(len(exp["files"]) for exp in experiments)
    downloaded = 0

    for exp in experiments:
        rbp = exp["target_label"]
        cell_line = exp["biosample"]

        for f in exp["files"]:
            downloaded += 1
            logger.info(
                "[%d/%d] %s / %s — %s",
                downloaded, total_files, rbp, cell_line, f["file_accession"],
            )

            df = download_narrowpeak(
                f["href"], f["file_accession"], cache_dir
            )
            if df is None or df.height == 0:
                continue

            df = df.with_columns([
                pl.lit(rbp).alias("rbp"),
                pl.lit(cell_line).alias("cell_line"),
            ])
            all_peaks.append(df)

            time.sleep(rate_limit)

    if not all_peaks:
        logger.error("No peaks downloaded successfully")
        return None

    combined = pl.concat(all_peaks)
    logger.info("Combined: %d peaks from %d files", combined.height, len(all_peaks))
    return combined


def postprocess_peaks(df: pl.DataFrame) -> pl.DataFrame:
    """Clean and enrich the combined peaks DataFrame.

    - Filter to canonical chromosomes
    - Compute neg_log10_pvalue
    - Select output columns
    """
    initial = df.height

    # Filter to canonical chromosomes
    df = df.filter(pl.col("chrom").is_in(list(CANONICAL_CHROMS)))
    logger.info(
        "Canonical chromosome filter: %d → %d peaks",
        initial, df.height,
    )

    # The narrowPeak BED format stores column 8 (p_value) as ALREADY
    # -log10 transformed per the UCSC spec. Use the value directly.
    # See: https://genome.ucsc.edu/FAQ/FAQformat.html#format12
    df = df.with_columns(
        pl.col("p_value").alias("neg_log10_pvalue")
    )

    # Select output columns
    df = df.select([
        "chrom", "start", "end", "rbp", "cell_line",
        "signal_value", "neg_log10_pvalue", "strand",
    ])

    return df


def print_summary(df: pl.DataFrame) -> None:
    """Print summary statistics of the aggregated peaks."""
    print("\n" + "=" * 60)
    print("eCLIP Peaks Summary")
    print("=" * 60)
    print(f"Total peaks:       {df.height:>10,}")
    print(f"Unique RBPs:       {df['rbp'].n_unique():>10,}")
    print(f"Unique cell lines: {df['cell_line'].n_unique():>10,}")
    print(f"Chromosomes:       {df['chrom'].n_unique():>10,}")
    print()

    # Per cell line
    print("Peaks per cell line:")
    cl_counts = df.group_by("cell_line").agg(
        pl.len().alias("n_peaks"),
        pl.col("rbp").n_unique().alias("n_rbps"),
    ).sort("n_peaks", descending=True)
    for row in cl_counts.iter_rows(named=True):
        print(f"  {row['cell_line']:>10s}: {row['n_peaks']:>8,} peaks, "
              f"{row['n_rbps']:>3d} RBPs")

    # Per chromosome
    print("\nPeaks per chromosome (top 10):")
    chrom_counts = df.group_by("chrom").agg(
        pl.len().alias("n_peaks")
    ).sort("n_peaks", descending=True)
    for row in chrom_counts.head(10).iter_rows(named=True):
        print(f"  {row['chrom']:>6s}: {row['n_peaks']:>8,}")

    # Signal distribution
    print(f"\nSignal value (fold-enrichment):")
    print(f"  Mean:   {df['signal_value'].mean():>8.2f}")
    print(f"  Median: {df['signal_value'].median():>8.2f}")
    print(f"  Max:    {df['signal_value'].max():>8.2f}")

    print(f"\n-log10(pValue):")
    print(f"  Mean:   {df['neg_log10_pvalue'].mean():>8.2f}")
    print(f"  Median: {df['neg_log10_pvalue'].median():>8.2f}")

    # Peak width distribution
    widths = (df["end"] - df["start"]).to_numpy()
    print(f"\nPeak width (bp):")
    print(f"  Mean:   {widths.mean():>8.1f}")
    print(f"  Median: {float(sorted(widths)[len(widths)//2]):>8.1f}")
    print(f"  Min:    {widths.min():>8d}")
    print(f"  Max:    {widths.max():>8d}")

    # Top RBPs by peak count
    print("\nTop 20 RBPs by peak count:")
    rbp_counts = df.group_by("rbp").agg(
        pl.len().alias("n_peaks")
    ).sort("n_peaks", descending=True)
    for row in rbp_counts.head(20).iter_rows(named=True):
        print(f"  {row['rbp']:>15s}: {row['n_peaks']:>8,}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and aggregate ENCODE eCLIP narrowPeak data."
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/mane/GRCh38/rbp_data"),
        help="Output directory for eclip_peaks.parquet",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory of pre-downloaded BED files (skip ENCODE API)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="TSV mapping filename→(rbp, cell_line) for local BED files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache downloaded BED files locally for re-runs",
    )
    parser.add_argument(
        "--cell-lines",
        nargs="+",
        default=["K562", "HepG2"],
        help="Cell lines to include (default: K562 HepG2)",
    )
    parser.add_argument(
        "--assembly",
        default="GRCh38",
        help="Genome assembly (default: GRCh38)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Seconds between ENCODE downloads (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query ENCODE API but don't download files",
    )
    args = parser.parse_args()

    if args.input_dir is not None:
        # Local mode: load from pre-downloaded files
        logger.info("Loading BED files from %s", args.input_dir)
        combined = load_local_beds(args.input_dir, args.metadata)
    else:
        # ENCODE API mode
        experiments = query_encode_eclip_experiments(
            cell_lines=tuple(args.cell_lines),
            assembly=args.assembly,
        )

        if args.dry_run:
            print(f"\nDry run: {len(experiments)} experiments found")
            for exp in experiments[:20]:
                print(f"  {exp['target_label']:>15s} / {exp['biosample']:>8s} "
                      f"({len(exp['files'])} files) — {exp['accession']}")
            if len(experiments) > 20:
                print(f"  ... and {len(experiments) - 20} more")
            return

        combined = aggregate_peaks(
            experiments,
            cache_dir=args.cache_dir,
            rate_limit=args.rate_limit,
        )

    if combined is None:
        logger.error("No peaks to aggregate")
        return

    # Post-process
    df = postprocess_peaks(combined)

    # Summary
    print_summary(df)

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / OUTPUT_FILENAME
    df.write_parquet(output_path)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("Saved %d peaks to %s (%.1f MB)", df.height, output_path, size_mb)


if __name__ == "__main__":
    main()
