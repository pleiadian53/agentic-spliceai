#!/usr/bin/env python
"""Download and filter ClinVar VCF for splice variant benchmarking.

Downloads the ClinVar VCF (GRCh38) from NCBI FTP, filters to
splice-relevant SNVs, and saves a filtered Parquet for fast reloading.

Filtering strategy:
  1. SNVs only (indels deferred)
  2. Pathogenic or Benign classification (skip VUS)
  3. Optionally: near annotated splice sites (±50bp)
  4. Optionally: ClinVar molecular consequence includes "splice"

Usage:
    # Download and filter
    python 02_clinvar_download.py --output-dir data/clinvar/

    # Filter existing VCF (skip download)
    python 02_clinvar_download.py \
        --vcf data/clinvar/clinvar_20260401.vcf.gz \
        --output-dir data/clinvar/

    # With splice site proximity filter
    python 02_clinvar_download.py \
        --output-dir data/clinvar/ \
        --splice-proximity 50
"""

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

log = logging.getLogger(__name__)

# ClinVar FTP URL (GRCh38)
CLINVAR_FTP = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
CLINVAR_TBI = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz.tbi"


def download_clinvar(output_dir: Path) -> Path:
    """Download ClinVar VCF from NCBI FTP.

    Returns path to the downloaded VCF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / "clinvar.vcf.gz"
    tbi_path = output_dir / "clinvar.vcf.gz.tbi"

    if vcf_path.exists():
        print(f"  ClinVar VCF already exists: {vcf_path}")
        return vcf_path

    print(f"  Downloading ClinVar VCF from {CLINVAR_FTP}...")
    urllib.request.urlretrieve(CLINVAR_FTP, str(vcf_path))
    print(f"  Downloaded: {vcf_path} ({vcf_path.stat().st_size / 1e6:.1f} MB)")

    print(f"  Downloading index...")
    urllib.request.urlretrieve(CLINVAR_TBI, str(tbi_path))
    print(f"  Downloaded: {tbi_path}")

    return vcf_path


def filter_clinvar(
    vcf_path: Path,
    output_dir: Path,
    min_stars: int = 1,
    splice_proximity: int = 0,
    splice_only_mc: bool = False,
) -> None:
    """Filter ClinVar VCF to splice-relevant SNVs and save as Parquet.

    Parameters
    ----------
    vcf_path : Path
        Path to ClinVar VCF (or .vcf.gz).
    output_dir : Path
        Output directory for filtered Parquet.
    min_stars : int
        Minimum review star count.
    splice_proximity : int
        If > 0, only keep variants within this distance of annotated
        splice sites.
    splice_only_mc : bool
        If True, only keep variants with splice-related molecular
        consequence in ClinVar.
    """
    from agentic_spliceai.splice_engine.meta_layer.data.clinvar_loader import (
        ClinVarLoader,
    )
    import polars as pl

    print(f"\n  Loading ClinVar VCF: {vcf_path}")
    loader = ClinVarLoader(vcf_path)
    stats = loader.get_statistics()

    print(f"  Total records:      {stats['total']:,}")
    print(f"  SNVs:               {stats['snvs']:,}")
    print(f"  Pathogenic:         {stats['n_pathogenic']:,}")
    print(f"  Benign:             {stats['n_benign']:,}")
    print(f"  VUS:                {stats['n_vus']:,}")
    print(f"  Pathogenic SNVs:    {stats['n_pathogenic_snvs']:,}")
    print(f"  Benign SNVs:        {stats['n_benign_snvs']:,}")
    print(f"  Genes:              {stats['n_genes']:,}")
    print(f"  Stars distribution: {stats['stars_distribution']}")

    # Get splice-relevant variants
    splice_site_positions = None
    if splice_proximity > 0:
        print(f"\n  Loading splice site positions for proximity filter (±{splice_proximity}bp)...")
        splice_site_positions = _load_splice_site_positions()
        print(f"  Loaded {len(splice_site_positions):,} splice site positions")

    records = loader.get_splice_relevant(
        splice_site_positions=splice_site_positions,
        max_distance_to_splice=splice_proximity if splice_proximity > 0 else 50,
        min_stars=min_stars,
        snvs_only=True,
    )

    # Optionally filter by molecular consequence
    if splice_only_mc:
        records = [r for r in records if r.is_splice_annotated]
        print(f"  After MC splice filter: {len(records):,}")

    print(f"\n  Filtered splice-relevant variants: {len(records):,}")
    n_path = sum(1 for r in records if r.is_pathogenic)
    n_ben = sum(1 for r in records if r.is_benign)
    print(f"    Pathogenic: {n_path:,}")
    print(f"    Benign:     {n_ben:,}")

    # Save as Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "clinvar_splice_snvs.parquet"

    rows = []
    for r in records:
        rows.append({
            "clinvar_id": r.clinvar_id,
            "chrom": r.chrom,
            "position": r.position,
            "ref_allele": r.ref_allele,
            "alt_allele": r.alt_allele,
            "gene": r.gene,
            "classification": r.classification,
            "review_stars": r.review_stars,
            "disease": r.disease,
            "molecular_consequence": r.molecular_consequence,
            "variant_type": r.variant_type,
        })

    df = pl.DataFrame(rows)
    df.write_parquet(str(parquet_path))
    print(f"\n  Saved: {parquet_path} ({len(df):,} variants)")

    # Also save stats
    import json
    stats_path = output_dir / "clinvar_filter_stats.json"
    filter_stats = {
        "source": str(vcf_path),
        "total_records": stats["total"],
        "filtered_splice_relevant": len(records),
        "n_pathogenic": n_path,
        "n_benign": n_ben,
        "min_stars": min_stars,
        "splice_proximity": splice_proximity,
        "splice_only_mc": splice_only_mc,
    }
    with open(stats_path, "w") as f:
        json.dump(filter_stats, f, indent=2)
    print(f"  Stats: {stats_path}")


def _load_splice_site_positions() -> set:
    """Load MANE splice site positions as a set of 'chrom:position' keys."""
    import polars as pl
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources("openspliceai")
    registry = resources.get_registry()
    ss_path = registry.resolve("splice_sites")

    if not ss_path:
        log.warning("Could not resolve splice sites path")
        return set()

    df = pl.read_csv(ss_path, separator="\t", columns=["chrom", "position"])
    positions = set()
    for row in df.iter_rows():
        chrom, pos = row
        positions.add(f"{chrom}:{pos}")
        # Also add the alternate chr format
        bare = chrom.replace("chr", "") if chrom.startswith("chr") else f"chr{chrom}"
        positions.add(f"{bare}:{pos}")
    return positions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and filter ClinVar VCF for splice benchmarking",
    )
    parser.add_argument(
        "--vcf", type=Path, default=None,
        help="Existing ClinVar VCF. If not provided, downloads from NCBI.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/clinvar/"),
        help="Output directory (default: data/clinvar/)",
    )
    parser.add_argument(
        "--min-stars", type=int, default=1,
        help="Minimum ClinVar review stars (default: 1)",
    )
    parser.add_argument(
        "--splice-proximity", type=int, default=0,
        help="If > 0, filter to variants within this distance of "
             "annotated splice sites (default: 0 = no filter)",
    )
    parser.add_argument(
        "--splice-only-mc", action="store_true",
        help="Only keep variants with splice-related molecular consequence",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("=" * 60)
    print("ClinVar Download & Filter for Splice Benchmarking")
    print("=" * 60)

    # Download or use existing VCF
    if args.vcf:
        vcf_path = args.vcf
        if not vcf_path.exists():
            print(f"ERROR: VCF not found: {vcf_path}")
            return 1
    else:
        vcf_path = download_clinvar(args.output_dir)

    # Filter and save
    filter_clinvar(
        vcf_path,
        args.output_dir,
        min_stars=args.min_stars,
        splice_proximity=args.splice_proximity,
        splice_only_mc=args.splice_only_mc,
    )

    print(f"\n{'='*60}")
    print("Done.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
