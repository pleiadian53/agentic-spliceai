#!/usr/bin/env python
"""Generate genome-wide ground truth splice site annotations for model evaluation.

This script produces the splice_sites_enhanced.tsv file used as ground truth when
evaluating base model predictions (SpliceAI, OpenSpliceAI, etc.). It extracts ALL
splice sites from the GTF annotation — no gene filtering.

The same result can be achieved via the CLI:
    agentic-spliceai-prepare --output <dir> --splice-sites-only

Or via the full pipeline script:
    python 03_full_data_pipeline.py --output <dir> --skip-sequences

Usage:
    # MANE / GRCh38 (curated, ~370K sites, ~19K genes)
    python 04_generate_ground_truth.py --output data/mane/GRCh38/

    # Ensembl / GRCh37 (comprehensive, ~2M sites, all transcripts)
    # (annotation_source and build are inferred from the output path)
    python 04_generate_ground_truth.py --output data/ensembl/GRCh37/

    # Arbitrary GTF (e.g., T2T-CHM13, pangenome, non-model organism)
    python 04_generate_ground_truth.py \\
        --gtf /path/to/chm13v2.0_RefSeq_Curated.gtf \\
        --output data/t2t_chm13/

    # Force re-extraction (overwrite cached file)
    python 04_generate_ground_truth.py --output data/mane/GRCh38/ --force
"""

import argparse
import sys
import time
from pathlib import Path
import polars as pl

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_splice_site_annotations,
)


def infer_from_output_path(output_path: Path) -> dict:
    """Infer annotation_source and build from output path components.

    Recognizes patterns like ``data/mane/GRCh38/`` or ``data/ensembl/GRCh37/``.

    Returns
    -------
    dict
        Keys ``annotation_source`` and ``build`` with inferred values,
        or empty dict if nothing could be inferred.
    """
    known_sources = {"mane", "ensembl", "gencode", "refseq", "t2t_chm13", "pangenome"}
    # Map lowercase → canonical mixed-case build names
    known_builds = {
        "grch37": "GRCh37", "grch38": "GRCh38",
        "t2t_chm13": "T2T-CHM13", "chm13": "T2T-CHM13",
        "t2t-chm13": "T2T-CHM13",
    }

    parts = [p.lower() for p in output_path.resolve().parts]
    result = {}
    for part in parts:
        if part in known_sources and "annotation_source" not in result:
            result["annotation_source"] = part
        if part in known_builds and "build" not in result:
            result["build"] = known_builds[part]
    return result


def main():
    """Generate genome-wide ground truth annotations."""
    parser = argparse.ArgumentParser(
        description="Generate genome-wide splice site ground truth for evaluation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (e.g., data/mane/GRCh38/)"
    )
    parser.add_argument(
        "--gtf",
        type=Path,
        default=None,
        help="Path to GTF annotation file. Bypasses registry lookup, "
             "enabling any genome build (T2T-CHM13, pangenome, non-model "
             "organisms). When provided, --build and --annotation-source "
             "are used only for labeling, not path resolution.",
    )
    parser.add_argument(
        "--build",
        default=None,
        help="Genome build (default: inferred from --output, else GRCh38). "
             "With --gtf, used only for metadata labeling.",
    )
    parser.add_argument(
        "--annotation-source",
        default=None,
        help="Annotation source (default: inferred from --output, else mane). "
             "With --gtf, can be any string (e.g., 't2t_chm13', 'pangenome').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cached file exists"
    )
    args = parser.parse_args()

    # Infer annotation_source / build from output path when not explicitly provided
    inferred = infer_from_output_path(args.output)

    if args.annotation_source is None:
        args.annotation_source = inferred.get("annotation_source", "mane")
        if "annotation_source" in inferred:
            print(f"(inferred --annotation-source={args.annotation_source} from output path)")

    if args.build is None:
        args.build = inferred.get("build", "GRCh38")
        if "build" in inferred:
            print(f"(inferred --build={args.build} from output path)")

    # Warn on mismatch between explicit args and path
    if inferred:
        if "annotation_source" in inferred and inferred["annotation_source"] != args.annotation_source:
            print(f"⚠️  Output path suggests source='{inferred['annotation_source']}' "
                  f"but --annotation-source={args.annotation_source}")
        if "build" in inferred and inferred["build"] != args.build:
            print(f"⚠️  Output path suggests build='{inferred['build']}' "
                  f"but --build={args.build}")

    # Validate --gtf if provided
    if args.gtf and not args.gtf.exists():
        print(f"ERROR: GTF file not found: {args.gtf}")
        return 1

    print("=" * 80)
    print("Generate Ground Truth: splice_sites_enhanced.tsv")
    print("=" * 80)
    print(f"\nBuild: {args.build}")
    print(f"Annotation Source: {args.annotation_source}")
    if args.gtf:
        print(f"GTF (custom): {args.gtf}")
    print(f"Output: {args.output}")
    print(f"Force: {args.force}")
    print()

    args.output.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    result = prepare_splice_site_annotations(
        output_dir=args.output,
        genes=None,           # No filter — genome-wide
        build=args.build,
        annotation_source=args.annotation_source,
        gtf_path=str(args.gtf) if args.gtf else None,
        force_extract=args.force,
        verbosity=2
    )
    elapsed = time.time() - t0

    if result['success']:
        print(f"\n{'=' * 80}")
        print(f"✅ Ground truth generated in {elapsed:.1f}s")
        print(f"{'=' * 80}")
        print(f"\n   File: {result['splice_sites_file']}")
        print(f"   Total sites: {result['n_sites']:,}")
        print(f"   Donors: {result['n_donors']:,}")
        print(f"   Acceptors: {result['n_acceptors']:,}")

        # Show per-chromosome breakdown
        df = result['splice_sites_df']
        chrom_counts = (
            df.group_by('chrom')
            .agg(pl.len().alias('count'))
            .sort('count', descending=True)
        )
        print(f"\n   Top chromosomes:")
        for row in chrom_counts.head(5).iter_rows(named=True):
            print(f"     {row['chrom']}: {row['count']:,} sites")
        print(f"     ... ({chrom_counts.height} chromosomes total)")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown')}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
