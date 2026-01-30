#!/usr/bin/env python
"""Phase 2 Example: Complete Data Preparation Pipeline.

Demonstrates the complete Phase 2 data preparation workflow:
1. Extract gene annotations
2. Extract DNA sequences
3. Extract splice site annotations
4. Save all data in organized format

This is equivalent to running the CLI:
    agentic-spliceai-base-prepare --genes <genes> --output <output>

Usage:
    python 03_full_data_pipeline.py --genes BRCA1 TP53 EGFR --output /tmp/full_pipeline/
    python 03_full_data_pipeline.py --genes BRCA1 --annotation-source mane --skip-sequences

Example:
    python 03_full_data_pipeline.py --genes BRCA1 TP53 --output /tmp/full_pipeline/
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_gene_data,
    prepare_splice_site_annotations,
)


def main():
    """Run complete data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Complete data preparation pipeline"
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="Gene symbols (e.g., BRCA1 TP53 EGFR)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory"
    )
    parser.add_argument(
        "--build",
        default="GRCh38",
        help="Genome build (default: GRCh38)"
    )
    parser.add_argument(
        "--annotation-source",
        default="mane",
        choices=["mane", "ensembl"],
        help="Annotation source (default: mane)"
    )
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip sequence extraction (faster, but incomplete)"
    )
    parser.add_argument(
        "--skip-splice-sites",
        action="store_true",
        help="Skip splice site extraction"
    )
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 2 Complete Data Preparation Pipeline")
    print("=" * 80)
    print(f"\nGenes: {', '.join(args.genes)}")
    print(f"Output: {args.output}")
    print(f"Build: {args.build}")
    print(f"Annotation Source: {args.annotation_source}")
    print(f"Skip Sequences: {args.skip_sequences}")
    print(f"Skip Splice Sites: {args.skip_splice_sites}")
    print()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Initialize summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "genes": args.genes,
        "build": args.build,
        "annotation_source": args.annotation_source,
        "results": {}
    }
    
    # Step 1: Prepare gene data
    print("=" * 80)
    print("Step 1/3: Preparing Gene Data")
    print("=" * 80)
    print()
    
    gene_result = prepare_gene_data(
        genes=args.genes,
        build=args.build,
        annotation_source=args.annotation_source
    )
    
    print(f"✅ Loaded {len(gene_result['genes'])} gene annotations")
    print(f"✅ Extracted {len(gene_result['sequences'])} sequences")
    
    # Save gene data
    genes_file = args.output / "genes.tsv"
    sequences_file = args.output / "sequences.tsv"
    
    gene_result['genes'].write_csv(genes_file, separator='\t')
    if not args.skip_sequences:
        gene_result['sequences'].write_csv(sequences_file, separator='\t')
    
    summary["results"]["genes"] = {
        "count": len(gene_result['genes']),
        "file": str(genes_file),
    }
    
    if not args.skip_sequences:
        summary["results"]["sequences"] = {
            "count": len(gene_result['sequences']),
            "file": str(sequences_file),
        }
    
    # Step 2: Prepare splice site annotations
    if not args.skip_splice_sites:
        print("\n" + "=" * 80)
        print("Step 2/3: Preparing Splice Site Annotations")
        print("=" * 80)
        print()
        
        splice_result = prepare_splice_site_annotations(
            output_dir=args.output,
            genes=args.genes,
            build=args.build,
            annotation_source=args.annotation_source,
            verbosity=1
        )
        
        if splice_result['success']:
            splice_sites = splice_result['splice_sites_df']
            donors = splice_sites.filter(pl.col('splice_site') == 'donor')
            acceptors = splice_sites.filter(pl.col('splice_site') == 'acceptor')
            
            print(f"✅ Extracted {len(splice_sites):,} splice sites")
            print(f"   - Donors: {len(donors):,}")
            print(f"   - Acceptors: {len(acceptors):,}")
            
            summary["results"]["splice_sites"] = {
                "total": len(splice_sites),
                "donors": len(donors),
                "acceptors": len(acceptors),
                "file": str(splice_result['output_file']),
            }
        else:
            print(f"❌ Splice site extraction failed: {splice_result.get('error', 'Unknown')}")
            summary["results"]["splice_sites"] = {"error": splice_result.get('error', 'Unknown')}
    
    # Step 3: Save summary
    print("\n" + "=" * 80)
    print("Step 3/3: Saving Summary")
    print("=" * 80)
    print()
    
    summary_file = args.output / "preparation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved to: {summary_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ Complete Data Preparation Pipeline Finished!")
    print("=" * 80)
    
    print(f"\nFiles created:")
    print(f"  - Genes: {genes_file}")
    if not args.skip_sequences:
        print(f"  - Sequences: {sequences_file}")
    if not args.skip_splice_sites and splice_result['success']:
        print(f"  - Splice sites: {splice_result['output_file']}")
    print(f"  - Summary: {summary_file}")
    
    print(f"\nStatistics:")
    print(f"  - Genes: {len(gene_result['genes'])} ({len(gene_result['sequences'])} with sequences)")
    if not args.skip_splice_sites and splice_result['success']:
        print(f"  - Splice sites: {len(splice_sites):,} ({len(donors):,} donors, {len(acceptors):,} acceptors)")
    
    return 0


if __name__ == "__main__":
    import polars as pl
    sys.exit(main())
