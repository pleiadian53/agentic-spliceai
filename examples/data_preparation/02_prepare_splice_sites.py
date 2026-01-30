#!/usr/bin/env python
"""Phase 2 Example: Splice Site Annotation Extraction.

Demonstrates splice site annotation extraction from GTF:
1. Parse GTF file for exon annotations
2. Derive splice sites from exon boundaries
3. Infer metadata (biotype, exon numbers)
4. Filter to specific genes
5. Save annotated splice sites

Usage:
    python 02_prepare_splice_sites.py --genes BRCA1 --output /tmp/splice_sites/
    python 02_prepare_splice_sites.py --genes BRCA1 TP53 --annotation-source mane

Example:
    python 02_prepare_splice_sites.py --genes BRCA1 --output /tmp/splice_sites/
"""

import argparse
import sys
from pathlib import Path

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.data.preparation import (
    prepare_splice_site_annotations
)


def main():
    """Run splice site annotation preparation example."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Extract and annotate splice sites"
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        required=True,
        help="Gene symbols (e.g., BRCA1 TP53)"
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
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 2 Example: Splice Site Annotation Extraction")
    print("=" * 80)
    print(f"\nGenes: {', '.join(args.genes)}")
    print(f"Output: {args.output}")
    print(f"Build: {args.build}")
    print(f"Annotation Source: {args.annotation_source}")
    print()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Extract splice sites
    print("🧬 Extracting splice site annotations...")
    result = prepare_splice_site_annotations(
        output_dir=args.output,
        genes=args.genes,
        build=args.build,
        annotation_source=args.annotation_source,
        verbosity=2
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    if result['success']:
        splice_sites = result['splice_sites_df']
        
        print(f"\n✅ Extracted {len(splice_sites):,} splice sites")
        
        # Count by type
        donors = splice_sites.filter(pl.col('splice_site') == 'donor')
        acceptors = splice_sites.filter(pl.col('splice_site') == 'acceptor')
        
        print(f"   - Donors: {len(donors):,}")
        print(f"   - Acceptors: {len(acceptors):,}")
        
        # Count by gene
        print(f"\n📊 Splice sites by gene:")
        for gene in sorted(args.genes):
            gene_sites = splice_sites.filter(pl.col('gene_name') == gene)
            if len(gene_sites) > 0:
                gene_donors = gene_sites.filter(pl.col('splice_site') == 'donor')
                gene_acceptors = gene_sites.filter(pl.col('splice_site') == 'acceptor')
                print(f"   - {gene}: {len(gene_sites):,} total ({len(gene_donors):,} donors, {len(gene_acceptors):,} acceptors)")
        
        # Show sample
        print(f"\n📋 Sample splice sites (first 5):")
        print(splice_sites.head(5))
        
        # Show metadata coverage
        print(f"\n🏷️  Metadata coverage:")
        print(f"   - gene_biotype: {splice_sites['gene_biotype'].n_unique()} unique values")
        print(f"   - transcript_biotype: {splice_sites['transcript_biotype'].n_unique()} unique values")
        print(f"   - exon_id: {splice_sites.filter(pl.col('exon_id') != '').height} annotated")
        print(f"   - exon_number: {splice_sites.filter(pl.col('exon_number') > 0).height} annotated")
        
        print(f"\n💾 Saved to: {result['output_file']}")
    else:
        print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
        return 1
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import polars as pl
    sys.exit(main())
