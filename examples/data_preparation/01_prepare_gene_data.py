#!/usr/bin/env python
"""Phase 2 Example: Gene Data Preparation.

Demonstrates gene annotation loading and sequence extraction:
1. Load gene annotations from GTF
2. Extract DNA sequences from FASTA
3. Save prepared data to files

Usage:
    python 01_prepare_gene_data.py --genes BRCA1 TP53 --output /tmp/gene_data/
    python 01_prepare_gene_data.py --genes BRCA1 --annotation-source mane

Example:
    python 01_prepare_gene_data.py --genes BRCA1 TP53 EGFR --output /tmp/gene_data/
"""

import argparse
import sys
from pathlib import Path

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.data.preparation import prepare_gene_data
import polars as pl


def main():
    """Run gene data preparation example."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Prepare gene annotations and sequences"
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
    args = parser.parse_args()
    
    print("=" * 80)
    print("Phase 2 Example: Gene Data Preparation")
    print("=" * 80)
    print(f"\nGenes: {', '.join(args.genes)}")
    print(f"Output: {args.output}")
    print(f"Build: {args.build}")
    print(f"Annotation Source: {args.annotation_source}")
    print()
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Prepare gene data
    print("🧬 Preparing gene data...")
    result = prepare_gene_data(
        genes=args.genes,
        build=args.build,
        annotation_source=args.annotation_source
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    
    print(f"\n✅ Loaded {len(result['genes'])} gene annotations")
    print(f"✅ Extracted {len(result['sequences'])} sequences")
    
    # Show gene details
    print(f"\n📋 Gene details:")
    for _, row in result['genes'].iter_rows(named=True):
        seq_status = "✓" if row['gene_name'] in result['sequences']['gene_name'].to_list() else "✗"
        print(f"   {seq_status} {row['gene_name']}: {row['seqname']}:{row['start']:,}-{row['end']:,} ({row['strand']})")
        if row['gene_name'] in result['sequences']['gene_name'].to_list():
            seq_row = result['sequences'].filter(pl.col('gene_name') == row['gene_name']).row(0, named=True)
            print(f"      Sequence length: {len(seq_row['sequence']):,} bp")
    
    # Save to files
    genes_file = args.output / "genes.tsv"
    sequences_file = args.output / "sequences.tsv"
    
    print(f"\n💾 Saving results...")
    result['genes'].write_csv(genes_file, separator='\t')
    result['sequences'].write_csv(sequences_file, separator='\t')
    
    print(f"   ✓ Genes: {genes_file}")
    print(f"   ✓ Sequences: {sequences_file}")
    
    print("\n" + "=" * 80)
    print("✅ Example complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
