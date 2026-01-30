"""CLI for data preparation - extract genomic features for splice site prediction.

This module provides command-line access to the data preparation capabilities,
allowing users to extract gene annotations, sequences, and splice site annotations
from GTF/FASTA files into ML-friendly formats.

Usage:
    agentic-spliceai-prepare --genes BRCA1 TP53 --output data/prepared/
    agentic-spliceai-prepare --chromosomes 21 --output data/prepared/
    agentic-spliceai-prepare --build GRCh38 --output data/prepared/
"""

import argparse
import sys
from typing import Optional, List
from pathlib import Path
import json
from datetime import datetime


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the data preparation CLI."""
    parser = argparse.ArgumentParser(
        description="Prepare genomic data for splice site prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract data for specific genes
  agentic-spliceai-prepare --genes BRCA1 TP53 --output data/prepared/
  
  # Extract data for chromosome
  agentic-spliceai-prepare --chromosomes 21 --output data/prepared/
  
  # Extract all data for a build
  agentic-spliceai-prepare --build GRCh38 --output data/ensembl/GRCh38/
  
  # Use custom GTF/FASTA files
  agentic-spliceai-prepare --genes BRCA1 \\
    --gtf /path/to/annotations.gtf \\
    --fasta /path/to/genome.fa \\
    --output data/prepared/
  
  # Extract only splice sites
  agentic-spliceai-prepare --build GRCh38 \\
    --output data/ensembl/GRCh38/ \\
    --splice-sites-only
  
  # Force re-extraction even if files exist
  agentic-spliceai-prepare --genes BRCA1 \\
    --output data/prepared/ \\
    --force
        """
    )
    
    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--genes",
        nargs="+",
        help="Gene symbols or IDs to extract (e.g., BRCA1 TP53)"
    )
    target_group.add_argument(
        "--chromosomes",
        nargs="+",
        help="Chromosomes to extract (e.g., 21 22 X Y)"
    )
    
    # Build and source
    parser.add_argument(
        "--build",
        default="GRCh38",
        help="Genome build (default: GRCh38). Options: GRCh38, GRCh37, GRCh38_MANE"
    )
    parser.add_argument(
        "--annotation-source",
        default="mane",
        help="Annotation source (default: mane). Options: mane, ensembl, gencode"
    )
    
    # Custom paths
    parser.add_argument(
        "--gtf",
        help="Custom path to GTF file (overrides build/source)"
    )
    parser.add_argument(
        "--fasta",
        help="Custom path to FASTA file (overrides build/source)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for extracted data"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if files exist"
    )
    
    # Content selection
    parser.add_argument(
        "--splice-sites-only",
        action="store_true",
        help="Extract only splice sites (skip genes and sequences)"
    )
    parser.add_argument(
        "--skip-sequences",
        action="store_true",
        help="Skip sequence extraction (faster, genes only)"
    )
    parser.add_argument(
        "--skip-splice-sites",
        action="store_true",
        help="Skip splice site extraction"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        default="tsv",
        choices=["tsv", "parquet", "both"],
        help="Output format (default: tsv)"
    )
    
    # Control
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Output verbosity: 0=minimal, 1=normal, 2=detailed"
    )
    
    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point for the data preparation CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        from agentic_spliceai.splice_engine.base_layer.data import (
            prepare_gene_data,
            prepare_splice_site_annotations
        )
        from agentic_spliceai.splice_engine.config import get_project_root
        
        # Resolve output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.verbosity >= 1:
            print("=" * 70)
            print("🧬 Genomic Data Preparation")
            print("=" * 70)
            print(f"\nBuild: {args.build}")
            print(f"Source: {args.annotation_source}")
            if args.genes:
                print(f"Genes: {', '.join(args.genes)}")
            elif args.chromosomes:
                print(f"Chromosomes: {', '.join(args.chromosomes)}")
            else:
                print("Target: All (full build)")
            print(f"Output: {output_dir}")
            print()
        
        # Track what we extracted
        extracted = {
            'timestamp': datetime.now().isoformat(),
            'build': args.build,
            'annotation_source': args.annotation_source,
            'genes': args.genes,
            'chromosomes': args.chromosomes,
            'files': {},
            'stats': {}
        }
        
        # Extract splice sites
        if not args.skip_splice_sites:
            if args.verbosity >= 1:
                print("=" * 70)
                print("📍 Extracting Splice Sites")
                print("=" * 70)
                print()
            
            splice_result = prepare_splice_site_annotations(
                output_dir=output_dir,
                genes=args.genes,
                chromosomes=args.chromosomes,
                build=args.build,
                annotation_source=args.annotation_source,
                gtf_path=args.gtf,
                force_extract=args.force,
                verbosity=args.verbosity
            )
            
            if splice_result['success']:
                extracted['files']['splice_sites'] = splice_result['splice_sites_file']
                extracted['stats']['splice_sites'] = {
                    'total': splice_result['n_sites'],
                    'donors': splice_result['n_donors'],
                    'acceptors': splice_result['n_acceptors']
                }
                
                # Convert to parquet if requested
                if args.format in ['parquet', 'both']:
                    import polars as pl
                    parquet_file = output_dir / 'splice_sites_enhanced.parquet'
                    splice_result['splice_sites_df'].write_parquet(parquet_file)
                    extracted['files']['splice_sites_parquet'] = str(parquet_file)
                    if args.verbosity >= 1:
                        print(f"✓ Saved Parquet: {parquet_file}")
            else:
                print(f"❌ Failed to extract splice sites: {splice_result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        # Exit early if only extracting splice sites
        if args.splice_sites_only:
            if args.verbosity >= 1:
                print("\n" + "=" * 70)
                print("✓ Splice sites extraction complete")
                print("=" * 70)
            
            # Save summary
            summary_file = output_dir / 'preparation_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(extracted, f, indent=2)
            
            if args.verbosity >= 1:
                print(f"\nSummary saved to: {summary_file}")
            
            return
        
        # Extract genes and sequences
        if args.verbosity >= 1:
            print("\n" + "=" * 70)
            print("🧬 Extracting Genes and Sequences")
            print("=" * 70)
            print()
        
        gene_df = prepare_gene_data(
            genes=args.genes,
            chromosomes=args.chromosomes,
            build=args.build,
            annotation_source=args.annotation_source,
            gtf_path=args.gtf,
            fasta_path=args.fasta,
            verbosity=args.verbosity
        )
        
        if gene_df.height == 0:
            print("❌ No genes found matching criteria")
            sys.exit(1)
        
        # Save gene annotations (without sequences)
        genes_file = output_dir / 'genes.tsv'
        gene_cols = [c for c in gene_df.columns if c != 'sequence']
        gene_df.select(gene_cols).write_csv(genes_file, separator='\t')
        extracted['files']['genes'] = str(genes_file)
        extracted['stats']['genes'] = {
            'total': gene_df.height,
            'with_sequences': gene_df.filter(gene_df['sequence'].is_not_null()).height
        }
        
        if args.verbosity >= 1:
            print(f"✓ Saved genes: {genes_file}")
        
        # Save sequences if not skipped
        if not args.skip_sequences:
            sequences_file = output_dir / 'sequences.tsv'
            seq_cols = ['gene_id', 'gene_name', 'seqname', 'start', 'end', 'strand', 'sequence']
            seq_cols = [c for c in seq_cols if c in gene_df.columns]
            gene_df.select(seq_cols).write_csv(sequences_file, separator='\t')
            extracted['files']['sequences'] = str(sequences_file)
            
            if args.verbosity >= 1:
                print(f"✓ Saved sequences: {sequences_file}")
        
        # Convert to parquet if requested
        if args.format in ['parquet', 'both']:
            import polars as pl
            genes_parquet = output_dir / 'genes.parquet'
            gene_df.select(gene_cols).write_parquet(genes_parquet)
            extracted['files']['genes_parquet'] = str(genes_parquet)
            
            if not args.skip_sequences:
                sequences_parquet = output_dir / 'sequences.parquet'
                gene_df.select(seq_cols).write_parquet(sequences_parquet)
                extracted['files']['sequences_parquet'] = str(sequences_parquet)
            
            if args.verbosity >= 1:
                print(f"✓ Saved Parquet files")
        
        # Save summary
        summary_file = output_dir / 'preparation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(extracted, f, indent=2)
        
        if args.verbosity >= 1:
            print("\n" + "=" * 70)
            print("✓ Data preparation complete!")
            print("=" * 70)
            print(f"\nFiles created:")
            for key, path in extracted['files'].items():
                print(f"  - {key}: {path}")
            print(f"\nSummary: {summary_file}")
        
        # Print statistics
        if args.verbosity >= 1 and extracted['stats']:
            print(f"\nStatistics:")
            if 'genes' in extracted['stats']:
                stats = extracted['stats']['genes']
                print(f"  Genes: {stats['total']} ({stats['with_sequences']} with sequences)")
            if 'splice_sites' in extracted['stats']:
                stats = extracted['stats']['splice_sites']
                print(f"  Splice sites: {stats['total']} ({stats['donors']} donors, {stats['acceptors']} acceptors)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
