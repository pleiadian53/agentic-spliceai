"""Validate MANE metadata inference against Ensembl ground truth.

This script extracts splice sites from both Ensembl and MANE for a set of test genes,
then compares the inferred MANE metadata against Ensembl's native metadata.
"""

import sys
from pathlib import Path

# Add project to path - using marker-based root finding
sys.path.insert(0, str(Path(__file__).parent.parent))
from _example_utils import setup_example_environment
setup_example_environment()

from agentic_spliceai.splice_engine.base_layer.data import prepare_splice_site_annotations
import polars as pl
import tempfile


# Test genes: well-documented in both Ensembl and MANE
TEST_GENES = [
    'BRCA1',    # Breast cancer gene
    'TP53',     # Tumor suppressor
    'EGFR',     # Epidermal growth factor receptor
    'MYC',      # Oncogene
    'KRAS',     # RAS oncogene
    'PTEN',     # Tumor suppressor
    'APC',      # Tumor suppressor
    'BRAF',     # Kinase
    'PIK3CA',   # Phosphoinositide 3-kinase
    'NRAS',     # RAS oncogene
]


def extract_splice_sites_for_comparison(genes, annotation_source, build='GRCh38'):
    """Extract splice sites from specified annotation source."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = prepare_splice_site_annotations(
            output_dir=tmpdir,
            genes=genes,
            build=build,
            annotation_source=annotation_source,
            verbosity=1
        )
        
        if not result['success']:
            print(f"Failed to extract from {annotation_source}")
            return None
        
        return result['splice_sites_df']


def compare_metadata(ensembl_df, mane_df, gene):
    """Compare metadata for a specific gene between Ensembl and MANE."""
    
    # Filter to specific gene
    ensembl_gene = ensembl_df.filter(pl.col('gene_name') == gene)
    mane_gene = mane_df.filter(pl.col('gene_name') == gene)
    
    if ensembl_gene.height == 0 or mane_gene.height == 0:
        return {
            'gene': gene,
            'status': 'MISSING',
            'ensembl_count': ensembl_gene.height,
            'mane_count': mane_gene.height,
        }
    
    # Get unique biotypes
    ensembl_biotypes = set(ensembl_gene['gene_biotype'].unique().to_list())
    mane_biotypes = set(mane_gene['gene_biotype'].unique().to_list())
    
    ensembl_transcript_biotypes = set(ensembl_gene['transcript_biotype'].unique().to_list())
    mane_transcript_biotypes = set(mane_gene['transcript_biotype'].unique().to_list())
    
    # Check if our MANE inference matches Ensembl reality
    biotype_match = bool(ensembl_biotypes & mane_biotypes)  # Any overlap
    transcript_biotype_match = bool(ensembl_transcript_biotypes & mane_transcript_biotypes)
    
    # Check exon numbering validity
    mane_exon_numbers = mane_gene['exon_number'].to_list()
    ensembl_exon_numbers = ensembl_gene['exon_number'].to_list()
    
    valid_mane_numbering = all(n > 0 for n in mane_exon_numbers)
    valid_ensembl_numbering = all(n > 0 for n in ensembl_exon_numbers)
    
    return {
        'gene': gene,
        'status': 'OK' if (biotype_match and valid_mane_numbering) else 'MISMATCH',
        'ensembl_splice_sites': ensembl_gene.height,
        'mane_splice_sites': mane_gene.height,
        'ensembl_gene_biotypes': ensembl_biotypes,
        'mane_gene_biotypes': mane_biotypes,
        'ensembl_transcript_biotypes': ensembl_transcript_biotypes,
        'mane_transcript_biotypes': mane_transcript_biotypes,
        'biotype_match': biotype_match,
        'transcript_biotype_match': transcript_biotype_match,
        'mane_numbering_valid': valid_mane_numbering,
        'ensembl_numbering_valid': valid_ensembl_numbering,
        'note': 'MANE expected to have fewer sites (curated subset)',
    }


def main():
    """Run validation."""
    print("=" * 80)
    print("MANE Metadata Validation")
    print("=" * 80)
    print(f"\nTest genes: {', '.join(TEST_GENES)}")
    print("\n" + "=" * 80)
    
    # Extract from Ensembl (ground truth)
    print("\n1. Extracting splice sites from Ensembl (ground truth)...")
    ensembl_df = extract_splice_sites_for_comparison(TEST_GENES, 'ensembl')
    
    if ensembl_df is None:
        print("Failed to extract from Ensembl")
        return
    
    print(f"   ✓ Extracted {ensembl_df.height} splice sites from Ensembl")
    
    # Extract from MANE (with our inference)
    print("\n2. Extracting splice sites from MANE (with inferred metadata)...")
    mane_df = extract_splice_sites_for_comparison(TEST_GENES, 'mane')
    
    if mane_df is None:
        print("Failed to extract from MANE")
        return
    
    print(f"   ✓ Extracted {mane_df.height} splice sites from MANE")
    
    # Compare metadata for each gene
    print("\n3. Comparing metadata gene-by-gene...")
    print("=" * 80)
    
    results = []
    for gene in TEST_GENES:
        comparison = compare_metadata(ensembl_df, mane_df, gene)
        results.append(comparison)
        
        print(f"\n{gene}:")
        print(f"  Status: {comparison['status']}")
        print(f"  Splice sites: Ensembl={comparison.get('ensembl_splice_sites', 0)}, MANE={comparison.get('mane_splice_sites', 0)}")
        
        if comparison['status'] == 'OK':
            print(f"  Gene biotype: {comparison.get('mane_gene_biotypes', set())}")
            print(f"  Transcript biotype: {comparison.get('mane_transcript_biotypes', set())}")
            print(f"  ✓ Biotype match: {comparison.get('biotype_match', False)}")
            print(f"  ✓ Exon numbering valid: {comparison.get('mane_numbering_valid', False)}")
        elif comparison['status'] == 'MISSING':
            print(f"  ⚠ Gene not found in one or both sources")
        else:
            print(f"  ❌ Metadata mismatch detected")
            print(f"     Ensembl biotypes: {comparison.get('ensembl_gene_biotypes', set())}")
            print(f"     MANE biotypes: {comparison.get('mane_gene_biotypes', set())}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    missing_count = sum(1 for r in results if r['status'] == 'MISSING')
    mismatch_count = sum(1 for r in results if r['status'] == 'MISMATCH')
    
    print(f"\nTotal genes tested: {len(TEST_GENES)}")
    print(f"  ✓ OK: {ok_count}")
    print(f"  ⚠ Missing: {missing_count}")
    print(f"  ❌ Mismatch: {mismatch_count}")
    
    # Check protein_coding assumption
    all_protein_coding = all(
        'protein_coding' in r.get('mane_gene_biotypes', set())
        for r in results if r['status'] == 'OK'
    )
    
    print(f"\n✓ MANE 'protein_coding' default valid: {all_protein_coding}")
    
    if all_protein_coding and ok_count >= 8:  # At least 80% success
        print("\n✅ VALIDATION PASSED")
        print("   MANE metadata inference is correct!")
        return True
    else:
        print("\n⚠ VALIDATION NEEDS REVIEW")
        print("   Some genes show unexpected metadata")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
