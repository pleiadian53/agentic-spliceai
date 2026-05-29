"""
Integration tests for Phase 2: Data Preparation Module

Tests the standalone data preparation functions that load annotations
and extract sequences for splice site prediction.

Test Cases:
-----------
1. test_prepare_gene_data_single_gene: Load data for BRCA1
2. test_prepare_gene_data_multiple_genes: Load data for multiple genes
3. test_prepare_gene_data_chromosome: Load all genes on chr21
4. test_load_gene_annotations: Test GTF loading
5. test_filter_functions: Test gene/chromosome filtering
6. test_helper_functions: Test utility functions
"""

import polars as pl
from pathlib import Path

from agentic_spliceai.splice_engine.base_layer.data import (
    prepare_gene_data,
    load_gene_annotations,
    filter_by_genes,
    filter_by_chromosomes,
    normalize_chromosome_names,
    get_gene_count,
    get_genes_by_chromosome,
    validate_gene_data
)


def test_prepare_gene_data_single_gene():
    """Test Phase 2: Load data for single gene (BRCA1)."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Prepare data for BRCA1")
    print("="*80)
    
    # Load data for BRCA1
    gene_df = prepare_gene_data(
        genes=['BRCA1'],
        build='GRCh38',
        annotation_source='mane',
        verbosity=1
    )
    
    # Verify results
    assert gene_df.height > 0, "No genes loaded"
    assert 'sequence' in gene_df.columns, "Missing sequence column"
    assert 'gene_name' in gene_df.columns, "Missing gene_name column"
    
    # Check BRCA1 was found
    brca1 = gene_df.filter(pl.col('gene_name').str.to_uppercase() == 'BRCA1')
    assert brca1.height > 0, "BRCA1 not found"
    
    # Check sequence exists
    seq = brca1['sequence'][0]
    assert seq is not None, "BRCA1 sequence is null"
    assert len(seq) > 0, "BRCA1 sequence is empty"
    
    print(f"\n✓ BRCA1 loaded successfully")
    print(f"  Sequence length: {len(seq):,} bp")
    print(f"  Chromosome: {brca1['seqname'][0]}")
    print(f"  Start: {brca1['start'][0]:,}")
    print(f"  End: {brca1['end'][0]:,}")
    print(f"  Strand: {brca1['strand'][0]}")
    
    return gene_df


def test_prepare_gene_data_multiple_genes():
    """Test Phase 2: Load data for multiple genes."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Prepare data for multiple genes")
    print("="*80)
    
    # Load data for multiple genes
    genes = ['BRCA1', 'TP53', 'EGFR', 'MYC']
    gene_df = prepare_gene_data(
        genes=genes,
        build='GRCh38',
        annotation_source='mane',
        verbosity=1
    )
    
    # Verify results
    assert gene_df.height > 0, "No genes loaded"
    
    # Check how many genes were found
    found_genes = set(gene_df['gene_name'].str.to_uppercase().to_list())
    requested_genes = set(g.upper() for g in genes)
    
    print(f"\n✓ Loaded {gene_df.height} genes")
    print(f"  Requested: {requested_genes}")
    print(f"  Found: {found_genes}")
    
    # All genes should have sequences
    with_seq = gene_df.filter(pl.col('sequence').is_not_null()).height
    print(f"  With sequences: {with_seq}/{gene_df.height}")
    assert with_seq == gene_df.height, "Some genes missing sequences"
    
    return gene_df


def test_prepare_gene_data_chromosome():
    """Test Phase 2: Load all genes on chromosome 21."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Prepare data for chr21")
    print("="*80)
    
    # Load all genes on chr21
    gene_df = prepare_gene_data(
        chromosomes=['21'],
        build='GRCh38',
        annotation_source='mane',
        verbosity=1
    )
    
    # Verify results
    assert gene_df.height > 0, "No genes loaded from chr21"
    
    # All should be on chr21
    chroms = set(gene_df['seqname'].to_list())
    print(f"\n✓ Loaded {gene_df.height} genes from chr21")
    print(f"  Chromosomes in result: {chroms}")
    assert len(chroms) == 1, "Multiple chromosomes in result"
    assert '21' in str(list(chroms)[0]), "Not chromosome 21"
    
    # Check some genes have sequences
    with_seq = gene_df.filter(pl.col('sequence').is_not_null()).height
    print(f"  With sequences: {with_seq}/{gene_df.height}")
    assert with_seq > 0, "No sequences extracted"
    
    # Show some example genes
    print(f"\n  Example genes:")
    example_genes = gene_df.select(['gene_name', 'start', 'end']).head(5)
    for row in example_genes.iter_rows(named=True):
        print(f"    - {row['gene_name']}: {row['start']:,} - {row['end']:,}")
    
    return gene_df


def test_load_gene_annotations():
    """Test Phase 2: Load gene annotations from GTF."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Load gene annotations")
    print("="*80)
    
    from agentic_spliceai.splice_engine.resources import get_genomic_registry
    
    # Get GTF path
    registry = get_genomic_registry(build='GRCh38_MANE', release='1.3')
    gtf_path = registry.get_gtf_path(validate=True)
    
    print(f"\nGTF: {gtf_path}")
    
    # Load annotations for specific genes
    genes_df = load_gene_annotations(
        gtf_path=gtf_path,
        genes=['BRCA1', 'TP53'],
        verbosity=1
    )
    
    assert genes_df.height > 0, "No genes loaded"
    assert 'gene_name' in genes_df.columns, "Missing gene_name column"
    
    print(f"\n✓ Loaded {genes_df.height} gene annotations")
    print(f"  Columns: {genes_df.columns}")
    
    return genes_df


def test_filter_functions():
    """Test Phase 2: Filter functions."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Filter functions")
    print("="*80)
    
    # Create test DataFrame
    test_df = pl.DataFrame({
        'gene_name': ['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS'],
        'seqname': ['17', '17', '7', '8', '12'],
        'start': [43044295, 7668402, 55019017, 127735434, 25245347],
        'end': [43125483, 7687550, 55211628, 127742951, 25250929]
    })
    
    print(f"\nOriginal DataFrame: {test_df.height} rows")
    
    # Test gene filtering
    print("\n1. Filter by genes:")
    filtered = filter_by_genes(test_df, ['BRCA1', 'TP53'])
    print(f"   Filtered to: {filtered.height} rows")
    assert filtered.height == 2, "Gene filtering failed"
    
    # Test chromosome filtering
    print("\n2. Filter by chromosomes:")
    filtered = filter_by_chromosomes(test_df, ['17'])
    print(f"   Filtered to: {filtered.height} rows")
    assert filtered.height == 2, "Chromosome filtering failed"
    
    # Test chromosome name normalization
    print("\n3. Normalize chromosome names:")
    normalized = normalize_chromosome_names(['1', 'chr21', 'X'])
    print(f"   Input: ['1', 'chr21', 'X']")
    print(f"   Output: {sorted(normalized)}")
    assert '1' in normalized, "Missing '1'"
    assert 'chr1' in normalized, "Missing 'chr1'"
    assert '21' in normalized, "Missing '21'"
    assert 'chr21' in normalized, "Missing 'chr21'"
    
    print("\n✓ All filter functions working")


def test_helper_functions():
    """Test Phase 2: Helper functions."""
    print("\n" + "="*80)
    print("TEST: Phase 2 - Helper functions")
    print("="*80)
    
    # Create test DataFrame with sequences
    test_df = pl.DataFrame({
        'gene_name': ['BRCA1', 'TP53', 'EGFR'],
        'gene_id': ['ENSG001', 'ENSG002', 'ENSG003'],
        'seqname': ['17', '17', '7'],
        'start': [1000, 2000, 3000],
        'end': [2000, 3000, 4000],
        'strand': ['+', '-', '+'],
        'sequence': ['ATCG'*100, 'GCTA'*100, None]
    })
    
    # Test get_gene_count
    print(f"\n1. Gene count: {get_gene_count(test_df)}")
    assert get_gene_count(test_df) == 3
    
    # Test get_genes_by_chromosome
    print("\n2. Genes by chromosome:")
    by_chrom = get_genes_by_chromosome(test_df)
    for chrom, count in by_chrom.items():
        print(f"   {chrom}: {count} genes")
    assert by_chrom['17'] == 2
    assert by_chrom['7'] == 1
    
    # Test validate_gene_data
    print("\n3. Validate gene data:")
    is_valid = validate_gene_data(test_df, verbosity=1)
    assert is_valid, "Validation failed"
    
    print("\n✓ All helper functions working")


def test_prepare_splice_site_annotations():
    """Test splice site annotation extraction."""
    print("\n" + "="*80)
    print("TEST 7: Splice Site Annotation Extraction")
    print("="*80)
    
    from agentic_spliceai.splice_engine.base_layer.data import prepare_splice_site_annotations
    import tempfile
    from pathlib import Path
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        print("\n📍 Extracting splice sites for BRCA1...")
        result = prepare_splice_site_annotations(
            output_dir=output_dir,
            genes=['BRCA1'],
            build='GRCh38',
            verbosity=1
        )
        
        # Verify result
        assert result['success'], "Splice site extraction failed"
        assert result['splice_sites_df'] is not None, "No splice sites DataFrame"
        assert result['n_sites'] > 0, "No splice sites found"
        assert result['n_donors'] > 0, "No donor sites found"
        assert result['n_acceptors'] > 0, "No acceptor sites found"
        
        df = result['splice_sites_df']
        
        # Verify DataFrame structure
        required_cols = ['chrom', 'position', 'site_type', 'gene_id', 'gene_name', 'strand']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
        
        # Verify site types
        site_types = set(df['site_type'].unique().to_list())
        assert 'donor' in site_types, "No donor sites"
        assert 'acceptor' in site_types, "No acceptor sites"
        
        # Verify gene name
        gene_names = set(df['gene_name'].unique().to_list())
        assert 'BRCA1' in gene_names, "BRCA1 not found in results"
        
        print(f"\n✅ Test 7 passed!")
        print(f"   Extracted {result['n_sites']} splice sites")
        print(f"   Donors: {result['n_donors']}")
        print(f"   Acceptors: {result['n_acceptors']}")
        print(f"   File: {result['splice_sites_file']}")
        
        return result


def run_all_tests():
    """Run all Phase 2 data preparation tests."""
    print("\n" + "="*80)
    print("PHASE 2: DATA PREPARATION MODULE - INTEGRATION TESTS")
    print("="*80)
    
    try:
        # Test 1: Single gene
        df1 = test_prepare_gene_data_single_gene()
        
        # Test 2: Multiple genes
        df2 = test_prepare_gene_data_multiple_genes()
        
        # Test 3: Chromosome
        df3 = test_prepare_gene_data_chromosome()
        
        # Test 4: Load annotations
        df4 = test_load_gene_annotations()
        
        # Test 5: Filter functions
        test_filter_functions()
        
        # Test 6: Helper functions
        test_helper_functions()
        
        # Test 7: Splice site annotations ⭐ NEW
        test_prepare_splice_site_annotations()
        
        print("\n" + "="*80)
        print("✅ ALL PHASE 2 TESTS PASSED")
        print("="*80)
        print("\nPhase 2 Data Preparation Module is working correctly!")
        print("\nKey capabilities:")
        print("  ✓ Load gene annotations from GTF")
        print("  ✓ Extract sequences from FASTA")
        print("  ✓ Extract splice site annotations ⭐ NEW")
        print("  ✓ Filter by genes and chromosomes")
        print("  ✓ Normalize chromosome names")
        print("  ✓ Helper utilities for data inspection")
        print("\nNext: Phase 3 (Workflow Orchestration) or Phase 4 (Artifact Management)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
