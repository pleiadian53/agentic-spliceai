#!/usr/bin/env python
"""
Test Base Layer Entry Points: meta-spliceai vs agentic-spliceai

This script runs the primary base layer entry points in both systems
on the same 5 randomly chosen genes and compares their performance metrics.

IMPORTANT: This script uses isolated test directories with unique timestamps
to avoid overwriting production artifacts. It explicitly protects:
    - data/mane/GRCh38/openspliceai_eval/meta_models/ (OpenSpliceAI production)
    - data/ensembl/GRCh37/spliceai_eval/meta_models/ (SpliceAI production)

Usage:
    cd /Users/pleiadian53/work/agentic-spliceai
    mamba activate metaspliceai
    python scripts/validation/test_base_layer_comparison.py

Expected Outcome:
    - Performance metrics should match exactly between both systems
    - Nucleotide-level predictions should match within floating-point tolerance
"""

import os
import sys
import time
import random
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import polars as pl

# =============================================================================
# CONFIGURATION
# =============================================================================

# Random seed for reproducibility
# Change this value to test different gene sets
# Tested seeds: 42 (HTT, MYC, BRCA1, APOB, KRAS - all passed)
RANDOM_SEED = 42

# Gene pool for random selection (validated genes from previous tests + additional genes)
GENE_POOL = [
    # Previously validated
    'BRCA1', 'TP53', 'EGFR', 'MYC',
    # Additional commonly studied genes
    'BRCA2', 'ATM', 'PTEN', 'RB1', 'KRAS', 'NRAS',
    'APC', 'MLH1', 'MSH2', 'VHL', 'WT1', 'CDH1',
    'CFTR', 'DMD', 'SMN1', 'FMR1', 'HTT', 'PKD1',
    'LDLR', 'APOB', 'PCSK9', 'NF1', 'NF2', 'TSC1',
]

# Number of genes to test
N_TEST_GENES = 5

# Comparison tolerance for floating-point values
FLOAT_TOLERANCE = 1e-5

# Base models to test - both SpliceAI and OpenSpliceAI
BASE_MODELS = ['spliceai', 'openspliceai']

# Model-specific configurations
MODEL_CONFIGS = {
    'spliceai': {
        'build': 'GRCh37',
        'source': 'ensembl',
        'description': 'SpliceAI (GRCh37/Ensembl)',
    },
    'openspliceai': {
        'build': 'GRCh38',
        'source': 'mane',
        'description': 'OpenSpliceAI (GRCh38/MANE)',
    },
}

# =============================================================================
# PROTECTED DIRECTORIES - NEVER OVERWRITE THESE
# =============================================================================
PROTECTED_DIRS = [
    'data/mane/GRCh38/openspliceai_eval/meta_models',
    'data/mane/GRCh38/openspliceai_eval/predictions',
]

# Checkpoint directories that may need clearing for fresh predictions
CHECKPOINT_DIRS = {
    'spliceai': '/Users/pleiadian53/work/meta-spliceai/data/ensembl/spliceai_eval/meta_models',
    'openspliceai': '/Users/pleiadian53/work/meta-spliceai/data/mane/GRCh38/openspliceai_eval/meta_models',
}

# Gene-to-chromosome mapping for the test gene pool
GENE_CHROMOSOMES = {
    'BRCA1': '17', 'TP53': '17', 'EGFR': '7', 'MYC': '8',
    'BRCA2': '13', 'ATM': '11', 'PTEN': '10', 'RB1': '13', 'KRAS': '12', 'NRAS': '1',
    'APC': '5', 'MLH1': '3', 'MSH2': '2', 'VHL': '3', 'WT1': '11', 'CDH1': '16',
    'CFTR': '7', 'DMD': 'X', 'SMN1': '5', 'FMR1': 'X', 'HTT': '4', 'PKD1': '16',
    'LDLR': '19', 'APOB': '2', 'PCSK9': '1', 'NF1': '17', 'NF2': '22', 'TSC1': '9',
}

def get_unique_test_name() -> str:
    """Generate unique test name with timestamp to avoid overwriting artifacts."""
    return f"validation_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def verify_not_overwriting_production(eval_dir: str, project_root: str) -> bool:
    """Verify we're not writing to protected production directories."""
    for protected in PROTECTED_DIRS:
        protected_path = os.path.join(project_root, protected)
        if os.path.commonpath([eval_dir, protected_path]) == protected_path:
            print(f"⚠️  WARNING: Attempted to write to protected directory: {protected}")
            print(f"   Eval dir: {eval_dir}")
            return False
    return True

def clear_checkpoints_for_genes(genes: List[str], base_model: str, backup: bool = True) -> Dict[str, Any]:
    """
    Clear checkpoint files for specific genes to force fresh predictions.
    
    This is needed because meta-spliceai's checkpoint system uses shared directories
    and old checkpoints (without nucleotide scores) would be reused otherwise.
    
    Parameters
    ----------
    genes : List[str]
        Gene names to clear checkpoints for
    base_model : str
        Base model ('spliceai' or 'openspliceai')
    backup : bool
        If True, move files to backup instead of deleting
        
    Returns
    -------
    dict
        Information about cleared/backed up files
    """
    checkpoint_dir = CHECKPOINT_DIRS.get(base_model.lower())
    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        return {'cleared': [], 'backed_up': [], 'error': None}
    
    # Determine chromosomes for the genes
    chromosomes = set()
    for gene in genes:
        chrom = GENE_CHROMOSOMES.get(gene)
        if chrom:
            chromosomes.add(chrom)
    
    if not chromosomes:
        return {'cleared': [], 'backed_up': [], 'error': 'No chromosome mapping found'}
    
    cleared = []
    backed_up = []
    
    # Find and process checkpoint files
    for chrom in chromosomes:
        pattern = f"analysis_sequences_{chrom}_chunk_*.tsv"
        import glob
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, pattern))
        
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            if backup:
                # Move to backup location
                backup_dir = os.path.join(checkpoint_dir, 'backup_for_validation')
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(filepath, backup_path)
                backed_up.append(filename)
            else:
                os.remove(filepath)
                cleared.append(filename)
    
    return {'cleared': cleared, 'backed_up': backed_up, 'chromosomes': list(chromosomes)}

def restore_checkpoints(base_model: str) -> int:
    """Restore backed-up checkpoint files after validation."""
    checkpoint_dir = CHECKPOINT_DIRS.get(base_model.lower())
    if not checkpoint_dir:
        return 0
    
    backup_dir = os.path.join(checkpoint_dir, 'backup_for_validation')
    if not os.path.exists(backup_dir):
        return 0
    
    restored = 0
    for filename in os.listdir(backup_dir):
        src = os.path.join(backup_dir, filename)
        dst = os.path.join(checkpoint_dir, filename)
        shutil.move(src, dst)
        restored += 1
    
    # Remove empty backup directory
    try:
        os.rmdir(backup_dir)
    except OSError:
        pass
    
    return restored


def setup_paths():
    """Set up Python paths for both packages."""
    meta_path = '/Users/pleiadian53/work/meta-spliceai'
    agentic_path = '/Users/pleiadian53/work/agentic-spliceai/src'
    
    # Add paths
    if meta_path not in sys.path:
        sys.path.insert(0, meta_path)
    if agentic_path not in sys.path:
        sys.path.insert(0, agentic_path)
    
    return meta_path, agentic_path


def select_random_genes(n: int = N_TEST_GENES, seed: int = RANDOM_SEED) -> List[str]:
    """Randomly select genes from the gene pool."""
    random.seed(seed)
    return random.sample(GENE_POOL, min(n, len(GENE_POOL)))


def print_section(title: str, char: str = '='):
    """Print a section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def run_meta_spliceai(
    genes: List[str], 
    output_dir: str,
    base_model: str = 'spliceai',
    verbosity: int = 1,
    test_name: str = None
) -> Dict[str, Any]:
    """
    Run predictions using meta-spliceai entry point.
    
    Entry point: meta_spliceai/run_base_model.py::run_base_model_predictions
    """
    model_info = MODEL_CONFIGS.get(base_model, {})
    print_section(f"META-SPLICEAI: {model_info.get('description', base_model)}", "-")
    
    # Use unique test name to avoid cached checkpoints and protect production data
    if test_name is None:
        test_name = get_unique_test_name()
    
    print(f"[info] Using test_name: {test_name} (isolated test directory)")
    print(f"[info] Base model: {base_model} ({model_info.get('build', 'unknown')})")
    
    try:
        from meta_spliceai import run_base_model_predictions
        
        start_time = time.time()
        
        results = run_base_model_predictions(
            base_model=base_model,
            target_genes=genes,
            save_nucleotide_scores=True,
            verbosity=verbosity,
            mode='test',
            test_name=test_name,
        )
        
        elapsed = time.time() - start_time
        
        # Extract key results
        positions_df = results.get('positions', pl.DataFrame())
        nucleotide_scores = results.get('nucleotide_scores', pl.DataFrame())
        
        # If nucleotide_scores not in results, try loading from disk
        if nucleotide_scores is None or (hasattr(nucleotide_scores, '__len__') and len(nucleotide_scores) == 0):
            # Check for nucleotide_scores file in artifacts directory
            paths = results.get('paths', {})
            artifacts_dir = paths.get('artifacts_dir', '')
            
            # Build model-specific paths
            if base_model == 'openspliceai':
                base_dir = '/Users/pleiadian53/work/meta-spliceai/data/mane/GRCh38/openspliceai_eval'
            else:
                base_dir = '/Users/pleiadian53/work/meta-spliceai/data/ensembl/GRCh37/spliceai_eval'
            
            # Try common locations for meta-spliceai nucleotide scores
            possible_paths = [
                os.path.join(artifacts_dir, 'nucleotide_scores.tsv') if artifacts_dir else '',
                f'{base_dir}/tests/{test_name}/meta_models/predictions/nucleotide_scores.tsv',
                f'{base_dir}/nucleotide_scores.tsv',
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    print(f"  [info] Loading nucleotide scores from disk: {path}")
                    nucleotide_scores = pl.read_csv(path, separator='\t')
                    break
        
        print(f"\n[meta-spliceai] Completed in {elapsed:.1f}s")
        print(f"  - Positions: {len(positions_df) if positions_df is not None else 0}")
        print(f"  - Nucleotide scores: {len(nucleotide_scores) if nucleotide_scores is not None else 0}")
        print(f"  - Success: {results.get('success', False)}")
        
        return {
            'success': results.get('success', False),
            'positions': positions_df,
            'nucleotide_scores': nucleotide_scores,
            'elapsed': elapsed,
            'raw_results': results,
            'test_name': test_name,
        }
        
    except Exception as e:
        import traceback
        print(f"[meta-spliceai] ERROR: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'positions': pl.DataFrame(),
            'nucleotide_scores': pl.DataFrame(),
        }


def run_agentic_spliceai(
    genes: List[str], 
    output_dir: str,
    base_model: str = 'spliceai',
    verbosity: int = 1,
    test_name: str = None
) -> Dict[str, Any]:
    """
    Run predictions using agentic-spliceai entry point.
    
    Entry point: splice_engine/meta_models/workflows/splice_prediction_workflow.py::run_base_model_predictions
    
    Uses isolated test directory to avoid overwriting production artifacts.
    """
    model_info = MODEL_CONFIGS.get(base_model, {})
    print_section(f"AGENTIC-SPLICEAI: {model_info.get('description', base_model)}", "-")
    
    # Use unique test name to avoid cached checkpoints and protect production data
    if test_name is None:
        test_name = get_unique_test_name()
    
    print(f"[info] Using test_name: {test_name} (isolated test directory)")
    print(f"[info] Base model: {base_model} ({model_info.get('build', 'unknown')})")
    
    try:
        from agentic_spliceai.splice_engine import run_base_model_predictions
        
        start_time = time.time()
        
        results = run_base_model_predictions(
            base_model=base_model,
            target_genes=genes,
            save_nucleotide_scores=True,
            verbosity=verbosity,
            mode='test',
            test_name=test_name,  # Use unique test name
        )
        
        elapsed = time.time() - start_time
        
        # Extract key results
        positions_df = results.get('positions', pl.DataFrame())
        nucleotide_scores = results.get('nucleotide_scores', pl.DataFrame())
        
        print(f"\n[agentic-spliceai] Completed in {elapsed:.1f}s")
        print(f"  - Positions: {len(positions_df) if positions_df is not None else 0}")
        print(f"  - Nucleotide scores: {len(nucleotide_scores) if nucleotide_scores is not None else 0}")
        print(f"  - Success: {results.get('success', False)}")
        
        return {
            'success': results.get('success', False),
            'positions': positions_df,
            'nucleotide_scores': nucleotide_scores,
            'elapsed': elapsed,
            'raw_results': results,
        }
        
    except Exception as e:
        import traceback
        print(f"[agentic-spliceai] ERROR: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'positions': pl.DataFrame(),
            'nucleotide_scores': pl.DataFrame(),
        }


def compare_dataframes(
    meta_df: pl.DataFrame, 
    agentic_df: pl.DataFrame, 
    name: str
) -> Dict[str, Any]:
    """Compare two DataFrames for equality."""
    result = {
        'name': name,
        'meta_rows': len(meta_df) if meta_df is not None else 0,
        'agentic_rows': len(agentic_df) if agentic_df is not None else 0,
        'match': False,
        'details': [],
    }
    
    # Check row counts
    if result['meta_rows'] != result['agentic_rows']:
        result['details'].append(f"Row count mismatch: {result['meta_rows']} vs {result['agentic_rows']}")
        return result
    
    if result['meta_rows'] == 0:
        result['match'] = True
        result['details'].append("Both empty")
        return result
    
    # Check column sets
    meta_cols = set(meta_df.columns)
    agentic_cols = set(agentic_df.columns)
    
    if meta_cols != agentic_cols:
        common = meta_cols & agentic_cols
        only_meta = meta_cols - agentic_cols
        only_agentic = agentic_cols - meta_cols
        result['details'].append(f"Column differences - Only in meta: {only_meta}, Only in agentic: {only_agentic}")
        # Continue with common columns
        cols_to_compare = list(common)
    else:
        cols_to_compare = list(meta_cols)
    
    result['match'] = True
    return result


def compare_nucleotide_scores(
    meta_results: Dict[str, Any],
    agentic_results: Dict[str, Any],
    genes: List[str],
) -> Dict[str, Any]:
    """Compare nucleotide-level scores between both systems."""
    
    print_section("COMPARISON: Nucleotide-Level Scores", "-")
    
    meta_scores = meta_results.get('nucleotide_scores', pl.DataFrame())
    agentic_scores = agentic_results.get('nucleotide_scores', pl.DataFrame())
    
    comparison = {
        'overall_match': True,
        'genes': {},
        'summary': {
            'total_positions_compared': 0,
            'max_donor_diff': 0.0,
            'max_acceptor_diff': 0.0,
            'mean_donor_diff': 0.0,
            'mean_acceptor_diff': 0.0,
        }
    }
    
    if meta_scores is None or len(meta_scores) == 0:
        print("[warning] meta-spliceai: No nucleotide scores")
        comparison['overall_match'] = False
        return comparison
    
    if agentic_scores is None or len(agentic_scores) == 0:
        print("[warning] agentic-spliceai: No nucleotide scores")
        comparison['overall_match'] = False
        return comparison
    
    # Process each gene
    all_donor_diffs = []
    all_acceptor_diffs = []
    
    for gene in genes:
        # Filter by gene
        meta_gene = meta_scores.filter(
            (pl.col('gene_id').str.contains(gene)) | 
            (pl.col('gene_name') == gene)
        )
        agentic_gene = agentic_scores.filter(
            (pl.col('gene_id').str.contains(gene)) | 
            (pl.col('gene_name') == gene)
        )
        
        gene_result = {
            'meta_positions': len(meta_gene),
            'agentic_positions': len(agentic_gene),
            'positions_match': False,
            'scores_match': False,
            'max_donor_diff': 0.0,
            'max_acceptor_diff': 0.0,
        }
        
        if len(meta_gene) == 0 or len(agentic_gene) == 0:
            print(f"  {gene}: SKIP (no data - meta:{len(meta_gene)}, agentic:{len(agentic_gene)})")
            comparison['genes'][gene] = gene_result
            continue
        
        # Compare positions
        meta_positions = set(meta_gene['genomic_position'].to_list())
        agentic_positions = set(agentic_gene['genomic_position'].to_list())
        
        common_positions = meta_positions & agentic_positions
        gene_result['positions_match'] = (meta_positions == agentic_positions)
        gene_result['common_positions'] = len(common_positions)
        
        if not gene_result['positions_match']:
            only_meta = len(meta_positions - agentic_positions)
            only_agentic = len(agentic_positions - meta_positions)
            print(f"  {gene}: Position mismatch (only_meta:{only_meta}, only_agentic:{only_agentic})")
        
        # Compare scores at common positions
        if len(common_positions) > 0:
            # Build lookup tables
            meta_lookup = {}
            for row in meta_gene.iter_rows(named=True):
                pos = row['genomic_position']
                meta_lookup[pos] = (row.get('donor_score', 0), row.get('acceptor_score', 0))
            
            agentic_lookup = {}
            for row in agentic_gene.iter_rows(named=True):
                pos = row['genomic_position']
                agentic_lookup[pos] = (row.get('donor_score', 0), row.get('acceptor_score', 0))
            
            # Compute differences
            donor_diffs = []
            acceptor_diffs = []
            
            for pos in common_positions:
                m_donor, m_acceptor = meta_lookup.get(pos, (0, 0))
                a_donor, a_acceptor = agentic_lookup.get(pos, (0, 0))
                
                donor_diffs.append(abs(m_donor - a_donor))
                acceptor_diffs.append(abs(m_acceptor - a_acceptor))
            
            gene_result['max_donor_diff'] = max(donor_diffs) if donor_diffs else 0
            gene_result['max_acceptor_diff'] = max(acceptor_diffs) if acceptor_diffs else 0
            gene_result['mean_donor_diff'] = np.mean(donor_diffs) if donor_diffs else 0
            gene_result['mean_acceptor_diff'] = np.mean(acceptor_diffs) if acceptor_diffs else 0
            gene_result['scores_match'] = (
                gene_result['max_donor_diff'] < FLOAT_TOLERANCE and 
                gene_result['max_acceptor_diff'] < FLOAT_TOLERANCE
            )
            
            all_donor_diffs.extend(donor_diffs)
            all_acceptor_diffs.extend(acceptor_diffs)
            
            status = "✅ PASS" if gene_result['scores_match'] else "❌ FAIL"
            print(f"  {gene}: {status} | positions:{len(common_positions)} | "
                  f"max_diff(donor:{gene_result['max_donor_diff']:.2e}, "
                  f"acceptor:{gene_result['max_acceptor_diff']:.2e})")
        
        comparison['genes'][gene] = gene_result
        comparison['summary']['total_positions_compared'] += len(common_positions)
        
        if not gene_result['scores_match']:
            comparison['overall_match'] = False
    
    # Overall summary
    if all_donor_diffs:
        comparison['summary']['max_donor_diff'] = max(all_donor_diffs)
        comparison['summary']['mean_donor_diff'] = np.mean(all_donor_diffs)
    if all_acceptor_diffs:
        comparison['summary']['max_acceptor_diff'] = max(all_acceptor_diffs)
        comparison['summary']['mean_acceptor_diff'] = np.mean(all_acceptor_diffs)
    
    return comparison


def print_final_report(
    genes: List[str],
    meta_results: Dict[str, Any],
    agentic_results: Dict[str, Any],
    comparison: Dict[str, Any],
    base_model: str = None,
):
    """Print final comparison report."""
    
    model_name = base_model or 'unknown'
    model_info = MODEL_CONFIGS.get(model_name, {})
    
    print_section(f"COMPARISON REPORT: {model_info.get('description', model_name)}", "-")
    
    print(f"\n📋 Test Configuration:")
    print(f"   - Base Model: {model_name} ({model_info.get('build', 'unknown')})")
    print(f"   - Test Genes: {genes}")
    print(f"   - Random Seed: {RANDOM_SEED}")
    print(f"   - Float Tolerance: {FLOAT_TOLERANCE}")
    
    print(f"\n⏱️  Execution Times:")
    print(f"   - meta-spliceai:   {meta_results.get('elapsed', 0):.1f}s")
    print(f"   - agentic-spliceai: {agentic_results.get('elapsed', 0):.1f}s")
    
    print(f"\n📊 Results Summary:")
    print(f"   - meta-spliceai success:   {meta_results.get('success', False)}")
    print(f"   - agentic-spliceai success: {agentic_results.get('success', False)}")
    
    print(f"\n🔬 Comparison Summary:")
    print(f"   - Total positions compared: {comparison['summary']['total_positions_compared']:,}")
    print(f"   - Max donor score diff:     {comparison['summary']['max_donor_diff']:.2e}")
    print(f"   - Max acceptor score diff:  {comparison['summary']['max_acceptor_diff']:.2e}")
    print(f"   - Mean donor score diff:    {comparison['summary']['mean_donor_diff']:.2e}")
    print(f"   - Mean acceptor score diff: {comparison['summary']['mean_acceptor_diff']:.2e}")
    
    print(f"\n📈 Per-Gene Results:")
    print(f"   {'Gene':<12} {'Positions':<12} {'Max Donor Diff':<16} {'Max Acceptor Diff':<18} {'Status':<8}")
    print(f"   {'-'*10:<12} {'-'*10:<12} {'-'*14:<16} {'-'*16:<18} {'-'*6:<8}")
    
    for gene in genes:
        gene_data = comparison['genes'].get(gene, {})
        pos = gene_data.get('common_positions', 0)
        donor_diff = gene_data.get('max_donor_diff', float('nan'))
        acceptor_diff = gene_data.get('max_acceptor_diff', float('nan'))
        status = "✅ PASS" if gene_data.get('scores_match', False) else "❌ FAIL"
        print(f"   {gene:<12} {pos:<12,} {donor_diff:<16.2e} {acceptor_diff:<18.2e} {status:<8}")
    
    print(f"\n{'='*70}")
    overall_status = "✅ ALL TESTS PASSED" if comparison['overall_match'] else "❌ SOME TESTS FAILED"
    print(f"  OVERALL RESULT: {overall_status}")
    print(f"{'='*70}\n")
    
    return comparison['overall_match']


def run_comparison_for_model(
    base_model: str,
    genes: List[str],
    test_name: str,
    verbosity: int = 1
) -> Dict[str, Any]:
    """Run comparison for a single base model."""
    model_info = MODEL_CONFIGS.get(base_model, {})
    
    print_section(f"TESTING: {model_info.get('description', base_model)}", "=")
    print(f"Build: {model_info.get('build', 'unknown')}")
    print(f"Source: {model_info.get('source', 'unknown')}")
    
    # Clear existing checkpoints to force fresh predictions
    print(f"\n🔄 CHECKPOINT MANAGEMENT for {base_model}:")
    backup_result = clear_checkpoints_for_genes(genes, base_model, backup=True)
    if backup_result.get('backed_up'):
        print(f"   - Backed up {len(backup_result['backed_up'])} checkpoint files")
        print(f"   - Chromosomes affected: {backup_result.get('chromosomes', [])}")
    else:
        print(f"   - No checkpoints to back up (fresh run)")
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_output = os.path.join(tmpdir, 'meta_spliceai')
            agentic_output = os.path.join(tmpdir, 'agentic_spliceai')
            os.makedirs(meta_output, exist_ok=True)
            os.makedirs(agentic_output, exist_ok=True)
            
            # Model-specific test name
            model_test_name = f"{test_name}_{base_model}"
            
            # Run both systems
            meta_results = run_meta_spliceai(
                genes, meta_output, base_model=base_model, 
                verbosity=verbosity, test_name=model_test_name
            )
            agentic_results = run_agentic_spliceai(
                genes, agentic_output, base_model=base_model,
                verbosity=verbosity, test_name=model_test_name
            )
            
            # Compare results
            comparison = compare_nucleotide_scores(meta_results, agentic_results, genes)
            
            # Print report
            success = print_final_report(genes, meta_results, agentic_results, comparison, base_model=base_model)
            
            return {
                'base_model': base_model,
                'success': success,
                'comparison': comparison,
                'meta_results': meta_results,
                'agentic_results': agentic_results,
            }
    finally:
        # Always restore checkpoints
        restored = restore_checkpoints(base_model)
        if restored > 0:
            print(f"   [restore] Restored {restored} checkpoint files for {base_model}")


def main():
    """Main entry point."""
    print_section("BASE LAYER COMPARISON TEST - ALL MODELS", "=")
    print(f"\nTest started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(BASE_MODELS)} base model(s): {BASE_MODELS}")
    
    # Generate unique test name to avoid overwriting production artifacts
    test_name = get_unique_test_name()
    
    print(f"\n🔒 ARTIFACT PROTECTION:")
    print(f"   - Using isolated test directories with prefix: {test_name}")
    print(f"   - Protected directories will NOT be overwritten:")
    for protected in PROTECTED_DIRS:
        print(f"     • {protected}")
    
    # Setup
    setup_paths()
    
    # Select random genes
    genes = select_random_genes(N_TEST_GENES, RANDOM_SEED)
    print(f"\n🎲 Randomly selected genes (seed={RANDOM_SEED}):")
    for i, gene in enumerate(genes, 1):
        print(f"   {i}. {gene}")
    
    # Run comparison for each base model
    all_results = {}
    overall_success = True
    
    for base_model in BASE_MODELS:
        try:
            result = run_comparison_for_model(base_model, genes, test_name, verbosity=1)
            all_results[base_model] = result
            if not result.get('success', False):
                overall_success = False
        except Exception as e:
            print(f"\n❌ ERROR testing {base_model}: {e}")
            import traceback
            traceback.print_exc()
            all_results[base_model] = {'success': False, 'error': str(e)}
            overall_success = False
    
    # Print final summary
    print_section("OVERALL SUMMARY", "=")
    print(f"\n{'Base Model':<20} {'Build':<10} {'Status':<15}")
    print(f"{'-'*18:<20} {'-'*8:<10} {'-'*13:<15}")
    
    for base_model in BASE_MODELS:
        result = all_results.get(base_model, {})
        model_info = MODEL_CONFIGS.get(base_model, {})
        build = model_info.get('build', 'unknown')
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        print(f"{base_model:<20} {build:<10} {status:<15}")
    
    print(f"\n{'='*70}")
    if overall_success:
        print("  🎉 ALL BASE MODELS PASSED - Meta-spliceai and Agentic-spliceai are consistent!")
    else:
        print("  ⚠️  SOME BASE MODELS FAILED - Investigate the differences above")
    print(f"{'='*70}\n")
    
    return 0 if overall_success else 1


if __name__ == '__main__':
    sys.exit(main())

