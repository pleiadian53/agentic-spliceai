#!/usr/bin/env python
"""Compare evaluation metrics between meta-spliceai and agentic-spliceai."""

import sys
import polars as pl
import numpy as np

# Load the existing test data
DATA_DIR = '/Users/pleiadian53/work/agentic-spliceai/data/test_runs/openspliceai/20251212_180602_genes_BRCA1_TP53'

nucleotide_scores = pl.read_csv(f'{DATA_DIR}/nucleotide_scores.tsv', separator='\t')
splice_sites = pl.read_csv(f'{DATA_DIR}/splice_sites_enhanced.tsv', separator='\t')

print('Loaded nucleotide_scores:', nucleotide_scores.shape)
print('Loaded splice_sites:', splice_sites.shape)

# Build predictions dict
predictions = {}
for (gene_id,), g in nucleotide_scores.group_by(['gene_id'], maintain_order=True):
    g_sorted = g.sort('genomic_position')
    first = g_sorted.row(0, named=True)
    positions = g_sorted['genomic_position'].to_list()
    gene_start = min(positions)
    gene_end = max(positions)
    strand = first.get('strand', '+')
    
    predictions[str(gene_id)] = {
        'gene_id': str(gene_id),
        'gene_name': first.get('gene_name', ''),
        'chromosome': first.get('chrom', ''),
        'strand': strand,
        'gene_start': gene_start,
        'gene_end': gene_end,
        'donor_prob': np.array(g_sorted['donor_score'].to_list()),
        'acceptor_prob': np.array(g_sorted['acceptor_score'].to_list()),
        'neither_prob': np.array(g_sorted['neither_score'].to_list()),
    }

print(f'\nBuilt predictions for {len(predictions)} genes')
for gid, gdata in predictions.items():
    print(f'  {gid}: {len(gdata["donor_prob"])} positions, strand={gdata["strand"]}')


def calculate_metrics(positions_df, label=''):
    """Calculate and print metrics from positions DataFrame."""
    tp = positions_df.filter(pl.col('pred_type') == 'TP').height
    fp = positions_df.filter(pl.col('pred_type') == 'FP').height
    fn = positions_df.filter(pl.col('pred_type') == 'FN').height
    tn = positions_df.filter(pl.col('pred_type') == 'TN').height
    
    print(f'\n{label} Overall: TP={tp}, FP={fp}, FN={fn}, TN={tn}')
    precision = recall = f1 = 0.0
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f'  Precision: {precision:.4f}')
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f'  Recall: {recall:.4f}')
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f'  F1: {f1:.4f}')
    
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'precision': precision, 'recall': recall, 'f1': f1}


# Test meta-spliceai evaluation
print('\n' + '='*60)
print('TESTING META-SPLICEAI EVALUATION')
print('='*60)

try:
    sys.path.insert(0, '/Users/pleiadian53/work/meta-spliceai')
    from meta_spliceai.splice_engine.meta_models.core.enhanced_evaluation import enhanced_evaluate_splice_site_errors
    
    # Standardize column names
    ss = splice_sites.clone()
    if 'site_type' in ss.columns:
        ss = ss.rename({'site_type': 'splice_type'})
    
    error_df, positions_df = enhanced_evaluate_splice_site_errors(
        annotations_df=ss,
        pred_results=predictions,
        threshold=0.5,
        consensus_window=2,
        collect_tn=True,
        no_tn_sampling=True,
        return_positions_df=True,
        verbose=0,
    )
    
    print(f'Positions shape: {positions_df.shape}')
    meta_metrics = calculate_metrics(positions_df, 'meta-spliceai')
    
except Exception as e:
    print(f'Error running meta-spliceai evaluation: {e}')
    import traceback
    traceback.print_exc()
    meta_metrics = None


# Test agentic-spliceai evaluation
print('\n' + '='*60)
print('TESTING AGENTIC-SPLICEAI EVALUATION')
print('='*60)

try:
    # Build predictions in agentic-spliceai format (uses genomic positions)
    agentic_predictions = {}
    for (gene_id,), g in nucleotide_scores.group_by(['gene_id'], maintain_order=True):
        g_sorted = g.sort('genomic_position')
        first = g_sorted.row(0, named=True)
        agentic_predictions[str(gene_id)] = {
            'gene_id': str(gene_id),
            'gene_name': first.get('gene_name', ''),
            'seqname': first.get('chrom', ''),
            'chrom': first.get('chrom', ''),
            'strand': first.get('strand', '+'),
            'positions': g_sorted['genomic_position'].to_list(),
            'donor_prob': g_sorted['donor_score'].to_list(),
            'acceptor_prob': g_sorted['acceptor_score'].to_list(),
            'neither_prob': g_sorted['neither_score'].to_list(),
        }
    
    from agentic_spliceai.splice_engine.base_layer.prediction.evaluation import evaluate_splice_site_predictions
    
    error_df, positions_df, pr_metrics = evaluate_splice_site_predictions(
        predictions=agentic_predictions,
        annotations_df=splice_sites,
        threshold=0.5,
        consensus_window=2,
        collect_tn=True,
        no_tn_sampling=True,
        verbosity=0,
        return_pr_metrics=True,
    )
    
    print(f'Positions shape: {positions_df.shape}')
    agentic_metrics = calculate_metrics(positions_df, 'agentic-spliceai')
    print(f'\nPR Metrics: {pr_metrics}')
    
except Exception as e:
    print(f'Error running agentic-spliceai evaluation: {e}')
    import traceback
    traceback.print_exc()
    agentic_metrics = None


# Compare results
print('\n' + '='*60)
print('COMPARISON')
print('='*60)

if meta_metrics and agentic_metrics:
    print(f'\n{"Metric":<15} {"meta-spliceai":>15} {"agentic-spliceai":>18} {"Match":>10}')
    print('-' * 60)
    for key in ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']:
        m_val = meta_metrics[key]
        a_val = agentic_metrics[key]
        if isinstance(m_val, float):
            match = '✓' if abs(m_val - a_val) < 0.001 else '✗'
            print(f'{key:<15} {m_val:>15.4f} {a_val:>18.4f} {match:>10}')
        else:
            match = '✓' if m_val == a_val else '✗'
            print(f'{key:<15} {m_val:>15} {a_val:>18} {match:>10}')
else:
    print('Could not compare - one or both evaluations failed')
