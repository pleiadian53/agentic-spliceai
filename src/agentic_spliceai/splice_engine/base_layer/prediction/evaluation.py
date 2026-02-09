"""
Evaluation functions for splice site predictions.

This module provides functions to evaluate SpliceAI predictions against
ground truth splice site annotations, classifying positions as TP/FP/FN/TN.

Simplified port from: meta_spliceai/splice_engine/meta_models/core/enhanced_evaluation.py
"""

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict

from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def evaluate_splice_site_predictions(
    predictions: Dict[str, Dict],
    annotations_df: pl.DataFrame,
    threshold: float = 0.5,
    consensus_window: int = 2,
    collect_tn: bool = True,
    no_tn_sampling: bool = False,
    tn_sample_ratio: float = 1.2,
    verbosity: int = 1,
    return_pr_metrics: bool = False,
) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, float]]:
    """
    Evaluate splice site predictions against ground truth annotations.
    
    Parameters
    ----------
    predictions : Dict[str, Dict]
        Output from predict_splice_sites_for_genes()
    annotations_df : pl.DataFrame
        Ground truth splice site annotations with columns:
        gene_id, chrom, position, strand, splice_type (donor/acceptor)
    threshold : float, default=0.5
        Score threshold for classifying a prediction as positive
    consensus_window : int, default=2
        Window around true splice sites for matching predictions
    collect_tn : bool, default=True
        Whether to collect true negative positions
    no_tn_sampling : bool, default=False
        If True, keep all TN positions (can be very large)
    tn_sample_ratio : float, default=1.2
        Ratio of TN to (TP+FP+FN) when sampling
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Tuple[pl.DataFrame, pl.DataFrame]
        (error_df, positions_df)
        - error_df: DataFrame with FP/FN positions and error windows
        - positions_df: DataFrame with all positions and their classifications
    """
    if verbosity >= 1:
        print("[eval] Evaluating splice site predictions...")
    
    # Standardize annotation column names
    annotations_df = _standardize_annotations(annotations_df)
    
    # Evaluate donor and acceptor sites separately
    donor_results = _evaluate_site_type(
        predictions, annotations_df, 'donor', threshold, 
        consensus_window, collect_tn, verbosity
    )
    
    acceptor_results = _evaluate_site_type(
        predictions, annotations_df, 'acceptor', threshold,
        consensus_window, collect_tn, verbosity
    )
    
    # Combine results
    all_positions = donor_results['positions'] + acceptor_results['positions']
    all_errors = donor_results['errors'] + acceptor_results['errors']
    
    # Create DataFrames
    if all_positions:
        positions_df = pl.DataFrame(all_positions)
    else:
        positions_df = pl.DataFrame()
    
    if all_errors:
        error_df = pl.DataFrame(all_errors)
    else:
        error_df = pl.DataFrame()
    
    # Sample TN if needed
    if collect_tn and not no_tn_sampling and positions_df.height > 0:
        positions_df = _sample_true_negatives(
            positions_df, tn_sample_ratio, verbosity
        )
    
    if verbosity >= 1:
        _print_evaluation_summary(positions_df)

    pr_metrics: Dict[str, float] = {}
    if return_pr_metrics:
        pr_metrics = compute_pr_metrics(positions_df)

    return error_df, positions_df, pr_metrics


def compute_pr_metrics(positions_df: pl.DataFrame) -> Dict[str, float]:
    """Compute PR-AUC and Average Precision (AP) from continuous scores.

    Metrics are computed separately for donor and acceptor using the
    corresponding score column, then macro-averaged.

    Positive class is defined as positions labeled TP/FN for that splice_type.
    """
    if positions_df.height == 0:
        return {
            "donor_ap": 0.0,
            "donor_pr_auc": 0.0,
            "acceptor_ap": 0.0,
            "acceptor_pr_auc": 0.0,
            "macro_ap": 0.0,
            "macro_pr_auc": 0.0,
        }

    required = {"splice_type", "pred_type", "donor_score", "acceptor_score"}
    if not required.issubset(set(positions_df.columns)):
        return {
            "donor_ap": 0.0,
            "donor_pr_auc": 0.0,
            "acceptor_ap": 0.0,
            "acceptor_pr_auc": 0.0,
            "macro_ap": 0.0,
            "macro_pr_auc": 0.0,
        }

    def _site_metrics(site_type: str, score_col: str) -> Tuple[float, float]:
        df = positions_df.filter(pl.col("splice_type") == site_type)
        if df.height == 0:
            return 0.0, 0.0

        y_true = (
            df.select(
                pl.col("pred_type").is_in(["TP", "FN"]).cast(pl.Int8).alias("y")
            )["y"].to_list()
        )
        y_score = df[score_col].to_list()

        if sum(y_true) == 0:
            return 0.0, 0.0

        ap = float(average_precision_score(y_true, y_score))
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = float(auc(recall, precision))
        return ap, pr_auc

    donor_ap, donor_pr_auc = _site_metrics("donor", "donor_score")
    acceptor_ap, acceptor_pr_auc = _site_metrics("acceptor", "acceptor_score")

    macro_ap = float(np.mean([donor_ap, acceptor_ap]))
    macro_pr_auc = float(np.mean([donor_pr_auc, acceptor_pr_auc]))

    return {
        "donor_ap": donor_ap,
        "donor_pr_auc": donor_pr_auc,
        "acceptor_ap": acceptor_ap,
        "acceptor_pr_auc": acceptor_pr_auc,
        "macro_ap": macro_ap,
        "macro_pr_auc": macro_pr_auc,
    }


def _standardize_annotations(df: pl.DataFrame) -> pl.DataFrame:
    """Standardize annotation column names."""
    # Handle common column name variations
    rename_map = {}
    
    if 'site_type' in df.columns and 'splice_type' not in df.columns:
        rename_map['site_type'] = 'splice_type'
    if 'seqname' in df.columns and 'chrom' not in df.columns:
        rename_map['seqname'] = 'chrom'
    if 'start' in df.columns and 'position' not in df.columns:
        rename_map['start'] = 'position'
    
    if rename_map:
        df = df.rename(rename_map)
    
    return df


def _evaluate_site_type(
    predictions: Dict[str, Dict],
    annotations_df: pl.DataFrame,
    site_type: str,
    threshold: float,
    consensus_window: int,
    collect_tn: bool,
    verbosity: int
) -> Dict[str, List]:
    """
    Evaluate predictions for a specific site type (donor or acceptor).
    
    The evaluation follows meta-spliceai's approach:
    1. For each TRUE splice site, check if ANY prediction within consensus_window
       exceeds threshold. If yes → TP, otherwise → FN.
    2. For each PREDICTED position (prob >= threshold) not near any true site → FP.
    3. All other positions → TN.
    
    This avoids the bug where positions adjacent to true sites were incorrectly
    counted as FN just because they didn't individually exceed threshold.
    """
    
    prob_col = f'{site_type}_prob'
    score_col = f'{site_type}_score'
    
    positions = []
    errors = []
    
    # Filter annotations to this site type
    site_annotations = annotations_df.filter(
        pl.col('splice_type').str.to_lowercase() == site_type
    )
    
    # Build lookup of true splice sites per gene
    true_sites_by_gene = defaultdict(set)
    for row in site_annotations.iter_rows(named=True):
        gene_id = row['gene_id']
        pos = row['position']
        true_sites_by_gene[gene_id].add(pos)
    
    # Process each gene's predictions
    for gene_id, gene_data in predictions.items():
        if prob_col not in gene_data and score_col not in gene_data:
            continue
        
        probs = gene_data.get(prob_col, gene_data.get(score_col, []))
        gene_positions = gene_data.get('positions', [])
        
        if len(probs) != len(gene_positions):
            continue
        
        true_sites = true_sites_by_gene.get(gene_id, set())
        strand = gene_data.get('strand', '+')
        chrom = gene_data.get('seqname', gene_data.get('chrom', ''))
        gene_name = gene_data.get('gene_name', '')
        
        # Get other probability scores
        donor_probs = gene_data.get('donor_prob', [])
        acceptor_probs = gene_data.get('acceptor_prob', [])
        neither_probs = gene_data.get('neither_prob', [])
        
        # Build position-to-index lookup for fast access
        pos_to_idx = {pos: i for i, pos in enumerate(gene_positions)}
        
        # Track which true sites have been matched (for TP/FN determination)
        matched_true_sites = set()
        
        # Track positions that are within consensus window of any true site
        # These should NOT be counted as FP even if predicted
        positions_near_true_sites = set()
        for true_pos in true_sites:
            for offset in range(-consensus_window, consensus_window + 1):
                positions_near_true_sites.add(true_pos + offset)
        
        # STEP 1: Evaluate each TRUE splice site (TP or FN)
        for true_pos in true_sites:
            # Find max probability within consensus window
            max_prob = 0.0
            max_prob_pos = true_pos
            best_idx = None
            
            for offset in range(-consensus_window, consensus_window + 1):
                check_pos = true_pos + offset
                if check_pos in pos_to_idx:
                    idx = pos_to_idx[check_pos]
                    if probs[idx] > max_prob:
                        max_prob = probs[idx]
                        max_prob_pos = check_pos
                        best_idx = idx
            
            # Get scores at the best position (or true position if not found)
            if best_idx is not None:
                donor_score = donor_probs[best_idx] if best_idx < len(donor_probs) else 0.0
                acceptor_score = acceptor_probs[best_idx] if best_idx < len(acceptor_probs) else 0.0
                neither_score = neither_probs[best_idx] if best_idx < len(neither_probs) else 0.0
            else:
                # True site not in predictions (edge case)
                donor_score = 0.0
                acceptor_score = 0.0
                neither_score = 1.0
            
            if max_prob >= threshold:
                pred_type = 'TP'
                matched_true_sites.add(true_pos)
            else:
                pred_type = 'FN'
                errors.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'strand': strand,
                    'position': true_pos,
                    'splice_type': site_type,
                    'error_type': 'FN',
                    'score': max_prob,
                })
            
            positions.append({
                'gene_id': gene_id,
                'gene_name': gene_name,
                'chrom': chrom,
                'strand': strand,
                'position': true_pos,
                'splice_type': site_type,
                'pred_type': pred_type,
                'donor_score': donor_score,
                'acceptor_score': acceptor_score,
                'neither_score': neither_score,
            })
        
        # STEP 2: Find FPs (predicted but not near any true site)
        for i, (pos, prob) in enumerate(zip(gene_positions, probs)):
            if prob >= threshold and pos not in positions_near_true_sites:
                # This is a false positive
                donor_score = donor_probs[i] if i < len(donor_probs) else 0.0
                acceptor_score = acceptor_probs[i] if i < len(acceptor_probs) else 0.0
                neither_score = neither_probs[i] if i < len(neither_probs) else 0.0
                
                positions.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'strand': strand,
                    'position': pos,
                    'splice_type': site_type,
                    'pred_type': 'FP',
                    'donor_score': donor_score,
                    'acceptor_score': acceptor_score,
                    'neither_score': neither_score,
                })
                errors.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'strand': strand,
                    'position': pos,
                    'splice_type': site_type,
                    'error_type': 'FP',
                    'score': prob,
                })
        
        # STEP 3: Collect TNs if requested (positions not near true sites and not predicted)
        if collect_tn:
            for i, (pos, prob) in enumerate(zip(gene_positions, probs)):
                if pos not in positions_near_true_sites and prob < threshold:
                    donor_score = donor_probs[i] if i < len(donor_probs) else 0.0
                    acceptor_score = acceptor_probs[i] if i < len(acceptor_probs) else 0.0
                    neither_score = neither_probs[i] if i < len(neither_probs) else 0.0
                    
                    positions.append({
                        'gene_id': gene_id,
                        'gene_name': gene_name,
                        'chrom': chrom,
                        'strand': strand,
                        'position': pos,
                        'splice_type': site_type,
                        'pred_type': 'TN',
                        'donor_score': donor_score,
                        'acceptor_score': acceptor_score,
                        'neither_score': neither_score,
                    })
    
    return {'positions': positions, 'errors': errors}


def _is_near_true_site(pos: int, true_sites: Set[int], window: int) -> bool:
    """Check if position is within window of any true site."""
    for true_pos in true_sites:
        if abs(pos - true_pos) <= window:
            return True
    return False


def _get_nearby_sites(pos: int, true_sites: Set[int], window: int) -> Set[int]:
    """Get true sites within window of position."""
    return {ts for ts in true_sites if abs(pos - ts) <= window}


def _sample_true_negatives(
    positions_df: pl.DataFrame,
    sample_ratio: float,
    verbosity: int
) -> pl.DataFrame:
    """Sample true negatives to balance dataset."""
    
    # Count non-TN positions
    non_tn = positions_df.filter(pl.col('pred_type') != 'TN')
    tn = positions_df.filter(pl.col('pred_type') == 'TN')
    
    n_non_tn = non_tn.height
    n_tn = tn.height
    
    if n_tn == 0:
        return positions_df
    
    # Calculate target TN count
    target_tn = int(n_non_tn * sample_ratio)
    
    if target_tn >= n_tn:
        return positions_df
    
    # Sample TNs
    sampled_tn = tn.sample(n=target_tn, seed=42)
    
    if verbosity >= 1:
        print(f"[eval] Sampled TN: {n_tn} -> {target_tn}")
    
    return pl.concat([non_tn, sampled_tn])


def _print_evaluation_summary(positions_df: pl.DataFrame):
    """Print summary of evaluation results."""
    if positions_df.height == 0:
        print("[eval] No positions evaluated")
        return
    
    print("\n[eval] Evaluation Summary:")
    
    # Count by pred_type
    counts = positions_df.group_by('pred_type').count().sort('pred_type')
    for row in counts.iter_rows(named=True):
        print(f"  {row['pred_type']}: {row['count']}")
    
    # Calculate metrics
    tp = positions_df.filter(pl.col('pred_type') == 'TP').height
    fp = positions_df.filter(pl.col('pred_type') == 'FP').height
    fn = positions_df.filter(pl.col('pred_type') == 'FN').height
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"  Precision: {precision:.4f}")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"  Recall: {recall:.4f}")
    
    if tp + fp > 0 and tp + fn > 0:
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  F1 Score: {f1:.4f}")


def compute_topk_accuracy(
    predictions: Dict[str, Dict],
    annotations_df: pl.DataFrame,
    k_multipliers: List[float] = [0.5, 1.0, 2.0, 4.0],
    min_score: float = 0.1,
    verbosity: int = 1
) -> Dict[str, Dict[float, float]]:
    """
    Compute top-k accuracy as defined in the SpliceAI paper.
    
    Top-k accuracy: For each gene, rank positions by score and check if true
    splice sites appear in the top-k predictions, where k = k_multiplier * k_true.
    
    This is the ranking-based metric from: Jaganathan et al., Cell 2019
    
    Definition:
    1. For each gene and splice type (donor/acceptor)
    2. Rank all positions by predicted score (descending)
    3. Take top k = k_multiplier * n_true_sites positions
    4. Top-k accuracy = fraction of true sites in that top-k set
    
    This measures: "Do true sites rank highly when we're allowed to predict
    as many (or more) sites as truly exist?"
    
    Parameters
    ----------
    predictions : Dict[str, Dict]
        Output from predict_splice_sites_for_genes()
    annotations_df : pl.DataFrame
        Ground truth splice site annotations
    k_multipliers : List[float], default=[0.5, 1.0, 2.0, 4.0]
        Multipliers for k_true to determine top-k cutoff
        k = k_multiplier * n_true_sites in gene
        Common values: 0.5 (strict), 1.0 (match true count), 2.0, 4.0 (lenient)
    min_score : float, default=0.1
        Minimum score threshold for considering a prediction
        Prevents trivially high accuracy from "any positive probability"
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Dict[float, float]]
        Nested dict with structure:
        {
            'donor': {0.5: 0.65, 1.0: 0.85, 2.0: 0.95, 4.0: 0.98},
            'acceptor': {0.5: 0.63, 1.0: 0.83, 2.0: 0.94, 4.0: 0.97},
            'overall': {0.5: 0.64, 1.0: 0.84, 2.0: 0.945, 4.0: 0.975}
        }
        Keys are k_multipliers, values are accuracies
        
    Examples
    --------
    >>> topk = compute_topk_accuracy(predictions, annotations, k_multipliers=[1.0, 2.0])
    >>> print(f"Top-k (k=1.0x true) accuracy (donor): {topk['donor'][1.0]:.2%}")
    Top-k (k=1.0x true) accuracy (donor): 85.00%
    
    Notes
    -----
    This implements the SpliceAI paper metric where k is a rank cutoff,
    not a nucleotide window. For each gene:
    - Count true sites: n_true
    - Rank all positions by score (descending)
    - Take top k = k_multiplier * n_true positions
    - Accuracy = fraction of true sites in that top-k set
    """
    if verbosity >= 1:
        print("[eval] Computing top-k accuracy (ranking-based)...")
    
    # Standardize annotations
    annotations_df = _standardize_annotations(annotations_df)
    
    results = {
        'donor': {m: [] for m in k_multipliers},  # Store per-gene accuracies
        'acceptor': {m: [] for m in k_multipliers},
        'overall': {}
    }
    
    # Group annotations by gene
    gene_annotations = {}
    for row in annotations_df.iter_rows(named=True):
        gene_id = row['gene_id']
        if gene_id not in gene_annotations:
            gene_annotations[gene_id] = {'donor': set(), 'acceptor': set()}
        splice_type = row['splice_type']
        gene_annotations[gene_id][splice_type].add(row['position'])
    
    # Process each gene
    for gene_id, gene_pred in predictions.items():
        if gene_id not in gene_annotations:
            continue
        
        positions = gene_pred.get('positions', [])
        if len(positions) == 0:
            continue
        
        # Process each splice type
        for splice_type in ['donor', 'acceptor']:
            true_sites = gene_annotations[gene_id][splice_type]
            n_true = len(true_sites)
            
            if n_true == 0:
                continue
            
            # Get scores for this splice type
            score_key = f'{splice_type}_prob'
            scores = gene_pred.get(score_key, [])
            
            if len(scores) == 0:
                # No predictions for this gene - 0% accuracy for all k
                for m in k_multipliers:
                    results[splice_type][m].append(0.0)
                continue
            
            # Filter by minimum score threshold
            scored_positions = [
                (pos, score) for pos, score in zip(positions, scores)
                if score >= min_score
            ]
            
            if len(scored_positions) == 0:
                # No predictions above threshold - 0% accuracy
                for m in k_multipliers:
                    results[splice_type][m].append(0.0)
                continue
            
            # Sort by score (descending)
            scored_positions.sort(key=lambda x: x[1], reverse=True)
            
            # For each k multiplier
            for m in k_multipliers:
                k = int(m * n_true)
                k = max(1, k)  # At least top-1
                
                # Take top-k positions
                top_k_positions = set(pos for pos, score in scored_positions[:k])
                
                # Count how many true sites are in top-k
                hits = len(true_sites & top_k_positions)
                accuracy = hits / n_true
                
                results[splice_type][m].append(accuracy)
    
    # Average across genes for each multiplier
    final_results = {
        'donor': {},
        'acceptor': {},
        'overall': {}
    }
    
    for splice_type in ['donor', 'acceptor']:
        for m in k_multipliers:
            accuracies = results[splice_type][m]
            avg_accuracy = np.mean(accuracies) if len(accuracies) > 0 else 0.0
            final_results[splice_type][m] = avg_accuracy
    
    # Compute overall (macro-average of donor and acceptor)
    for m in k_multipliers:
        donor_acc = final_results['donor'].get(m, 0.0)
        acceptor_acc = final_results['acceptor'].get(m, 0.0)
        final_results['overall'][m] = (donor_acc + acceptor_acc) / 2.0
    
    if verbosity >= 1:
        print(f"[eval] Top-k accuracy computed for k_multipliers={k_multipliers}")
    
    return final_results


def compute_windowed_recall(
    predictions: Dict[str, Dict],
    annotations_df: pl.DataFrame,
    k_values: List[int] = [1, 2, 5, 10, 20, 50],
    min_score: float = 0.1,
    verbosity: int = 1
) -> Dict[str, Dict[int, float]]:
    """
    Compute windowed recall (tolerance-based metric).
    
    Windowed recall: For each true splice site, check if there's a predicted
    site within ±k nucleotides with score >= min_score.
    
    This is a position-tolerance metric (NOT the SpliceAI top-k accuracy).
    
    Definition:
    - For each true splice site at position p
    - Check if ANY prediction in window [p-k, p+k] has score >= min_score
    - Windowed recall = fraction of true sites with nearby prediction
    
    Parameters
    ----------
    predictions : Dict[str, Dict]
        Output from predict_splice_sites_for_genes()
    annotations_df : pl.DataFrame
        Ground truth splice site annotations
    k_values : List[int], default=[1, 2, 5, 10, 20, 50]
        List of window sizes (±k nucleotides) to compute recall for
    min_score : float, default=0.1
        Minimum score threshold for a prediction to count
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Dict[int, float]]
        Nested dict with structure:
        {
            'donor': {1: 0.75, 2: 0.85, 5: 0.92, ...},
            'acceptor': {1: 0.72, 2: 0.83, 5: 0.90, ...},
            'overall': {1: 0.735, 2: 0.84, 5: 0.91, ...}
        }
        
    Examples
    --------
    >>> windowed = compute_windowed_recall(predictions, annotations, k_values=[2, 5])
    >>> print(f"Recall within ±5bp: {windowed['donor'][5]:.2%}")
    Recall within ±5bp: 92.00%
    
    Notes
    -----
    This metric is useful for understanding localization ability and
    practical tolerance, but is NOT the SpliceAI paper's top-k accuracy.
    """
    if verbosity >= 1:
        print("[eval] Computing windowed recall (position tolerance)...")
    
    # Standardize annotations
    annotations_df = _standardize_annotations(annotations_df)
    
    results = {
        'donor': {},
        'acceptor': {},
        'overall': {}
    }
    
    # Process each splice type
    for splice_type in ['donor', 'acceptor']:
        type_annotations = annotations_df.filter(
            pl.col('splice_type') == splice_type
        )
        
        if type_annotations.height == 0:
            for k in k_values:
                results[splice_type][k] = 0.0
            continue
        
        # For each k value
        for k in k_values:
            correct = 0
            total = 0
            
            # Check each true splice site
            for row in type_annotations.iter_rows(named=True):
                gene_id = row['gene_id']
                true_pos = row['position']
                
                if gene_id not in predictions:
                    total += 1
                    continue
                
                gene_pred = predictions[gene_id]
                positions = gene_pred.get('positions', [])
                
                # Get scores for this splice type
                score_key = f'{splice_type}_prob'
                scores = gene_pred.get(score_key, [])
                
                if len(positions) == 0 or len(scores) == 0:
                    total += 1
                    continue
                
                # Find predictions within ±k window above threshold
                window_start = true_pos - k
                window_end = true_pos + k
                
                found = False
                for pos, score in zip(positions, scores):
                    if window_start <= pos <= window_end and score >= min_score:
                        found = True
                        break
                
                if found:
                    correct += 1
                
                total += 1
            
            # Calculate recall for this k
            recall = correct / total if total > 0 else 0.0
            results[splice_type][k] = recall
    
    # Compute overall (macro-average of donor and acceptor)
    for k in k_values:
        donor_recall = results['donor'].get(k, 0.0)
        acceptor_recall = results['acceptor'].get(k, 0.0)
        results['overall'][k] = (donor_recall + acceptor_recall) / 2.0
    
    if verbosity >= 1:
        print(f"[eval] Windowed recall computed for k_values={k_values}")
    
    return results


def filter_annotations_by_transcript(
    annotations_df: pl.DataFrame,
    mode: str = 'canonical',
    transcript_ids: Optional[List[str]] = None,
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Filter annotations to specific transcript(s).
    
    Many genes have multiple transcripts, each contributing splice site
    annotations. SpliceAI primarily predicts canonical splice sites from the
    major transcript. Evaluating against ALL transcripts inflates the
    denominator and deflates recall.
    
    This function filters annotations to reduce evaluation to the most
    relevant splice sites for a given analysis context.
    
    Parameters
    ----------
    annotations_df : pl.DataFrame
        Full annotations with columns: gene_id, transcript_id, position, 
        site_type/splice_type, etc.
    mode : str, default='canonical'
        Filtering mode:
        - 'all': No filtering (return as-is)
        - 'canonical': Keep only the transcript with the most splice sites
          per gene (proxy for canonical/principal transcript)
        - 'shared': Keep only positions that appear in 2+ transcripts
          (high-confidence sites shared across isoforms)
        - 'custom': Use the transcript_ids parameter
    transcript_ids : list of str, optional
        Explicit transcript IDs to keep (only used when mode='custom')
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        Filtered annotations
        
    Notes
    -----
    The 'canonical' mode uses heuristics (most splice sites = likely
    principal transcript). For precise filtering, use MANE Select or
    APPRIS principal annotations if available.
    """
    # Standardize column names for internal processing
    site_col = 'splice_type' if 'splice_type' in annotations_df.columns else 'site_type'
    
    if mode == 'all':
        if verbosity >= 1:
            print("[filter] Using all transcripts (no filtering)")
        return annotations_df
    
    if mode == 'custom':
        if transcript_ids is None:
            raise ValueError("transcript_ids required when mode='custom'")
        filtered = annotations_df.filter(pl.col('transcript_id').is_in(transcript_ids))
        if verbosity >= 1:
            print(f"[filter] Custom transcript filter: {annotations_df.height} -> {filtered.height} annotations")
        return filtered
    
    if mode == 'canonical':
        # For each gene, find the transcript with the most annotated sites
        # This is a heuristic proxy for the canonical/principal transcript
        if 'transcript_id' not in annotations_df.columns or 'gene_id' not in annotations_df.columns:
            if verbosity >= 1:
                print("[filter] Cannot filter: missing transcript_id or gene_id column")
            return annotations_df
        
        # Count unique positions per (gene, transcript)
        gene_transcript_counts = (
            annotations_df
            .group_by(['gene_id', 'transcript_id'])
            .agg(pl.col('position').n_unique().alias('n_sites'))
        )
        
        # For each gene, keep the transcript with the most sites
        canonical_transcripts = (
            gene_transcript_counts
            .sort(['gene_id', 'n_sites'], descending=[False, True])
            .group_by('gene_id')
            .first()
            .select(['gene_id', 'transcript_id'])
        )
        
        # Join back to filter annotations
        filtered = annotations_df.join(
            canonical_transcripts, on=['gene_id', 'transcript_id'], how='inner'
        )
        
        if verbosity >= 1:
            n_genes = canonical_transcripts.height
            print(f"[filter] Canonical transcript filter: {annotations_df.height} -> {filtered.height} annotations "
                  f"({n_genes} gene(s), 1 transcript each)")
            for row in canonical_transcripts.iter_rows(named=True):
                gene_sites = filtered.filter(pl.col('gene_id') == row['gene_id'])
                n_unique = gene_sites['position'].n_unique()
                print(f"   {row['gene_id']}: {row['transcript_id']} ({n_unique} unique sites)")
        
        return filtered
    
    if mode == 'shared':
        # Keep only positions that appear in 2+ transcripts
        # These are high-confidence sites shared across isoforms
        if 'transcript_id' not in annotations_df.columns:
            if verbosity >= 1:
                print("[filter] Cannot filter: missing transcript_id column")
            return annotations_df
        
        # Count transcripts per (gene, position, site_type)
        pos_transcript_counts = (
            annotations_df
            .group_by(['gene_id', 'position', site_col])
            .agg(pl.col('transcript_id').n_unique().alias('n_transcripts'))
        )
        
        # Keep positions with 2+ transcripts
        shared_positions = pos_transcript_counts.filter(pl.col('n_transcripts') >= 2)
        
        filtered = annotations_df.join(
            shared_positions.select(['gene_id', 'position', site_col]),
            on=['gene_id', 'position', site_col],
            how='inner'
        )
        
        if verbosity >= 1:
            n_unique_before = annotations_df.select(['gene_id', 'position', site_col]).unique().height
            n_unique_after = shared_positions.height
            print(f"[filter] Shared-position filter: {n_unique_before} -> {n_unique_after} unique sites "
                  f"(positions in 2+ transcripts)")
        
        return filtered
    
    raise ValueError(f"Unknown filter mode: {mode}. Use 'all', 'canonical', 'shared', or 'custom'.")


def splice_site_gap_analysis(
    predictions: Dict[str, Dict],
    annotations_df: pl.DataFrame,
    threshold: float = 0.5,
    nearby_window: int = 5,
    verbosity: int = 1
) -> Dict[str, Any]:
    """
    Analyze the gap between base model predictions and ground truth annotations.
    
    For each gene, categorizes missed splice sites by:
    - How many transcripts share that position (canonical vs alternative)
    - Whether ANY nearby signal exists (meta-layer recovery potential)
    - Score distribution at missed sites
    
    This analysis directly informs what the meta/agentic layers need to address.
    
    Parameters
    ----------
    predictions : Dict[str, Dict]
        Output from predict_splice_sites_for_genes()
    annotations_df : pl.DataFrame
        Ground truth annotations (all transcripts)
    threshold : float, default=0.5
        Score threshold for detection
    nearby_window : int, default=5
        Window (bp) to search for nearby signals at missed sites
        
    Returns
    -------
    Dict with keys:
        'summary': Overall summary dict
        'per_gene': Per-gene breakdown dicts
        'missed_sites': DataFrame of missed sites with metadata
        'recovery_potential': Assessment of what meta/agentic layers can address
    
    Notes
    -----
    This is designed to produce the kind of analysis shown in the example:
    
        "The 14 missed donor positions:
         - 8 from a single minor transcript (score = 0.0 within ±5bp)
         - 6 from 2-4 alternative transcripts with non-canonical patterns"
    
    It also assesses recovery potential for the meta layer by checking if there
    is ANY signal (however weak) near missed sites that could serve as features.
    """
    # Standardize column names
    annotations_df = _standardize_annotations(annotations_df)
    site_col = 'splice_type'
    
    all_missed_sites = []
    per_gene_results = {}
    
    # Build transcript mapping for each position
    pos_transcript_map = {}  # (gene_id, position, splice_type) -> set of transcript_ids
    for row in annotations_df.iter_rows(named=True):
        key = (row['gene_id'], row['position'], row[site_col])
        if key not in pos_transcript_map:
            pos_transcript_map[key] = set()
        if 'transcript_id' in row and row['transcript_id']:
            pos_transcript_map[key].add(row['transcript_id'])
    
    for gene_id, gene_data in predictions.items():
        positions = gene_data.get('positions', [])
        strand = gene_data.get('strand', '+')
        gene_name = gene_data.get('gene_name', gene_id)
        
        if len(positions) == 0:
            continue
        
        pos_to_idx = {p: i for i, p in enumerate(positions)}
        gene_result = {'gene_name': gene_name, 'strand': strand}
        
        for splice_type in ['donor', 'acceptor']:
            probs = np.array(gene_data.get(f'{splice_type}_prob', []))
            if len(probs) == 0:
                continue
            
            # Get unique true positions for this gene/type
            gene_annots = annotations_df.filter(
                (pl.col('gene_id') == gene_id) &
                (pl.col(site_col).str.to_lowercase() == splice_type)
            )
            unique_positions = sorted(set(gene_annots['position'].to_list()))
            
            detected = []
            missed = []
            
            for true_pos in unique_positions:
                key = (gene_id, true_pos, splice_type)
                transcripts = pos_transcript_map.get(key, set())
                n_transcripts = len(transcripts)
                
                # Check score at this position and nearby
                score_at_pos = 0.0
                max_nearby_score = 0.0
                max_nearby_offset = 0
                nearby_signals = []  # Positions with score >= 0.01
                
                if true_pos in pos_to_idx:
                    idx = pos_to_idx[true_pos]
                    score_at_pos = float(probs[idx])
                
                for off in range(-nearby_window, nearby_window + 1):
                    check_pos = true_pos + off
                    if check_pos in pos_to_idx:
                        s = float(probs[pos_to_idx[check_pos]])
                        if s > max_nearby_score:
                            max_nearby_score = s
                            max_nearby_offset = off
                        if s >= 0.01:
                            nearby_signals.append((off, s))
                
                site_info = {
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'strand': strand,
                    'splice_type': splice_type,
                    'position': true_pos,
                    'score_at_position': score_at_pos,
                    'max_nearby_score': max_nearby_score,
                    'max_nearby_offset': max_nearby_offset,
                    'n_transcripts': n_transcripts,
                    'n_nearby_signals': len(nearby_signals),
                    'has_weak_signal': max_nearby_score >= 0.01,
                    'has_moderate_signal': max_nearby_score >= 0.1,
                }
                
                if score_at_pos >= threshold:
                    detected.append(site_info)
                else:
                    missed.append(site_info)
                    all_missed_sites.append(site_info)
            
            # Categorize missed sites
            single_transcript_missed = [m for m in missed if m['n_transcripts'] <= 1]
            multi_transcript_missed = [m for m in missed if m['n_transcripts'] >= 2]
            no_signal_missed = [m for m in missed if not m['has_weak_signal']]
            weak_signal_missed = [m for m in missed if m['has_weak_signal'] and not m['has_moderate_signal']]
            moderate_signal_missed = [m for m in missed if m['has_moderate_signal'] and m['score_at_position'] < threshold]
            
            gene_result[splice_type] = {
                'total_unique': len(unique_positions),
                'detected': len(detected),
                'missed': len(missed),
                'recall': len(detected) / len(unique_positions) if unique_positions else 0,
                'single_transcript_missed': len(single_transcript_missed),
                'multi_transcript_missed': len(multi_transcript_missed),
                'no_signal': len(no_signal_missed),
                'weak_signal_only': len(weak_signal_missed),
                'moderate_signal': len(moderate_signal_missed),
                'missed_details': missed,
            }
        
        per_gene_results[gene_id] = gene_result
    
    # Overall summary
    total_unique = sum(g.get(st, {}).get('total_unique', 0) 
                       for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_detected = sum(g.get(st, {}).get('detected', 0) 
                         for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_missed = sum(g.get(st, {}).get('missed', 0) 
                       for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_no_signal = sum(g.get(st, {}).get('no_signal', 0) 
                          for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_weak = sum(g.get(st, {}).get('weak_signal_only', 0) 
                     for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_moderate = sum(g.get(st, {}).get('moderate_signal', 0) 
                         for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_single_tx = sum(g.get(st, {}).get('single_transcript_missed', 0) 
                          for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    total_multi_tx = sum(g.get(st, {}).get('multi_transcript_missed', 0) 
                         for g in per_gene_results.values() for st in ['donor', 'acceptor'])
    
    summary = {
        'total_unique_sites': total_unique,
        'detected': total_detected,
        'missed': total_missed,
        'overall_recall': total_detected / total_unique if total_unique > 0 else 0,
        'missed_no_signal': total_no_signal,
        'missed_weak_signal': total_weak,
        'missed_moderate_signal': total_moderate,
        'missed_single_transcript': total_single_tx,
        'missed_multi_transcript': total_multi_tx,
    }
    
    # Recovery potential assessment
    recovery = {
        'score_based_recoverable': total_moderate,
        'context_feature_potential': total_weak + total_no_signal,
        'no_base_model_signal': total_no_signal,
        'assessment': [],
    }
    
    if total_moderate > 0:
        recovery['assessment'].append(
            f"{total_moderate} site(s) have moderate signal (0.1-{threshold}) - "
            f"meta layer can likely recover these using score-based features"
        )
    if total_weak > 0:
        recovery['assessment'].append(
            f"{total_weak} site(s) have only weak signal (0.01-0.1) - "
            f"meta layer may recover with contextual sequence features"
        )
    if total_no_signal > 0:
        recovery['assessment'].append(
            f"{total_no_signal} site(s) have NO base model signal (<0.01 within ±{nearby_window}bp) - "
            f"requires external evidence (conservation, epigenomics, variant data) or "
            f"alternative base models"
        )
    
    # Create missed sites DataFrame
    missed_df = pl.DataFrame(all_missed_sites) if all_missed_sites else pl.DataFrame()
    
    # Print report if verbose
    if verbosity >= 1:
        _print_gap_analysis_report(summary, per_gene_results, recovery, threshold, nearby_window)
    
    return {
        'summary': summary,
        'per_gene': per_gene_results,
        'missed_sites': missed_df,
        'recovery_potential': recovery,
    }


def _print_gap_analysis_report(
    summary: Dict,
    per_gene: Dict,
    recovery: Dict,
    threshold: float,
    nearby_window: int
):
    """Print formatted gap analysis report."""
    print(f"\n{'='*70}")
    print(f"  Splice Site Gap Analysis (Base Model vs All Transcripts)")
    print(f"{'='*70}")
    
    total = summary['total_unique_sites']
    detected = summary['detected']
    missed = summary['missed']
    recall = summary['overall_recall']
    
    print(f"\n  Overall: {detected}/{total} unique sites detected ({recall:.1%})")
    print(f"  Missed:  {missed} sites")
    
    if missed > 0:
        print(f"\n  Missed site breakdown:")
        st = summary['missed_single_transcript']
        mt = summary['missed_multi_transcript']
        ns = summary['missed_no_signal']
        ws = summary['missed_weak_signal']
        ms = summary['missed_moderate_signal']
        
        print(f"    By transcript sharing:")
        print(f"      Single-transcript only: {st:>3} ({100*st/missed:.0f}%) - alternative isoform-specific")
        print(f"      Multi-transcript (2+):  {mt:>3} ({100*mt/missed:.0f}%) - shared but below threshold")
        print(f"    By base model signal (within ±{nearby_window}bp):")
        print(f"      No signal (<0.01):      {ns:>3} ({100*ns/missed:.0f}%) - invisible to base model")
        print(f"      Weak (0.01-0.1):        {ws:>3} ({100*ws/missed:.0f}%) - faint hint exists")
        print(f"      Moderate (0.1-{threshold}):     {ms:>3} ({100*ms/missed:.0f}%) - near threshold")
    
    # Per-gene details
    for gene_id, gene_data in per_gene.items():
        gene_name = gene_data.get('gene_name', gene_id)
        strand = gene_data.get('strand', '?')
        print(f"\n  --- {gene_name} ({gene_id}, {strand} strand) ---")
        
        for splice_type in ['donor', 'acceptor']:
            if splice_type not in gene_data:
                continue
            st_data = gene_data[splice_type]
            total_st = st_data['total_unique']
            det_st = st_data['detected']
            recall_st = st_data['recall']
            
            print(f"    {splice_type.capitalize()}: {det_st}/{total_st} ({recall_st:.0%})")
            
            missed_details = st_data.get('missed_details', [])
            if missed_details:
                for m in missed_details:
                    sig = "no signal" if not m['has_weak_signal'] else \
                          f"weak ({m['max_nearby_score']:.3f}@{m['max_nearby_offset']:+d})" if not m['has_moderate_signal'] else \
                          f"moderate ({m['score_at_position']:.3f}, nearby max {m['max_nearby_score']:.3f}@{m['max_nearby_offset']:+d})"
                    print(f"      {m['position']:>10}: {m['n_transcripts']} transcript(s), {sig}")
    
    # Recovery potential
    if recovery['assessment']:
        print(f"\n  Meta/Agentic Layer Recovery Potential:")
        for line in recovery['assessment']:
            print(f"    - {line}")
    
    print(f"\n{'='*70}")


def add_derived_features(
    positions_df: pl.DataFrame,
    verbosity: int = 1
) -> pl.DataFrame:
    """
    Add derived probability features useful for meta-modeling.
    
    Parameters
    ----------
    positions_df : pl.DataFrame
        Positions DataFrame with donor_score, acceptor_score, neither_score
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    pl.DataFrame
        DataFrame with additional derived features
    """
    if positions_df.height == 0:
        return positions_df
    
    if verbosity >= 1:
        print("[features] Adding derived probability features...")
    
    epsilon = 1e-10
    
    # Add derived features
    positions_df = positions_df.with_columns([
        # Normalized probabilities
        (pl.col("donor_score") / (
            pl.col("donor_score") + pl.col("acceptor_score") + epsilon
        )).alias("relative_donor_probability"),
        
        # Splice probability (donor + acceptor)
        ((pl.col("donor_score") + pl.col("acceptor_score")) / (
            pl.col("donor_score") + pl.col("acceptor_score") + 
            pl.col("neither_score") + epsilon
        )).alias("splice_probability"),
        
        # Donor-acceptor difference
        ((pl.col("donor_score") - pl.col("acceptor_score")) / (
            pl.max_horizontal([pl.col("donor_score"), pl.col("acceptor_score")]) + epsilon
        )).alias("donor_acceptor_diff"),
        
        # Log-odds transformations
        (pl.col("donor_score").add(epsilon).log() - 
         pl.col("acceptor_score").add(epsilon).log()
        ).alias("donor_acceptor_logodds"),
        
        # Entropy (uncertainty measure)
        (-pl.col("donor_score") * pl.col("donor_score").add(epsilon).log()
         -pl.col("acceptor_score") * pl.col("acceptor_score").add(epsilon).log()
         -pl.col("neither_score") * pl.col("neither_score").add(epsilon).log()
        ).alias("probability_entropy"),
    ])
    
    return positions_df
