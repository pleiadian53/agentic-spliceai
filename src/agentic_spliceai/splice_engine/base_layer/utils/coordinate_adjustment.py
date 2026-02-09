"""
Automated coordinate adjustment detection for splice site predictions.

This module provides utilities to:
1. Detect optimal position adjustments by comparing predictions to annotations
2. Apply adjustments to score arrays using np.roll()
3. Cache detected adjustments for reuse

Key difference from simple position shifting:
- Applies np.roll() to score arrays BEFORE peak detection
- Allows creation of new peaks that weren't visible at original positions
- Critical for models like SpliceAI with systematic coordinate offsets

Ported from: meta_spliceai/splice_engine/meta_models/utils/infer_splice_site_adjustments.py
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import copy

import numpy as np
import polars as pl


# ============================================================================
# Utility Functions
# ============================================================================

def normalize_strand(strand: str) -> str:
    """Normalize strand notation to '+' or '-'."""
    if strand in ['+', '1', 1, 'plus', 'forward']:
        return '+'
    elif strand in ['-', '-1', -1, 'minus', 'reverse']:
        return '-'
    else:
        raise ValueError(f"Invalid strand value: {strand}")


# ============================================================================
# Core Adjustment Functions
# ============================================================================

def adjust_scores_hardcoded(
    scores: np.ndarray,
    strand: str,
    splice_type: str,
    is_neither_prob: bool = False,
    normalize_edges: bool = True
) -> np.ndarray:
    """
    Apply hardcoded SpliceAI adjustments to score arrays.
    
    These offsets are empirically validated from extensive testing:
    - Donor + strand: predicts 2nt upstream → roll forward by +2
    - Donor - strand: predicts 1nt upstream → roll forward by +1
    - Acceptor + strand: exact (no offset) → no roll
    - Acceptor - strand: predicts 1nt downstream → roll backward by -1
    
    Parameters
    ----------
    scores : np.ndarray
        Array of splice site probabilities (donor, acceptor, or neither)
    strand : str
        Gene strand ('+' or '-')
    splice_type : str
        Site type ('donor' or 'acceptor')
    is_neither_prob : bool
        Whether this array represents 'neither' probabilities
    normalize_edges : bool
        Whether to set edge values (0 for splice probs, 1 for neither)
        
    Returns
    -------
    np.ndarray
        Adjusted scores with np.roll applied
        
    Notes
    -----
    Uses np.roll() which shifts the entire array circularly.
    Edge positions are then set to 0 (splice) or 1 (neither) to avoid artifacts.
    """
    adjusted_scores = scores.copy()
    
    if splice_type == 'donor':
        if strand == '+':
            adjusted_scores = np.roll(adjusted_scores, 2)  # Shift forward by 2nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[:2] = 1.0
                else:
                    adjusted_scores[:2] = 0
            else:
                adjusted_scores[:2] = 0
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, 1)  # Shift forward by 1nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[:1] = 1.0
                else:
                    adjusted_scores[:1] = 0
            else:
                adjusted_scores[:1] = 0
        else:
            raise ValueError(f"Invalid strand value: {strand}")
            
    elif splice_type == 'acceptor':
        if strand == '+':
            # No adjustment for + strand acceptor sites in SpliceAI
            pass
        elif strand == '-':
            adjusted_scores = np.roll(adjusted_scores, -1)  # Shift backward by 1nt
            if normalize_edges:
                if is_neither_prob:
                    adjusted_scores[-1:] = 1.0
                else:
                    adjusted_scores[-1:] = 0
            else:
                adjusted_scores[-1:] = 0
        else:
            raise ValueError(f"Invalid strand value: {strand}")
    else:
        raise ValueError(f"Invalid splice type: {splice_type}")
    
    return adjusted_scores


def apply_custom_adjustments(
    scores: np.ndarray,
    strand: str,
    splice_type: str,
    adjustment_dict: Dict[str, Dict[str, int]],
    is_neither_prob: bool = False,
    normalize_edges: bool = True
) -> np.ndarray:
    """
    Apply custom position adjustments from a dictionary.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of splice site probabilities
    strand : str
        Gene strand ('+' or '-')
    splice_type : str
        Site type ('donor' or 'acceptor')
    adjustment_dict : dict
        Adjustments in format:
        {'donor': {'plus': offset, 'minus': offset},
         'acceptor': {'plus': offset, 'minus': offset}}
    is_neither_prob : bool
        Whether this array represents 'neither' probabilities
    normalize_edges : bool
        Whether to set edge values after rolling
        
    Returns
    -------
    np.ndarray
        Adjusted scores
    """
    # Convert to numpy array if needed
    if isinstance(scores, list):
        scores = np.array(scores)
    
    # Make a copy to avoid modifying the original
    adjusted_scores = scores.copy()
    
    # No adjustments if dict is not provided
    if adjustment_dict is None:
        return adjusted_scores
    
    # Normalize strand and get the corresponding key
    norm_strand = normalize_strand(strand)
    strand_key = 'plus' if norm_strand == '+' else 'minus'
    
    # Check if we have an adjustment for this site type and strand
    if splice_type in adjustment_dict and strand_key in adjustment_dict[splice_type]:
        offset = adjustment_dict[splice_type][strand_key]
        
        # Skip if no adjustment needed
        if offset == 0:
            return adjusted_scores
        
        # Apply the roll/shift
        adjusted_scores = np.roll(adjusted_scores, offset)
        
        # Zero out the wrapped-around values
        if offset > 0:
            if normalize_edges and is_neither_prob:
                # For neither probabilities at the edge, set to 1.0
                adjusted_scores[:offset] = 1.0
            else:
                # For donor or acceptor probabilities at the edge, set to 0
                adjusted_scores[:offset] = 0
        elif offset < 0:
            if normalize_edges and is_neither_prob:
                # For neither probabilities at the edge, set to 1.0
                adjusted_scores[offset:] = 1.0
            else:
                # For donor or acceptor probabilities at the edge, set to 0
                adjusted_scores[offset:] = 0
    
    return adjusted_scores


# ============================================================================
# Empirical Detection (Complex - port from MetaSpliceAI)
# ============================================================================

def empirical_infer_adjustments(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Any],
    search_range: Tuple[int, int] = (-5, 5),
    min_genes_per_category: int = 3,
    consensus_window: int = 2,
    probability_threshold: float = 0.4,
    min_improvement: float = 0.2,
    verbose: bool = False
) -> Tuple[Dict[str, Dict[str, int]], Dict]:
    """
    Empirically infer optimal position adjustments by testing different offsets.
    
    Algorithm:
    1. Group genes by strand and splice type
    2. For each combination (donor/acceptor × +/- strand):
       a. Test offsets from search_range[0] to search_range[1]
       b. Apply np.roll() to score arrays
       c. Find peaks and compare to ground truth
       d. Calculate F1 score
    3. Select offset with highest F1 score (if improvement > threshold)
    
    Parameters
    ----------
    annotations_df : pl.DataFrame
        Ground truth annotations with columns: gene_id, site_type, strand, position
    pred_results : dict
        Predictions in format: {gene_id: {'donor_prob': array, 'acceptor_prob': array, ...}}
    search_range : tuple
        (min_offset, max_offset) to test (inclusive)
    min_genes_per_category : int
        Minimum genes required for each strand+type combination
    consensus_window : int
        Window size for matching predictions to truth (±window bp)
    probability_threshold : float
        Minimum score to consider a peak
    min_improvement : float
        Minimum F1 improvement required to accept an offset
    verbose : bool
        Print progress information
        
    Returns
    -------
    tuple
        - optimal_adjustments: dict with format {'donor': {'plus': int, 'minus': int}, ...}
        - adjustment_stats: detailed statistics for all tested offsets
        
    Notes
    -----
    This function is computationally expensive (~30-60s for 50 genes).
    Results should be cached to avoid repeated computation.
    """
    if verbose:
        print(f"\n🔍 Starting empirical adjustment detection")
        print(f"   Analyzing {len(pred_results)} genes")
        print(f"   Testing offsets: {search_range[0]} to {search_range[1]}")
    
    # Extract gene metadata (strand, splice types)
    gene_metadata = {}
    gene_ids = sorted(pred_results.keys())
    
    for gene_id in gene_ids:
        gene_annots = annotations_df.filter(pl.col("gene_id") == gene_id)
        if gene_annots.height == 0:
            continue
        
        strands = gene_annots["strand"].unique().to_list()
        strand = strands[0]  # Take first if multiple
        
        splice_types = gene_annots["site_type"].unique().to_list()
        
        gene_metadata[gene_id] = {
            "strand": strand,
            "splice_types": splice_types
        }
    
    # Group genes by strand and site type
    plus_strand_genes = [g for g, meta in gene_metadata.items() if meta["strand"] == "+"]
    minus_strand_genes = [g for g, meta in gene_metadata.items() if meta["strand"] == "-"]
    
    donor_genes = [g for g, meta in gene_metadata.items() if "donor" in meta["splice_types"]]
    acceptor_genes = [g for g, meta in gene_metadata.items() if "acceptor" in meta["splice_types"]]
    
    # Create category groups
    categories = {
        "donor_plus": [g for g in donor_genes if g in plus_strand_genes],
        "donor_minus": [g for g in donor_genes if g in minus_strand_genes],
        "acceptor_plus": [g for g in acceptor_genes if g in plus_strand_genes],
        "acceptor_minus": [g for g in acceptor_genes if g in minus_strand_genes]
    }
    
    if verbose:
        print(f"\n   Gene counts by category:")
        for category, genes in categories.items():
            print(f"      {category}: {len(genes)} genes")
    
    # Check for insufficient genes
    insufficient = [cat for cat, genes in categories.items() if len(genes) < min_genes_per_category]
    if insufficient and verbose:
        print(f"   ⚠️  Insufficient genes in: {insufficient}")
        print(f"      Consider more genes or lower min_genes_per_category (current: {min_genes_per_category})")
    
    # Define combinations to test
    combinations = [
        {"splice_type": "donor", "strand": "+"},
        {"splice_type": "donor", "strand": "-"},
        {"splice_type": "acceptor", "strand": "+"},
        {"splice_type": "acceptor", "strand": "-"}
    ]
    
    # Initialize results tracking
    adjustment_stats = {
        "donor": {"plus": {}, "minus": {}},
        "acceptor": {"plus": {}, "minus": {}}
    }
    
    # Test each combination
    for combo in combinations:
        splice_type = combo["splice_type"]
        strand = combo["strand"]
        strand_key = "plus" if strand == "+" else "minus"
        
        category_genes = categories[f"{splice_type}_{strand_key}"]
        if len(category_genes) < min_genes_per_category:
            if verbose:
                print(f"\n   ⏭️  Skipping {splice_type} {strand} (insufficient genes)")
            continue
        
        if verbose:
            print(f"\n   📊 Testing {splice_type} sites on {strand} strand ({len(category_genes)} genes)")
        
        # Filter annotations to this category
        category_annots = annotations_df.filter(
            (pl.col("site_type") == splice_type) & 
            (pl.col("strand") == strand) &
            (pl.col("gene_id").is_in(category_genes))
        )
        
        # Test each offset
        for offset in range(search_range[0], search_range[1] + 1):
            # Create adjustment dict for this test
            test_adjustment = {
                "donor": {"plus": 0, "minus": 0},
                "acceptor": {"plus": 0, "minus": 0}
            }
            test_adjustment[splice_type][strand_key] = offset
            
            # Evaluate with this adjustment (simplified version - full evaluation would use enhanced_evaluate)
            # For now, we'll do a basic TP/FP/FN count
            tp_count, fp_count, fn_count = _evaluate_with_adjustment(
                category_annots,
                {gene_id: pred_results[gene_id] for gene_id in category_genes},
                test_adjustment,
                splice_type,
                strand,
                probability_threshold,
                consensus_window
            )
            
            # Calculate metrics
            total_relevant = tp_count + fn_count
            recall = tp_count / total_relevant if total_relevant > 0 else 0
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store stats
            adjustment_stats[splice_type][strand_key][offset] = {
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
            
            if verbose:
                print(f"      Offset {offset:+3d}: TP={tp_count:3d}, FP={fp_count:3d}, FN={fn_count:3d}, F1={f1:.3f}")
    
    # Determine optimal adjustments
    optimal_adjustments = {
        "donor": {"plus": 0, "minus": 0},
        "acceptor": {"plus": 0, "minus": 0}
    }
    
    for splice_type in ["donor", "acceptor"]:
        for strand_key in ["plus", "minus"]:
            # Check if we have stats
            if not adjustment_stats[splice_type][strand_key]:
                continue
            
            # Find offset with highest F1
            best_offset = max(
                adjustment_stats[splice_type][strand_key].keys(),
                key=lambda k: adjustment_stats[splice_type][strand_key][k]["f1"]
            )
            
            # Compare to baseline (offset=0)
            baseline_f1 = adjustment_stats[splice_type][strand_key].get(0, {"f1": 0})["f1"]
            best_f1 = adjustment_stats[splice_type][strand_key][best_offset]["f1"]
            
            # Only apply if improvement is significant
            if best_f1 > baseline_f1 + min_improvement:
                optimal_adjustments[splice_type][strand_key] = best_offset
                if verbose:
                    improvement = best_f1 - baseline_f1
                    print(f"\n   ✅ Selected {best_offset:+d} for {splice_type} {strand_key} (F1 +{improvement:.3f})")
            else:
                if verbose:
                    print(f"\n   ⏺️  No improvement for {splice_type} {strand_key}, keeping 0")
    
    if verbose:
        print(f"\n🎯 Final adjustments:")
        print(f"   Donor:    +{optimal_adjustments['donor']['plus']:+d} (+ strand), {optimal_adjustments['donor']['minus']:+d} (- strand)")
        print(f"   Acceptor: {optimal_adjustments['acceptor']['plus']:+d} (+ strand), {optimal_adjustments['acceptor']['minus']:+d} (- strand)")
    
    return optimal_adjustments, adjustment_stats


def _evaluate_with_adjustment(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Any],
    adjustment_dict: Dict,
    splice_type: str,
    strand: str,
    threshold: float,
    window: int
) -> Tuple[int, int, int]:
    """
    Helper function to evaluate predictions with adjustments.
    
    This is a simplified version - full implementation would use enhanced_evaluate
    from the evaluation module.
    
    Returns
    -------
    tuple
        (tp_count, fp_count, fn_count)
    """
    # Get true positions
    true_positions = set(annotations_df['position'].to_list())
    
    # Get predicted positions with adjustment applied
    predicted_positions = set()
    
    for gene_id, gene_pred in pred_results.items():
        # Get relevant score array
        prob_key = f'{splice_type}_prob'
        if prob_key not in gene_pred:
            continue
        
        scores = gene_pred[prob_key]
        if isinstance(scores, list):
            scores = np.array(scores)
        
        # Apply adjustment
        adjusted_scores = apply_custom_adjustments(
            scores, strand, splice_type, adjustment_dict,
            is_neither_prob=False, normalize_edges=True
        )
        
        # Find peaks above threshold
        peaks = np.where(adjusted_scores >= threshold)[0]
        
        # Convert relative positions to absolute (if gene has position info)
        if 'positions' in gene_pred:
            abs_positions = [gene_pred['positions'][p] for p in peaks if p < len(gene_pred['positions'])]
            predicted_positions.update(abs_positions)
    
    # Calculate TP/FP/FN with window matching
    tp_set = set()
    for true_pos in true_positions:
        for pred_pos in predicted_positions:
            if abs(true_pos - pred_pos) <= window:
                tp_set.add(true_pos)
                break
    
    tp_count = len(tp_set)
    fn_count = len(true_positions) - tp_count
    fp_count = len(predicted_positions) - tp_count
    
    return tp_count, fp_count, fn_count


# ============================================================================
# High-Level Auto-Detection
# ============================================================================

def auto_detect_adjustments(
    annotations_df: pl.DataFrame,
    pred_results: Dict[str, Any],
    use_empirical: bool = True,
    consensus_window: int = 2,
    threshold: float = 0.4,
    verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Automatically detect optimal adjustments (empirical or hardcoded).
    
    Parameters
    ----------
    annotations_df : pl.DataFrame
        Ground truth annotations
    pred_results : dict
        Prediction results
    use_empirical : bool
        If True, detect empirically from data
        If False, use hardcoded SpliceAI pattern
    consensus_window : int
        Window for matching (±window bp)
    threshold : float
        Minimum probability threshold
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Adjustments: {'donor': {'plus': int, 'minus': int}, 'acceptor': {...}}
    """
    if use_empirical:
        if verbose:
            print("🔬 Using empirical (data-driven) adjustment detection")
        
        adjustments, _ = empirical_infer_adjustments(
            annotations_df=annotations_df,
            pred_results=pred_results,
            consensus_window=consensus_window,
            probability_threshold=threshold,
            verbose=verbose
        )
        
        return adjustments
    else:
        # Use hardcoded SpliceAI pattern
        spliceai_pattern = {
            'donor': {'plus': 2, 'minus': 1},
            'acceptor': {'plus': 0, 'minus': -1}
        }
        
        if verbose:
            print("📚 Using hardcoded SpliceAI adjustment pattern")
            print(f"   Donor:    +{spliceai_pattern['donor']['plus']} (+ strand), +{spliceai_pattern['donor']['minus']} (- strand)")
            print(f"   Acceptor: +{spliceai_pattern['acceptor']['plus']} (+ strand), {spliceai_pattern['acceptor']['minus']} (- strand)")
        
        return spliceai_pattern


# ============================================================================
# Caching Functions
# ============================================================================

def get_adjustment_cache_path(
    model_type: str,
    build: str,
    annotation_source: str,
    data_root: Optional[Path] = None
) -> Path:
    """
    Get path to adjustment cache file.
    
    Parameters
    ----------
    model_type : str
        Model name (e.g., 'spliceai', 'openspliceai')
    build : str
        Genome build (e.g., 'GRCh37', 'GRCh38')
    annotation_source : str
        Annotation source (e.g., 'ensembl', 'mane')
    data_root : Path, optional
        Root data directory (if None, uses model resources)
        
    Returns
    -------
    Path
        Path to cache file: {data_root}/{annotation_source}/{build}/{model_type}_coordinate_adjustments.json
    """
    if data_root is None:
        # Use model resources to get the appropriate directory
        from agentic_spliceai.splice_engine.resources import get_model_resources
        
        resources = get_model_resources(model_type)
        cache_dir = resources.get_eval_dir().parent  # e.g., data/ensembl/GRCh37/
    else:
        cache_dir = Path(data_root) / annotation_source / build
    
    return cache_dir / f"{model_type}_coordinate_adjustments.json"


def load_cached_adjustments(cache_path: Path, verbose: bool = False) -> Optional[Dict]:
    """
    Load adjustments from cache file if it exists.
    
    Parameters
    ----------
    cache_path : Path
        Path to cache file
    verbose : bool
        Print status messages
        
    Returns
    -------
    dict or None
        Adjustment dictionary if found, None otherwise
    """
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                adjustments = json.load(f)
            
            if verbose:
                print(f"✅ Loaded cached adjustments from: {cache_path}")
            
            return adjustments
        except Exception as e:
            if verbose:
                print(f"⚠️  Failed to load cache: {e}")
            return None
    else:
        if verbose:
            print(f"📭 No cache found at: {cache_path}")
        return None


def save_adjustments_to_cache(
    adjustments: Dict,
    cache_path: Path,
    verbose: bool = False
):
    """
    Save adjustments to cache file.
    
    Parameters
    ----------
    adjustments : dict
        Adjustment dictionary to save
    cache_path : Path
        Path to cache file
    verbose : bool
        Print status messages
    """
    try:
        # Create directory if needed
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON
        with open(cache_path, 'w') as f:
            json.dump(adjustments, f, indent=2)
        
        if verbose:
            print(f"💾 Saved adjustments to cache: {cache_path}")
            print(f"   Donor:    +{adjustments['donor']['plus']} (+ strand), +{adjustments['donor']['minus']} (- strand)")
            print(f"   Acceptor: +{adjustments['acceptor']['plus']} (+ strand), +{adjustments['acceptor']['minus']} (- strand)")
    except Exception as e:
        if verbose:
            print(f"⚠️  Failed to save cache: {e}")


# ============================================================================
# Convenience Function
# ============================================================================

def get_or_detect_adjustments(
    model_type: str,
    build: str,
    annotation_source: str,
    annotations_df: Optional[pl.DataFrame] = None,
    pred_results: Optional[Dict] = None,
    force_redetect: bool = False,
    use_empirical: bool = True,
    verbose: bool = False
) -> Dict[str, Dict[str, int]]:
    """
    Get adjustments from cache OR detect them if needed.
    
    This is the main convenience function that:
    1. Checks cache first
    2. If cache miss, detects adjustments
    3. Saves to cache
    4. Returns adjustments
    
    Parameters
    ----------
    model_type : str
        Model name
    build : str
        Genome build
    annotation_source : str
        Annotation source
    annotations_df : pl.DataFrame, optional
        Ground truth (required if cache miss)
    pred_results : dict, optional
        Predictions (required if cache miss and use_empirical=True)
    force_redetect : bool
        Force re-detection even if cache exists
    use_empirical : bool
        Use empirical detection (vs hardcoded)
    verbose : bool
        Print progress
        
    Returns
    -------
    dict
        Adjustment dictionary
    """
    # 1. Check cache (unless forcing redetection)
    if not force_redetect:
        cache_path = get_adjustment_cache_path(model_type, build, annotation_source)
        adjustments = load_cached_adjustments(cache_path, verbose)
        
        if adjustments is not None:
            return adjustments
    
    # 2. Cache miss or forced - need to detect
    if verbose:
        print("🔍 Detecting coordinate adjustments...")
    
    if annotations_df is None:
        raise ValueError("annotations_df required for adjustment detection")
    
    if use_empirical and pred_results is None:
        raise ValueError("pred_results required for empirical adjustment detection")
    
    # 3. Detect adjustments
    adjustments = auto_detect_adjustments(
        annotations_df=annotations_df,
        pred_results=pred_results,
        use_empirical=use_empirical,
        verbose=verbose
    )
    
    # 4. Save to cache
    cache_path = get_adjustment_cache_path(model_type, build, annotation_source)
    save_adjustments_to_cache(adjustments, cache_path, verbose)
    
    return adjustments
