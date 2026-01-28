"""
Enhanced SpliceAI prediction workflow for meta models.

This module provides the core prediction workflow for splice site prediction,
supporting both standard and full coverage modes.

STANDALONE VERSION - No dependencies on meta-spliceai.
"""

import os
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import polars as pl
import numpy as np

from agentic_spliceai.splice_engine.utils.display import print_emphasized, print_with_indent, print_section_separator
from agentic_spliceai.splice_engine.base_layer import BaseModelConfig, SpliceAIConfig

from agentic_spliceai.splice_engine.base_layer.data import (
    extract_gene_annotations,
    extract_splice_sites,
    extract_gene_sequences,
    one_hot_encode,
    GeneManifest,
    GeneManifestEntry,
    # Position coordinate utilities
    PositionType,
    absolute_to_relative,
)

from agentic_spliceai.splice_engine.base_layer.prediction import (
    prepare_input_sequence,
    predict_with_model,
    SPLICEAI_CONTEXT as PREDICTION_CONTEXT,
    SPLICEAI_BLOCK_SIZE,
)

from agentic_spliceai.splice_engine.meta_models.workflows.data_preparation import (
    prepare_gene_annotations,
    prepare_splice_site_annotations,
    prepare_genomic_sequences,
    handle_overlapping_genes,
    load_spliceai_models
)

from agentic_spliceai.splice_engine.meta_models.utils.chrom_utils import (
    normalize_chromosome_names,
    determine_target_chromosomes
)


# SpliceAI context length (5000 bp on each side)
SPLICEAI_CONTEXT = 5000


def run_enhanced_splice_prediction_workflow(
    config: Optional[Union[SpliceAIConfig, BaseModelConfig]] = None, 
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None, 
    verbosity: int = 1,
    no_final_aggregate: bool = False,
    no_tn_sampling: bool = False,
    position_id_mode: str = 'genomic',
    **kwargs
) -> Dict[str, Any]:
    """
    Run the enhanced SpliceAI prediction workflow.
    
    This is the main entry point for running splice site predictions with
    support for full coverage mode (nucleotide-level scores).
    
    Parameters
    ----------
    config : SpliceAIConfig or BaseModelConfig, optional
        Configuration for SpliceAI workflow. If None, uses default config.
    target_genes : List[str], optional
        List of gene symbols or IDs to focus on.
    target_chromosomes : List[str], optional
        List of chromosomes to focus on.
    verbosity : int, default=1
        Controls output verbosity (0=minimal, 1=normal, 2=detailed)
    no_final_aggregate : bool, default=False
        If True, skip final aggregation of results.
    no_tn_sampling : bool, default=False
        If True, preserve all TN positions without sampling.
    position_id_mode : str, default='genomic'
        Position identification strategy ('genomic' or 'relative').
    **kwargs : dict
        Additional parameters for SpliceAI workflow.
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing workflow results:
        - 'success': bool - Whether workflow completed successfully
        - 'error_analysis': Error analysis DataFrame
        - 'positions': Enhanced positions DataFrame
        - 'analysis_sequences': Analysis sequences DataFrame
        - 'gene_manifest': Gene processing manifest DataFrame
        - 'nucleotide_scores': Full nucleotide-level scores (if enabled)
        - 'paths': Dictionary of output file paths
    """
    run_start_time = time.time()
    
    if verbosity >= 1:
        print_section_separator()
        print_emphasized("[workflow] Starting Enhanced Splice Prediction Workflow")
        print_emphasized("[standalone] Using agentic-spliceai native implementation")
        print_section_separator()
    
    # Initialize config if not provided
    if config is None:
        config = BaseModelConfig()
    
    # Extract key parameters from config
    gtf_file = config.gtf_file
    genome_fasta = config.genome_fasta
    eval_dir = config.eval_dir
    base_model = config.base_model
    save_nucleotide_scores = getattr(config, 'save_nucleotide_scores', False)
    
    # Create output directory
    os.makedirs(eval_dir, exist_ok=True)
    
    # Initialize result containers
    gene_manifest = GeneManifest()
    full_positions_df = None
    full_error_df = None
    full_nucleotide_scores_df = None
    
    # =========================================================================
    # PHASE 1: Data Preparation
    # =========================================================================
    if verbosity >= 1:
        print_emphasized("[phase 1] Data Preparation")
    
    # 1.1 Prepare gene annotations
    annot_result = prepare_gene_annotations(
        local_dir=eval_dir,
        gtf_file=gtf_file,
        do_extract=True,
        target_chromosomes=target_chromosomes,
        verbosity=verbosity
    )
    
    if not annot_result['success']:
        return _error_result("Failed to prepare gene annotations", annot_result.get('error'))
    
    gene_df = annot_result['annotation_df']
    
    # Filter to target genes if specified
    if target_genes:
        if verbosity >= 1:
            print(f"[filter] Filtering to {len(target_genes)} target genes...")
        
        if isinstance(gene_df, pl.DataFrame):
            gene_df = gene_df.filter(
                pl.col('gene_id').is_in(target_genes) | 
                pl.col('gene_name').is_in(target_genes)
            )
        else:
            import pandas as pd
            gene_df = gene_df[
                gene_df['gene_id'].isin(target_genes) | 
                gene_df['gene_name'].isin(target_genes)
            ]
        
        if verbosity >= 1:
            print(f"[filter] Found {len(gene_df)} matching genes")
    
    # 1.2 Prepare splice site annotations
    ss_result = prepare_splice_site_annotations(
        local_dir=eval_dir,
        gtf_file=gtf_file,
        do_extract=True,
        gene_annotations_df=gene_df,
        target_chromosomes=target_chromosomes,
        verbosity=verbosity
    )
    
    if not ss_result['success']:
        return _error_result("Failed to prepare splice site annotations", ss_result.get('error'))
    
    splice_sites_df = ss_result['splice_sites_df']
    
    # 1.3 Prepare genomic sequences
    seq_result = prepare_genomic_sequences(
        local_dir=eval_dir,
        gtf_file=gtf_file,
        genome_fasta=genome_fasta,
        chromosomes=target_chromosomes,
        genes=target_genes,
        do_extract=True,
        verbosity=verbosity
    )
    
    if not seq_result['success']:
        return _error_result("Failed to prepare genomic sequences", seq_result.get('error'))
    
    sequences_df = seq_result.get('sequences_df')
    
    # =========================================================================
    # PHASE 2: Model Loading
    # =========================================================================
    if verbosity >= 1:
        print_emphasized("[phase 2] Model Loading")
    
    model_result = load_spliceai_models(verbosity=verbosity)
    
    if not model_result['success']:
        if verbosity >= 1:
            print_with_indent(f"[warning] Model loading failed: {model_result.get('error')}", indent=2)
            print_with_indent("[info] Continuing without models - predictions will be skipped", indent=2)
        models = None
    else:
        models = model_result['models']
    
    # =========================================================================
    # PHASE 3: Prediction Loop
    # =========================================================================
    if verbosity >= 1:
        print_emphasized("[phase 3] Running Predictions")
    
    # Determine chromosomes to process
    if sequences_df is not None and len(sequences_df) > 0:
        if isinstance(sequences_df, pl.DataFrame):
            chroms_to_process = sequences_df['chrom'].unique().to_list()
        else:
            chroms_to_process = sequences_df['chrom'].unique().tolist()
    else:
        chroms_to_process = target_chromosomes or []
    
    if verbosity >= 1:
        print(f"[info] Processing {len(chroms_to_process)} chromosomes")
    
    total_genes_processed = 0
    all_positions = []
    all_nucleotide_scores = []
    
    for chrom in chroms_to_process:
        if verbosity >= 1:
            print(f"\n[chromosome] Processing {chrom}...")
        
        # Get genes for this chromosome
        if sequences_df is not None:
            if isinstance(sequences_df, pl.DataFrame):
                chrom_seqs = sequences_df.filter(pl.col('chrom') == chrom)
            else:
                chrom_seqs = sequences_df[sequences_df['chrom'] == chrom]
            
            n_genes = len(chrom_seqs)
        else:
            n_genes = 0
        
        if n_genes == 0:
            if verbosity >= 1:
                print(f"[skip] No genes found for chromosome {chrom}")
            continue
        
        # Process each gene
        if isinstance(chrom_seqs, pl.DataFrame):
            genes_list = chrom_seqs.to_dicts()
        else:
            genes_list = chrom_seqs.to_dict('records')
        
        for gene_info in genes_list:
            gene_id = gene_info['gene_id']
            gene_name = gene_info.get('gene_name', gene_id)
            sequence = gene_info.get('sequence', '')
            
            # Register gene in manifest
            gene_manifest.add_requested_genes([gene_id])
            
            if not sequence or len(sequence) < 100:
                gene_manifest.mark_failed(gene_id, gene_name, status='skipped', reason='Sequence too short')
                continue
            
            # Run prediction if models available
            if models:
                try:
                    predictions = _predict_gene(
                        sequence=sequence,
                        models=models,
                        gene_id=gene_id,
                        gene_info=gene_info,
                        save_nucleotide_scores=save_nucleotide_scores,
                        verbosity=verbosity
                    )
                    
                    if predictions:
                        all_positions.extend(predictions.get('positions', []))
                        if save_nucleotide_scores:
                            all_nucleotide_scores.extend(predictions.get('nucleotide_scores', []))

                        if splice_sites_df is not None and isinstance(splice_sites_df, pl.DataFrame) and splice_sites_df.height > 0:
                            gene_splice_sites = splice_sites_df.filter(pl.col('gene_id') == gene_id).height
                        else:
                            gene_splice_sites = 0
                        
                        gene_manifest.mark_processed(
                            gene_id, gene_name,
                            num_positions=len(predictions.get('positions', [])),
                            num_nucleotides=len(sequence),
                            num_splice_sites=gene_splice_sites
                        )
                        total_genes_processed += 1
                    else:
                        gene_manifest.mark_failed(gene_id, gene_name, status='prediction_failed', reason='No predictions')
                        
                except Exception as e:
                    gene_manifest.mark_failed(gene_id, gene_name, status='prediction_failed', reason=str(e))
                    if verbosity >= 2:
                        print(f"[error] Failed to predict {gene_id}: {e}")
            else:
                # No models - mark as skipped
                gene_manifest.mark_failed(gene_id, gene_name, status='skipped', reason='Models not loaded')
        
        if verbosity >= 1:
            print(f"[chromosome] Completed {chrom}: {n_genes} genes")
    
    # =========================================================================
    # PHASE 4: Aggregation and Output
    # =========================================================================
    if verbosity >= 1:
        print_emphasized("[phase 4] Aggregating Results")
    
    # Create positions DataFrame
    if all_positions:
        full_positions_df = pl.DataFrame(all_positions)
        if verbosity >= 1:
            print(f"[info] Total positions: {len(full_positions_df)}")
    else:
        full_positions_df = pl.DataFrame()
    
    # Create nucleotide scores DataFrame
    if save_nucleotide_scores and all_nucleotide_scores:
        full_nucleotide_scores_df = pl.DataFrame(all_nucleotide_scores)
        if verbosity >= 1:
            print(f"[info] Total nucleotide scores: {len(full_nucleotide_scores_df)}")
    else:
        full_nucleotide_scores_df = pl.DataFrame()
    
    # Save outputs
    if not no_final_aggregate and full_positions_df.height > 0:
        positions_path = os.path.join(eval_dir, 'full_splice_positions_enhanced.tsv')
        full_positions_df.write_csv(positions_path, separator='\t')
        if verbosity >= 1:
            print(f"[saved] Positions: {positions_path}")
    
    if save_nucleotide_scores and full_nucleotide_scores_df.height > 0:
        scores_path = os.path.join(eval_dir, 'nucleotide_scores.tsv')
        full_nucleotide_scores_df.write_csv(scores_path, separator='\t')
        if verbosity >= 1:
            print(f"[saved] Nucleotide scores: {scores_path}")
    
    # Save manifest
    manifest_path = os.path.join(eval_dir, 'gene_manifest.tsv')
    gene_manifest.save(manifest_path, use_polars=True)
    if verbosity >= 1:
        summary = gene_manifest.get_summary()
        print(f"[saved] Manifest: {manifest_path}")
        print(f"[summary] Processed: {summary['processed_genes']}/{summary['total_genes']} genes")
    
    # Calculate runtime
    runtime_min = (time.time() - run_start_time) / 60.0
    if verbosity >= 1:
        print_section_separator()
        print_emphasized(f"[complete] Total runtime: {runtime_min:.1f} min")
        print_emphasized(f"[complete] Genes processed: {total_genes_processed}")
    
    return {
        "success": True,
        "error_analysis": full_error_df if full_error_df is not None else pl.DataFrame(),
        "positions": full_positions_df,
        "analysis_sequences": pl.DataFrame(),
        "gene_manifest": gene_manifest.to_dataframe(use_polars=True),
        "nucleotide_scores": full_nucleotide_scores_df,
        "paths": {
            "eval_dir": eval_dir,
            "positions": os.path.join(eval_dir, 'full_splice_positions_enhanced.tsv'),
            "manifest": manifest_path,
        },
        "manifest_summary": gene_manifest.get_summary()
    }


def _predict_gene(
    sequence: str,
    models: List,
    gene_id: str,
    gene_info: Dict,
    save_nucleotide_scores: bool = False,
    verbosity: int = 1,
    context: int = 10000
) -> Optional[Dict[str, Any]]:
    """
    Run SpliceAI prediction for a single gene using block-based processing.
    
    Parameters
    ----------
    sequence : str
        Gene sequence
    models : List
        List of loaded SpliceAI models
    gene_id : str
        Gene identifier
    gene_info : Dict
        Gene metadata
    save_nucleotide_scores : bool
        Whether to save nucleotide-level scores
    verbosity : int
        Verbosity level
    context : int
        Context length for SpliceAI (default 10000 for SpliceAI-10k)
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Prediction results or None if failed
    """
    from collections import defaultdict
    
    try:
        seq_len = len(sequence)
        
        # Minimum sequence length check
        if seq_len < 100:
            return None
        
        # Prepare input blocks using the proper block-based approach
        input_blocks = prepare_input_sequence(sequence, context)
        
        if verbosity >= 2:
            print(f"[predict] Gene {gene_id}: {seq_len} bp -> {len(input_blocks)} blocks")
        
        # Dictionary to store merged results by position
        merged_results = defaultdict(lambda: {'donor_prob': [], 'acceptor_prob': [], 'neither_prob': []})
        
        gene_start = gene_info.get('start', 0)
        gene_end = gene_info.get('end', gene_start + seq_len)
        strand = gene_info.get('strand', '+')
        chrom = gene_info.get('chrom', '')
        gene_name = gene_info.get('gene_name', gene_id)
        
        # Process each block
        for block_index, block in enumerate(input_blocks):
            x = block[None, :]  # Add batch dimension
            
            # Average predictions across all models
            y = np.mean([predict_with_model(model, x) for model in models], axis=0)
            
            # Extract probabilities
            # SpliceAI output channels: [neither, acceptor, donor]
            donor_prob = y[0, :, 2]
            acceptor_prob = y[0, :, 1]
            neither_prob = y[0, :, 0]
            
            # Calculate block start position
            block_start = block_index * SPLICEAI_BLOCK_SIZE
            
            # Store results with adjusted positions
            for i in range(len(donor_prob)):
                if strand == '+':
                    absolute_position = gene_start + (block_start + i)
                else:
                    absolute_position = gene_end - (block_start + i)
                
                # Only keep positions within gene boundaries
                if gene_start <= absolute_position <= gene_end:
                    merged_results[absolute_position]['donor_prob'].append(float(donor_prob[i]))
                    merged_results[absolute_position]['acceptor_prob'].append(float(acceptor_prob[i]))
                    merged_results[absolute_position]['neither_prob'].append(float(neither_prob[i]))
        
        # Average overlapping predictions and build output
        threshold = 0.1
        positions = []  # High-scoring positions only (for backward compatibility)
        nucleotide_scores = []  # ALL positions (full coverage mode)
        
        for pos in sorted(merged_results.keys()):
            data = merged_results[pos]
            avg_donor = np.mean(data['donor_prob'])
            avg_acceptor = np.mean(data['acceptor_prob'])
            avg_neither = np.mean(data['neither_prob'])
            
            # FULL COVERAGE: Always add nucleotide-level scores for every position
            if save_nucleotide_scores:
                # Convert ABSOLUTE position to RELATIVE position
                # See base_layer/data/position_types.py for coordinate system documentation
                rel_position = absolute_to_relative(
                    pos, 
                    gene_start=gene_start, 
                    gene_end=gene_end, 
                    strand=strand
                )
                
                nucleotide_scores.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'strand': strand,
                    'position': rel_position,  # RELATIVE position (1-indexed, 5' to 3' in transcription space)
                    'genomic_position': pos,   # ABSOLUTE genomic coordinate
                    'donor_score': avg_donor,
                    'acceptor_score': avg_acceptor,
                    'neither_score': avg_neither,
                })
            
            # HIGH-SCORING POSITIONS: Only positions above threshold (sparse output)
            if avg_donor > threshold or avg_acceptor > threshold:
                if avg_donor > avg_acceptor:
                    splice_type = 'donor'
                    score = avg_donor
                else:
                    splice_type = 'acceptor'
                    score = avg_acceptor
                
                positions.append({
                    'gene_id': gene_id,
                    'gene_name': gene_name,
                    'chrom': chrom,
                    'strand': strand,
                    'position': pos,
                    'splice_type': splice_type,
                    'score': score,
                    'donor_score': avg_donor,
                    'acceptor_score': avg_acceptor,
                    'neither_score': avg_neither,
                })
        
        result = {'positions': positions}
        if save_nucleotide_scores:
            result['nucleotide_scores'] = nucleotide_scores
            if verbosity >= 2:
                print(f"[full_coverage] Gene {gene_id}: {len(nucleotide_scores)} nucleotide scores")
        
        return result
        
    except Exception as e:
        if verbosity >= 2:
            import traceback
            print(f"[error] Prediction failed for {gene_id}: {e}")
            traceback.print_exc()
        return None


def _error_result(message: str, error: Optional[str] = None) -> Dict[str, Any]:
    """Create an error result dictionary."""
    return {
        "success": False,
        "error": error or message,
        "error_analysis": pl.DataFrame(),
        "positions": pl.DataFrame(),
        "analysis_sequences": pl.DataFrame(),
        "gene_manifest": pl.DataFrame(),
        "nucleotide_scores": pl.DataFrame(),
        "paths": {}
    }


def run_base_model_predictions(
    base_model: str = 'spliceai',
    target_genes: Optional[List[str]] = None,
    target_chromosomes: Optional[List[str]] = None,
    config: Optional[BaseModelConfig] = None,
    verbosity: int = 1,
    no_tn_sampling: bool = False,
    save_nucleotide_scores: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    User-friendly interface for running base model predictions.
    
    Parameters
    ----------
    base_model : str, default='spliceai'
        Base model to use ('spliceai' or 'openspliceai')
    target_genes : List[str], optional
        List of gene symbols or IDs to process
    target_chromosomes : List[str], optional
        List of chromosomes to process
    config : BaseModelConfig, optional
        Configuration object
    verbosity : int, default=1
        Verbosity level
    no_tn_sampling : bool, default=False
        If True, preserve all TN positions
    save_nucleotide_scores : bool, default=False
        If True, save full nucleotide-level scores
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary
    """
    if config is None:
        config = BaseModelConfig(base_model=base_model, **kwargs)
    
    if save_nucleotide_scores:
        config_dict = asdict(config)
        config_dict['save_nucleotide_scores'] = True
        config = BaseModelConfig(**config_dict)
    
    if verbosity >= 1:
        print_emphasized(f"[base_model] Running {base_model} predictions...")
        if target_genes:
            print_with_indent(f"Target genes: {target_genes[:5]}{'...' if len(target_genes) > 5 else ''}", indent=2)
        if target_chromosomes:
            print_with_indent(f"Target chromosomes: {target_chromosomes}", indent=2)
        if save_nucleotide_scores:
            print_with_indent("[mode] Full coverage mode enabled", indent=2)
    
    return run_enhanced_splice_prediction_workflow(
        config=config,
        target_genes=target_genes,
        target_chromosomes=target_chromosomes,
        verbosity=verbosity,
        no_tn_sampling=no_tn_sampling,
        **kwargs
    )


def validate_workflow_config(config: BaseModelConfig, verbosity: int = 1) -> Dict[str, Any]:
    """
    Validate workflow configuration before running predictions.
    
    Parameters
    ----------
    config : BaseModelConfig
        Configuration to validate
    verbosity : int, default=1
        Verbosity level
        
    Returns
    -------
    Dict[str, Any]
        Validation result with 'valid' bool and 'errors' list
    """
    errors = []
    warnings = []
    
    if config.gtf_file is None:
        errors.append("GTF file not specified")
    elif not os.path.exists(config.gtf_file):
        errors.append(f"GTF file not found: {config.gtf_file}")
    
    if config.genome_fasta is None:
        errors.append("Genome FASTA not specified")
    elif not os.path.exists(config.genome_fasta):
        errors.append(f"Genome FASTA not found: {config.genome_fasta}")
    
    valid_models = ['spliceai', 'openspliceai']
    if config.base_model.lower() not in valid_models:
        errors.append(f"Invalid base_model: {config.base_model}. Must be one of {valid_models}")
    
    try:
        os.makedirs(config.eval_dir, exist_ok=True)
    except Exception as e:
        errors.append(f"Cannot create eval_dir: {config.eval_dir}. Error: {e}")
    
    if getattr(config, 'save_nucleotide_scores', False):
        warnings.append("save_nucleotide_scores=True will generate large output files.")
    
    if verbosity >= 1:
        if errors:
            print_emphasized("[validation] Configuration errors:")
            for err in errors:
                print_with_indent(f"❌ {err}", indent=2)
        if warnings:
            print_emphasized("[validation] Warnings:")
            for warn in warnings:
                print_with_indent(f"⚠️ {warn}", indent=2)
        if not errors:
            print_emphasized("[validation] ✅ Configuration is valid")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }
