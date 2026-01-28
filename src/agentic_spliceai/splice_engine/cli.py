"""
CLI interface for splice site prediction using base models.

This module provides command-line access to the splice prediction capabilities
using standalone agentic-spliceai implementation (no meta-spliceai dependency).
"""

import argparse
import sys
from typing import Optional, List
import json


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the splice prediction CLI."""
    parser = argparse.ArgumentParser(
        description="Splice Site Prediction using Base Models (SpliceAI, OpenSpliceAI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict for specific genes
  agentic-spliceai-predict --genes BRCA1 TP53 UNC13A
  
  # Predict for a chromosome
  agentic-spliceai-predict --chromosomes 21
  
  # Use SpliceAI instead of OpenSpliceAI
  agentic-spliceai-predict --base-model spliceai --genes BRCA1
  
  # Test mode with sample data
  agentic-spliceai-predict --mode test --coverage sample --genes BRCA1
  
  # Production run for full genome
  agentic-spliceai-predict --mode production --coverage full_genome
        """
    )
    
    # Target selection
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--genes",
        nargs="+",
        help="Gene symbols or IDs to analyze (e.g., BRCA1 TP53)"
    )
    target_group.add_argument(
        "--chromosomes",
        nargs="+",
        help="Chromosomes to process (e.g., 21 22 X Y)"
    )
    
    # Model selection
    parser.add_argument(
        "--base-model",
        default="openspliceai",
        choices=["openspliceai", "spliceai"],
        help="Base model to use (default: openspliceai)"
    )
    
    # Mode and coverage
    parser.add_argument(
        "--mode",
        default="test",
        choices=["test", "production"],
        help="Execution mode: test (overwritable) or production (immutable)"
    )
    parser.add_argument(
        "--coverage",
        default="gene_subset",
        choices=["gene_subset", "chromosome", "full_genome", "sample"],
        help="Analysis coverage level"
    )
    
    # Output control
    parser.add_argument(
        "--output-dir",
        help="Output directory for results (default: auto-detected)"
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Output verbosity: 0=minimal, 1=normal, 2=detailed"
    )
    
    # Advanced options
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Splice site score threshold (default: 0.5)"
    )
    parser.add_argument(
        "--no-tn-sampling",
        action="store_true",
        help="Preserve all true negative positions (memory intensive)"
    )
    parser.add_argument(
        "--save-nucleotide-scores",
        action="store_true",
        help="Save per-nucleotide scores (generates large files)"
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions vs splice site annotations (requires --save-nucleotide-scores)"
    )
    
    # Output format
    parser.add_argument(
        "--format",
        default="summary",
        choices=["summary", "json", "paths"],
        help="Output format"
    )
    
    return parser


def main(argv: Optional[List[str]] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    try:
        # Import the prediction function
        from agentic_spliceai.splice_engine import run_base_model_predictions
        from agentic_spliceai.splice_engine.config import get_project_root
        from datetime import datetime
        from pathlib import Path
        import os
        import polars as pl
        from agentic_spliceai.splice_engine.base_layer.prediction import evaluate_splice_site_predictions
        
        # Prepare kwargs
        kwargs = {
            "verbosity": args.verbosity,
            "threshold": args.threshold,
            "mode": args.mode,
            "coverage": args.coverage,
            "no_tn_sampling": args.no_tn_sampling,
            "save_nucleotide_scores": args.save_nucleotide_scores,
        }
        
        if args.output_dir:
            kwargs["eval_dir"] = args.output_dir
        elif args.mode == "test":
            project_root = Path(get_project_root())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.genes:
                target_label = "genes_" + "_".join(args.genes[:3])
                if len(args.genes) > 3:
                    target_label += "_etc"
            elif args.chromosomes:
                target_label = "chroms_" + "_".join(args.chromosomes[:3])
                if len(args.chromosomes) > 3:
                    target_label += "_etc"
            else:
                target_label = f"coverage_{args.coverage}"

            kwargs["eval_dir"] = str(
                project_root
                / "data"
                / "test_runs"
                / args.base_model
                / f"{timestamp}_{target_label}"
            )
        
        # Run predictions
        if args.verbosity >= 1:
            print(f"Running {args.base_model} predictions...")
            if args.genes:
                print(f"Target genes: {', '.join(args.genes)}")
            elif args.chromosomes:
                print(f"Target chromosomes: {', '.join(args.chromosomes)}")
            else:
                print(f"Coverage: {args.coverage}")
            if args.mode == "test" and not args.output_dir:
                print(f"Test mode: writing outputs to {kwargs.get('eval_dir')}")
            print()
        
        results = run_base_model_predictions(
            base_model=args.base_model,
            target_genes=args.genes,
            target_chromosomes=args.chromosomes,
            **kwargs
        )

        if args.evaluate:
            if not args.save_nucleotide_scores:
                raise ValueError("--evaluate requires --save-nucleotide-scores (full coverage output)")

            eval_dir = results.get("paths", {}).get("eval_dir", kwargs.get("eval_dir"))
            if not eval_dir:
                raise ValueError("Cannot determine eval_dir for evaluation")

            scores_path = os.path.join(eval_dir, "nucleotide_scores.tsv")
            splice_sites_path = os.path.join(eval_dir, "splice_sites_enhanced.tsv")

            if not os.path.exists(scores_path):
                raise FileNotFoundError(f"Missing nucleotide scores file: {scores_path}")
            if not os.path.exists(splice_sites_path):
                raise FileNotFoundError(f"Missing splice sites annotation file: {splice_sites_path}")

            if args.verbosity >= 1:
                print("[eval] Loading nucleotide scores and splice site annotations...")

            nucleotide_scores = pl.read_csv(scores_path, separator='\t')
            splice_sites = pl.read_csv(splice_sites_path, separator='\t')

            required_score_cols = {"gene_id", "gene_name", "chrom", "strand", "genomic_position", "donor_score", "acceptor_score", "neither_score"}
            missing = required_score_cols - set(nucleotide_scores.columns)
            if missing:
                raise ValueError(f"nucleotide_scores.tsv is missing required columns: {sorted(missing)}")

            predictions = {}
            for (gene_id,), g in nucleotide_scores.group_by(["gene_id"], maintain_order=True):
                g_sorted = g.sort("genomic_position")
                first = g_sorted.row(0, named=True)
                predictions[str(gene_id)] = {
                    "gene_id": str(gene_id),
                    "gene_name": first.get("gene_name", ""),
                    "seqname": first.get("chrom", ""),
                    "chrom": first.get("chrom", ""),
                    "strand": first.get("strand", "+"),
                    "positions": g_sorted["genomic_position"].to_list(),
                    "donor_prob": g_sorted["donor_score"].to_list(),
                    "acceptor_prob": g_sorted["acceptor_score"].to_list(),
                    "neither_prob": g_sorted["neither_score"].to_list(),
                }

            error_df, positions_df, pr_metrics = evaluate_splice_site_predictions(
                predictions=predictions,
                annotations_df=splice_sites,
                threshold=args.threshold,
                consensus_window=2,
                collect_tn=True,
                no_tn_sampling=args.no_tn_sampling,
                verbosity=args.verbosity,
                return_pr_metrics=True,
            )

            eval_positions_path = os.path.join(eval_dir, "evaluation_positions.tsv")
            eval_errors_path = os.path.join(eval_dir, "evaluation_errors.tsv")
            positions_df.write_csv(eval_positions_path, separator='\t')
            error_df.write_csv(eval_errors_path, separator='\t')

            results.setdefault("paths", {})["evaluation_positions"] = eval_positions_path
            results.setdefault("paths", {})["evaluation_errors"] = eval_errors_path
            results.setdefault("metrics", {}).update(pr_metrics)
            if args.verbosity >= 1:
                print(f"[eval] Saved evaluation positions: {eval_positions_path}")
                print(f"[eval] Saved evaluation errors: {eval_errors_path}")
                if pr_metrics:
                    print("[eval] PR metrics:")
                    print(f"  donor_ap: {pr_metrics.get('donor_ap', 0.0):.4f}")
                    print(f"  donor_pr_auc: {pr_metrics.get('donor_pr_auc', 0.0):.4f}")
                    print(f"  acceptor_ap: {pr_metrics.get('acceptor_ap', 0.0):.4f}")
                    print(f"  acceptor_pr_auc: {pr_metrics.get('acceptor_pr_auc', 0.0):.4f}")
                    print(f"  macro_ap: {pr_metrics.get('macro_ap', 0.0):.4f}")
                    print(f"  macro_pr_auc: {pr_metrics.get('macro_pr_auc', 0.0):.4f}")
        
        # Output results
        if args.format == "json":
            # JSON output (paths only, not DataFrames)
            output = {
                "success": results.get("success", False),
                "paths": results.get("paths", {}),
                "manifest_summary": results.get("manifest_summary", {})
            }
            print(json.dumps(output, indent=2))
            
        elif args.format == "paths":
            # Just print paths
            paths = results.get("paths", {})
            for key, value in paths.items():
                print(f"{key}: {value}")
                
        else:  # summary
            # Human-readable summary
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            
            if results.get("success"):
                print("✓ Predictions completed successfully")
                
                # Summary statistics
                manifest = results.get("manifest_summary", {})
                if manifest:
                    print(f"\nProcessed: {manifest.get('processed_genes', 0)} genes")
                    print(f"Total positions: {manifest.get('total_positions', 0)}")
                
                # Output paths
                paths = results.get("paths", {})
                if paths:
                    print("\nOutput files:")
                    for key, value in paths.items():
                        if value and "artifact" in key:
                            print(f"  - {key}: {value}")
            else:
                print("✗ Predictions failed")
                if "error" in results:
                    print(f"Error: {results['error']}")
            
            print("="*60)
        
        return 0 if results.get("success") else 1
        
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("\nRequired dependencies are missing.", file=sys.stderr)
        print("Please ensure agentic-spliceai is properly installed.", file=sys.stderr)
        return 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbosity >= 2:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
