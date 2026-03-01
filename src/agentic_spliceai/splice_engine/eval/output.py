"""Evaluation output writing and artifact management.

Saves structured evaluation artifacts:
- metrics.json: Full metrics payload with metadata
- gene_list.txt: Gene list for reproducibility
- summary.txt: Human-readable summary

Decoupled from argparse — accepts an eval_config dict so it works
from CLI tools, notebooks, and UI apps alike.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from .metrics import extract_metrics_from_eval


class EvaluationOutputWriter:
    """Write evaluation results to structured output directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory to write output files to. Created if it doesn't exist.

    Examples
    --------
    >>> writer = EvaluationOutputWriter("output/openspliceai_eval")
    >>> writer.save(
    ...     eval_results=eval_results,
    ...     model_name="openspliceai",
    ...     build="GRCh38",
    ...     annotation_source="mane",
    ...     gene_list=["TP53", "BRCA1"],
    ...     eval_config={"threshold": 0.5, "consensus_window": 2},
    ... )
    """

    def __init__(self, output_dir):
        self.output_path = Path(output_dir)

    def save(
        self,
        eval_results: Dict[str, Dict],
        model_name: str,
        build: str,
        annotation_source: str,
        gene_list: List[str],
        eval_config: Dict[str, Any],
        runtime_seconds: Optional[float] = None,
    ) -> Path:
        """Save all evaluation output files.

        Parameters
        ----------
        eval_results : dict
            Evaluation results keyed by filter mode ('canonical', 'all', etc.).
        model_name : str
            Base model name (e.g., 'openspliceai', 'spliceai').
        build : str
            Genomic build (e.g., 'GRCh38', 'GRCh37').
        annotation_source : str
            Annotation source (e.g., 'mane', 'ensembl').
        gene_list : list of str
            Genes that were evaluated.
        eval_config : dict
            Evaluation configuration. Expected keys: threshold,
            consensus_window, transcript_filter. Optional: random_seed.
        runtime_seconds : float, optional
            Total evaluation runtime.

        Returns
        -------
        Path
            The output directory where files were saved.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        metrics_data = extract_metrics_from_eval(eval_results)

        output = {
            'metadata': {
                'model': model_name,
                'build': build,
                'annotation_source': annotation_source,
                'threshold': eval_config.get('threshold'),
                'consensus_window': eval_config.get('consensus_window'),
                'transcript_filter': eval_config.get('transcript_filter'),
                'n_genes': len(gene_list),
                'random_seed': eval_config.get('random_seed'),
                'genes': gene_list,
                'timestamp': datetime.now().isoformat(),
                'runtime_seconds': round(runtime_seconds, 1) if runtime_seconds else None,
            },
            'metrics': metrics_data,
        }

        # metrics.json
        metrics_file = self.output_path / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n   Saved metrics: {metrics_file}")

        # gene_list.txt
        gene_file = self.output_path / 'gene_list.txt'
        with open(gene_file, 'w') as f:
            f.write('\n'.join(gene_list) + '\n')
        print(f"   Saved gene list: {gene_file}")

        # summary.txt
        summary_file = self.output_path / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write(f"Evaluation Summary - {model_name}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Model: {model_name} ({build}, {annotation_source})\n")
            f.write(f"Genes ({len(gene_list)}): {', '.join(gene_list)}\n")
            f.write(f"Threshold: {eval_config.get('threshold')}, "
                    f"Window: +/-{eval_config.get('consensus_window')}bp\n")
            if runtime_seconds:
                f.write(f"Runtime: {runtime_seconds:.1f}s\n")
            f.write(f"Timestamp: {output['metadata']['timestamp']}\n\n")

            for filter_mode, mode_metrics in metrics_data.items():
                label = 'Canonical' if filter_mode == 'canonical' else 'All Transcripts'
                f.write(f"\n{label}\n{'-'*40}\n")
                for site_type in ['donor', 'acceptor', 'overall']:
                    m = mode_metrics.get(site_type, {})
                    tp, fn = m.get('tp', 0), m.get('fn', 0)
                    f.write(f"  {site_type.capitalize():>10}: {tp}/{tp+fn} "
                            f"(R={m.get('recall',0):.1%}, P={m.get('precision',0):.1%}, "
                            f"F1={m.get('f1',0):.1%})\n")
                pr = mode_metrics.get('pr_metrics', {})
                if pr:
                    f.write(f"  Macro AP: {pr.get('macro_ap',0):.4f}, "
                            f"PR-AUC: {pr.get('macro_pr_auc',0):.4f}\n")
        print(f"   Saved summary: {summary_file}")

        return self.output_path
