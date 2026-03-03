"""Chunked prediction workflow for genome-scale base layer predictions.

Orchestrates the prediction pipeline with:
- Gene-level chunking (configurable batch size)
- Per-chunk checkpointing (resume after failure)
- Raw prediction persistence (meta layer input)
- GeneManifest tracking

The output serves as precomputed input for the meta layer, which
consumes per-nucleotide ``donor_prob``, ``acceptor_prob``, and
``neither_prob`` as base features.

Usage::

    from agentic_spliceai.splice_engine.base_layer.models.config import create_workflow_config
    from agentic_spliceai.splice_engine.base_layer.workflows import PredictionWorkflow

    config = create_workflow_config(base_model='openspliceai', chunk_size=500)
    workflow = PredictionWorkflow(config)
    result = workflow.run(genes=['BRCA1', 'TP53'])
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import polars as pl

from ..data.types import GeneManifest
from ..io.artifacts import ArtifactManager
from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class WorkflowResult:
    """Container for prediction workflow results.

    Attributes
    ----------
    success : bool
        Whether the workflow completed without fatal errors.
    predictions : pl.DataFrame
        Aggregated raw predictions (all chunks). Columns:
        gene_id, gene_name, chrom, position, strand, gene_start,
        gene_end, donor_prob, acceptor_prob, neither_prob.
    manifest : GeneManifest
        Gene processing manifest with per-gene status.
    chunks_processed : int
        Number of chunks that were actually predicted (not resumed).
    chunks_skipped : int
        Number of chunks skipped via resume.
    total_chunks : int
        Total number of chunks.
    runtime_seconds : float
        Total wall-clock time.
    output_dir : Optional[Path]
        Directory where artifacts were saved.
    error : Optional[str]
        Error message if workflow failed.
    """

    success: bool
    predictions: pl.DataFrame
    manifest: GeneManifest
    chunks_processed: int = 0
    chunks_skipped: int = 0
    total_chunks: int = 0
    runtime_seconds: float = 0.0
    output_dir: Optional[Path] = None
    error: Optional[str] = None

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for JSON serialization."""
        manifest_summary = self.manifest.get_summary()
        return {
            "success": self.success,
            "total_predictions": self.predictions.height,
            "total_chunks": self.total_chunks,
            "chunks_processed": self.chunks_processed,
            "chunks_skipped": self.chunks_skipped,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "manifest": manifest_summary,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Workflow orchestrator
# ---------------------------------------------------------------------------


class PredictionWorkflow:
    """Orchestrate genome-scale predictions with chunking and checkpointing.

    This workflow wraps the existing prediction pipeline
    (``predict_splice_sites_for_genes``) with:

    1. **Chunking** — process genes in configurable batches
    2. **Checkpointing** — per-chunk TSV artifacts, skip on resume
    3. **Persistence** — raw per-nucleotide scores for meta layer consumption
    4. **Tracking** — GeneManifest records per-gene processing status

    Parameters
    ----------
    config : WorkflowConfig
        Workflow configuration (chunk size, output dir, resume, etc.).
    """

    def __init__(self, config: WorkflowConfig) -> None:
        self.config = config

        # Resolve output directory
        if config.output_dir is not None:
            raw = Path(config.output_dir).expanduser()
            if config.mode == "test" and not raw.is_absolute():
                # Relative paths in test mode resolve under output/
                self._output_dir = Path("output") / raw
            else:
                self._output_dir = raw
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._output_dir = Path("output") / f"{config.base_model}_{config.genomic_build}_{ts}"

        self._artifact_manager = ArtifactManager(
            output_dir=self._output_dir,
            model_name=config.base_model,
            genomic_build=config.genomic_build,
            mode=config.mode,
        )
        self._manifest = GeneManifest(
            base_model=config.base_model,
            genomic_build=config.genomic_build,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        genes: Optional[List[str]] = None,
        chromosomes: Optional[List[str]] = None,
    ) -> WorkflowResult:
        """Execute the chunked prediction workflow.

        Parameters
        ----------
        genes : list of str, optional
            Target gene IDs or symbols. ``None`` means all genes in the
            annotation (filtered by *chromosomes* if given).
        chromosomes : list of str, optional
            Restrict to these chromosomes (e.g., ``['chr21', 'chr22']``).

        Returns
        -------
        WorkflowResult
            Aggregated predictions, manifest, and statistics.
        """
        start_time = time.time()
        cfg = self.config

        logger.info(
            "Starting prediction workflow: model=%s, build=%s, chunk_size=%d, resume=%s",
            cfg.base_model, cfg.genomic_build, cfg.chunk_size, cfg.resume,
        )
        if cfg.verbosity >= 1:
            print(f"\n{'='*70}")
            print(f"  Prediction Workflow: {cfg.base_model} / {cfg.genomic_build}")
            print(f"  Chunk size: {cfg.chunk_size} genes | Resume: {cfg.resume}")
            print(f"  Mode: {cfg.mode} | Output: {self._output_dir}")
            print(f"{'='*70}\n")

        try:
            # 1. Prepare gene data (annotations + sequences)
            gene_df = self._prepare_gene_data(genes, chromosomes)
            if gene_df is None or gene_df.height == 0:
                return self._make_error_result(start_time, "No genes found in annotations")

            # 2. Load models (once for all chunks)
            models = self._load_models()
            if not models:
                return self._make_error_result(start_time, f"Failed to load {cfg.base_model} models")

            # 3. Build chunks
            chunks = self._build_chunks(gene_df)
            total_chunks = len(chunks)
            logger.info("Split %d genes into %d chunks", gene_df.height, total_chunks)

            if cfg.verbosity >= 1:
                print(f"  Genes: {gene_df.height} | Chunks: {total_chunks}")

            # 4. Process each chunk
            chunk_frames: List[pl.DataFrame] = []
            chunks_processed = 0
            chunks_skipped = 0

            for chunk_idx, chunk_gene_df in enumerate(chunks):
                chunk_genes = chunk_gene_df["gene_name"].to_list()

                # 4a. Resume check
                if cfg.resume and self._artifact_manager.chunk_exists(chunk_idx, "predictions"):
                    if cfg.verbosity >= 1:
                        print(f"  Chunk {chunk_idx+1}/{total_chunks}: "
                              f"skipped (checkpoint exists, {len(chunk_genes)} genes)")
                    logger.info("Chunk %d: skipping (checkpoint exists)", chunk_idx)

                    loaded = self._artifact_manager.load_chunk(chunk_idx, "predictions")
                    chunk_frames.append(loaded)
                    chunks_skipped += 1

                    # Mark genes from loaded chunk as processed in manifest
                    for gname in chunk_genes:
                        self._manifest.mark_processed(gname, gene_name=gname)
                    continue

                # 4b. Run predictions for this chunk
                chunk_start = time.time()
                if cfg.verbosity >= 1:
                    print(f"  Chunk {chunk_idx+1}/{total_chunks}: "
                          f"predicting {len(chunk_genes)} genes...")

                chunk_predictions_df = self._predict_chunk(chunk_gene_df, models, chunk_idx)
                chunk_elapsed = time.time() - chunk_start

                if chunk_predictions_df.height == 0:
                    if cfg.verbosity >= 1:
                        print(f"    0 positions ({chunk_elapsed:.1f}s)")
                    for gname in chunk_genes:
                        self._manifest.mark_failed(gname, gene_name=gname, reason="no predictions")
                    chunks_processed += 1
                    continue

                # 4c. Update manifest
                predicted_genes = chunk_predictions_df["gene_name"].unique().to_list()
                for gname in predicted_genes:
                    gene_rows = chunk_predictions_df.filter(pl.col("gene_name") == gname)
                    self._manifest.mark_processed(
                        gname,
                        gene_name=gname,
                        num_positions=gene_rows.height,
                        processing_time=chunk_elapsed / max(len(predicted_genes), 1),
                    )
                for gname in set(chunk_genes) - set(predicted_genes):
                    self._manifest.mark_failed(gname, gene_name=gname, reason="no predictions")

                # 4d. Save chunk artifact
                if cfg.save_predictions:
                    self._artifact_manager.save_chunk(
                        chunk_predictions_df, chunk_idx, "predictions"
                    )

                chunk_frames.append(chunk_predictions_df)
                chunks_processed += 1

                if cfg.verbosity >= 1:
                    print(f"    {chunk_predictions_df.height:,} positions ({chunk_elapsed:.1f}s)")

            # 5. Aggregate all chunks
            if chunk_frames:
                all_predictions = pl.concat(chunk_frames)
            else:
                all_predictions = pl.DataFrame()

            # 6. Save final artifacts
            if cfg.save_predictions and all_predictions.height > 0:
                self._artifact_manager.save_aggregated(all_predictions, "predictions")

            manifest_df = self._manifest.to_dataframe()
            self._artifact_manager.save_manifest(manifest_df)

            elapsed = time.time() - start_time
            summary = {
                "model": cfg.base_model,
                "genomic_build": cfg.genomic_build,
                "chunk_size": cfg.chunk_size,
                "total_genes": gene_df.height,
                "total_predictions": all_predictions.height,
                "total_chunks": total_chunks,
                "chunks_processed": chunks_processed,
                "chunks_skipped": chunks_skipped,
                "runtime_seconds": round(elapsed, 2),
                "timestamp": datetime.now().isoformat(),
                "manifest_summary": self._manifest.get_summary(),
            }
            self._artifact_manager.save_summary(summary)

            if cfg.verbosity >= 1:
                ms = self._manifest.get_summary()
                print(f"\n{'='*70}")
                print(f"  Workflow complete!")
                print(f"  Genes: {ms['processed_genes']} processed, {ms['failed_genes']} failed")
                print(f"  Predictions: {all_predictions.height:,} total positions")
                print(f"  Chunks: {chunks_processed} processed, {chunks_skipped} resumed")
                print(f"  Runtime: {elapsed:.1f}s")
                print(f"  Output: {self._output_dir}")
                print(f"{'='*70}\n")

            return WorkflowResult(
                success=True,
                predictions=all_predictions,
                manifest=self._manifest,
                chunks_processed=chunks_processed,
                chunks_skipped=chunks_skipped,
                total_chunks=total_chunks,
                runtime_seconds=elapsed,
                output_dir=self._output_dir,
            )

        except Exception as e:
            logger.exception("Workflow failed")
            return self._make_error_result(start_time, str(e))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_gene_data(
        self,
        genes: Optional[List[str]],
        chromosomes: Optional[List[str]],
    ) -> Optional[pl.DataFrame]:
        """Load gene annotations and extract sequences.

        When both *genes* and *chromosomes* are supplied the result is a
        **union**: all genes on the specified chromosomes **plus** any
        explicitly named genes (which may reside on other chromosomes).

        Delegates to the same resource resolution and extraction logic
        used by :class:`BaseModelRunner`.
        """
        from ...resources import get_model_resources, get_genomic_registry
        from ..data.sequence_extraction import extract_gene_sequences
        from ..data.genomic_extraction import extract_gene_annotations

        cfg = self.config
        model_resources = get_model_resources(cfg.base_model)
        build = model_resources.build
        annotation_source = model_resources.annotation_source

        # Registry resolution
        if annotation_source.lower() == "mane":
            registry_build = build if "_MANE" in build else f"{build}_MANE"
            registry = get_genomic_registry(build=registry_build, release="1.3")
        else:
            registry = get_genomic_registry(build=build, release="87")

        gtf_path = registry.get_gtf_path(validate=True)
        fasta_path = registry.get_fasta_path(validate=True)

        if cfg.verbosity >= 1:
            print(f"  Loading annotations: {gtf_path.name}")
            print(f"  Loading sequences:   {fasta_path.name}")

        if genes and chromosomes:
            # Union: all genes on specified chromosomes + explicitly named genes
            genes_df = extract_gene_annotations(
                gtf_file=str(gtf_path), verbosity=0,
            )
            if genes_df is None or genes_df.height == 0:
                return None
            genes_df = genes_df.filter(
                pl.col("chrom").is_in(chromosomes)
                | pl.col("gene_id").is_in(genes)
                | pl.col("gene_name").is_in(genes)
            )
        else:
            # Chromosomes-only or genes-only (or neither = all genes)
            genes_df = extract_gene_annotations(
                gtf_file=str(gtf_path),
                chromosomes=chromosomes,
                verbosity=0,
            )
            if genes_df is None or genes_df.height == 0:
                return None
            if genes:
                genes_df = genes_df.filter(
                    pl.col("gene_id").is_in(genes) | pl.col("gene_name").is_in(genes)
                )

        if genes_df.height == 0:
            return None

        # Extract sequences
        genes_df = extract_gene_sequences(
            gene_df=genes_df,
            fasta_file=str(fasta_path),
            verbosity=0,
        )
        genes_df = genes_df.filter(pl.col("sequence").is_not_null())

        if cfg.verbosity >= 1:
            print(f"  Loaded {genes_df.height} genes with sequences")

        return genes_df

    def _load_models(self) -> Optional[list]:
        """Load the base model weights (once for all chunks)."""
        from ..prediction.core import load_spliceai_models
        from ...resources import get_model_resources

        cfg = self.config
        model_resources = get_model_resources(cfg.base_model)

        if cfg.verbosity >= 1:
            print(f"  Loading {cfg.base_model} models...")

        models = load_spliceai_models(
            model_type=cfg.base_model,
            build=model_resources.build,
            verbosity=cfg.verbosity,
        )
        return models if models else None

    def _build_chunks(self, gene_df: pl.DataFrame) -> List[pl.DataFrame]:
        """Split gene DataFrame into chunks of ``chunk_size``.

        Genes are sorted by chromosome (natural order) then by start
        position, so chunks tend to be chromosome-contiguous.  This
        improves FASTA access locality.
        """
        # Sort by chromosome and position for locality
        chrom_col = "chrom" if "chrom" in gene_df.columns else "seqname"
        if chrom_col in gene_df.columns and "start" in gene_df.columns:
            gene_df = gene_df.sort([chrom_col, "start"])

        chunk_size = self.config.chunk_size
        n_genes = gene_df.height
        chunks = []
        for start_idx in range(0, n_genes, chunk_size):
            end_idx = min(start_idx + chunk_size, n_genes)
            chunks.append(gene_df.slice(start_idx, end_idx - start_idx))
        return chunks

    def _predict_chunk(
        self,
        chunk_gene_df: pl.DataFrame,
        models: list,
        chunk_idx: int,
    ) -> pl.DataFrame:
        """Run predictions for a single chunk and return raw scores DataFrame."""
        from ..prediction.core import predict_splice_sites_for_genes

        # Pass verbosity so tqdm shows per-gene progress within the chunk.
        # verbosity=0: silent, 1: tqdm progress bar, 2: per-gene logging
        predict_verbosity = self.config.verbosity

        predictions_dict = predict_splice_sites_for_genes(
            gene_df=chunk_gene_df,
            models=models,
            context=10000,
            output_format="dict",
            verbosity=predict_verbosity,
        )

        if not predictions_dict:
            return pl.DataFrame()

        return self._predictions_dict_to_dataframe(predictions_dict)

    @staticmethod
    def _predictions_dict_to_dataframe(predictions_dict: Dict[str, Dict]) -> pl.DataFrame:
        """Convert the per-gene predictions dict to a flat Polars DataFrame.

        This is the same conversion as ``BaseModelRunner._predictions_to_dataframe``
        but kept here to avoid a dependency on the runner instance.
        """
        records: List[Dict[str, Any]] = []
        for gene_id, pred in predictions_dict.items():
            gene_name = pred.get("gene_name", gene_id)
            chrom = pred.get("chrom", pred.get("seqname", ""))
            strand = pred.get("strand", "+")
            gene_start = pred.get("gene_start", 0)
            gene_end = pred.get("gene_end", 0)

            positions = pred.get("positions", [])
            donor_probs = pred.get("donor_prob", [])
            acceptor_probs = pred.get("acceptor_prob", [])
            neither_probs = pred.get("neither_prob", [])

            for i, pos in enumerate(positions):
                records.append({
                    "gene_id": gene_id,
                    "gene_name": gene_name,
                    "chrom": chrom,
                    "position": pos,
                    "strand": strand,
                    "gene_start": gene_start,
                    "gene_end": gene_end,
                    "donor_prob": donor_probs[i] if i < len(donor_probs) else 0.0,
                    "acceptor_prob": acceptor_probs[i] if i < len(acceptor_probs) else 0.0,
                    "neither_prob": neither_probs[i] if i < len(neither_probs) else 0.0,
                })

        return pl.DataFrame(records) if records else pl.DataFrame()

    def _make_error_result(self, start_time: float, error_msg: str) -> WorkflowResult:
        """Create a failed WorkflowResult."""
        elapsed = time.time() - start_time
        if self.config.verbosity >= 1:
            print(f"\n  Workflow failed: {error_msg}")
        return WorkflowResult(
            success=False,
            predictions=pl.DataFrame(),
            manifest=self._manifest,
            runtime_seconds=elapsed,
            output_dir=self._output_dir,
            error=error_msg,
        )
