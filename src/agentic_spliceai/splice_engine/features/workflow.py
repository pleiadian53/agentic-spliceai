"""Feature engineering workflow — genome-scale processing with I/O management.

Wraps the FeaturePipeline with chromosome-level chunked I/O, artifact
management (atomic writes, resume support), and progress reporting.
Reads base layer prediction artifacts and writes analysis_sequences output.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

from .pipeline import FeaturePipeline, FeaturePipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureWorkflowResult:
    """Container for feature workflow results.

    Attributes
    ----------
    success : bool
        Whether the workflow completed without fatal errors.
    chromosomes_processed : list of str
        Chromosomes that were actually processed (not resumed).
    chromosomes_skipped : list of str
        Chromosomes skipped via resume.
    total_positions : int
        Total number of positions in the output.
    output_dir : Path or None
        Directory where artifacts were saved.
    runtime_seconds : float
        Total wall-clock time.
    error : str or None
        Error message if workflow failed.
    pipeline_schema : dict
        Output column schema grouped by modality.
    """

    success: bool
    chromosomes_processed: List[str] = field(default_factory=list)
    chromosomes_skipped: List[str] = field(default_factory=list)
    total_positions: int = 0
    output_dir: Optional[Path] = None
    runtime_seconds: float = 0.0
    error: Optional[str] = None
    pipeline_schema: Dict[str, List[str]] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary dict suitable for JSON serialization."""
        return {
            "success": self.success,
            "chromosomes_processed": self.chromosomes_processed,
            "chromosomes_skipped": self.chromosomes_skipped,
            "total_positions": self.total_positions,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "runtime_seconds": round(self.runtime_seconds, 2),
            "pipeline_schema": self.pipeline_schema,
            "error": self.error,
        }


class FeatureWorkflow:
    """Orchestrate feature engineering at genome scale.

    Reads base layer predictions (from PredictionWorkflow output),
    applies the FeaturePipeline, and saves analysis_sequences artifacts
    per chromosome.

    Parameters
    ----------
    pipeline_config : FeaturePipelineConfig
        Configuration for the feature engineering pipeline.
    input_dir : str or Path
        Directory containing base layer prediction artifacts
        (predictions.tsv or predictions_chunk_*.tsv files).
    output_dir : str or Path, optional
        Directory for output artifacts. If None, defaults to
        ``{input_dir}/../analysis_sequences/``.
    resume : bool
        If True, skip chromosomes that already have output files.

    Examples
    --------
    >>> config = FeaturePipelineConfig(modalities=['base_scores', 'annotation'])
    >>> workflow = FeatureWorkflow(config, input_dir='data/.../precomputed/')
    >>> result = workflow.run(chromosomes=['chr22'])
    """

    def __init__(
        self,
        pipeline_config: FeaturePipelineConfig,
        input_dir: Path | str,
        output_dir: Optional[Path | str] = None,
        resume: bool = False,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.input_dir = Path(input_dir)
        self.resume = resume

        if output_dir is not None:
            self.output_dir = Path(output_dir)
        else:
            # Default: sibling directory to input
            self.output_dir = self.input_dir.parent / "analysis_sequences"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import modalities registry (triggers auto-registration)
        from . import modalities as _  # noqa: F401

        self._pipeline = FeaturePipeline(pipeline_config)

    def run(
        self,
        chromosomes: Optional[List[str]] = None,
    ) -> FeatureWorkflowResult:
        """Run the feature engineering workflow.

        Parameters
        ----------
        chromosomes : list of str, optional
            Chromosomes to process. If None, processes all chromosomes
            found in the input predictions.

        Returns
        -------
        FeatureWorkflowResult
            Workflow results including processed chromosomes and output paths.
        """
        t0 = time.time()
        result = FeatureWorkflowResult(
            success=False,
            output_dir=self.output_dir,
            pipeline_schema=self._pipeline.get_output_schema(),
        )

        try:
            # Load input predictions
            predictions = self._load_predictions()
            if predictions.height == 0:
                result.error = "No predictions found in input directory."
                return result

            # Determine chromosomes to process
            available_chroms = predictions["chrom"].unique().sort().to_list()
            if chromosomes is not None:
                # Normalize chromosome names
                target_chroms = [
                    c if c.startswith("chr") else f"chr{c}" for c in chromosomes
                ]
                missing = set(target_chroms) - set(available_chroms)
                if missing:
                    logger.warning(
                        "Requested chromosomes not in predictions: %s", missing
                    )
                target_chroms = [c for c in target_chroms if c in available_chroms]
            else:
                target_chroms = available_chroms

            logger.info(
                "Feature workflow: %d chromosomes, %d total positions",
                len(target_chroms),
                predictions.height,
            )

            total_positions = 0
            fmt = self.pipeline_config.output_format

            for chrom in target_chroms:
                output_path = self._get_chrom_output_path(chrom, fmt)

                # Resume support
                if self.resume and output_path.exists():
                    logger.info("Resuming: skipping %s (output exists)", chrom)
                    result.chromosomes_skipped.append(chrom)
                    # Count positions in existing file
                    existing = self._read_output(output_path, fmt)
                    total_positions += existing.height
                    continue

                # Filter to this chromosome
                chrom_df = predictions.filter(pl.col("chrom") == chrom)
                logger.info(
                    "Processing %s: %d positions...", chrom, chrom_df.height
                )

                # Apply pipeline
                enriched = self._pipeline.transform(chrom_df)

                # Save atomically
                self._atomic_write(enriched, output_path, fmt)
                logger.info(
                    "Saved %s: %d positions, %d columns → %s",
                    chrom,
                    enriched.height,
                    enriched.width,
                    output_path,
                )

                result.chromosomes_processed.append(chrom)
                total_positions += enriched.height

            result.total_positions = total_positions
            result.success = True

            # Save workflow summary
            result.runtime_seconds = time.time() - t0
            self._save_summary(result)

        except Exception as e:
            logger.error("Feature workflow failed: %s", e, exc_info=True)
            result.error = str(e)
            result.runtime_seconds = time.time() - t0

        return result

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_predictions(self) -> pl.DataFrame:
        """Load base layer predictions from the input directory.

        Looks for aggregated ``predictions.tsv`` first, then falls back
        to concatenating ``predictions_chunk_*.tsv`` files.
        """
        from agentic_spliceai.splice_engine.resources import ensure_chrom_column

        aggregated = self.input_dir / "predictions.tsv"
        if aggregated.exists():
            logger.info("Loading aggregated predictions: %s", aggregated)
            df = pl.read_csv(aggregated, separator="\t")
            return ensure_chrom_column(df)

        # Fall back to chunk files
        chunks = sorted(self.input_dir.glob("predictions_chunk_*.tsv"))
        if not chunks:
            return pl.DataFrame()

        logger.info("Loading %d prediction chunk files...", len(chunks))
        frames = [pl.read_csv(f, separator="\t") for f in chunks]
        return ensure_chrom_column(pl.concat(frames))

    def _get_chrom_output_path(self, chrom: str, fmt: str) -> Path:
        """Get the output path for a chromosome artifact."""
        ext = "parquet" if fmt == "parquet" else "tsv"
        return self.output_dir / f"analysis_sequences_{chrom}.{ext}"

    def _atomic_write(
        self, df: pl.DataFrame, path: Path, fmt: str
    ) -> None:
        """Write DataFrame atomically (temp file + rename)."""
        if fmt == "parquet":
            tmp = path.with_suffix(".parquet.tmp")
            df.write_parquet(tmp)
        else:
            tmp = path.with_suffix(".tsv.tmp")
            df.write_csv(tmp, separator="\t")
        tmp.rename(path)

    def _read_output(self, path: Path, fmt: str) -> pl.DataFrame:
        """Read a previously saved output file."""
        if fmt == "parquet" or path.suffix == ".parquet":
            return pl.read_parquet(path)
        return pl.read_csv(path, separator="\t")

    def _save_summary(self, result: FeatureWorkflowResult) -> None:
        """Save workflow summary as JSON."""
        summary = result.get_summary()
        summary["timestamp"] = datetime.now().isoformat()
        summary["pipeline_config"] = {
            "base_model": self.pipeline_config.base_model,
            "modalities": self.pipeline_config.modalities,
            "output_format": self.pipeline_config.output_format,
        }

        path = self.output_dir / "feature_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved feature summary → %s", path)
