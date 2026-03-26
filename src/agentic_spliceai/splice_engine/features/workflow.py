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
from .sampling import PositionSamplingConfig, sample_positions
from ..utils.memory_monitor import MemoryLimitExceeded, MemoryMonitor

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
    sampling_config : PositionSamplingConfig, optional
        Position sampling configuration. If None, no sampling is applied
        (all positions retained). Enable to reduce storage from ~271 GB
        to ~1-5 GB for full genome by keeping only splice-relevant
        positions and a configurable background sample.
    memory_limit_gb : float, optional
        Memory limit in GB. If set, a background MemoryMonitor thread
        logs RSS periodically and the workflow checks between chromosomes.
        When exceeded, the current chromosome is saved and the workflow
        exits gracefully. Default: None (no monitoring).

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
        sampling_config: Optional[PositionSamplingConfig] = None,
        memory_limit_gb: Optional[float] = None,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.input_dir = Path(input_dir)
        self.resume = resume
        self.sampling_config = sampling_config
        self.memory_limit_gb = memory_limit_gb

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

        Predictions are loaded lazily — only one chromosome is materialized
        in memory at a time.  This keeps peak memory proportional to the
        largest single chromosome (~1.7 GB for chr1) rather than the full
        genome (~40 GB), enabling genome-scale runs on memory-constrained
        machines.

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

        # Start memory monitor if configured
        monitor: Optional[MemoryMonitor] = None
        if self.memory_limit_gb is not None:
            monitor = MemoryMonitor(
                limit_gb=self.memory_limit_gb,
                abort_on_exceed=True,
            )
            monitor.start()

        try:
            # Build lazy scanner (no data loaded yet)
            lazy = self._build_lazy_scanner()
            if lazy is None:
                result.error = "No predictions found in input directory."
                return result

            # Discover available chromosomes (lightweight — reads only chrom column)
            available_chroms = (
                lazy.select("chrom").unique().sort("chrom").collect()["chrom"].to_list()
            )

            if chromosomes is not None:
                # Use chromosomes as-is — caller is responsible for
                # build-appropriate naming (see normalize_chromosomes_for_build)
                target_chroms = list(chromosomes)
                missing = set(target_chroms) - set(available_chroms)
                if missing:
                    logger.warning(
                        "Requested chromosomes not in predictions: %s", missing
                    )
                target_chroms = [c for c in target_chroms if c in available_chroms]
            else:
                target_chroms = available_chroms

            logger.info(
                "Feature workflow: %d chromosomes to process", len(target_chroms),
            )

            total_positions = 0
            fmt = self.pipeline_config.output_format

            for chrom in target_chroms:
                # Memory check between chromosomes (safe checkpoint boundary)
                if monitor is not None:
                    try:
                        snap = monitor.check()
                        logger.info(
                            "Memory check before %s: RSS=%.2f GB",
                            chrom,
                            snap.rss_gb,
                        )
                    except MemoryLimitExceeded:
                        logger.warning(
                            "Memory limit exceeded before %s — saving progress "
                            "and exiting gracefully. Completed: %s",
                            chrom,
                            result.chromosomes_processed,
                        )
                        result.total_positions = total_positions
                        result.success = True  # partial success
                        result.error = (
                            f"Memory limit ({self.memory_limit_gb} GB) exceeded "
                            f"before {chrom}. Processed {len(result.chromosomes_processed)} "
                            f"chromosomes. Re-run with --resume to continue."
                        )
                        result.runtime_seconds = time.time() - t0
                        self._save_summary(result)
                        return result

                output_path = self._get_chrom_output_path(chrom, fmt)

                # Resume support
                if self.resume and output_path.exists():
                    logger.info("Resuming: skipping %s (output exists)", chrom)
                    result.chromosomes_skipped.append(chrom)
                    # Count positions without loading full file
                    total_positions += self._count_output(output_path, fmt)
                    continue

                # Lazy filter + collect: only this chromosome is materialized
                chrom_df = lazy.filter(pl.col("chrom") == chrom).collect()
                logger.info(
                    "Processing %s: %d positions...", chrom, chrom_df.height
                )

                # Early sampling: reduce positions BEFORE expensive modalities
                early = (
                    self.sampling_config is not None
                    and getattr(self.sampling_config, "early", False)
                )
                if early:
                    n_before = chrom_df.height
                    chrom_df = sample_positions(chrom_df, self.sampling_config)
                    logger.info(
                        "Early sampling %s: %d → %d positions (%.1f%%) "
                        "before feature engineering",
                        chrom,
                        n_before,
                        chrom_df.height,
                        100 * chrom_df.height / n_before if n_before > 0 else 0,
                    )

                # Apply pipeline
                enriched = self._pipeline.transform(chrom_df)

                # Free input before writing (only enriched is needed)
                del chrom_df

                # Late sampling (only if early sampling was not applied)
                if self.sampling_config is not None and not early:
                    n_before = enriched.height
                    enriched = sample_positions(enriched, self.sampling_config)
                    logger.info(
                        "Sampled %s: %d → %d positions (%.1f%%)",
                        chrom,
                        n_before,
                        enriched.height,
                        100 * enriched.height / n_before if n_before > 0 else 0,
                    )

                # Save atomically
                self._atomic_write(enriched, output_path, fmt)
                logger.info(
                    "Saved %s: %d positions, %d columns -> %s",
                    chrom,
                    enriched.height,
                    enriched.width,
                    output_path,
                )

                result.chromosomes_processed.append(chrom)
                total_positions += enriched.height

                # Free enriched before next chromosome
                del enriched

            result.total_positions = total_positions
            result.success = True

            # Save workflow summary
            result.runtime_seconds = time.time() - t0
            self._save_summary(result)

        except MemoryLimitExceeded as e:
            logger.warning("Memory limit exceeded during processing: %s", e)
            result.error = str(e) + " Re-run with --resume to continue."
            result.runtime_seconds = time.time() - t0
            # Still save summary for partial progress
            if result.chromosomes_processed:
                result.success = True
                self._save_summary(result)

        except Exception as e:
            logger.error("Feature workflow failed: %s", e, exc_info=True)
            result.error = str(e)
            result.runtime_seconds = time.time() - t0

        finally:
            if monitor is not None:
                monitor.stop()

        return result

    # ------------------------------------------------------------------
    # Augmentation — add new modalities to existing artifacts
    # ------------------------------------------------------------------

    def augment(
        self,
        chromosomes: List[str],
        new_modalities: List[str],
    ) -> FeatureWorkflowResult:
        """Add new modality columns to existing analysis_sequences parquets.

        Loads each chromosome's parquet, applies only the specified new
        modalities, and overwrites the parquet with the augmented columns.
        No predictions are needed — the existing parquet provides all
        required inputs (chrom, position, prior modality outputs).

        Parameters
        ----------
        chromosomes : list of str
            Chromosomes to augment.
        new_modalities : list of str
            Modality names to add (must be in FeaturePipeline registry).

        Returns
        -------
        FeatureWorkflowResult
            Augmentation results.
        """
        t0 = time.time()

        # Build sub-pipeline with only the new modalities
        augment_config = FeaturePipelineConfig(
            base_model=self.pipeline_config.base_model,
            modalities=new_modalities,
            modality_configs={
                k: v for k, v in self.pipeline_config.modality_configs.items()
                if k in new_modalities
            },
            output_format=self.pipeline_config.output_format,
        )

        # Import modalities registry (triggers auto-registration)
        from . import modalities as _  # noqa: F401

        sub_pipeline = FeaturePipeline(augment_config)
        sub_schema = sub_pipeline.get_output_schema()
        new_cols_flat = {
            col for cols in sub_schema.values() for col in cols
        }

        result = FeatureWorkflowResult(
            success=False,
            output_dir=self.output_dir,
            pipeline_schema=sub_schema,
        )

        logger.info(
            "Augmenting with %d new modalities: %s",
            len(new_modalities), new_modalities,
        )

        total_positions = 0
        fmt = self.pipeline_config.output_format

        for chrom in chromosomes:
            output_path = self._get_chrom_output_path(chrom, fmt)

            if not output_path.exists():
                logger.warning(
                    "No existing artifact for %s at %s — skipping.",
                    chrom, output_path,
                )
                result.chromosomes_skipped.append(chrom)
                continue

            # Load existing parquet
            existing_df = pl.read_parquet(output_path)
            existing_cols = set(existing_df.columns)
            n_positions = existing_df.height

            # Idempotency: check if new columns already exist
            already_present = new_cols_flat & existing_cols
            if already_present == new_cols_flat:
                logger.info(
                    "Skipping %s — all %d new columns already present",
                    chrom, len(new_cols_flat),
                )
                result.chromosomes_skipped.append(chrom)
                total_positions += n_positions
                del existing_df
                continue

            if already_present:
                logger.info(
                    "Partial overlap on %s: %d/%d new columns already exist. "
                    "Dropping existing to re-augment.",
                    chrom, len(already_present), len(new_cols_flat),
                )
                existing_df = existing_df.drop(list(already_present))

            # Validate required inputs
            for mod in sub_pipeline._modalities:
                missing = mod.meta.required_inputs - set(existing_df.columns)
                if missing:
                    raise ValueError(
                        f"Modality '{mod.meta.name}' requires columns "
                        f"{sorted(missing)} not found in {output_path.name}. "
                        f"Run full pipeline first."
                    )

            # Apply new modalities
            logger.info(
                "Augmenting %s: %d positions, adding %d columns...",
                chrom, n_positions, len(new_cols_flat - already_present),
            )
            augmented = sub_pipeline.transform(existing_df)
            del existing_df

            # Atomic write (overwrite original)
            self._atomic_write(augmented, output_path, fmt)
            logger.info(
                "Saved augmented %s: %d positions, %d columns -> %s",
                chrom, augmented.height, augmented.width, output_path,
            )

            result.chromosomes_processed.append(chrom)
            total_positions += augmented.height
            del augmented

        result.total_positions = total_positions
        result.success = True
        result.runtime_seconds = time.time() - t0

        # Update summary with augmentation info
        self._update_summary_after_augment(result, new_modalities)

        return result

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _build_lazy_scanner(self) -> Optional[pl.LazyFrame]:
        """Build a lazy scanner over prediction files.

        Returns a LazyFrame that can be filtered per-chromosome and
        collected on demand — no data is loaded into memory until
        ``.collect()`` is called with a filter.

        Resolution order:
        1. Per-chromosome parquet files (``predictions_chr*.parquet``)
        2. Aggregated ``predictions.tsv``
        3. Chunk TSV files (``predictions_chunk_*.tsv``)
        """
        # 1. Per-chromosome parquet (best case — zero unnecessary I/O)
        # Match both chr-prefixed (GRCh38: predictions_chr22.parquet) and
        # bare names (GRCh37: predictions_22.parquet). Exclude chunk files.
        chrom_parquets = sorted([
            f for f in self.input_dir.glob("predictions_*.parquet")
            if not f.name.startswith("predictions_chunk")
        ])
        if chrom_parquets:
            logger.info(
                "Using %d per-chromosome parquet files", len(chrom_parquets)
            )
            lazy = pl.concat([pl.scan_parquet(f) for f in chrom_parquets])
            return self._ensure_chrom_lazy(lazy)

        # 2. Aggregated TSV (common case — lazy scan streams through file)
        aggregated = self.input_dir / "predictions.tsv"
        if aggregated.exists():
            logger.info("Lazy-scanning aggregated predictions: %s", aggregated)
            lazy = pl.scan_csv(aggregated, separator="\t")
            return self._ensure_chrom_lazy(lazy)

        # 3. Chunk TSV files
        chunks = sorted(self.input_dir.glob("predictions_chunk_*.tsv"))
        if not chunks:
            return None

        logger.info("Lazy-scanning %d prediction chunk files...", len(chunks))
        lazy = pl.concat([pl.scan_csv(f, separator="\t") for f in chunks])
        return self._ensure_chrom_lazy(lazy)

    def _ensure_chrom_lazy(self, lazy: pl.LazyFrame) -> pl.LazyFrame:
        """Ensure the LazyFrame has a ``chrom`` column (may be ``seqname``).

        Also casts ``chrom`` to Utf8 — Polars infers bare numeric chromosome
        names (e.g., ``22``) as Int64, but all downstream code uses strings.
        """
        cols = lazy.collect_schema().names()
        if "chrom" not in cols and "seqname" in cols:
            lazy = lazy.rename({"seqname": "chrom"})
        # Ensure chrom is always string (Polars infers "22" as Int64 from TSV)
        lazy = lazy.with_columns(pl.col("chrom").cast(pl.Utf8))
        return lazy

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

    def _count_output(self, path: Path, fmt: str) -> int:
        """Count rows in an output file without loading it into memory."""
        if fmt == "parquet" or path.suffix == ".parquet":
            return pl.scan_parquet(path).select(pl.len()).collect().item()
        return pl.scan_csv(path, separator="\t").select(pl.len()).collect().item()

    def _save_summary(self, result: FeatureWorkflowResult) -> None:
        """Save workflow summary as JSON."""
        summary = result.get_summary()
        summary["timestamp"] = datetime.now().isoformat()
        summary["pipeline_config"] = {
            "base_model": self.pipeline_config.base_model,
            "modalities": self.pipeline_config.modalities,
            "output_format": self.pipeline_config.output_format,
        }
        if self.sampling_config is not None and self.sampling_config.enabled:
            summary["sampling"] = {
                "score_threshold": self.sampling_config.score_threshold,
                "proximity_window": self.sampling_config.proximity_window,
                "background_rate": self.sampling_config.background_rate,
            }

        path = self.output_dir / "feature_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved feature summary → %s", path)

    def _update_summary_after_augment(
        self,
        result: FeatureWorkflowResult,
        new_modalities: List[str],
    ) -> None:
        """Update feature_summary.json after augmentation.

        Merges new modalities into the existing summary rather than
        overwriting. Adds an 'augmentations' history log.
        """
        summary_path = self.output_dir / "feature_summary.json"

        # Load existing summary or start fresh
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        else:
            summary = {}

        # Merge modalities
        existing_mods = summary.get("pipeline_config", {}).get("modalities", [])
        all_mods = list(dict.fromkeys(existing_mods + new_modalities))
        summary.setdefault("pipeline_config", {})["modalities"] = all_mods

        # Merge schema
        existing_schema = summary.get("pipeline_schema", {})
        existing_schema.update(result.pipeline_schema)
        summary["pipeline_schema"] = existing_schema

        # Append augmentation log
        augment_entry = {
            "timestamp": datetime.now().isoformat(),
            "new_modalities": new_modalities,
            "chromosomes_augmented": result.chromosomes_processed,
            "chromosomes_skipped": result.chromosomes_skipped,
            "runtime_seconds": round(result.runtime_seconds, 2),
        }
        summary.setdefault("augmentations", []).append(augment_entry)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Updated feature summary with augmentation → %s", summary_path)


def detect_existing_modalities(output_dir: Path) -> set[str]:
    """Detect which modalities are present in existing artifacts.

    Strategy (in priority order):
    1. Read feature_summary.json → pipeline_config.modalities
    2. Fallback: read first parquet's columns, match against registry

    Parameters
    ----------
    output_dir : Path
        Directory containing analysis_sequences parquets.

    Returns
    -------
    set of str
        Names of modalities whose output columns are present.
    """
    summary_path = output_dir / "feature_summary.json"

    # Strategy 1: summary file, cross-checked against actual parquet columns.
    # The summary may be stale (e.g., columns were stripped manually), so
    # verify that declared modalities actually have columns in the parquets.
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        declared_mods = summary.get("pipeline_config", {}).get("modalities", [])
        schema_map = summary.get("pipeline_schema", {})

        if declared_mods and schema_map:
            # Cross-check: read columns from the parquet with the fewest
            # columns (the least-augmented file). Using the first file
            # alphabetically can give false positives when some chromosomes
            # have been augmented but others haven't yet.
            parquets = sorted(output_dir.glob("analysis_sequences_*.parquet"))
            if parquets:
                min_parquet = min(
                    parquets,
                    key=lambda p: len(pl.scan_parquet(p).collect_schema().names()),
                )
                actual_cols = set(
                    pl.scan_parquet(min_parquet).collect_schema().names()
                )
                verified = set()
                for mod_name in declared_mods:
                    mod_cols = schema_map.get(mod_name, [])
                    if mod_cols and set(mod_cols).issubset(actual_cols):
                        verified.add(mod_name)
                    elif not mod_cols:
                        # No schema info — trust the summary
                        verified.add(mod_name)
                    else:
                        logger.info(
                            "Modality '%s' declared in summary but columns "
                            "missing from %s — treating as absent.",
                            mod_name, min_parquet.name,
                        )
                logger.info(
                    "Detected %d existing modalities (summary + column check): %s",
                    len(verified), sorted(verified),
                )
                return verified

    # Strategy 2: column matching against registry
    parquets = sorted(output_dir.glob("analysis_sequences_*.parquet"))
    if not parquets:
        logger.info("No existing parquets found in %s", output_dir)
        return set()

    # Read columns from first parquet (lightweight)
    existing_cols = set(pl.scan_parquet(parquets[0]).collect_schema().names())

    # Import modalities to populate registry
    from . import modalities as _  # noqa: F401

    detected = set()
    for name in FeaturePipeline.available_modalities():
        mod_cls, cfg_cls = FeaturePipeline._REGISTRY[name]
        mod = mod_cls(mod_cls.default_config())
        output_cols = set(mod.meta.output_columns)
        if output_cols and output_cols.issubset(existing_cols):
            detected.add(name)

    logger.info(
        "Detected %d existing modalities by column matching: %s",
        len(detected), sorted(detected),
    )
    return detected
