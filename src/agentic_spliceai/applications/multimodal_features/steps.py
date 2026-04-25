"""Feature-generation pipeline steps.

Thin wrappers over existing library code under
``splice_engine.features``. No new feature logic lives here — this module
orchestrates existing primitives and records outputs in the manifest.

Steps implemented
-----------------

- ``step_fetch_conservation`` — download PhyloP / PhastCons bigWigs to the
  configured cache (skipped if already present)
- ``step_generate_features`` — run ``FeatureWorkflow`` for a profile over
  selected chromosomes; writes ``analysis_sequences_{chrom}.parquet``
- ``step_validate`` — read-only sanity check (alignment columns present,
  non-empty rows)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import (
    FeatureArtifactRecord,
    FeatureManifest,
    TrackReference,
)
from .profiles import FeatureProfile
from .tracks import TrackRecord, fetch_conservation_tracks

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    step_name: str
    success: bool
    artifact_name: Optional[str] = None
    artifact_path: Optional[Path] = None
    rows: Optional[int] = None
    skipped: bool = False
    skipped_reason: Optional[str] = None
    error: Optional[str] = None
    extras: Dict[str, Any] = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# step: fetch conservation
# ---------------------------------------------------------------------------


def step_fetch_conservation(
    *,
    build: str,
    cache_dir: Optional[Path] = None,
    force: bool = False,
    verbosity: int = 1,
) -> StepResult:
    """Download PhyloP + PhastCons bigWigs for ``build`` into the cache.

    Existing files are preserved unless ``force=True``.
    """
    try:
        updated = fetch_conservation_tracks(
            build=build, dest_dir=cache_dir, force=force, verbosity=verbosity,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("fetch_conservation: failed")
        return StepResult(
            step_name="fetch_conservation", success=False, error=str(exc),
        )

    n_cached = sum(1 for t in updated if t.is_cached)
    return StepResult(
        step_name="fetch_conservation",
        success=(n_cached > 0),
        rows=n_cached,
        extras={"tracks": [t.name for t in updated if t.is_cached]},
    )


# ---------------------------------------------------------------------------
# step: generate features
# ---------------------------------------------------------------------------


def step_generate_features(
    *,
    profile: FeatureProfile,
    input_dir: Path,
    output_dir: Path,
    chromosomes: Optional[List[str]] = None,
    resume: bool = True,
    memory_limit_gb: Optional[float] = None,
    verbosity: int = 1,
) -> StepResult:
    """Run ``FeatureWorkflow`` for a profile over selected chromosomes.

    ``resume=True`` (default) matches the library's behaviour: chromosomes
    that already have output parquets are skipped. This is production-safe
    for the canonical ``analysis_sequences/`` directory.
    """
    try:
        from agentic_spliceai.splice_engine.features import (
            FeaturePipelineConfig, FeatureWorkflow,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_features: could not import FeatureWorkflow")
        return StepResult(
            step_name="generate_features", success=False, error=str(exc),
        )

    try:
        pipeline_config = _build_pipeline_config_from_profile(
            profile, FeaturePipelineConfig,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_features: profile conversion failed")
        return StepResult(
            step_name="generate_features", success=False, error=str(exc),
        )

    try:
        workflow = FeatureWorkflow(
            pipeline_config=pipeline_config,
            input_dir=Path(input_dir),
            output_dir=Path(output_dir),
            resume=resume,
            memory_limit_gb=memory_limit_gb,
        )
        if verbosity >= 1:
            logger.info(
                "generate_features: running profile=%r, modalities=%s, "
                "chromosomes=%s",
                profile.name, profile.modalities,
                chromosomes or "<all>",
            )
        workflow_result = workflow.run(chromosomes=chromosomes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate_features: workflow.run failed")
        return StepResult(
            step_name="generate_features", success=False, error=str(exc),
        )

    return StepResult(
        step_name="generate_features",
        success=bool(workflow_result.success),
        rows=int(workflow_result.total_positions or 0),
        skipped=(not workflow_result.chromosomes_processed
                 and bool(workflow_result.chromosomes_skipped)),
        skipped_reason=(
            "all chromosomes already present"
            if not workflow_result.chromosomes_processed
               and workflow_result.chromosomes_skipped
            else None
        ),
        error=workflow_result.error,
        extras={
            "chromosomes_processed": list(workflow_result.chromosomes_processed),
            "chromosomes_skipped": list(workflow_result.chromosomes_skipped),
            "pipeline_schema": dict(workflow_result.pipeline_schema),
        },
    )


# ---------------------------------------------------------------------------
# step: validate (read-only)
# ---------------------------------------------------------------------------


def step_validate(
    *,
    output_dir: Path,
    chromosomes: Optional[List[str]] = None,
    verbosity: int = 1,
) -> StepResult:
    """Read-only: inspect generated parquets for basic integrity.

    Checks that each expected per-chrom parquet has non-zero rows and
    carries the alignment columns (``chrom``, ``position``).
    """
    import polars as pl

    output_dir = Path(output_dir)
    files = sorted(output_dir.glob("analysis_sequences_*.parquet"))
    if chromosomes is not None:
        wanted = set()
        for c in chromosomes:
            wanted.add(str(c))
            wanted.add(f"chr{c}" if not str(c).startswith("chr") else str(c))
            wanted.add(str(c).removeprefix("chr"))
        files = [
            f for f in files
            if any(w in f.name for w in wanted)
        ]

    if not files:
        return StepResult(
            step_name="validate",
            success=False,
            error=f"No analysis_sequences_*.parquet files in {output_dir}.",
        )

    total_rows = 0
    for fp in files:
        try:
            df = pl.read_parquet(fp, n_rows=1)
            cols = set(df.columns)
            if "chrom" not in cols or "position" not in cols:
                return StepResult(
                    step_name="validate",
                    success=False,
                    error=(
                        f"{fp.name} missing required alignment columns "
                        f"('chrom', 'position'). Found: {sorted(cols)[:10]}"
                    ),
                )
            total_rows += pl.scan_parquet(fp).select(pl.len()).collect().item()
        except Exception as exc:  # noqa: BLE001
            return StepResult(
                step_name="validate",
                success=False,
                error=f"Failed to read {fp.name}: {exc}",
            )

    if verbosity >= 1:
        logger.info(
            "validate: %d parquets ok, total %d rows", len(files), total_rows,
        )
    return StepResult(
        step_name="validate", success=True, rows=total_rows,
        extras={"n_files": len(files)},
    )


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------


def record_features_artifacts(
    manifest: FeatureManifest,
    output_dir: Path,
    *,
    processed_chromosomes: List[str],
    hash_artifacts: bool = True,
) -> None:
    """Walk ``output_dir`` for per-chrom parquets and update the manifest."""
    output_dir = Path(output_dir)
    for chrom in processed_chromosomes:
        # Library currently writes both ``analysis_sequences_{chrom}.parquet``
        # and occasionally a ``chr``-prefixed variant depending on build.
        for fname in (
            f"analysis_sequences_{chrom}.parquet",
            f"analysis_sequences_chr{chrom}.parquet",
            f"analysis_sequences_{chrom.removeprefix('chr')}.parquet"
            if chrom.startswith("chr") else None,
        ):
            if not fname:
                continue
            candidate = output_dir / fname
            if candidate.exists():
                record = FeatureArtifactRecord.from_path(
                    candidate,
                    chromosome=chrom,
                    step="generate_features",
                    hash_file=hash_artifacts,
                )
                manifest.set_artifact(chrom, record)
                break


def record_tracks_used(
    manifest: FeatureManifest,
    tracks: List[TrackRecord],
) -> None:
    """Attach cached-track references to the manifest."""
    for t in tracks:
        if not t.is_cached or not t.cached_path:
            continue
        manifest.add_track(TrackReference(
            modality=t.modality,
            name=t.name,
            path=str(t.cached_path),
            accession=t.accession,
        ))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pipeline_config_from_profile(
    profile: FeatureProfile,
    FeaturePipelineConfig,
):
    """Map a :class:`FeatureProfile` onto the library's pipeline config.

    Mirrors ``examples/features/config_loader.py``: each modality's raw
    config dict is instantiated into the typed ``ModalityConfig`` subclass
    looked up from ``FeaturePipeline._REGISTRY``. Unknown keys are dropped
    with a warning (forward-compat).
    """
    from dataclasses import fields as dataclass_fields

    from agentic_spliceai.splice_engine.features import FeaturePipeline

    modality_configs: Dict[str, Any] = {}
    for name in profile.modalities:
        if name not in profile.modality_configs:
            continue  # Pipeline falls back to default_config()
        cfg_dict = profile.modality_configs[name]
        if name not in FeaturePipeline._REGISTRY:
            raise ValueError(
                f"Profile {profile.name!r} references unknown modality {name!r}. "
                f"Registered: {FeaturePipeline.available_modalities()}"
            )
        _, config_cls = FeaturePipeline._REGISTRY[name]
        valid = {f.name for f in dataclass_fields(config_cls)}
        filtered = {k: v for k, v in (cfg_dict or {}).items() if k in valid}
        dropped = [k for k in (cfg_dict or {}) if k not in valid]
        if dropped:
            logger.warning(
                "Profile %s/%s: ignoring unknown config keys %s",
                profile.name, name, dropped,
            )
        modality_configs[name] = config_cls(**filtered)

    return FeaturePipelineConfig(
        base_model=profile.base_model or "openspliceai",
        modalities=list(profile.modalities),
        modality_configs=modality_configs,
    )
