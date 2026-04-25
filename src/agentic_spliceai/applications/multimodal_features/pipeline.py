"""Orchestrator for the multimodal-features application.

``prepare_features`` is the single high-level entry point. It composes
the step functions from :mod:`steps`, records artifacts in a
:class:`FeatureManifest`, and returns a :class:`PreparationResult`.

Production safety
-----------------

The pipeline never writes outside ``output_dir``. ``--inplace`` (CLI-only)
resolves ``output_dir`` to the resource-manager-resolved production path
for the chosen build, but existing artifacts are preserved unless
``force=True``. The underlying ``FeatureWorkflow`` already defaults to
``resume=True`` so per-chromosome parquets are skipped when present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .manifest import FeatureManifest
from .profiles import FeatureProfile, load_profile
from .steps import (
    StepResult,
    record_features_artifacts,
    record_tracks_used,
    step_fetch_conservation,
    step_generate_features,
)
from .tracks import list_tracks

logger = logging.getLogger(__name__)


DEFAULT_STEPS = ("fetch_conservation", "generate_features")


def resolve_canonical_features_dir(
    *, build: str, annotation_source: str, base_model: str = "openspliceai",
) -> Path:
    """Return the conventional features output dir for a build.

    Mirrors where the library writes by default:

        data/<source>/<build>/<base_model>_eval/analysis_sequences/

    The CLI writes here only when the user passes ``--inplace``.
    """
    from agentic_spliceai.splice_engine.resources import get_genomic_registry

    if build == "GRCh38" and annotation_source == "mane":
        registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
    else:
        registry = get_genomic_registry(build=build)
    return Path(registry.data_dir) / f"{base_model}_eval" / "analysis_sequences"


@dataclass
class PreparationResult:
    """High-level summary of a feature-preparation run."""

    build: str
    profile: str
    output_dir: Path
    manifest_path: Optional[Path] = None
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True

    def summary(self) -> str:
        lines = [
            f"Feature preparation — build={self.build}, profile={self.profile}",
            f"Output dir: {self.output_dir}",
            f"Manifest:   {self.manifest_path}",
            "Steps:",
        ]
        for step in self.steps:
            if step.skipped:
                status = f"skipped ({step.skipped_reason or 'n/a'})"
            elif step.success:
                rows = f", rows={step.rows}" if step.rows is not None else ""
                extras = step.extras or {}
                extra_bits = []
                if extras.get("chromosomes_processed") is not None:
                    extra_bits.append(
                        f"processed={extras['chromosomes_processed']}"
                    )
                if extras.get("chromosomes_skipped") is not None:
                    extra_bits.append(
                        f"skipped={extras['chromosomes_skipped']}"
                    )
                extras_tail = f"  [{', '.join(extra_bits)}]" if extra_bits else ""
                status = f"ok{rows}{extras_tail}"
            else:
                status = f"FAILED — {step.error}"
            lines.append(f"  [{step.step_name}] {status}")
        lines.append(f"Overall success: {self.success}")
        return "\n".join(lines)


def prepare_features(
    *,
    profile: str,
    build: str,
    input_dir: Path,
    output_dir: Path,
    chromosomes: Optional[List[str]] = None,
    fetch_missing_tracks: bool = False,
    skip_steps: Optional[List[str]] = None,
    only_steps: Optional[List[str]] = None,
    resume: bool = True,
    memory_limit_gb: Optional[float] = None,
    hash_artifacts: bool = True,
    track: bool = False,
    tracking_project: Optional[str] = None,
    tracking_tags: Optional[List[str]] = None,
    verbosity: int = 1,
) -> PreparationResult:
    """Run the feature-preparation pipeline.

    Parameters
    ----------
    profile
        Profile name (YAML basename under ``examples/features/configs/``).
    build
        Genomic build identifier (``"GRCh38"``, ``"GRCh37"``).
    input_dir
        Directory containing base-layer prediction artifacts
        (``predictions_{chrom}.parquet`` or ``predictions.tsv``).
    output_dir
        Directory to write per-chromosome feature parquets. Required.
    chromosomes
        Chromosomes to process. ``None`` = all available in ``input_dir``.
    fetch_missing_tracks
        When ``True``, download conservation bigWigs to the configured
        cache before running the feature workflow. Off by default; the
        library's ``remote_fallback`` handles cache misses at run time
        already.
    skip_steps, only_steps
        Control which steps run. Names: ``fetch_conservation``,
        ``generate_features``.
    resume
        Skip per-chromosome feature parquets that already exist.
    memory_limit_gb
        Passed to ``FeatureWorkflow`` — graceful abort if RSS exceeds.
    """
    profile_obj: FeatureProfile = load_profile(profile)

    output_dir = Path(output_dir).resolve()
    if not output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output_dir does not exist: {output_dir.parent}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load/create manifest. Refuse to overwrite an incompatible manifest.
    existing = FeatureManifest.try_load(output_dir)
    if existing is not None:
        if existing.build != build or existing.profile != profile_obj.name:
            raise ValueError(
                f"Existing manifest in {output_dir} is for build={existing.build}, "
                f"profile={existing.profile}; refusing to mix with build={build}, "
                f"profile={profile_obj.name}. Choose a different output_dir."
            )
        manifest = existing
    else:
        manifest = FeatureManifest(
            build=build,
            profile=profile_obj.name,
            modalities=list(profile_obj.modalities),
            chromosomes=list(chromosomes or []),
            base_predictions_dir=str(input_dir),
            output_dir=str(output_dir),
            tool_version=_get_tool_version(),
        )

    steps_to_run = _resolve_step_order(skip_steps, only_steps)

    # Conservation fetch is opt-in via fetch_missing_tracks.
    if "fetch_conservation" in steps_to_run and not fetch_missing_tracks:
        steps_to_run = [s for s in steps_to_run if s != "fetch_conservation"]

    result = PreparationResult(
        build=build, profile=profile_obj.name, output_dir=output_dir,
    )

    tracker = _start_tracker_if_requested(
        track=track,
        build=build,
        profile=profile_obj,
        output_dir=output_dir,
        chromosomes=chromosomes,
        steps_to_run=steps_to_run,
        project=tracking_project,
        tags=tracking_tags,
    )

    try:
        import time
        for step_name in steps_to_run:
            t0 = time.time()
            if step_name == "fetch_conservation":
                sr = step_fetch_conservation(
                    build=build, force=False, verbosity=verbosity,
                )
            elif step_name == "generate_features":
                sr = step_generate_features(
                    profile=profile_obj,
                    input_dir=Path(input_dir),
                    output_dir=output_dir,
                    chromosomes=chromosomes,
                    resume=resume,
                    memory_limit_gb=memory_limit_gb,
                    verbosity=verbosity,
                )
                # If the workflow succeeded, record per-chromosome artifacts.
                if sr.success:
                    processed = (sr.extras or {}).get("chromosomes_processed", []) or []
                    skipped = (sr.extras or {}).get("chromosomes_skipped", []) or []
                    record_features_artifacts(
                        manifest, output_dir,
                        processed_chromosomes=list(processed) + list(skipped),
                        hash_artifacts=hash_artifacts,
                    )
            else:
                logger.warning("Unknown step %r — skipping", step_name)
                continue

            result.steps.append(sr)
            if not sr.success and not sr.skipped:
                result.success = False

            if tracker is not None:
                tracker.log_metrics({
                    f"step/{step_name}/seconds": round(time.time() - t0, 3),
                    f"step/{step_name}/rows": sr.rows or 0,
                    f"step/{step_name}/skipped": int(sr.skipped),
                    f"step/{step_name}/success": int(bool(sr.success)),
                })

        # Always record tracks that are already cached, so the manifest stays
        # informative even when fetch_conservation wasn't run.
        cached = [t for t in list_tracks(build=build) if t.is_cached]
        record_tracks_used(manifest, cached)

        result.manifest_path = manifest.save(output_dir)
        if verbosity >= 1:
            logger.info("Manifest written to %s", result.manifest_path)

        if tracker is not None:
            tracker.log_metrics({
                "run/overall_success": int(bool(result.success)),
                "run/n_artifacts": len(manifest.artifacts),
                "run/n_tracks_used": len(manifest.tracks_used),
            })
            if result.manifest_path is not None:
                tracker.log_artifact(
                    result.manifest_path,
                    artifact_type="feature_manifest",
                )
    finally:
        if tracker is not None:
            tracker.finish()

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_step_order(
    skip_steps: Optional[List[str]],
    only_steps: Optional[List[str]],
) -> List[str]:
    if only_steps and skip_steps:
        raise ValueError("Use only_steps OR skip_steps, not both.")
    if only_steps:
        unknown = [s for s in only_steps if s not in DEFAULT_STEPS]
        if unknown:
            raise ValueError(
                f"Unknown steps: {unknown}. Known: {list(DEFAULT_STEPS)}"
            )
        return list(only_steps)
    skip = set(skip_steps or ())
    return [s for s in DEFAULT_STEPS if s not in skip]


def _start_tracker_if_requested(
    *,
    track: bool,
    build: str,
    profile: FeatureProfile,
    output_dir: Path,
    chromosomes: Optional[List[str]],
    steps_to_run: List[str],
    project: Optional[str],
    tags: Optional[List[str]],
):
    """Return a :class:`WandbTracker` (or ``None``) for this run."""
    if not track:
        return None
    from .._common.tracking import start_run

    config = {
        "build": build,
        "profile": profile.name,
        "modalities": list(profile.modalities),
        "base_model": profile.base_model,
        "output_dir": str(output_dir),
        "steps": list(steps_to_run),
    }
    if chromosomes is not None:
        config["chromosomes"] = list(chromosomes)
    return start_run(
        app="multimodal_features",
        run_kind="prepare",
        config=config,
        project=project,
        tags=tags,
    )


def _get_tool_version() -> str:
    try:
        from importlib.metadata import version

        return f"agentic-spliceai {version('agentic-spliceai')}"
    except Exception:  # noqa: BLE001
        return "agentic-spliceai (version unknown)"
