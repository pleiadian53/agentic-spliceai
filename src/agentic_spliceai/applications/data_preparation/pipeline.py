"""Pipeline orchestrator for the data-preparation application.

``prepare_build`` is the single high-level entry point. It composes the
step functions from :mod:`steps`, records artifacts in an
:class:`IngestManifest`, and returns a :class:`PreparationResult`
summarising what ran, what was skipped, and what failed.

Production safety
-----------------

The pipeline never writes outside ``output_dir``. It accepts explicit
``gtf_path`` / ``fasta_path`` overrides — otherwise it resolves paths via
the resource manager for reading only. The caller is responsible for
choosing a safe ``output_dir`` (the CLI enforces this).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import IngestManifest, InputRecord
from .steps import (
    StepResult,
    record_step_artifact,
    step_chromosome_split,
    step_gene_features,
    step_splice_sites,
)

logger = logging.getLogger(__name__)


# The default step list. Callers can disable individual steps via
# ``skip_steps=[...]`` on prepare_build.
DEFAULT_STEPS = ("gene_features", "splice_sites", "chromosome_split")


def resolve_canonical_output_dir(
    *, build: str, annotation_source: str,
) -> Path:
    """Return the resource-manager-resolved data dir for a build.

    This is the *production* path where artifacts conventionally live
    (e.g., ``data/mane/GRCh38/``). The CLI only writes here when the user
    passes ``--inplace`` — otherwise ``--output-dir`` is required.

    Useful for completeness checks: ``status`` and ``prepare --inplace
    --dry-run`` can report what is missing at the canonical path without
    writing anything.
    """
    from agentic_spliceai.splice_engine.resources import get_genomic_registry

    if build == "GRCh38" and annotation_source == "mane":
        registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
    else:
        registry = get_genomic_registry(build=build)
    return Path(registry.data_dir)


@dataclass
class PreparationResult:
    """High-level summary of a preparation run."""

    build: str
    annotation_source: str
    output_dir: Path
    manifest_path: Optional[Path] = None
    steps: List[StepResult] = field(default_factory=list)
    success: bool = True

    def summary(self) -> str:
        lines = [
            f"Preparation — build={self.build}, source={self.annotation_source}",
            f"Output dir: {self.output_dir}",
            f"Manifest:   {self.manifest_path}",
            "Steps:",
        ]
        for step in self.steps:
            if step.skipped:
                status = f"skipped ({step.skipped_reason or 'n/a'})"
            elif step.success:
                rows = f", rows={step.rows}" if step.rows is not None else ""
                status = f"ok -> {step.artifact_path}{rows}"
            else:
                status = f"FAILED — {step.error}"
            lines.append(f"  [{step.step_name}] {status}")
        lines.append(f"Overall success: {self.success}")
        return "\n".join(lines)


def prepare_build(
    *,
    build: str,
    annotation_source: str,
    output_dir: Path,
    gtf_path: Optional[Path] = None,
    fasta_path: Optional[Path] = None,
    skip_steps: Optional[List[str]] = None,
    only_steps: Optional[List[str]] = None,
    chromosome_split_strategy: str = "spliceai",
    force: bool = False,
    hash_artifacts: bool = True,
    track: bool = False,
    tracking_project: Optional[str] = None,
    tracking_tags: Optional[List[str]] = None,
    verbosity: int = 1,
) -> PreparationResult:
    """Run the preparation pipeline for a build.

    Parameters
    ----------
    build
        Genomic build identifier (e.g., ``"GRCh38"``, ``"GRCh37"``).
    annotation_source
        Annotation source (e.g., ``"mane"``, ``"ensembl"``).
    output_dir
        Directory to write all artifacts. Required and non-empty: the
        caller must choose this explicitly — the pipeline will not write
        to resource-manager-resolved production paths implicitly.
    gtf_path, fasta_path
        Optional explicit paths to override resource-manager lookups.
        Used when the user provides a custom annotation/genome.
    skip_steps, only_steps
        Control which steps run. Step names: ``gene_features``,
        ``splice_sites``, ``chromosome_split``.
    chromosome_split_strategy
        ``"spliceai"`` (default, paper convention) or ``"balanced"``.
    force
        Re-run steps even when output files already exist.
    hash_artifacts
        If ``True``, compute SHA-256 for each artifact recorded in the
        manifest. Slight I/O cost; disable for large outputs.
    verbosity
        0 silent, 1 info, 2 debug.
    """
    output_dir = Path(output_dir).resolve()
    if not output_dir.parent.exists():
        raise FileNotFoundError(
            f"Parent of output_dir does not exist: {output_dir.parent}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build or load manifest. Reloading preserves prior run history when
    # re-preparing into the same dir.
    existing = IngestManifest.try_load(output_dir)
    if existing is not None:
        if existing.build != build or existing.annotation_source != annotation_source:
            raise ValueError(
                f"Existing manifest in {output_dir} is for "
                f"build={existing.build} source={existing.annotation_source}; "
                f"refusing to mix with build={build} source={annotation_source}. "
                f"Choose a different output_dir or clear it first."
            )
        manifest = existing
    else:
        manifest = IngestManifest(
            build=build,
            annotation_source=annotation_source,
            tool_version=_get_tool_version(),
        )

    # Record inputs. We hash them (once) so the manifest pins which exact
    # FASTA/GTF versions produced these artifacts.
    _record_inputs(
        manifest=manifest,
        build=build,
        annotation_source=annotation_source,
        gtf_path=gtf_path,
        fasta_path=fasta_path,
        hash_artifacts=hash_artifacts,
        verbosity=verbosity,
    )

    steps_to_run = _resolve_step_order(skip_steps, only_steps)

    result = PreparationResult(
        build=build, annotation_source=annotation_source,
        output_dir=output_dir,
    )

    # Optional W&B tracker. start_run returns None when wandb is
    # unavailable or no API key is set — logging calls below are no-ops
    # in that case.
    tracker = _start_tracker_if_requested(
        track=track,
        build=build,
        annotation_source=annotation_source,
        output_dir=output_dir,
        steps_to_run=steps_to_run,
        project=tracking_project,
        tags=tracking_tags,
    )

    try:
        import time
        for step_name in steps_to_run:
            t0 = time.time()
            if step_name == "gene_features":
                sr = step_gene_features(
                    build=build,
                    annotation_source=annotation_source,
                    output_dir=output_dir,
                    gtf_path=gtf_path,
                    fasta_path=fasta_path,
                    force=force,
                    verbosity=verbosity,
                )
            elif step_name == "splice_sites":
                sr = step_splice_sites(
                    build=build,
                    annotation_source=annotation_source,
                    output_dir=output_dir,
                    gtf_path=gtf_path,
                    force=force,
                    verbosity=verbosity,
                )
            elif step_name == "chromosome_split":
                sr = step_chromosome_split(
                    build=build,
                    output_dir=output_dir,
                    strategy=chromosome_split_strategy,
                    force=force,
                    verbosity=verbosity,
                )
            else:
                logger.warning("Unknown step %r — skipping", step_name)
                continue

            result.steps.append(sr)
            if sr.success:
                record_step_artifact(manifest, sr, hash_artifacts=hash_artifacts)
            else:
                result.success = False

            if tracker is not None:
                tracker.log_metrics({
                    f"step/{step_name}/seconds": round(time.time() - t0, 3),
                    f"step/{step_name}/rows": sr.rows or 0,
                    f"step/{step_name}/skipped": int(sr.skipped),
                    f"step/{step_name}/success": int(bool(sr.success)),
                })

        result.manifest_path = manifest.save(output_dir)

        if tracker is not None:
            tracker.log_metrics({
                "run/overall_success": int(bool(result.success)),
                "run/n_artifacts": len(manifest.artifacts),
            })
            if result.manifest_path is not None:
                tracker.log_artifact(
                    result.manifest_path,
                    artifact_type="ingest_manifest",
                )
    finally:
        if tracker is not None:
            tracker.finish()

    if verbosity >= 1:
        logger.info("Manifest written to %s", result.manifest_path)

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
                f"Unknown steps in only_steps: {unknown}. "
                f"Known: {list(DEFAULT_STEPS)}"
            )
        return list(only_steps)
    skip = set(skip_steps or ())
    return [s for s in DEFAULT_STEPS if s not in skip]


def _record_inputs(
    *,
    manifest: IngestManifest,
    build: str,
    annotation_source: str,
    gtf_path: Optional[Path],
    fasta_path: Optional[Path],
    hash_artifacts: bool,
    verbosity: int,
) -> None:
    """Attach input-file records (FASTA + GTF) to the manifest."""
    # If not provided, resolve via the resource manager (read-only).
    if gtf_path is None or fasta_path is None:
        try:
            from agentic_spliceai.splice_engine.resources import (
                get_genomic_registry,
            )

            if build == "GRCh38" and annotation_source == "mane":
                registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
            else:
                registry = get_genomic_registry(build=build)
            if gtf_path is None:
                gtf_path = registry.get_gtf_path(validate=True)
            if fasta_path is None:
                fasta_path = registry.get_fasta_path(validate=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not resolve input paths from registry: %s. Proceeding "
                "without input-record hashes (step failures may follow).", exc,
            )

    if gtf_path is not None:
        try:
            manifest.set_input(
                "gtf",
                InputRecord.from_path(Path(gtf_path), hash_file=hash_artifacts),
            )
        except FileNotFoundError as exc:
            logger.warning("GTF not found: %s", exc)
    if fasta_path is not None:
        try:
            manifest.set_input(
                "fasta",
                InputRecord.from_path(Path(fasta_path), hash_file=hash_artifacts),
            )
        except FileNotFoundError as exc:
            logger.warning("FASTA not found: %s", exc)


def _start_tracker_if_requested(
    *,
    track: bool,
    build: str,
    annotation_source: str,
    output_dir: Path,
    steps_to_run: List[str],
    project: Optional[str],
    tags: Optional[List[str]],
):
    """Return a :class:`WandbTracker` (or ``None``) for this run."""
    if not track:
        return None
    from .._common.tracking import start_run

    return start_run(
        app="data_preparation",
        run_kind="prepare",
        config={
            "build": build,
            "annotation_source": annotation_source,
            "output_dir": str(output_dir),
            "steps": list(steps_to_run),
        },
        project=project,
        tags=tags,
    )


def _get_tool_version() -> str:
    try:
        from importlib.metadata import version

        return f"agentic-spliceai {version('agentic-spliceai')}"
    except Exception:  # noqa: BLE001
        return "agentic-spliceai (version unknown)"
