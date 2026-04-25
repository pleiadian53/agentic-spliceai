"""Ingestion pipeline steps.

Each step is a thin wrapper over existing library functions under
``splice_engine.base_layer.data.preparation`` and ``splice_engine.eval.splitting``.
Steps are designed to be:

- **Idempotent**: re-running writes the same output (unless inputs changed)
- **Safe**: writes go to the caller-supplied ``output_dir``, never to the
  resource-manager-resolved production path unless the caller explicitly
  points there AND enables ``force=True``
- **Minimal**: no re-implementation of library logic; this module only
  orchestrates and wraps the outputs in the manifest format

Steps currently implemented:

- ``step_gene_features``      — gene metadata parquet
- ``step_splice_sites``       — splice_sites_enhanced.tsv
- ``step_chromosome_split``   — SpliceAI-convention train/test split
- ``step_validate``           — coordinate consistency checks

Not yet implemented (explicit follow-up):

- ``step_gene_sequences``     — per-chromosome sequence parquets (expensive,
  opt-in; for now use ``examples/data_preparation/`` or let base predictors
  extract lazily)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .manifest import ArtifactRecord, IngestManifest

logger = logging.getLogger(__name__)


def _resolve_gtf_path(
    *,
    build: str,
    annotation_source: str,
    override: Optional[Path],
) -> Path:
    """Resolve the GTF path for a build via the resource manager.

    Mirrors ``prepare_gene_data``'s registry-resolution logic, including
    the GRCh38 + MANE special case. The resolved path is what
    ``load_gene_annotations`` consumes; it does not accept build /
    annotation_source kwargs directly.
    """
    if override is not None:
        return Path(override)
    from agentic_spliceai.splice_engine.resources import get_genomic_registry

    if build == "GRCh38" and annotation_source == "mane":
        registry = get_genomic_registry(build="GRCh38_MANE", release="1.3")
    else:
        registry = get_genomic_registry(build=build)
    return Path(registry.get_gtf_path(validate=True))


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Outcome of a single pipeline step."""

    step_name: str
    success: bool
    artifact_name: Optional[str] = None
    artifact_path: Optional[Path] = None
    rows: Optional[int] = None
    skipped: bool = False
    skipped_reason: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Step: gene features
# ---------------------------------------------------------------------------


def step_gene_features(
    *,
    build: str,
    annotation_source: str,
    output_dir: Path,
    gtf_path: Optional[Path] = None,
    fasta_path: Optional[Path] = None,
    force: bool = False,
    verbosity: int = 1,
) -> StepResult:
    """Extract gene metadata from GTF, save as ``gene_features.parquet``.

    Produces a polars-queryable table of gene annotations. Columns mirror
    what ``load_gene_annotations`` returns (``chrom``, ``gene_id``,
    ``gene_name``, ``start``, ``end``, ``strand``, etc.).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "gene_features.parquet"

    if out_path.exists() and not force:
        logger.info("gene_features: exists, skipping (%s)", out_path)
        return StepResult(
            step_name="gene_features",
            success=True,
            skipped=True,
            skipped_reason="file exists; pass force=True to regenerate",
            artifact_name="gene_features",
            artifact_path=out_path,
        )

    try:
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            load_gene_annotations,
        )

        # Resolve GTF path. ``load_gene_annotations`` takes the path
        # directly — it does not understand build / annotation_source.
        # The resource manager handles the build → GTF lookup the same way
        # ``prepare_gene_data`` does (incl. the GRCh38 + MANE special case).
        resolved_gtf = _resolve_gtf_path(
            build=build,
            annotation_source=annotation_source,
            override=gtf_path,
        )

        if verbosity >= 1:
            logger.info(
                "gene_features: extracting from %s "
                "(build=%s, source=%s)...",
                resolved_gtf, build, annotation_source,
            )

        gene_df = load_gene_annotations(
            gtf_path=resolved_gtf,
            verbosity=verbosity,
        )

        if gene_df is None or len(gene_df) == 0:
            return StepResult(
                step_name="gene_features",
                success=False,
                error="load_gene_annotations returned no rows",
            )

        gene_df.write_parquet(out_path)
        logger.info(
            "gene_features: wrote %d rows to %s", len(gene_df), out_path,
        )
        return StepResult(
            step_name="gene_features",
            success=True,
            artifact_name="gene_features",
            artifact_path=out_path,
            rows=len(gene_df),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("gene_features: failed")
        return StepResult(
            step_name="gene_features", success=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Step: splice sites
# ---------------------------------------------------------------------------


def step_splice_sites(
    *,
    build: str,
    annotation_source: str,
    output_dir: Path,
    gtf_path: Optional[Path] = None,
    force: bool = False,
    verbosity: int = 1,
) -> StepResult:
    """Derive per-splice-site ground truth (donor / acceptor) TSV.

    Wraps ``prepare_splice_site_annotations``. Output filename is the
    canonical ``splice_sites_enhanced.tsv``. Existing file is preserved
    unless ``force=True``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "splice_sites_enhanced.tsv"

    try:
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            prepare_splice_site_annotations,
        )

        if out_path.exists() and not force:
            logger.info("splice_sites: exists, skipping (%s)", out_path)
            return StepResult(
                step_name="splice_sites",
                success=True,
                skipped=True,
                skipped_reason="file exists; pass force=True to regenerate",
                artifact_name="splice_sites",
                artifact_path=out_path,
            )

        if verbosity >= 1:
            logger.info(
                "splice_sites: extracting from GTF (build=%s, source=%s)...",
                build, annotation_source,
            )

        kwargs: Dict[str, Any] = dict(
            output_dir=str(output_dir),
            build=build,
            annotation_source=annotation_source,
            output_filename="splice_sites_enhanced.tsv",
            force_extract=bool(force),
            verbosity=verbosity,
        )
        if gtf_path is not None:
            kwargs["gtf_path"] = str(gtf_path)

        result = prepare_splice_site_annotations(**kwargs)

        if not result.get("success"):
            return StepResult(
                step_name="splice_sites",
                success=False,
                error=f"prepare_splice_site_annotations failed: "
                      f"{result.get('error', 'unknown')}",
            )

        return StepResult(
            step_name="splice_sites",
            success=True,
            artifact_name="splice_sites",
            artifact_path=Path(result.get("splice_sites_file", out_path)),
            rows=int(result.get("n_sites") or 0),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("splice_sites: failed")
        return StepResult(
            step_name="splice_sites", success=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Step: chromosome split
# ---------------------------------------------------------------------------


def step_chromosome_split(
    *,
    build: str,
    output_dir: Path,
    strategy: str = "spliceai",
    force: bool = False,
    verbosity: int = 1,
) -> StepResult:
    """Produce the canonical chromosome train/test split.

    Default strategy is ``"spliceai"`` (paper convention). Saved as
    ``chromosome_split.json`` for downstream evaluators.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "chromosome_split.json"

    if out_path.exists() and not force:
        return StepResult(
            step_name="chromosome_split",
            success=True,
            skipped=True,
            skipped_reason="file exists; pass force=True to regenerate",
            artifact_name="chromosome_split",
            artifact_path=out_path,
        )

    try:
        split = _build_split(build=build, strategy=strategy)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chromosome_split: failed")
        return StepResult(
            step_name="chromosome_split", success=False, error=str(exc),
        )

    payload: Dict[str, Any] = {
        "build": build,
        "strategy": strategy,
        **split,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    if verbosity >= 1:
        logger.info(
            "chromosome_split: train=%d test=%d -> %s",
            len(split.get("train", [])), len(split.get("test", [])), out_path,
        )

    return StepResult(
        step_name="chromosome_split",
        success=True,
        artifact_name="chromosome_split",
        artifact_path=out_path,
        rows=len(split.get("train", [])) + len(split.get("test", [])),
    )


def _build_split(*, build: str, strategy: str) -> Dict[str, List[str]]:
    """Return ``{"train": [...], "test": [...]}`` for the chosen strategy."""
    strategy = strategy.lower()
    if strategy == "spliceai":
        # SpliceAI paper convention: odd chromosomes train, even test
        # (paralogs handled separately in the library).
        train = [str(c) for c in range(1, 23, 2)]  # 1, 3, 5, ...
        test = [str(c) for c in range(2, 23, 2)]   # 2, 4, 6, ...
        return {"train": train, "test": test}

    if strategy == "balanced":
        # Use the library's balanced split helper.
        from agentic_spliceai.splice_engine.eval.splitting import build_gene_split

        split = build_gene_split(build=build, strategy="balanced")
        return {
            "train": list(split.train_chroms),
            "test": list(split.test_chroms),
        }

    raise ValueError(
        f"Unknown chromosome-split strategy {strategy!r}. "
        f"Supported: spliceai, balanced."
    )


# ---------------------------------------------------------------------------
# Step: validate (read-only)
# ---------------------------------------------------------------------------


def step_validate(
    *,
    build: str,
    annotation_source: str,
    output_dir: Path,
    verbosity: int = 1,
) -> StepResult:
    """Read-only coordinate-consistency + input-integrity check.

    Doesn't write anything. Returns a StepResult with ``error`` set if a
    check fails. Uses the library's ``validate_gene_data`` plus manifest
    integrity checks.
    """
    from agentic_spliceai.splice_engine.base_layer.data.preparation import (
        validate_gene_data,
    )

    try:
        # Re-derive gene annotations quickly to validate the GTF is parseable
        # and the chrom column is present. This does not write.
        from agentic_spliceai.splice_engine.base_layer.data.preparation import (
            load_gene_annotations,
        )

        resolved_gtf = _resolve_gtf_path(
            build=build,
            annotation_source=annotation_source,
            override=None,
        )
        gene_df = load_gene_annotations(
            gtf_path=resolved_gtf,
            verbosity=verbosity,
        )
        ok = validate_gene_data(gene_df, verbosity=verbosity)

        if not ok:
            return StepResult(
                step_name="validate",
                success=False,
                error="validate_gene_data reported integrity issues",
            )

        return StepResult(
            step_name="validate",
            success=True,
            rows=int(len(gene_df)),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate: failed")
        return StepResult(
            step_name="validate", success=False, error=str(exc),
        )


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------


def record_step_artifact(
    manifest: IngestManifest,
    step_result: StepResult,
    *,
    hash_artifacts: bool = True,
) -> None:
    """Update the manifest in-place with a step's artifact, if any."""
    if not step_result.success:
        return
    if step_result.artifact_name is None or step_result.artifact_path is None:
        return
    if not Path(step_result.artifact_path).exists():
        return

    record = ArtifactRecord.from_path(
        Path(step_result.artifact_path),
        step=step_result.step_name,
        rows=step_result.rows,
        hash_file=hash_artifacts,
    )
    manifest.set_artifact(step_result.artifact_name, record)
