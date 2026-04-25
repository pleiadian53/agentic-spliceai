"""Ingestion-layer REST endpoints for the Bioinformatics Lab UI.

Read-only endpoints wrapping the application-level status APIs:

- ``data_preparation.get_status()``   — gene_features / splice_sites / chromosome_split readiness
- ``multimodal_features.get_status()`` — per-chromosome feature-parquet readiness
- ``multimodal_features.list_profiles()`` — feature-profile catalog
- ``multimodal_features.list_tracks()``   — external-track catalog

Phase D3 of the deployment plan. These endpoints let the Lab UI render
"ready to run base layer?" / "ready to train meta layer?" signals
without invoking the Python API directly from the browser.

Safety
------

All endpoints are read-only. Writes (``prepare`` subcommands) live on the
CLI only; the service deliberately does not expose them. A future
async-job endpoint would be a separate router.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ingest", tags=["ingest"])


# ---------------------------------------------------------------------------
# Health / routing
# ---------------------------------------------------------------------------


@router.get("/health")
async def ingest_health() -> dict:
    """Liveness check for the ingestion endpoints."""
    return {"status": "ok", "service": "ingest_api", "version": 1}


# ---------------------------------------------------------------------------
# Data preparation (FASTA + GTF → gene_features, splice_sites, ...)
# ---------------------------------------------------------------------------


@router.get("/data-prep/builds")
async def data_prep_list_builds() -> dict:
    """List configured base models / builds from settings.yaml.

    Mirrors ``agentic-spliceai-ingest list-builds``.
    """
    try:
        from agentic_spliceai.splice_engine.resources.model_resources import (
            get_model_info,
            list_available_models,
        )

        names = list_available_models()
        return {
            "builds": [
                {"name": n, **(get_model_info(n) or {})} for n in names
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("data_prep/builds failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/data-prep/status")
async def data_prep_status(
    build: Optional[str] = Query(None, description="Genome build (e.g., GRCh38)"),
    annotation_source: Optional[str] = Query(
        None, description="Annotation source (e.g., mane, ensembl)",
    ),
    output_dir: Optional[str] = Query(
        None,
        description=(
            "Explicit output directory to inspect. Mutually exclusive with "
            "build/annotation_source (which resolves the canonical path)."
        ),
    ),
) -> dict:
    """Read-only readiness for a data-preparation output directory.

    Two modes, mirroring the CLI:

    - **Canonical path**: pass ``build`` + ``annotation_source`` → resolves
      to ``data/<source>/<build>/`` via the resource manager.
    - **Explicit path**: pass ``output_dir`` to inspect any directory.
    """
    try:
        from agentic_spliceai.applications.data_preparation import (
            get_status,
            resolve_canonical_output_dir,
        )

        target = _resolve_target(
            build=build,
            annotation_source=annotation_source,
            output_dir=output_dir,
            canonical_resolver=lambda: resolve_canonical_output_dir(
                build=build, annotation_source=annotation_source,  # type: ignore[arg-type]
            ),
        )
        status = get_status(target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("data_prep/status failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "output_dir": str(status.output_dir),
        "manifest_present": status.manifest_present,
        "build": status.build,
        "annotation_source": status.annotation_source,
        "manifest_updated_at": status.manifest_updated_at,
        "ready": status.ready,
        "missing": status.missing,
        "stale": status.stale,
        "artifacts": {
            name: {
                "path": str(a.path) if a.path else None,
                "exists": a.exists,
                "stale": a.stale,
                "stale_reason": a.stale_reason,
            }
            for name, a in status.artifacts.items()
        },
    }


# ---------------------------------------------------------------------------
# Multimodal features (modalities → per-chromosome feature parquets)
# ---------------------------------------------------------------------------


@router.get("/features/profiles")
async def features_list_profiles() -> dict:
    """List feature profiles under examples/features/configs/."""
    try:
        from agentic_spliceai.applications.multimodal_features import (
            list_profiles,
        )

        profiles = list_profiles()
        return {
            "profiles": [
                {
                    "name": p.name,
                    "path": str(p.path),
                    "modalities": list(p.modalities),
                    "base_model": p.base_model,
                    "n_modalities": p.n_modalities,
                }
                for p in profiles
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("features/profiles failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/features/tracks")
async def features_list_tracks(
    build: Optional[str] = Query(None),
    modality: Optional[str] = Query(None),
) -> dict:
    """List registered external tracks (conservation, epigenetic, ...)."""
    try:
        from agentic_spliceai.applications.multimodal_features import list_tracks

        tracks = list_tracks(build=build, modality=modality)
        return {
            "tracks": [
                {
                    "modality": t.modality,
                    "build": t.build,
                    "name": t.name,
                    "url": t.url,
                    "accession": t.accession,
                    "filename": t.filename,
                    "alignment": t.alignment,
                    "is_cached": t.is_cached,
                    "cached_path": str(t.cached_path) if t.cached_path else None,
                    "extra": dict(t.extra) if t.extra else {},
                }
                for t in tracks
            ],
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("features/tracks failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/features/status")
async def features_status(
    build: Optional[str] = Query(None),
    annotation_source: Optional[str] = Query("mane"),
    base_model: Optional[str] = Query("openspliceai"),
    chromosomes: Optional[str] = Query(
        None,
        description=(
            "Comma-separated expected chromosomes (e.g., '1,2,21,22,X,Y'). "
            "Drives the missing-list; when omitted, takes what's on disk."
        ),
    ),
    output_dir: Optional[str] = Query(None),
) -> dict:
    """Read-only per-chromosome readiness for feature parquets."""
    try:
        from agentic_spliceai.applications.multimodal_features import get_status
        from agentic_spliceai.applications.multimodal_features.pipeline import (
            resolve_canonical_features_dir,
        )

        expected = [c.strip() for c in chromosomes.split(",")] if chromosomes else None

        target = _resolve_target(
            build=build,
            annotation_source=annotation_source,
            output_dir=output_dir,
            canonical_resolver=lambda: resolve_canonical_features_dir(
                build=build,  # type: ignore[arg-type]
                annotation_source=annotation_source or "mane",
                base_model=base_model or "openspliceai",
            ),
        )
        status = get_status(target, expected_chromosomes=expected)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.exception("features/status failed")
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "output_dir": str(status.output_dir),
        "manifest_present": status.manifest_present,
        "build": status.build,
        "profile": status.profile,
        "modalities": list(status.modalities),
        "manifest_updated_at": status.manifest_updated_at,
        "expected_chromosomes": list(status.expected_chromosomes),
        "tracks_cached": list(status.tracks_cached),
        "ready": status.ready,
        "missing": list(status.missing),
        "chromosomes": {
            c: {
                "exists": a.exists,
                "path": str(a.path) if a.path else None,
                "size_bytes": a.size_bytes,
            }
            for c, a in status.chromosomes.items()
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_target(
    *,
    build: Optional[str],
    annotation_source: Optional[str],
    output_dir: Optional[str],
    canonical_resolver,
) -> Path:
    """Pick canonical-path or explicit output_dir; error on ambiguity."""
    if output_dir is not None:
        if build is not None or annotation_source is not None:
            raise ValueError(
                "Pass either `output_dir` OR `build` + `annotation_source`, "
                "not both."
            )
        return Path(output_dir)

    if not build:
        raise ValueError(
            "Must pass either `output_dir` or `build` + `annotation_source`."
        )

    try:
        return Path(canonical_resolver())
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"Could not resolve canonical dir for build={build!r}, "
            f"annotation_source={annotation_source!r}: {exc}"
        )
