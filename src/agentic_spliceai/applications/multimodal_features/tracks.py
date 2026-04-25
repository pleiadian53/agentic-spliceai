"""External-track catalog + fetch/verify helpers.

The catalog is read from ``settings.yaml`` under ``external_tracks``.
Currently surfaced modalities:

- ``conservation`` (PhyloP, PhastCons from UCSC)
- ``epigenetic``   (ENCODE ChIP-seq — listed here, fetched via the modality's
  own registry since ENCODE URLs are accession-driven)

Only conservation has a simple URL-based fetcher here. Epigenetic
downloads are delegated to ``splice_engine.features.modalities.epigenetic``
which already has an auto-resolution path (``remote_fallback=True``).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TrackRecord:
    """A single external track entry from the catalog."""

    modality: str
    build: str
    name: str  # e.g. "phylop", "phastcons"
    url: Optional[str] = None
    filename: Optional[str] = None
    accession: Optional[str] = None  # for ENCODE tracks
    alignment: Optional[str] = None  # e.g. "100way"
    cached_path: Optional[Path] = None
    is_cached: bool = False
    extra: Dict[str, str] = field(default_factory=dict)


def list_tracks(
    *,
    build: Optional[str] = None,
    modality: Optional[str] = None,
) -> List[TrackRecord]:
    """Read the external-track catalog from ``settings.yaml``.

    Both filters are optional; omit them to enumerate all registered
    tracks. ``is_cached`` / ``cached_path`` are populated by probing the
    configured bigwig cache directory.
    """
    cfg = _load_settings()
    tracks_cfg = cfg.get("external_tracks", {}) or {}
    cache_dir = _resolve_cache_dir(cfg)

    out: List[TrackRecord] = []

    # Conservation: cleanly keyed by build -> mark -> {url, filename}
    for track_modality, per_build in tracks_cfg.items():
        if modality and track_modality != modality:
            continue
        if not isinstance(per_build, dict):
            continue
        for b, entries in per_build.items():
            if build and b != build:
                continue

            if track_modality == "conservation":
                _append_conservation_tracks(
                    out, b, entries, cache_dir=cache_dir,
                )
            elif track_modality == "epigenetic":
                _append_epigenetic_tracks(out, b, entries)
            else:
                # Unknown modality layout: surface as best-effort, no caching
                for name, payload in (entries or {}).items():
                    if not isinstance(payload, dict):
                        continue
                    out.append(TrackRecord(
                        modality=track_modality,
                        build=b,
                        name=str(name),
                        url=payload.get("url"),
                        filename=payload.get("filename"),
                        accession=payload.get("accession"),
                    ))

    return out


def fetch_conservation_tracks(
    *,
    build: str,
    dest_dir: Optional[Path] = None,
    force: bool = False,
    verbosity: int = 1,
) -> List[TrackRecord]:
    """Download PhyloP + PhastCons bigWigs to the configured cache.

    Simple URL fetch. Existing files are preserved unless ``force=True``.
    Returns the updated list of ``TrackRecord`` for the conservation modality.
    """
    cfg = _load_settings()
    cache_dir = Path(dest_dir or _resolve_cache_dir(cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)

    entries = cfg.get("external_tracks", {}).get("conservation", {}).get(build)
    if not entries:
        raise ValueError(
            f"No conservation tracks configured for build={build} in settings.yaml."
        )

    updated: List[TrackRecord] = []
    for name, payload in entries.items():
        url = payload.get("url")
        filename = payload.get("filename")
        if not url or not filename:
            logger.warning("Skipping %s/%s: missing url or filename", build, name)
            continue

        target = cache_dir / filename
        rec = TrackRecord(
            modality="conservation",
            build=build,
            name=name,
            url=url,
            filename=filename,
            alignment=payload.get("alignment"),
            cached_path=target,
            is_cached=target.exists(),
        )

        if target.exists() and not force:
            if verbosity >= 1:
                logger.info(
                    "conservation/%s: cached at %s (size=%d)",
                    name, target, target.stat().st_size,
                )
            rec.is_cached = True
            updated.append(rec)
            continue

        if verbosity >= 1:
            logger.info("conservation/%s: downloading %s -> %s", name, url, target)
        _download_url(url, target)
        rec.is_cached = target.exists()
        rec.cached_path = target
        updated.append(rec)

    return updated


# ---------------------------------------------------------------------------
# ENCODE accession-driven fetch (epigenetic + chromatin-accessibility marks)
# ---------------------------------------------------------------------------


# ENCODE file download URL pattern. Accessions are of the form
# ``ENCFF814IYI``; the canonical download path is:
#     https://www.encodeproject.org/files/<accession>/@@download/<accession>.<ext>
# For bigWig fold-change-over-control signal files we use ``.bigWig``.
ENCODE_DOWNLOAD_URL = (
    "https://www.encodeproject.org/files/{accession}/@@download/{accession}.bigWig"
)


def fetch_encode_tracks(
    *,
    build: str,
    modality: str = "epigenetic",
    cell_lines: Optional[List[str]] = None,
    marks: Optional[List[str]] = None,
    dest_dir: Optional[Path] = None,
    force: bool = False,
    verbosity: int = 1,
) -> List[TrackRecord]:
    """Download ENCODE bigWig signal files by accession.

    Resolves (cell_line × mark) accessions from ``settings.yaml`` and
    downloads each to ``<cache_dir>/<accession>.bigWig`` (or the
    ``filename`` specified in settings, if any). Existing files are
    preserved unless ``force=True``.

    Parameters
    ----------
    build
        Genome build (ENCODE tracks are GRCh38 only in our catalog).
    modality
        Currently only ``"epigenetic"`` is supported. ``"chrom_access"``
        may be added when its catalog lands in ``settings.yaml``.
    cell_lines
        Optional filter by cell line name (e.g., ``["K562", "HepG2"]``).
    marks
        Optional filter by histone mark (e.g., ``["h3k4me3"]``).

    Returns
    -------
    list of :class:`TrackRecord`
        One record per (cell_line × mark) attempt, with ``is_cached`` /
        ``cached_path`` filled in on success.

    Raises
    ------
    ValueError
        If ``modality`` is unsupported or no accessions are configured
        for ``build``.
    """
    if modality != "epigenetic":
        raise ValueError(
            f"fetch_encode_tracks: modality={modality!r} not supported yet. "
            f"Supported: 'epigenetic'."
        )

    cfg = _load_settings()
    cache_dir = Path(dest_dir or _resolve_cache_dir(cfg))
    cache_dir.mkdir(parents=True, exist_ok=True)

    per_build = cfg.get("external_tracks", {}).get(modality, {}).get(build)
    if not per_build:
        raise ValueError(
            f"No {modality} tracks configured for build={build} in settings.yaml."
        )

    configured_marks: List[str] = list(per_build.get("marks") or [])
    wanted_marks = set(marks) if marks else set(configured_marks)
    wanted_cells = set(cell_lines) if cell_lines else None

    updated: List[TrackRecord] = []
    for cell in per_build.get("cell_lines") or []:
        if not isinstance(cell, dict):
            continue
        cell_name = cell.get("name")
        if wanted_cells is not None and cell_name not in wanted_cells:
            continue
        for mark in configured_marks:
            if mark not in wanted_marks:
                continue
            payload = cell.get(mark)
            if not isinstance(payload, dict):
                continue
            accession = payload.get("accession")
            if not accession:
                logger.warning(
                    "Skipping %s/%s/%s: no accession in settings.yaml",
                    build, cell_name, mark,
                )
                continue

            url = ENCODE_DOWNLOAD_URL.format(accession=accession)
            filename = payload.get("filename") or f"{accession}.bigWig"
            target = cache_dir / filename

            rec = TrackRecord(
                modality=modality,
                build=build,
                name=f"{cell_name}/{mark}",
                url=url,
                filename=filename,
                accession=accession,
                cached_path=target,
                is_cached=target.exists(),
                extra={
                    "experiment": payload.get("experiment", ""),
                    "tissue": cell.get("tissue", ""),
                    "cell_line": cell_name or "",
                    "mark": mark,
                },
            )

            if target.exists() and not force:
                if verbosity >= 1:
                    logger.info(
                        "%s/%s: cached at %s (size=%d)",
                        rec.name, accession, target, target.stat().st_size,
                    )
                rec.is_cached = True
                updated.append(rec)
                continue

            if verbosity >= 1:
                logger.info(
                    "%s/%s: downloading %s -> %s",
                    rec.name, accession, url, target,
                )
            try:
                _download_url(url, target)
                rec.is_cached = target.exists()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to download %s (%s): %s",
                    rec.name, accession, exc,
                )
                rec.is_cached = False
            rec.cached_path = target
            updated.append(rec)

    return updated


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _append_conservation_tracks(
    out: List[TrackRecord],
    build: str,
    entries: Dict,
    *,
    cache_dir: Path,
) -> None:
    for name, payload in (entries or {}).items():
        if not isinstance(payload, dict):
            continue
        filename = payload.get("filename")
        cached_path = (cache_dir / filename) if filename else None
        is_cached = bool(cached_path and cached_path.exists())
        out.append(TrackRecord(
            modality="conservation",
            build=build,
            name=str(name),
            url=payload.get("url"),
            filename=filename,
            alignment=payload.get("alignment"),
            cached_path=cached_path,
            is_cached=is_cached,
        ))


def _append_epigenetic_tracks(
    out: List[TrackRecord],
    build: str,
    entries: Dict,
) -> None:
    """ENCODE panel: one record per (cell line × mark)."""
    marks = entries.get("marks") or []
    for cell in entries.get("cell_lines") or []:
        if not isinstance(cell, dict):
            continue
        cell_name = cell.get("name")
        for mark in marks:
            payload = cell.get(mark)
            if not isinstance(payload, dict):
                continue
            out.append(TrackRecord(
                modality="epigenetic",
                build=build,
                name=f"{cell_name}/{mark}",
                accession=payload.get("accession"),
                extra={
                    "experiment": payload.get("experiment", ""),
                    "tissue": cell.get("tissue", ""),
                },
            ))


def _load_settings() -> Dict:
    """Read settings.yaml directly (co-located with the config module)."""
    import yaml
    from agentic_spliceai.splice_engine import config as _cfg_pkg

    settings_path = Path(_cfg_pkg.__file__).parent / "settings.yaml"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"settings.yaml not found at {settings_path}"
        )
    with open(settings_path, "r") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_cache_dir(cfg: Dict) -> Path:
    """Return the configured bigwig cache directory."""
    cache_cfg = (cfg.get("cache") or {})
    raw = cache_cfg.get("bigwig_dir") or "data/cache/bigwig"
    p = Path(raw)
    if p.is_absolute():
        return p
    # Resolve relative to the repo root.
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return repo_root / p


def _download_url(url: str, target: Path) -> None:
    """Atomic URL download with a ``.part`` staging file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(staging, "wb") as out:
        shutil.copyfileobj(resp, out, length=4 * 1024 * 1024)
    staging.replace(target)
