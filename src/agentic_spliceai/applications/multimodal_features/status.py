"""Read-only status query for a feature-prep output directory.

Mirrors :mod:`agentic_spliceai.applications.data_preparation.status`. The
future ingestion UI polls this to decide whether the meta-layer can be
run yet.

What counts as "ready"
----------------------

1. A :class:`FeatureManifest` is present, OR the filesystem has
   ``analysis_sequences_*.parquet`` files (degraded, no manifest).
2. Every requested chromosome has a corresponding
   ``analysis_sequences_{chrom}.parquet``.
3. Staleness: no hash comparison on feature parquets yet — a stale
   check would require manifest + base-layer-prediction hashes, deferred.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import MANIFEST_FILENAME, FeatureManifest

logger = logging.getLogger(__name__)


@dataclass
class ChromosomeArtifactStatus:
    """Per-chromosome readiness."""

    chromosome: str
    path: Optional[Path] = None
    exists: bool = False
    size_bytes: Optional[int] = None


@dataclass
class FeaturePrepStatus:
    """Readiness report for a feature output directory."""

    output_dir: Path
    manifest_present: bool
    build: Optional[str] = None
    profile: Optional[str] = None
    modalities: List[str] = field(default_factory=list)
    manifest_updated_at: Optional[str] = None
    expected_chromosomes: List[str] = field(default_factory=list)
    chromosomes: Dict[str, ChromosomeArtifactStatus] = field(default_factory=dict)
    tracks_cached: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    ready: bool = False

    def summary(self) -> str:
        lines = [
            f"Status for: {self.output_dir}",
            f"Manifest:   {'present' if self.manifest_present else 'absent'}",
        ]
        if self.build or self.profile:
            lines.append(
                f"Build:      {self.build}  (profile={self.profile}, "
                f"updated {self.manifest_updated_at})"
            )
        if self.modalities:
            lines.append(f"Modalities: {self.modalities}")
        if self.chromosomes:
            lines.append("Chromosomes:")
            for chrom, art in sorted(self.chromosomes.items(), key=_chrom_sort_key):
                tag = "ok" if art.exists else "missing"
                size = (
                    f" ({_format_size(art.size_bytes)})"
                    if art.exists and art.size_bytes is not None else ""
                )
                lines.append(
                    f"  [{chrom:>5}] {tag}{size}  {art.path or ''}"
                )
        if self.tracks_cached:
            lines.append(f"Tracks cached: {self.tracks_cached}")
        lines.append(f"Ready: {self.ready}")
        return "\n".join(lines)


def get_status(
    output_dir: Path,
    *,
    expected_chromosomes: Optional[List[str]] = None,
) -> FeaturePrepStatus:
    """Inspect ``output_dir`` and return a :class:`FeaturePrepStatus`.

    Read-only. Safe to call repeatedly and from a web UI. The
    ``expected_chromosomes`` hint lets callers set what "ready" means
    (usually ``[1..22, X, Y]``); when omitted, we take the intersection
    of the manifest's recorded chromosomes and what is on disk.
    """
    output_dir = Path(output_dir).resolve()

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_present = manifest_path.exists()

    if not manifest_present:
        return _filesystem_only_status(
            output_dir, expected_chromosomes=expected_chromosomes,
        )

    manifest = FeatureManifest.load(output_dir)
    status = FeaturePrepStatus(
        output_dir=output_dir,
        manifest_present=True,
        build=manifest.build,
        profile=manifest.profile,
        modalities=list(manifest.modalities),
        manifest_updated_at=manifest.updated_at,
        tracks_cached=[f"{t.modality}/{t.name}" for t in manifest.tracks_used],
    )

    status.expected_chromosomes = list(
        expected_chromosomes
        or manifest.chromosomes
        or _detect_chromosomes_from_disk(output_dir)
    )
    if not status.expected_chromosomes:
        status.expected_chromosomes = _detect_chromosomes_from_disk(output_dir)

    for chrom in status.expected_chromosomes:
        art = _resolve_chromosome_artifact(output_dir, chrom, manifest)
        status.chromosomes[chrom] = art
        if not art.exists:
            status.missing.append(chrom)

    status.ready = bool(status.chromosomes) and not status.missing
    return status


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _filesystem_only_status(
    output_dir: Path,
    *,
    expected_chromosomes: Optional[List[str]] = None,
) -> FeaturePrepStatus:
    status = FeaturePrepStatus(output_dir=output_dir, manifest_present=False)
    detected = _detect_chromosomes_from_disk(output_dir)
    expected = list(expected_chromosomes or detected)
    status.expected_chromosomes = expected
    for chrom in expected:
        art = _resolve_chromosome_artifact(output_dir, chrom, manifest=None)
        status.chromosomes[chrom] = art
        if not art.exists:
            status.missing.append(chrom)
    status.ready = bool(status.chromosomes) and not status.missing
    return status


def _detect_chromosomes_from_disk(output_dir: Path) -> List[str]:
    """Extract chromosome identifiers from ``analysis_sequences_*.parquet``."""
    out: List[str] = []
    if not output_dir.exists():
        return out
    for fp in output_dir.glob("analysis_sequences_*.parquet"):
        stem = fp.stem  # analysis_sequences_chr22
        if stem.startswith("analysis_sequences_"):
            chrom = stem[len("analysis_sequences_"):]
            out.append(chrom)
    return sorted(out, key=_chrom_key)


def _resolve_chromosome_artifact(
    output_dir: Path, chrom: str, manifest: Optional[FeatureManifest],
) -> ChromosomeArtifactStatus:
    # Manifest record wins when present.
    if manifest is not None:
        rec = manifest.artifacts.get(chrom)
        if rec is not None:
            path = Path(rec.path)
            return ChromosomeArtifactStatus(
                chromosome=chrom, path=path,
                exists=path.exists(),
                size_bytes=(path.stat().st_size if path.exists() else None),
            )

    # Fallback to filesystem probe (cover both ``chr22`` and ``22`` forms).
    candidates = [
        output_dir / f"analysis_sequences_{chrom}.parquet",
        output_dir / f"analysis_sequences_chr{chrom.removeprefix('chr')}.parquet"
        if not chrom.startswith("chr") else
        output_dir / f"analysis_sequences_{chrom.removeprefix('chr')}.parquet",
    ]
    for cand in candidates:
        if cand.exists():
            return ChromosomeArtifactStatus(
                chromosome=chrom, path=cand, exists=True,
                size_bytes=cand.stat().st_size,
            )
    return ChromosomeArtifactStatus(chromosome=chrom, path=None, exists=False)


def _chrom_sort_key(item):
    return _chrom_key(item[0])


def _chrom_key(s: str):
    bare = s.removeprefix("chr")
    # Always return a 3-tuple so numeric and non-numeric chromosomes compare.
    if bare.isdigit():
        return (0, int(bare), s)
    return (1, 0, s)


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f}PB"
