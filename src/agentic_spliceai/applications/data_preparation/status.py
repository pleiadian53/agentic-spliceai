"""Read-only status query over a prepared output directory.

:class:`DataPrepStatus` is what the future ingestion UI (Phase D3 in the
deployment plan) will poll. It answers:

- Is there a manifest at all?
- Which expected artifacts exist, which are missing?
- Are existing artifacts stale relative to the inputs they were derived from?

Staleness is a hash comparison: if the current FASTA/GTF differ from the
hashes recorded in the manifest, artifacts are flagged stale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .manifest import IngestManifest, MANIFEST_FILENAME

logger = logging.getLogger(__name__)


# Artifact name -> expected relative filename(s) inside output_dir. If an
# artifact has alternatives (e.g., .tsv vs .parquet), list the preferred
# first.
EXPECTED_ARTIFACTS: Dict[str, List[str]] = {
    "gene_features": ["gene_features.parquet", "gene_features.tsv"],
    "splice_sites": ["splice_sites_enhanced.tsv", "splice_sites_enhanced.parquet"],
    "chromosome_split": ["chromosome_split.json"],
}


@dataclass
class ArtifactStatus:
    """Per-artifact readiness record."""

    name: str
    path: Optional[Path] = None
    exists: bool = False
    stale: bool = False
    stale_reason: Optional[str] = None


@dataclass
class DataPrepStatus:
    """Readiness report for a prepared output directory."""

    output_dir: Path
    manifest_present: bool
    build: Optional[str] = None
    annotation_source: Optional[str] = None
    manifest_updated_at: Optional[str] = None
    artifacts: Dict[str, ArtifactStatus] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    stale: List[str] = field(default_factory=list)
    ready: bool = False

    def summary(self) -> str:
        lines = [
            f"Status for: {self.output_dir}",
            f"Manifest:   {'present' if self.manifest_present else 'absent'}",
        ]
        if self.build:
            lines.append(
                f"Build:      {self.build} (source={self.annotation_source}, "
                f"updated {self.manifest_updated_at})"
            )
        if self.artifacts:
            lines.append("Artifacts:")
            for name, art in self.artifacts.items():
                if art.exists and not art.stale:
                    tag = "ok"
                elif art.exists and art.stale:
                    tag = f"stale — {art.stale_reason}"
                else:
                    tag = "missing"
                lines.append(f"  [{name}] {tag}  {art.path or ''}")
        lines.append(f"Ready: {self.ready}")
        return "\n".join(lines)


def get_status(output_dir: Path) -> DataPrepStatus:
    """Inspect ``output_dir`` and return a :class:`DataPrepStatus`.

    Does not write anything. Safe to call repeatedly and from a web UI.
    """
    output_dir = Path(output_dir).resolve()

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_present = manifest_path.exists()

    if not manifest_present:
        # Partial state — try to report what we can infer from the
        # filesystem alone, with no manifest.
        return _filesystem_only_status(output_dir)

    manifest = IngestManifest.load(output_dir)
    status = DataPrepStatus(
        output_dir=output_dir,
        manifest_present=True,
        build=manifest.build,
        annotation_source=manifest.annotation_source,
        manifest_updated_at=manifest.updated_at,
    )

    current_input_hashes = _rehash_current_inputs(manifest)

    for name, filenames in EXPECTED_ARTIFACTS.items():
        art = _resolve_artifact(output_dir, name, filenames, manifest)
        if art.exists and current_input_hashes is not None:
            _mark_stale_if_input_changed(
                art=art,
                manifest=manifest,
                current_input_hashes=current_input_hashes,
            )
        status.artifacts[name] = art
        if not art.exists:
            status.missing.append(name)
        elif art.stale:
            status.stale.append(name)

    status.ready = not status.missing and not status.stale
    return status


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _filesystem_only_status(output_dir: Path) -> DataPrepStatus:
    """Best-effort status when no manifest exists."""
    status = DataPrepStatus(output_dir=output_dir, manifest_present=False)
    for name, filenames in EXPECTED_ARTIFACTS.items():
        found = None
        for fname in filenames:
            candidate = output_dir / fname
            if candidate.exists():
                found = candidate
                break
        art = ArtifactStatus(name=name, path=found, exists=bool(found))
        status.artifacts[name] = art
        if not art.exists:
            status.missing.append(name)
    status.ready = not status.missing and output_dir.exists()
    return status


def _resolve_artifact(
    output_dir: Path,
    name: str,
    filenames: List[str],
    manifest: IngestManifest,
) -> ArtifactStatus:
    """Resolve a single expected artifact against the manifest + filesystem."""
    # Manifest record wins when present.
    record = manifest.artifacts.get(name)
    if record is not None:
        path = Path(record.path)
        return ArtifactStatus(name=name, path=path, exists=path.exists())

    # Fall back to filesystem scan.
    for fname in filenames:
        candidate = output_dir / fname
        if candidate.exists():
            return ArtifactStatus(name=name, path=candidate, exists=True)
    return ArtifactStatus(name=name, path=None, exists=False)


def _rehash_current_inputs(manifest: IngestManifest) -> Optional[Dict[str, str]]:
    """Re-hash current FASTA/GTF to compare with manifest.

    Returns ``None`` on any IO error — we report staleness as *unknown*
    rather than raising, so ``get_status`` stays safe to call.
    """
    from .manifest import _sha256_file

    current: Dict[str, str] = {}
    for name, record in manifest.inputs.items():
        if not record.path:
            continue
        p = Path(record.path)
        if not p.exists():
            logger.debug("Input %s no longer at %s", name, p)
            return None
        try:
            current[name] = _sha256_file(p)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to hash %s: %s", p, exc)
            return None
    return current


def _mark_stale_if_input_changed(
    *,
    art: ArtifactStatus,
    manifest: IngestManifest,
    current_input_hashes: Dict[str, str],
) -> None:
    """Flag an artifact as stale if any input hash has drifted."""
    for name, record in manifest.inputs.items():
        current = current_input_hashes.get(name)
        if record.sha256 and current and record.sha256 != current:
            art.stale = True
            art.stale_reason = f"input {name!r} changed since preparation"
            return
