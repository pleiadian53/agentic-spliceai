"""Feature-manifest — record of what was generated, with which profile,
inputs, and chromosomes.

Schema
------

    {
      "manifest_version": 1,
      "build": "GRCh38",
      "profile": "full_stack",
      "modalities": ["base_scores", "annotation", ...],
      "chromosomes": ["22"],
      "base_predictions_dir": "/...",
      "output_dir": "/...",
      "created_at": "...",
      "updated_at": "...",
      "artifacts": {
        "chr22": {"path": "...", "sha256": "...", "rows": 562341, "step": "..."}
      },
      "tracks_used": [
        {"modality": "conservation", "name": "phylop",
         "path": "...", "sha256": "..."}
      ],
      "tool_version": "agentic-spliceai ..."
    }

The on-disk file is ``<output_dir>/feature_manifest.json``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Reuse the hashing + timestamp helpers from the data-preparation package
# to keep the two manifests schema-adjacent.
from ..data_preparation.manifest import _now, _sha256_file

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "feature_manifest.json"


@dataclass
class FeatureArtifactRecord:
    """A generated per-chromosome feature parquet (or similar)."""

    path: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    rows: Optional[int] = None
    step: Optional[str] = None
    chromosome: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        chromosome: Optional[str] = None,
        step: Optional[str] = None,
        rows: Optional[int] = None,
        hash_file: bool = True,
    ) -> "FeatureArtifactRecord":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing: {path}")
        return cls(
            path=str(path),
            sha256=_sha256_file(path) if hash_file else None,
            size_bytes=path.stat().st_size,
            rows=rows,
            step=step,
            chromosome=chromosome,
            created_at=_now(),
        )


@dataclass
class TrackReference:
    """Minimal reference to an external track used in a feature run."""

    modality: str
    name: str
    path: Optional[str] = None
    sha256: Optional[str] = None
    accession: Optional[str] = None


@dataclass
class FeatureManifest:
    """Top-level manifest for a feature-generation run."""

    build: str
    profile: str
    modalities: List[str] = field(default_factory=list)
    chromosomes: List[str] = field(default_factory=list)
    base_predictions_dir: Optional[str] = None
    output_dir: Optional[str] = None
    artifacts: Dict[str, FeatureArtifactRecord] = field(default_factory=dict)
    tracks_used: List[TrackReference] = field(default_factory=list)
    manifest_version: int = MANIFEST_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    tool_version: Optional[str] = None
    notes: Optional[str] = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        path = output_dir / MANIFEST_FILENAME
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, output_dir: Path) -> "FeatureManifest":
        path = Path(output_dir) / MANIFEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No feature manifest at {path}. Run `agentic-spliceai-features "
                f"prepare --output-dir {output_dir}` first."
            )
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def try_load(cls, output_dir: Path) -> Optional["FeatureManifest"]:
        try:
            return cls.load(output_dir)
        except FileNotFoundError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "build": self.build,
            "profile": self.profile,
            "modalities": list(self.modalities),
            "chromosomes": list(self.chromosomes),
            "base_predictions_dir": self.base_predictions_dir,
            "output_dir": self.output_dir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_version": self.tool_version,
            "notes": self.notes,
            "artifacts": {k: asdict(v) for k, v in self.artifacts.items()},
            "tracks_used": [asdict(t) for t in self.tracks_used],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureManifest":
        version = int(data.get("manifest_version", 1))
        if version > MANIFEST_VERSION:
            raise ValueError(
                f"Manifest version {version} is newer than this tool "
                f"(supports up to {MANIFEST_VERSION}). Upgrade the package."
            )
        artifacts = {
            k: FeatureArtifactRecord(**v)
            for k, v in (data.get("artifacts") or {}).items()
        }
        tracks = [
            TrackReference(**t) for t in (data.get("tracks_used") or [])
        ]
        return cls(
            build=data["build"],
            profile=data["profile"],
            modalities=list(data.get("modalities", [])),
            chromosomes=list(data.get("chromosomes", [])),
            base_predictions_dir=data.get("base_predictions_dir"),
            output_dir=data.get("output_dir"),
            artifacts=artifacts,
            tracks_used=tracks,
            manifest_version=version,
            created_at=data.get("created_at") or _now(),
            updated_at=data.get("updated_at") or _now(),
            tool_version=data.get("tool_version"),
            notes=data.get("notes"),
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_artifact(self, name: str, record: FeatureArtifactRecord) -> None:
        self.artifacts[name] = record

    def add_track(self, ref: TrackReference) -> None:
        # Deduplicate by (modality, name).
        for existing in self.tracks_used:
            if existing.modality == ref.modality and existing.name == ref.name:
                existing.path = ref.path or existing.path
                existing.sha256 = ref.sha256 or existing.sha256
                return
        self.tracks_used.append(ref)
