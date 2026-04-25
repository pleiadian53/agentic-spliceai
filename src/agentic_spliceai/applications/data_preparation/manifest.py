"""Ingest manifest — durable record of what was prepared, from what, when.

The manifest lives at ``<output_dir>/ingest_manifest.json`` and is the
single source of truth for "what artifacts exist, from which inputs,
with what hashes". Downstream apps and UIs read this to check readiness.

Schema (stable, versioned via ``MANIFEST_VERSION``)
---------------------------------------------------

    {
      "manifest_version": 1,
      "build": "GRCh38",
      "annotation_source": "mane",
      "created_at": "2026-04-19T...",
      "updated_at": "2026-04-19T...",
      "inputs": {
        "fasta": {"path": "...", "sha256": "...", "size_bytes": ...},
        "gtf":   {"path": "...", "sha256": "...", "size_bytes": ...}
      },
      "artifacts": {
        "gene_features": {"path": "...", "sha256": "...", "rows": 19288, "step": "..."}
        ...
      },
      "tool_version": "agentic-spliceai <version>"
    }
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

MANIFEST_VERSION = 1
MANIFEST_FILENAME = "ingest_manifest.json"


# ---------------------------------------------------------------------------
# Helpers (defined up-front so dataclass defaults can reference them)
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """Streaming SHA-256 of a file. Chunked for memory safety."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


@dataclass
class InputRecord:
    """User-provided input file (FASTA or annotation)."""

    path: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    @classmethod
    def from_path(cls, path: Path, *, hash_file: bool = True) -> "InputRecord":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        size = path.stat().st_size
        sha = _sha256_file(path) if hash_file else None
        return cls(path=str(path), sha256=sha, size_bytes=size)


@dataclass
class ArtifactRecord:
    """A derived artifact produced by the pipeline."""

    path: str
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    rows: Optional[int] = None
    step: Optional[str] = None
    created_at: Optional[str] = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        step: Optional[str] = None,
        rows: Optional[int] = None,
        hash_file: bool = True,
    ) -> "ArtifactRecord":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing: {path}")
        return cls(
            path=str(path),
            sha256=_sha256_file(path) if hash_file else None,
            size_bytes=path.stat().st_size,
            rows=rows,
            step=step,
            created_at=_now(),
        )


@dataclass
class IngestManifest:
    """Top-level manifest serialised to ``ingest_manifest.json``."""

    build: str
    annotation_source: str
    inputs: Dict[str, InputRecord] = field(default_factory=dict)
    artifacts: Dict[str, ArtifactRecord] = field(default_factory=dict)
    manifest_version: int = MANIFEST_VERSION
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    tool_version: Optional[str] = None
    notes: Optional[str] = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, output_dir: Path) -> Path:
        """Write the manifest to ``<output_dir>/ingest_manifest.json``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        path = output_dir / MANIFEST_FILENAME
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)
        return path

    @classmethod
    def load(cls, output_dir: Path) -> "IngestManifest":
        """Load an existing manifest from ``<output_dir>``."""
        path = Path(output_dir) / MANIFEST_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No ingest manifest at {path}. Run `agentic-spliceai-ingest "
                f"prepare --output-dir {output_dir}` first."
            )
        with open(path, "r") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def try_load(cls, output_dir: Path) -> Optional["IngestManifest"]:
        """Load, returning ``None`` if the manifest doesn't exist."""
        try:
            return cls.load(output_dir)
        except FileNotFoundError:
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "build": self.build,
            "annotation_source": self.annotation_source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tool_version": self.tool_version,
            "notes": self.notes,
            "inputs": {k: asdict(v) for k, v in self.inputs.items()},
            "artifacts": {k: asdict(v) for k, v in self.artifacts.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestManifest":
        version = int(data.get("manifest_version", 1))
        if version > MANIFEST_VERSION:
            raise ValueError(
                f"Manifest version {version} is newer than this tool "
                f"(supports up to {MANIFEST_VERSION}). Upgrade the package."
            )
        inputs = {
            k: InputRecord(**v) for k, v in (data.get("inputs") or {}).items()
        }
        artifacts = {
            k: ArtifactRecord(**v) for k, v in (data.get("artifacts") or {}).items()
        }
        return cls(
            build=data["build"],
            annotation_source=data["annotation_source"],
            inputs=inputs,
            artifacts=artifacts,
            manifest_version=version,
            created_at=data.get("created_at") or _now(),
            updated_at=data.get("updated_at") or _now(),
            tool_version=data.get("tool_version"),
            notes=data.get("notes"),
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def set_input(self, name: str, record: InputRecord) -> None:
        self.inputs[name] = record

    def set_artifact(self, name: str, record: ArtifactRecord) -> None:
        self.artifacts[name] = record

    def has_artifact(self, name: str) -> bool:
        return name in self.artifacts and Path(self.artifacts[name].path).exists()
