"""Manifest schema for output artifacts.

Each artifact directory under ``output/<topic>/<artifact>/`` carries a
``MANIFEST.yaml`` describing its status, provenance, and human-meaningful
notes. The presence of a MANIFEST is what defines a directory as an
"artifact" for registry purposes — :mod:`agentic_spliceai.registry.discovery`
walks the tree looking for MANIFESTs and stops descending past them.

See ``docs/system_design/output_management.md`` for the convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "MANIFEST.yaml"

ALLOWED_STATUSES: tuple[str, ...] = (
    "active",
    "baseline",
    "experimental",
    "archived",
    "placeholder",
)


class ManifestError(ValueError):
    """Raised when a MANIFEST.yaml is missing, unparseable, or schema-invalid."""


@dataclass
class Manifest:
    """A single artifact's MANIFEST.yaml, parsed.

    Attributes
    ----------
    path : Path
        Directory containing the MANIFEST (the artifact dir itself).
    status : str
        One of ``ALLOWED_STATUSES``. See the convention doc for semantics.
    produced_by : list of str
        Command(s) or script(s) that produced this artifact, in order.
        Single-producer artifacts use a one-element list.
    superseded_by : str or None
        Name of the artifact that replaces this one (typically a sibling
        directory name), or ``None``. Used to chain version history.
    created : str or None
        Optional ISO date when this artifact was produced.
    notes : str
        Free-form human description. May be multi-paragraph.
    tags : list of str
        Free-form labels for filtered registry views, e.g.
        ``["demo:ui_integration", "meta:v4"]``. Convention: ``namespace:value``.
    referenced_by : list of str
        Optional list of code/doc paths that depend on this artifact's
        identity (e.g. ``settings.yaml meta_models.m1s_v4_cleanannot``).
    """

    path: Path
    status: str
    produced_by: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    created: Optional[str] = None
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    referenced_by: List[str] = field(default_factory=list)

    @property
    def topic(self) -> str:
        """The topic this artifact belongs to.

        Conventionally the directory two levels above the MANIFEST — for
        ``output/meta_layer/m1s_v4_cleanannot/MANIFEST.yaml`` that's
        ``meta_layer``. For top-level artifacts directly under ``output/``
        (rare) the topic is ``"(root)"``.
        """
        parts = self.path.parts
        try:
            output_idx = parts.index("output")
        except ValueError:
            return "(unknown)"
        if output_idx + 2 < len(parts):
            return parts[output_idx + 1]
        return "(root)"

    @property
    def name(self) -> str:
        """The artifact's directory name."""
        return self.path.name

    @classmethod
    def load(cls, manifest_path: Path) -> "Manifest":
        """Load and validate a MANIFEST.yaml file.

        Parameters
        ----------
        manifest_path : Path
            Path to the MANIFEST.yaml (not the artifact dir).

        Raises
        ------
        ManifestError
            On missing file, parse error, or schema violation.
        """
        if not manifest_path.exists():
            raise ManifestError(f"MANIFEST not found: {manifest_path}")

        try:
            with manifest_path.open() as f:
                raw = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ManifestError(f"Invalid YAML in {manifest_path}: {e}") from e

        if not isinstance(raw, dict):
            raise ManifestError(
                f"MANIFEST must be a YAML mapping (got {type(raw).__name__}): {manifest_path}"
            )

        # Validate + normalize fields.
        status = raw.get("status")
        if status not in ALLOWED_STATUSES:
            raise ManifestError(
                f"Invalid status {status!r} in {manifest_path}. "
                f"Allowed: {list(ALLOWED_STATUSES)}"
            )

        produced_by = raw.get("produced_by", [])
        if isinstance(produced_by, str):
            produced_by = [produced_by]
        if not isinstance(produced_by, list):
            raise ManifestError(
                f"`produced_by` must be a string or list of strings in {manifest_path}"
            )

        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            raise ManifestError(f"`tags` must be a list in {manifest_path}")

        referenced_by = raw.get("referenced_by") or []
        if not isinstance(referenced_by, list):
            raise ManifestError(f"`referenced_by` must be a list in {manifest_path}")

        notes = raw.get("notes") or ""
        if not isinstance(notes, str):
            raise ManifestError(f"`notes` must be a string in {manifest_path}")

        return cls(
            path=manifest_path.parent,
            status=status,
            produced_by=[str(p) for p in produced_by],
            superseded_by=raw.get("superseded_by"),
            created=raw.get("created"),
            notes=notes.strip(),
            tags=[str(t) for t in tags],
            referenced_by=[str(r) for r in referenced_by],
        )

    def to_yaml(self) -> str:
        """Render this manifest back to canonical YAML (round-tripping friendly)."""
        data = {
            "status": self.status,
            "produced_by": self.produced_by,
            "superseded_by": self.superseded_by,
        }
        if self.created:
            data["created"] = self.created
        if self.notes:
            data["notes"] = self.notes
        if self.tags:
            data["tags"] = self.tags
        if self.referenced_by:
            data["referenced_by"] = self.referenced_by
        return yaml.safe_dump(data, sort_keys=False, width=80)


def starter_manifest(
    path: Path,
    status: str = "active",
    produced_by: Optional[List[str]] = None,
    notes: str = "",
    tags: Optional[List[str]] = None,
) -> Manifest:
    """Make a minimal Manifest suitable for writing out as a starter MANIFEST.yaml.

    Used by the ``registry add`` CLI subcommand.
    """
    if status not in ALLOWED_STATUSES:
        raise ManifestError(f"Invalid status: {status}")
    return Manifest(
        path=path,
        status=status,
        produced_by=produced_by or [],
        notes=notes,
        tags=tags or [],
    )
