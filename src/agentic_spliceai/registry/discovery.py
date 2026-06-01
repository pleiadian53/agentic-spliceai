"""Walk ``output/`` to find artifact directories.

An "artifact directory" is one that contains a ``MANIFEST.yaml`` directly
(see :mod:`agentic_spliceai.registry.manifest`). Discovery stops descending
into an artifact dir — nested children are part of that artifact, not
separate artifacts themselves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional

from .manifest import MANIFEST_FILENAME, Manifest, ManifestError

logger = logging.getLogger(__name__)


def iter_artifact_dirs(output_root: Path) -> Iterator[Path]:
    """Yield every directory under ``output_root`` that contains a MANIFEST.yaml.

    Descent stops at each artifact dir (its children are not separate
    artifacts). Hidden dirs (``.foo``) and the registry's own outputs
    (``REGISTRY.md`` at the root) are ignored automatically because they
    aren't directories with MANIFESTs.
    """
    output_root = output_root.resolve()
    if not output_root.is_dir():
        return

    stack: List[Path] = [output_root]
    while stack:
        current = stack.pop()
        manifest = current / MANIFEST_FILENAME
        if manifest.is_file():
            yield current
            continue  # don't descend into an artifact
        try:
            for child in current.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    stack.append(child)
        except (PermissionError, OSError) as e:
            logger.warning("Could not list %s: %s", current, e)


def load_all_manifests(output_root: Path) -> List[Manifest]:
    """Load every artifact's MANIFEST under ``output_root``.

    Manifests that fail to load are logged and skipped (so a single broken
    file doesn't block the rest). Callers wanting strict behavior should
    use :func:`agentic_spliceai.registry.validator.validate` instead.
    """
    manifests: List[Manifest] = []
    for artifact_dir in iter_artifact_dirs(output_root):
        manifest_path = artifact_dir / MANIFEST_FILENAME
        try:
            manifests.append(Manifest.load(manifest_path))
        except ManifestError as e:
            logger.error("Skipping invalid manifest: %s", e)
    return manifests


def find_unmanaged_dirs(
    output_root: Path,
    *,
    artifact_depth: int = 2,
) -> List[Path]:
    """Find directories that look like artifact candidates but lack a MANIFEST.

    Heuristic: any directory at exactly ``artifact_depth`` levels below
    ``output_root`` (default 2: ``output/<topic>/<artifact>/``) that
    doesn't contain a MANIFEST.yaml. Also flags top-level directories
    directly under ``output_root`` if they have non-directory files but
    no MANIFEST (suggesting they ARE an artifact, just unmanaged).

    Used by the validator to catch newly-produced output dirs that the
    user forgot to manifest.
    """
    output_root = output_root.resolve()
    if not output_root.is_dir():
        return []

    unmanaged: List[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue

        topic_dir = child
        # If the topic dir itself has a MANIFEST (rare; means it IS the artifact)
        # or has children with MANIFESTs, walk into it.
        has_own_manifest = (topic_dir / MANIFEST_FILENAME).is_file()
        if has_own_manifest:
            continue

        # Walk one level deeper: <topic>/<artifact>/
        try:
            for grandchild in sorted(topic_dir.iterdir()):
                if not grandchild.is_dir() or grandchild.name.startswith("."):
                    continue
                if not (grandchild / MANIFEST_FILENAME).is_file():
                    unmanaged.append(grandchild)
        except (PermissionError, OSError):
            continue

        # Also flag the topic dir itself if it contains FILES (suggesting
        # it's actually a top-level artifact, not a grouper).
        has_files = any(
            f.is_file() and f.name != MANIFEST_FILENAME
            for f in topic_dir.iterdir()
        )
        has_artifact_subdirs = any(
            (g / MANIFEST_FILENAME).is_file()
            for g in topic_dir.iterdir() if g.is_dir()
        )
        if has_files and not has_artifact_subdirs:
            unmanaged.append(topic_dir)

    return unmanaged


def default_output_root() -> Path:
    """Best-effort default for the ``output/`` root, assuming cwd is the project."""
    return Path("output").resolve()
