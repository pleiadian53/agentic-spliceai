"""Artifact registry for ``output/`` directories.

Each artifact dir (``output/<topic>/<artifact>/``) carries a
``MANIFEST.yaml`` describing its status, provenance, and notes. The
registry tools (CLI and library API) walk those manifests to:

- generate a human-readable ``output/REGISTRY.md`` index,
- validate that everything on disk is properly manifested + cross-checks
  ``status: active`` against ``settings.yaml``,
- filter / list artifacts by tag, status, or topic (e.g., to show only
  the artifacts relevant to a particular demo presentation).

See ``docs/system_design/output_management.md`` for the convention and
``python -m agentic_spliceai.registry --help`` for the CLI.

Library API
-----------
::

    from agentic_spliceai.registry import (
        Manifest, build_registry, validate, load_all_manifests,
    )

    manifests = load_all_manifests(Path("output"))
    text = render_registry(manifests)  # if you need to render without writing
"""

from .builder import build_registry, render_registry
from .discovery import (
    default_output_root,
    find_unmanaged_dirs,
    iter_artifact_dirs,
    load_all_manifests,
)
from .manifest import (
    ALLOWED_STATUSES,
    MANIFEST_FILENAME,
    Manifest,
    ManifestError,
    starter_manifest,
)
from .validator import Issue, has_errors, validate

__all__ = [
    # manifest
    "ALLOWED_STATUSES",
    "MANIFEST_FILENAME",
    "Manifest",
    "ManifestError",
    "starter_manifest",
    # discovery
    "default_output_root",
    "find_unmanaged_dirs",
    "iter_artifact_dirs",
    "load_all_manifests",
    # builder
    "build_registry",
    "render_registry",
    # validator
    "Issue",
    "has_errors",
    "validate",
]
