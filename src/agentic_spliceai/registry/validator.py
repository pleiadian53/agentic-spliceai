"""Validate the artifact registry: every artifact dir has a valid MANIFEST,
status values are correct, and ``status: active`` cross-checks against
``settings.yaml meta_models:`` / ``base_models:`` where applicable.

Returns a list of issues; an empty list means valid. The CLI reports each
issue and exits nonzero if any are found, so this is CI-ready.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from .discovery import find_unmanaged_dirs, iter_artifact_dirs, load_all_manifests
from .manifest import ALLOWED_STATUSES, MANIFEST_FILENAME, Manifest, ManifestError

logger = logging.getLogger(__name__)


@dataclass
class Issue:
    """A single registry validation issue."""

    severity: str  # "error" or "warning"
    path: Path  # the artifact dir (or candidate)
    message: str

    def __str__(self) -> str:
        rel = self.path
        return f"[{self.severity.upper()}] {rel}: {self.message}"


def _load_settings(settings_yaml: Path) -> dict:
    if not settings_yaml.is_file():
        return {}
    try:
        with settings_yaml.open() as f:
            return yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Could not read %s: %s", settings_yaml, e)
        return {}


def validate(
    output_root: Path,
    settings_yaml: Optional[Path] = None,
) -> List[Issue]:
    """Run all registry validations.

    Parameters
    ----------
    output_root : Path
        The ``output/`` directory to validate.
    settings_yaml : Path, optional
        Path to ``settings.yaml`` for cross-checking ``status: active``
        models. If ``None``, the active-vs-settings check is skipped.

    Returns
    -------
    list of Issue
        Empty if everything is valid.
    """
    issues: List[Issue] = []

    # 1. Find unmanaged dirs (artifact-shaped, no MANIFEST).
    for unmanaged in find_unmanaged_dirs(output_root):
        issues.append(
            Issue(
                severity="error",
                path=unmanaged,
                message=f"missing {MANIFEST_FILENAME} (looks like an artifact dir but is unmanaged)",
            )
        )

    # 2. Walk known manifests, catching parse/schema errors.
    parsed: List[Manifest] = []
    for artifact_dir in iter_artifact_dirs(output_root):
        manifest_path = artifact_dir / MANIFEST_FILENAME
        try:
            parsed.append(Manifest.load(manifest_path))
        except ManifestError as e:
            issues.append(
                Issue(severity="error", path=artifact_dir, message=str(e))
            )

    # 3. Cross-check `status: active` against settings.yaml.
    if settings_yaml is not None and parsed:
        settings = _load_settings(settings_yaml)
        active_meta = set((settings.get("meta_models") or {}).keys())
        active_base = set((settings.get("base_models") or {}).keys())

        for m in parsed:
            if m.status != "active":
                continue
            # Only require settings membership for likely-runtime artifacts:
            # those whose name is conventionally a meta or base model name.
            # We can't perfectly classify; this is a soft check (warning).
            looks_like_meta = m.name.startswith(("m1s", "m2s", "m3s", "m4s", "m1p", "m2p"))
            looks_like_base = m.topic in {"splice_classifier"} or m.name.startswith(("spliceai", "openspliceai", "splicebert"))
            if looks_like_meta and m.name not in active_meta:
                issues.append(
                    Issue(
                        severity="warning",
                        path=m.path,
                        message=(
                            f"status=active but not present in settings.yaml meta_models:. "
                            "Either add it there (so the resource manager can find it) "
                            "or change status to baseline/experimental."
                        ),
                    )
                )
            if looks_like_base and m.name not in active_base:
                issues.append(
                    Issue(
                        severity="warning",
                        path=m.path,
                        message=(
                            f"status=active but not present in settings.yaml base_models:. "
                            "Likely the base-model protocol integration is pending."
                        ),
                    )
                )

    # 4. Superseded_by references should point at things that exist
    #    (any manifest's name in any topic).
    if parsed:
        known = {m.name for m in parsed}
        for m in parsed:
            if m.superseded_by and m.superseded_by not in known:
                issues.append(
                    Issue(
                        severity="warning",
                        path=m.path,
                        message=(
                            f"superseded_by={m.superseded_by!r} doesn't match any "
                            "known artifact name; check the spelling."
                        ),
                    )
                )

    return issues


def has_errors(issues: List[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)
