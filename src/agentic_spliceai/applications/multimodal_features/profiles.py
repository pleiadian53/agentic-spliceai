"""Feature-profile catalog reader.

Profiles live in ``examples/features/configs/*.yaml``. They select which
modalities to compute and hyperparameters per modality. This module reads
them read-only — it never modifies the YAMLs — and exposes them through a
``FeatureProfile`` dataclass for programmatic use.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Convention: profiles ship under examples/features/configs/. Callers can
# override (e.g., custom profiles) by passing ``profiles_dir=...``.
DEFAULT_PROFILES_DIR = Path("examples/features/configs")


@dataclass
class FeatureProfile:
    """A parsed feature-profile YAML."""

    name: str
    path: Path
    modalities: List[str] = field(default_factory=list)
    base_model: Optional[str] = None
    modality_configs: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_modalities(self) -> int:
        return len(self.modalities)


def list_profiles(
    profiles_dir: Optional[Path] = None,
) -> List[FeatureProfile]:
    """Return all profiles under ``profiles_dir`` (read-only)."""
    profiles_dir = Path(profiles_dir or _resolve_default_profiles_dir())
    if not profiles_dir.exists():
        logger.warning("profiles_dir not found: %s", profiles_dir)
        return []

    out: List[FeatureProfile] = []
    for path in sorted(profiles_dir.glob("*.yaml")):
        try:
            out.append(_load_profile(path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to parse profile %s: %s", path, exc)
    return out


def load_profile(
    name: str,
    profiles_dir: Optional[Path] = None,
) -> FeatureProfile:
    """Load a single profile by name (the YAML basename, no extension)."""
    profiles_dir = Path(profiles_dir or _resolve_default_profiles_dir())
    path = profiles_dir / f"{name}.yaml"
    if not path.exists():
        # Be helpful about what *is* available.
        available = [p.stem for p in sorted(profiles_dir.glob("*.yaml"))]
        raise FileNotFoundError(
            f"Profile {name!r} not found at {path}. "
            f"Available in {profiles_dir}: {available}"
        )
    return _load_profile(path)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_profile(path: Path) -> FeatureProfile:
    """Parse a profile YAML into a :class:`FeatureProfile`."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    pipeline = raw.get("pipeline", {}) or {}
    modalities = _sanitize_modalities(pipeline.get("modalities", []))
    base_model = pipeline.get("base_model")
    modality_configs = raw.get("modality_configs", {}) or {}

    return FeatureProfile(
        name=path.stem,
        path=path,
        modalities=modalities,
        base_model=base_model,
        modality_configs=modality_configs,
        raw=raw,
    )


def _sanitize_modalities(items) -> List[str]:
    """Accept list of plain strings or dicts keyed by modality name.

    Commented-out entries (in the YAML source) are already dropped by
    ``yaml.safe_load``, so the only thing left is to normalise types.
    """
    out: List[str] = []
    for item in items or []:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict):
            # ``{<modality>: {...cfg...}}`` variant — take the key.
            if len(item) == 1:
                out.append(next(iter(item.keys())))
    return out


def _resolve_default_profiles_dir() -> Path:
    """Find ``examples/features/configs/`` relative to the repo root."""
    # applications/multimodal_features/profiles.py
    # parents[0]=multimodal_features parents[1]=applications
    # parents[2]=agentic_spliceai parents[3]=src parents[4]=repo root
    here = Path(__file__).resolve()
    repo_root = here.parents[4]
    return repo_root / DEFAULT_PROFILES_DIR
