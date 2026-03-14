"""YAML config loader for multimodal feature engineering workflows.

Loads a YAML config file and maps it to existing dataclasses:
- FeaturePipelineConfig (pipeline section)
- PositionSamplingConfig (sampling section)
- Per-modality configs (resolved dynamically from FeaturePipeline._REGISTRY)

No modality-specific imports needed — the registry provides the config class.

Usage:
    from config_loader import load_workflow_config

    pipeline_config, sampling_config, workflow_opts = load_workflow_config()
    # or: load_workflow_config(config_path="configs/full_stack.yaml")
    # or: load_workflow_config(config_name="lightweight")
"""

import logging
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from agentic_spliceai.splice_engine.features import (
    FeaturePipeline,
    FeaturePipelineConfig,
    PositionSamplingConfig,
)
from agentic_spliceai.splice_engine.features.modality import ModalityConfig

logger = logging.getLogger(__name__)

# Default config directory (sibling to this file)
_CONFIGS_DIR = Path(__file__).parent / "configs"


def load_workflow_config(
    config_path: Optional[Path | str] = None,
    *,
    config_name: str = "default",
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[FeaturePipelineConfig, Optional[PositionSamplingConfig], Dict[str, Any]]:
    """Load and validate a workflow YAML config.

    Parameters
    ----------
    config_path : Path or str, optional
        Explicit path to a YAML file. If None, resolves from config_name.
    config_name : str
        Config variant name (e.g., 'default', 'full_stack'). Resolved as
        ``configs/{config_name}.yaml`` relative to examples/features/.
    overrides : dict, optional
        Runtime overrides applied on top of the YAML (e.g., from CLI).
        Supports dotted keys: ``{"pipeline.base_model": "spliceai"}``.

    Returns
    -------
    tuple of (FeaturePipelineConfig, PositionSamplingConfig | None, dict)
        Pipeline config, sampling config (None if disabled), and the
        raw workflow section for script-level settings.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If a modality in the config is not registered, or if config
        fields don't match the dataclass.
    """
    # Resolve path
    if config_path is None:
        config_path = _CONFIGS_DIR / f"{config_name}.yaml"
    config_path = Path(config_path)

    if not config_path.exists():
        available = [p.stem for p in _CONFIGS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Config not found: {config_path}\n"
            f"Available configs: {available}"
        )

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    logger.info("Loaded workflow config: %s", config_path)

    # Apply overrides
    if overrides:
        _apply_overrides(raw, overrides)

    # Trigger auto-registration of all modalities
    from agentic_spliceai.splice_engine.features import modalities as _  # noqa: F401

    # ── Validate modality names ───────────────────────────────────────
    pipeline_section = raw.get("pipeline", {})
    modality_names = pipeline_section.get("modalities", ["base_scores"])
    available = FeaturePipeline.available_modalities()
    unknown = [m for m in modality_names if m not in available]
    if unknown:
        raise ValueError(
            f"Unknown modalities in config: {unknown}. "
            f"Registered: {available}"
        )

    # ── Build per-modality configs from registry ──────────────────────
    modality_configs_section = raw.get("modality_configs", {})
    modality_configs: Dict[str, ModalityConfig] = {}

    for name in modality_names:
        if name in modality_configs_section:
            cfg_dict = modality_configs_section[name]
            modality_configs[name] = _instantiate_modality_config(name, cfg_dict)
        # else: FeaturePipeline will use default_config()

    # ── Build FeaturePipelineConfig ───────────────────────────────────
    pipeline_config = FeaturePipelineConfig(
        base_model=pipeline_section.get("base_model", "openspliceai"),
        modalities=modality_names,
        modality_configs=modality_configs,
        output_format=pipeline_section.get("output_format", "parquet"),
        verbosity=pipeline_section.get("verbosity", 1),
    )

    # ── Build PositionSamplingConfig ──────────────────────────────────
    sampling_section = raw.get("sampling", {})
    sampling_config = None
    if sampling_section.get("enabled", False):
        sampling_config = _instantiate_dataclass(
            PositionSamplingConfig, sampling_section
        )

    # ── Workflow section (pass-through) ───────────────────────────────
    workflow_section = raw.get("workflow", {})

    return pipeline_config, sampling_config, workflow_section


def _instantiate_modality_config(
    name: str, cfg_dict: Dict[str, Any]
) -> ModalityConfig:
    """Instantiate a modality config from the registry + a dict of values.

    Looks up the config class from FeaturePipeline._REGISTRY[name],
    then instantiates it with the provided dict values. Unknown keys
    are warned about and ignored (forward-compatible).
    """
    if name not in FeaturePipeline._REGISTRY:
        raise ValueError(f"Modality '{name}' not in registry")

    _, config_cls = FeaturePipeline._REGISTRY[name]
    return _instantiate_dataclass(config_cls, cfg_dict)


def _instantiate_dataclass(cls: type, values: Dict[str, Any]) -> Any:
    """Instantiate a dataclass from a dict, ignoring unknown fields.

    Parameters
    ----------
    cls : type
        A dataclass type to instantiate.
    values : dict
        Key-value pairs to pass as constructor arguments.

    Returns
    -------
    instance of cls
        The instantiated dataclass with known fields populated.
    """
    valid_fields = {f.name for f in dataclass_fields(cls)}
    filtered = {}
    for k, v in values.items():
        if k in valid_fields:
            # Convert lists to tuples for frozen dataclass fields
            field_info = {f.name: f for f in dataclass_fields(cls)}
            if k in field_info and hasattr(field_info[k].type, "__origin__"):
                # Handle tuple fields that come as lists from YAML
                pass
            filtered[k] = v
        else:
            logger.warning(
                "Ignoring unknown config field '%s' for %s", k, cls.__name__
            )
    return cls(**filtered)


def _apply_overrides(raw: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Apply dotted-key overrides to the raw YAML dict.

    Example: {"pipeline.base_model": "spliceai"} sets
    raw["pipeline"]["base_model"] = "spliceai".
    """
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        target = raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
