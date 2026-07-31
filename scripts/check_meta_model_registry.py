#!/usr/bin/env python
"""Consistency check for meta-layer model naming.

Verifies that every ``meta_models`` entry in ``settings.yaml`` says the truth
about itself. The naming convention is
``<variant>.<arch>.<corpus>`` (see ``docs/meta_layer/methods/naming_convention.md``),
and each axis is checked against an independent source:

  ==================  ========================================================
  Claim               Checked against
  ==================  ========================================================
  key prefix          the entry's own ``variant`` field
  key ``arch`` part   the entry's ``arch`` field, and ``ARCH_REGISTRY``
  ``arch`` field      the class pickled in the checkpoint's ``config.pt``
  ``variant`` field   ``cfg.variant`` inside the checkpoint
  key ``corpus`` part the entry's ``corpus`` field
  ``dir``             the filesystem
  aliases             resolve to a configured key
  ==================  ========================================================

The checkpoint is authoritative for ``arch`` and ``variant``: a directory name
or a registry line is hand-maintained and can drift, while ``config.pt`` records
what actually trained. This check exists because the earlier ordinal scheme did
drift — ``m3_v1`` was tagged ``meta:v1`` in the output registry while its
``train.log`` read ``Arch: v3, variant: M3-S``.

Usage
-----
    conda run -n agentic-spliceai python scripts/check_meta_model_registry.py
    conda run -n agentic-spliceai python scripts/check_meta_model_registry.py --no-checkpoints

Exit code 0 = consistent, 1 = at least one problem. Safe to wire into CI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("check_meta_model_registry")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_key_shape(key: str, spec: dict, arch_registry: tuple) -> list[str]:
    """Validate ``<variant>.<arch>.<corpus>`` against the entry's own fields."""
    problems: list[str] = []
    parts = key.split(".")
    if len(parts) < 3:
        return [
            f"{key}: key must be <variant>.<arch>.<corpus>[.<run>]; "
            f"got {len(parts)} part(s). No bare version ordinals — see "
            f"naming_convention.md."
        ]

    variant_part, arch_part, corpus_part = parts[0], parts[1], parts[2]

    declared_variant = spec.get("variant")
    if not declared_variant:
        problems.append(f"{key}: missing `variant` field")
    else:
        # "M1-S" -> "m1s"
        expected = declared_variant.lower().replace("-", "")
        if variant_part != expected:
            problems.append(
                f"{key}: key variant {variant_part!r} != `variant: {declared_variant}` "
                f"(expected key prefix {expected!r})"
            )

    declared_arch = spec.get("arch")
    if not declared_arch:
        problems.append(f"{key}: missing `arch` field")
    else:
        if declared_arch not in arch_registry:
            problems.append(
                f"{key}: arch {declared_arch!r} not in ARCH_REGISTRY {arch_registry}"
            )
        if arch_part != declared_arch:
            problems.append(
                f"{key}: key arch {arch_part!r} != `arch: {declared_arch}`"
            )

    declared_corpus = spec.get("corpus")
    if not declared_corpus:
        problems.append(f"{key}: missing `corpus` field")
    elif corpus_part != declared_corpus:
        problems.append(
            f"{key}: key corpus {corpus_part!r} != `corpus: {declared_corpus}`"
        )

    return problems


def _check_checkpoint(key: str, spec: dict) -> list[str]:
    """Validate the declared arch/variant against the trained checkpoint."""
    import torch

    from agentic_spliceai.splice_engine.meta_layer.models.loader import (
        CONFIG_TYPE_TO_ARCH,
        arch_of_config,
    )

    problems: list[str] = []
    model_dir = PROJECT_ROOT / spec["dir"]
    config_pt = model_dir / "config.pt"
    if not config_pt.exists():
        return [f"{key}: no config.pt at {spec['dir']}"]

    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_model_v3 import (
        MetaSpliceConfig,
    )
    from agentic_spliceai.splice_engine.meta_layer.models.meta_splice_v4_xattn import (
        MetaSpliceXAttnConfig,
    )
    torch.serialization.add_safe_globals([MetaSpliceConfig, MetaSpliceXAttnConfig])
    cfg = torch.load(config_pt, map_location="cpu", weights_only=True)

    try:
        actual_arch = arch_of_config(cfg)
    except TypeError as exc:
        return [f"{key}: {exc} (known: {sorted(CONFIG_TYPE_TO_ARCH)})"]

    if spec.get("arch") != actual_arch:
        problems.append(
            f"{key}: declares arch {spec.get('arch')!r} but config.pt holds "
            f"{type(cfg).__name__} = {actual_arch!r}"
        )
    if spec.get("variant") != cfg.variant:
        problems.append(
            f"{key}: declares variant {spec.get('variant')!r} but config.pt "
            f"holds {cfg.variant!r}"
        )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--no-checkpoints", action="store_true",
        help="Skip the config.pt cross-check (metadata-only; no torch import, "
             "no checkpoint files needed).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from agentic_spliceai.splice_engine.meta_layer.models.factory import ARCH_REGISTRY
    from agentic_spliceai.splice_engine.resources import (
        META_MODEL_ALIASES,
        get_meta_model_config,
        list_available_meta_models,
    )

    keys = list_available_meta_models(status=None)
    if not keys:
        logger.error("No meta_models configured in settings.yaml")
        return 1

    problems: list[str] = []
    logger.info("Checking %d meta-model entries\n", len(keys))

    for key in keys:
        spec = get_meta_model_config(key)
        entry_problems = _check_key_shape(key, spec, ARCH_REGISTRY)

        model_dir = PROJECT_ROOT / spec.get("dir", "")
        if not spec.get("dir"):
            entry_problems.append(f"{key}: missing `dir` field")
        elif not model_dir.is_dir():
            entry_problems.append(f"{key}: dir does not exist: {spec['dir']}")
        elif not args.no_checkpoints:
            entry_problems.extend(_check_checkpoint(key, spec))

        status = "FAIL" if entry_problems else "ok"
        logger.info(
            "  [%-4s] %-32s arch=%-14s corpus=%-12s -> %s",
            status, key, spec.get("arch", "?"), spec.get("corpus", "?"),
            spec.get("dir", "?"),
        )
        problems.extend(entry_problems)

    # Retired keys must still resolve, or ~100 existing references break.
    logger.info("\nChecking %d retired aliases", len(META_MODEL_ALIASES))
    for old, new in sorted(META_MODEL_ALIASES.items()):
        if new not in keys:
            problems.append(f"alias {old!r} -> {new!r}: target is not configured")
        else:
            logger.info("  [ok  ] %-22s -> %s", old, new)

    if problems:
        logger.error("\n%d problem(s):", len(problems))
        for p in problems:
            logger.error("  - %s", p)
        return 1

    logger.info("\nAll meta-model names are internally consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
