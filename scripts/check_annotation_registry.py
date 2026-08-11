#!/usr/bin/env python
"""Consistency check for model-to-data version binding.

Two independent config surfaces describe the same annotations:

  ================================  =========================================
  ``settings.yaml``                 the core registry (``builds``,
                                    ``base_models``, ``meta_models``), used by
                                    the prediction and feature pipelines
  ``server/bio/config.ANNOTATIONS``  the Bio Lab registry, used by the gene
                                    browser and the genome view's truth sets
  ================================  =========================================

They resolve through different roots (``data_root`` versus ``PROJECT_ROOT/data``)
and are maintained by hand, so they can drift silently: bump ``default_release``
in one and the other keeps serving the old file while its label still claims the
new version. Nothing fails; the numbers are just quietly computed against a
different annotation than the one named on screen.

This checks that a model resolves to exactly one data version, by verifying:

  ==========================  ==============================================
  Claim                       Checked against
  ==========================  ==============================================
  ``base_models.<m>.release``  present, i.e. pinned rather than inherited
  each model's build key      a configured ``builds`` entry
  each model's GTF            the filesystem
  Bio Lab GTF path            the same file the core registry resolves
  Bio Lab display name        its own ``release`` (derived, so it cannot lie)
  ``meta_models.<k>``          declares ``train_build`` + ``train_annotation``,
                              and that pair has a Bio Lab entry
  truth-set availability      the track parquet actually on disk
  ==========================  ==============================================

A pinned ``release`` is the point of the first row. Without it the version comes
from a default chain (model → ``builds.<key>.default_release`` → top-level
``default_release``), so editing a default silently re-points every model that
inherited it.

Usage
-----
    conda run -n agentic-spliceai python scripts/check_annotation_registry.py
    conda run -n agentic-spliceai python scripts/check_annotation_registry.py -v

Exit code 0 = consistent, 1 = at least one problem. Safe to wire into CI.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("check_annotation_registry")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _core_gtf(build_key: str, release: str | None) -> tuple[Path | None, str]:
    """GTF the core registry resolves for a build key, plus the release it used."""
    from agentic_spliceai.splice_engine.resources.registry import get_genomic_registry

    reg = get_genomic_registry(build=build_key, release=release)
    return reg.get_gtf_path(validate=False), reg.cfg.release


def check_base_models() -> list[str]:
    """Every base model pins a release and resolves to a GTF that exists."""
    from agentic_spliceai.splice_engine.config.genomic_config import load_config
    from agentic_spliceai.splice_engine.resources.model_resources import (
        get_model_resources,
        list_available_models,
    )

    problems: list[str] = []
    builds = load_config().builds or {}

    for name in list_available_models():
        res = get_model_resources(name)

        if not res.release:
            problems.append(
                f"base_models.{name}: no `release`. The data version would come "
                f"from a default chain, so editing a default silently re-points "
                f"this model. Pin it explicitly."
            )

        if res.build_key not in builds:
            problems.append(
                f"base_models.{name}: build key '{res.build_key}' "
                f"({res.build}/{res.annotation_source}) is not in `builds`. "
                f"Configured: {sorted(builds)}"
            )
            continue

        gtf, _ = _core_gtf(res.build_key, res.release)
        if gtf is None or not gtf.exists():
            problems.append(
                f"base_models.{name}: GTF not found for {res.build_key} "
                f"release {res.release} ({gtf})"
            )
        else:
            logger.debug("%s -> %s (%s)", name, gtf.name, res.build_key)

    return problems


def check_bio_lab_registry() -> list[str]:
    """Bio Lab annotations agree with the core registry, file for file."""
    from agentic_spliceai.splice_engine.config.genomic_config import load_config
    from agentic_spliceai.splice_engine.resources.model_resources import registry_build_key
    from server.bio import config as bio_config

    problems: list[str] = []
    builds = load_config().builds or {}

    for key, spec in bio_config.ANNOTATIONS.items():
        source, build = spec["source"], spec["build"]

        if key != f"{source}.{build}":
            problems.append(
                f"ANNOTATIONS['{key}']: key must be <source>.<build>, "
                f"got source={source} build={build}"
            )

        gtf = spec["gtf"]
        if not gtf.exists():
            problems.append(f"ANNOTATIONS['{key}']: GTF missing at {gtf}")

        # The version must appear in the filename it labels. Derived in
        # config.py, so this catches a template that stops matching reality
        # rather than a typo.
        release = spec.get("release")
        if not release:
            problems.append(f"ANNOTATIONS['{key}']: no `release`")
        elif release not in gtf.name:
            problems.append(
                f"ANNOTATIONS['{key}']: release '{release}' does not appear in "
                f"the resolved filename '{gtf.name}'"
            )
        elif release not in spec["name"]:
            problems.append(
                f"ANNOTATIONS['{key}']: display name '{spec['name']}' omits "
                f"release '{release}'"
            )

        # The cross-registry claim: same <source>.<build>, same file on disk.
        build_key = registry_build_key(build, source)
        if build_key not in builds:
            problems.append(
                f"ANNOTATIONS['{key}']: no core `builds` entry for "
                f"{build}/{source} (looked for '{build_key}'). The Bio Lab can "
                f"serve it but the prediction pipeline cannot resolve it."
            )
            continue

        core_gtf, core_release = _core_gtf(build_key, release)
        if core_gtf is None:
            problems.append(
                f"ANNOTATIONS['{key}']: core registry resolves no GTF for "
                f"{build_key} release {release}"
            )
        elif core_gtf.resolve() != gtf.resolve():
            problems.append(
                f"ANNOTATIONS['{key}']: registries disagree.\n"
                f"    Bio Lab: {gtf.resolve()}\n"
                f"    core   : {core_gtf.resolve()}"
            )
        else:
            logger.debug("%s -> %s (both registries, release %s)",
                         key, gtf.name, core_release)

    return problems


def check_meta_models() -> list[str]:
    """Meta models declare a build and annotation that the Bio Lab can serve."""
    from agentic_spliceai.splice_engine.config.genomic_config import load_config
    from server.bio import config as bio_config

    problems: list[str] = []
    for key, spec in (load_config().meta_models or {}).items():
        build = spec.get("train_build")
        annot = spec.get("train_annotation")
        if not build or not annot:
            problems.append(
                f"meta_models.{key}: missing "
                f"{'train_build' if not build else 'train_annotation'}. The "
                f"genome view resolves its ground truth from these; without "
                f"them it falls back to the base model's annotation."
            )
            continue
        if f"{annot}.{build}" not in bio_config.ANNOTATIONS:
            problems.append(
                f"meta_models.{key}: train_annotation '{annot}' on '{build}' "
                f"has no ANNOTATIONS entry, so it cannot be scored against its "
                f"own training annotation."
            )
    return problems


def check_truth_sets() -> list[str]:
    """Offered truth sets match the track parquets actually present."""
    from server.bio import annotation_tracks
    from server.bio import config as bio_config

    problems: list[str] = []
    for build in sorted({s["build"] for s in bio_config.ANNOTATIONS.values()}):
        offered = annotation_tracks.truth_sets_for_build(build)
        for source in offered:
            spec = bio_config.ANNOTATIONS[f"{source}.{build}"]
            if not spec["sites"] or not Path(spec["sites"]).exists():
                problems.append(
                    f"truth set '{source}' offered on {build} but its track "
                    f"parquet is missing ({spec['sites']})"
                )
        logger.debug("%s truth sets: %s", build, list(offered) or "(none)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log every resolved path, not just problems")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    problems: list[str] = []
    for label, check in (
        ("base models", check_base_models),
        ("Bio Lab registry", check_bio_lab_registry),
        ("meta models", check_meta_models),
        ("truth sets", check_truth_sets),
    ):
        found = check()
        logger.info("%-18s %s", label, "OK" if not found else f"{len(found)} problem(s)")
        problems.extend(found)

    if problems:
        logger.error("")
        for p in problems:
            logger.error("  %s", p)
        logger.error("")
        logger.error("%d problem(s). A model may be resolving to a data version "
                     "other than the one it is labelled with.", len(problems))
        return 1

    logger.info("")
    logger.info("Model-to-data binding is consistent across both registries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
