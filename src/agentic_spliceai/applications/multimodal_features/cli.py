"""CLI for the multimodal-features application.

Subcommands
-----------

- ``list-profiles``     — show feature-profile YAMLs under examples/features/configs/
- ``list-tracks``       — show registered external tracks (bigWigs, ENCODE accessions)
- ``fetch-tracks``      — download conservation bigWigs to the configured cache
- ``status``            — read-only readiness report for a feature output dir
- ``prepare``           — run FeatureWorkflow for a profile + chromosomes
- ``validate``          — read-only parquet alignment + row check

Entry point: ``agentic-spliceai-features`` (see ``pyproject.toml``).

Examples
--------

    agentic-spliceai-features list-profiles
    agentic-spliceai-features list-tracks --build GRCh38

    agentic-spliceai-features status \\
        --output-dir data/mane/GRCh38/openspliceai_eval/analysis_sequences

    agentic-spliceai-features prepare --profile full_stack \\
        --build GRCh38 --chromosomes 22 \\
        --input-dir  data/mane/GRCh38/openspliceai_eval/precomputed \\
        --output-dir output/features/chr22_full_stack

Production safety
-----------------

``prepare`` requires either ``--output-dir`` or ``--inplace``. In
``--inplace`` mode the CLI writes to the resource-manager-resolved
production features directory but the underlying workflow preserves
existing per-chromosome parquets (``resume=True``) — use ``--force``
to regenerate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .pipeline import (
    DEFAULT_STEPS, prepare_features, resolve_canonical_features_dir,
)
from .profiles import list_profiles
from .status import get_status
from .steps import step_validate
from .tracks import fetch_conservation_tracks, fetch_encode_tracks, list_tracks

logger = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=_log_level(args.verbosity),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.command == "list-profiles":
        return _cmd_list_profiles(args)
    if args.command == "list-tracks":
        return _cmd_list_tracks(args)
    if args.command == "fetch-tracks":
        return _cmd_fetch_tracks(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "validate":
        return _cmd_validate(args)

    parser.print_help()
    return 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-spliceai-features",
        description=(
            "Multimodal features application: feature-profile catalog, "
            "external-track ingestion, per-chromosome feature generation."
        ),
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="0 = silent, 1 = info (default), 2 = debug.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # list-profiles -------------------------------------------------------
    p_lp = sub.add_parser("list-profiles", help="List feature-profile YAMLs.")
    p_lp.add_argument(
        "--format", choices=("table", "json"), default="table",
    )

    # list-tracks ---------------------------------------------------------
    p_lt = sub.add_parser(
        "list-tracks", help="List registered external tracks.",
    )
    p_lt.add_argument("--build", default=None, help="Filter by build.")
    p_lt.add_argument("--modality", default=None, help="Filter by modality.")
    p_lt.add_argument(
        "--format", choices=("table", "json"), default="table",
    )

    # fetch-tracks --------------------------------------------------------
    p_ft = sub.add_parser(
        "fetch-tracks",
        help="Download external bigWigs to the configured cache.",
    )
    p_ft.add_argument("--build", required=True)
    p_ft.add_argument(
        "--modality", default="conservation",
        choices=("conservation", "epigenetic"),
        help=(
            "conservation: PhyloP / PhastCons (UCSC URL). "
            "epigenetic: ENCODE ChIP-seq by accession."
        ),
    )
    p_ft.add_argument(
        "--cell-lines", nargs="*", default=None,
        help="Filter ENCODE fetch by cell line (epigenetic only).",
    )
    p_ft.add_argument(
        "--marks", nargs="*", default=None,
        help="Filter ENCODE fetch by histone mark (epigenetic only).",
    )
    p_ft.add_argument("--force", action="store_true")
    p_ft.add_argument("--dest", type=Path, default=None, help="Override cache dir.")

    # status --------------------------------------------------------------
    p_st = sub.add_parser("status", help="Read-only readiness report.")
    stat_dest = p_st.add_mutually_exclusive_group(required=True)
    stat_dest.add_argument(
        "--output-dir", type=Path,
        help="Feature output directory.",
    )
    stat_dest.add_argument(
        "--canonical", action="store_true",
        help="Check the canonical features dir for --build / --source.",
    )
    p_st.add_argument("--build", default=None, help="Required with --canonical.")
    p_st.add_argument(
        "--annotation-source", default="mane",
        help="Required with --canonical (default: mane).",
    )
    p_st.add_argument(
        "--base-model", default="openspliceai",
        help="Base model subdir for canonical path (default: openspliceai).",
    )
    p_st.add_argument(
        "--chromosomes", nargs="*", default=None,
        help="Expected chromosomes (drives the missing-list).",
    )
    p_st.add_argument(
        "--format", choices=("table", "json"), default="table",
    )

    # prepare -------------------------------------------------------------
    p_pr = sub.add_parser("prepare", help="Run the feature workflow.")
    p_pr.add_argument("--profile", required=True)
    p_pr.add_argument("--build", required=True)
    prep_dest = p_pr.add_mutually_exclusive_group(required=True)
    prep_dest.add_argument(
        "--output-dir", type=Path,
        help="Directory for per-chromosome feature parquets.",
    )
    prep_dest.add_argument(
        "--inplace", action="store_true",
        help="Write to the canonical features dir "
             "(requires --annotation-source; existing parquets preserved).",
    )
    p_pr.add_argument(
        "--annotation-source", default="mane",
        help="Required with --inplace (default: mane).",
    )
    p_pr.add_argument(
        "--base-model", default="openspliceai",
        help="Used for canonical path resolution (default: openspliceai).",
    )
    p_pr.add_argument(
        "--input-dir", required=True, type=Path,
        help="Directory containing base-layer prediction artifacts.",
    )
    p_pr.add_argument("--chromosomes", nargs="*", default=None)
    p_pr.add_argument(
        "--fetch-missing-tracks", action="store_true",
        help="Download conservation bigWigs before running.",
    )
    p_pr.add_argument(
        "--no-resume", action="store_true",
        help="Disable library-level resume (regenerate existing parquets).",
    )
    p_pr.add_argument(
        "--skip-steps", nargs="*", default=None,
        choices=list(DEFAULT_STEPS),
    )
    p_pr.add_argument(
        "--only-steps", nargs="*", default=None,
        choices=list(DEFAULT_STEPS),
    )
    p_pr.add_argument("--memory-limit-gb", type=float, default=None)
    p_pr.add_argument(
        "--track", action="store_true",
        help="Log step metrics + manifest artifact to Weights & Biases.",
    )
    p_pr.add_argument(
        "--tracking-project", default=None,
        help="W&B project (falls back to $WANDB_PROJECT, then app default).",
    )
    p_pr.add_argument(
        "--tracking-tags", nargs="*", default=None,
        help="Extra W&B tags (app/run_kind tags added automatically).",
    )
    p_pr.add_argument(
        "--no-hash", action="store_true",
        help="Skip SHA-256 hashing of feature parquets (faster, less safe).",
    )
    p_pr.add_argument(
        "--dry-run", action="store_true",
        help="Resolve paths and show readiness; do not run.",
    )

    # validate ------------------------------------------------------------
    p_va = sub.add_parser("validate", help="Parquet alignment + row check.")
    p_va.add_argument("--output-dir", required=True, type=Path)
    p_va.add_argument("--chromosomes", nargs="*", default=None)

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_list_profiles(args: argparse.Namespace) -> int:
    profiles = list_profiles()
    if args.format == "json":
        payload = [
            {
                "name": p.name,
                "path": str(p.path),
                "modalities": p.modalities,
                "base_model": p.base_model,
                "n_modalities": p.n_modalities,
            }
            for p in profiles
        ]
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print()
    print(f"{'Profile':<22} {'Base model':<14} {'Modalities':<4} Modality list")
    print("-" * 90)
    for p in profiles:
        print(
            f"{p.name:<22} {str(p.base_model or ''):<14} "
            f"{p.n_modalities:<4} {', '.join(p.modalities)}"
        )
    print()
    return 0


def _cmd_list_tracks(args: argparse.Namespace) -> int:
    tracks = list_tracks(build=args.build, modality=args.modality)

    if args.format == "json":
        payload = [
            {
                "modality": t.modality,
                "build": t.build,
                "name": t.name,
                "url": t.url,
                "accession": t.accession,
                "filename": t.filename,
                "alignment": t.alignment,
                "is_cached": t.is_cached,
                "cached_path": str(t.cached_path) if t.cached_path else None,
                "extra": t.extra,
            }
            for t in tracks
        ]
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print()
    header = f"{'Build':<8} {'Modality':<14} {'Track':<32} {'Cached':<7} Source"
    print(header)
    print("-" * len(header))
    for t in tracks:
        cached = "yes" if t.is_cached else "-"
        source = t.url or t.accession or ""
        print(
            f"{t.build:<8} {t.modality:<14} {t.name[:32]:<32} {cached:<7} "
            f"{source[:50]}"
        )
    print()
    return 0


def _cmd_fetch_tracks(args: argparse.Namespace) -> int:
    try:
        if args.modality == "conservation":
            if args.cell_lines or args.marks:
                logger.warning(
                    "--cell-lines / --marks ignored for modality=conservation"
                )
            updated = fetch_conservation_tracks(
                build=args.build, dest_dir=args.dest, force=args.force,
                verbosity=args.verbosity,
            )
        elif args.modality == "epigenetic":
            updated = fetch_encode_tracks(
                build=args.build,
                modality="epigenetic",
                cell_lines=args.cell_lines,
                marks=args.marks,
                dest_dir=args.dest,
                force=args.force,
                verbosity=args.verbosity,
            )
        else:
            logger.error("Unsupported modality: %s", args.modality)
            return 2
    except Exception as exc:  # noqa: BLE001
        logger.error("fetch-tracks failed: %s", exc)
        return 1

    for t in updated:
        print(
            f"[{t.modality}/{t.name}] "
            f"{'cached' if t.is_cached else 'failed'}  "
            f"{t.cached_path or ''}"
        )
    return 0 if all(t.is_cached for t in updated) else 1


def _cmd_status(args: argparse.Namespace) -> int:
    if args.canonical:
        if not args.build:
            logger.error("--canonical requires --build")
            return 2
        try:
            out = resolve_canonical_features_dir(
                build=args.build,
                annotation_source=args.annotation_source,
                base_model=args.base_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to resolve canonical dir: %s", exc)
            return 2
        print(
            f"[--canonical] Canonical features dir for build={args.build} "
            f"source={args.annotation_source} base_model={args.base_model}: {out}"
        )
        status = get_status(out, expected_chromosomes=args.chromosomes)
    else:
        status = get_status(args.output_dir, expected_chromosomes=args.chromosomes)

    if args.format == "json":
        payload = {
            "output_dir": str(status.output_dir),
            "manifest_present": status.manifest_present,
            "build": status.build,
            "profile": status.profile,
            "modalities": status.modalities,
            "ready": status.ready,
            "missing": status.missing,
            "expected_chromosomes": status.expected_chromosomes,
            "tracks_cached": status.tracks_cached,
            "chromosomes": {
                c: {
                    "exists": a.exists,
                    "path": str(a.path) if a.path else None,
                    "size_bytes": a.size_bytes,
                }
                for c, a in status.chromosomes.items()
            },
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(status.summary())

    return 0 if status.ready else 1


def _cmd_prepare(args: argparse.Namespace) -> int:
    if args.inplace:
        try:
            out = resolve_canonical_features_dir(
                build=args.build,
                annotation_source=args.annotation_source,
                base_model=args.base_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to resolve canonical dir: %s", exc)
            return 2
        print(
            f"[--inplace] Canonical features dir: {out}"
        )
    else:
        out = args.output_dir

    if args.dry_run:
        print()
        print("Dry run — showing readiness, no files will be written.")
        status = get_status(out, expected_chromosomes=args.chromosomes)
        print()
        print(status.summary())
        print()
        print(
            f"Planned steps: {', '.join(args.only_steps or DEFAULT_STEPS)} "
            f"(profile={args.profile}, fetch_missing_tracks="
            f"{bool(args.fetch_missing_tracks)})"
        )
        return 0

    try:
        result = prepare_features(
            profile=args.profile,
            build=args.build,
            input_dir=args.input_dir,
            output_dir=out,
            chromosomes=args.chromosomes,
            fetch_missing_tracks=bool(args.fetch_missing_tracks),
            skip_steps=args.skip_steps,
            only_steps=args.only_steps,
            resume=(not args.no_resume),
            memory_limit_gb=args.memory_limit_gb,
            hash_artifacts=(not args.no_hash),
            track=bool(args.track),
            tracking_project=args.tracking_project,
            tracking_tags=args.tracking_tags,
            verbosity=args.verbosity,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    if args.verbosity >= 1:
        print()
        print(result.summary())

    return 0 if result.success else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    result = step_validate(
        output_dir=args.output_dir,
        chromosomes=args.chromosomes,
        verbosity=args.verbosity,
    )
    if result.success:
        extras = result.extras or {}
        print(
            f"validate: ok — {extras.get('n_files', 0)} parquet(s), "
            f"{result.rows} total rows"
        )
        return 0
    print(f"validate: FAILED — {result.error}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_level(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


if __name__ == "__main__":
    raise SystemExit(main())
