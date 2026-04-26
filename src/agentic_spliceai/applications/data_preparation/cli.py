"""CLI for the data-preparation application.

Subcommands
-----------

- ``prepare``      — run the ingestion pipeline for a build into an output dir
- ``status``       — read-only readiness report for an output dir
- ``validate``     — coordinate-consistency and input-integrity checks (read-only)
- ``list-builds``  — show configured base models / builds from settings.yaml

Entry point: ``agentic-spliceai-ingest`` (see ``pyproject.toml``).

Examples
--------

Prepare a GRCh38/MANE build into a throwaway output dir::

    agentic-spliceai-ingest prepare \\
        --build GRCh38 --annotation-source mane \\
        --output-dir output/ingest/GRCh38_mane

Use user-provided FASTA / GTF::

    agentic-spliceai-ingest prepare \\
        --build GRCh38 --annotation-source mane \\
        --fasta /path/to/custom.fa --gtf /path/to/custom.gtf \\
        --output-dir output/ingest/custom

Check readiness before running the base layer::

    agentic-spliceai-ingest status --output-dir output/ingest/GRCh38_mane

Production safety
-----------------

``--output-dir`` is **required** — the CLI never writes implicitly to
resource-manager-resolved paths (e.g. ``data/mane/GRCh38/``). To write
there deliberately, pass that path explicitly AND ``--force``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .pipeline import DEFAULT_STEPS, prepare_build, resolve_canonical_output_dir
from .status import get_status
from .steps import step_validate

logger = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=_log_level(args.verbosity),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.command == "prepare":
        return _cmd_prepare(args)
    if args.command == "status":
        return _cmd_status(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "list-builds":
        return _cmd_list_builds(args)

    parser.print_help()
    return 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-spliceai-ingest",
        description=(
            "Data preparation application: ingest FASTA + GTF/GFF, derive "
            "artifacts needed by downstream applications."
        ),
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="0 = silent, 1 = info (default), 2 = debug.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # prepare --------------------------------------------------------------
    p_prep = sub.add_parser("prepare", help="Run the ingestion pipeline.")
    p_prep.add_argument("--build", required=True, help="Genome build (e.g., GRCh38).")
    p_prep.add_argument(
        "--annotation-source", required=True,
        help="Annotation source (e.g., mane, ensembl).",
    )
    # Output-dir resolution: exactly one of --output-dir or --inplace
    dest = p_prep.add_mutually_exclusive_group(required=True)
    dest.add_argument(
        "--output-dir", type=Path,
        help=(
            "Directory for all artifacts. The CLI never writes implicitly "
            "to resource-manager-resolved production paths — use --inplace "
            "if you explicitly want that."
        ),
    )
    dest.add_argument(
        "--inplace", action="store_true",
        help=(
            "Write to the resource-manager-resolved canonical dir for the "
            "given build (e.g., data/mane/GRCh38/). Existing artifacts are "
            "preserved; missing ones are gap-filled. Pass --force to "
            "regenerate existing artifacts."
        ),
    )
    p_prep.add_argument("--fasta", type=Path, default=None, help="Override FASTA path.")
    p_prep.add_argument("--gtf", type=Path, default=None, help="Override GTF/GFF path.")
    p_prep.add_argument(
        "--dry-run", action="store_true",
        help="Resolve paths and show what would run; do not write.",
    )
    p_prep.add_argument(
        "--force", action="store_true",
        help="Regenerate artifacts even if they already exist.",
    )
    p_prep.add_argument(
        "--no-hash", action="store_true",
        help="Skip SHA-256 hashing of inputs/artifacts (faster, less safe).",
    )
    p_prep.add_argument(
        "--skip-steps", nargs="*", default=None,
        choices=list(DEFAULT_STEPS),
        help="Steps to skip.",
    )
    p_prep.add_argument(
        "--only-steps", nargs="*", default=None,
        choices=list(DEFAULT_STEPS),
        help="Only run these steps.",
    )
    p_prep.add_argument(
        "--chromosome-split-strategy",
        default="spliceai", choices=("spliceai", "balanced"),
        help="Chromosome-split strategy (default: spliceai).",
    )
    p_prep.add_argument(
        "--track", action="store_true",
        help="Log step metrics + manifest artifact to Weights & Biases.",
    )
    p_prep.add_argument(
        "--tracking-project", default=None,
        help="W&B project (falls back to $WANDB_PROJECT, then app default).",
    )
    p_prep.add_argument(
        "--tracking-tags", nargs="*", default=None,
        help="Extra W&B tags (app/run_kind tags added automatically).",
    )

    # status ---------------------------------------------------------------
    p_status = sub.add_parser("status", help="Read-only readiness report.")
    # Either point at an output_dir or resolve via --build / --source
    stat_dest = p_status.add_mutually_exclusive_group(required=True)
    stat_dest.add_argument(
        "--output-dir", type=Path,
        help="Output directory previously populated by `prepare`.",
    )
    stat_dest.add_argument(
        "--canonical", action="store_true",
        help=(
            "Check the resource-manager-resolved canonical dir for "
            "--build / --annotation-source (read-only)."
        ),
    )
    p_status.add_argument(
        "--build", default=None,
        help="Required with --canonical.",
    )
    p_status.add_argument(
        "--annotation-source", default=None,
        help="Required with --canonical.",
    )
    p_status.add_argument(
        "--format", choices=("table", "json"), default="table",
        help="Output format (default: table).",
    )

    # validate -------------------------------------------------------------
    p_val = sub.add_parser(
        "validate",
        help="Read-only coordinate and input-integrity checks.",
    )
    p_val.add_argument("--build", required=True)
    p_val.add_argument("--annotation-source", required=True)
    p_val.add_argument(
        "--output-dir", required=True, type=Path,
        help="Output directory used for this build.",
    )

    # list-builds ----------------------------------------------------------
    p_list = sub.add_parser(
        "list-builds",
        help="List configured base models / builds from settings.yaml.",
    )
    p_list.add_argument(
        "--format", choices=("table", "json"), default="table",
    )

    return parser


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_prepare(args: argparse.Namespace) -> int:
    # Resolve output_dir — explicit, or derived from build for --inplace.
    if args.inplace:
        try:
            out = resolve_canonical_output_dir(
                build=args.build, annotation_source=args.annotation_source,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to resolve canonical dir: %s", exc)
            return 2
        print(
            f"[--inplace] Canonical dir for build={args.build} "
            f"source={args.annotation_source}: {out}",
            file=sys.stderr,
        )
    else:
        out = args.output_dir

    # Dry-run: just show readiness and the plan, no side effects.
    if args.dry_run:
        print()
        print("Dry run — showing readiness, no files will be written.")
        status = get_status(out)
        print()
        print(status.summary())
        print()
        print(
            f"Planned steps: {', '.join(args.only_steps or DEFAULT_STEPS)}"
            f"{' (force)' if args.force else ' (skip existing)'}"
        )
        return 0

    if not args.force and out.exists() and any(out.iterdir()):
        _guard_overwrite(out)

    try:
        result = prepare_build(
            build=args.build,
            annotation_source=args.annotation_source,
            output_dir=out,
            gtf_path=args.gtf,
            fasta_path=args.fasta,
            skip_steps=args.skip_steps,
            only_steps=args.only_steps,
            chromosome_split_strategy=args.chromosome_split_strategy,
            force=args.force,
            hash_artifacts=not args.no_hash,
            track=bool(args.track),
            tracking_project=args.tracking_project,
            tracking_tags=args.tracking_tags,
            verbosity=args.verbosity,
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    if args.verbosity >= 1:
        print()
        print(result.summary())

    return 0 if result.success else 1


def _cmd_status(args: argparse.Namespace) -> int:
    if args.canonical:
        if not args.build or not args.annotation_source:
            logger.error(
                "--canonical requires --build and --annotation-source",
            )
            return 2
        try:
            out = resolve_canonical_output_dir(
                build=args.build, annotation_source=args.annotation_source,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to resolve canonical dir: %s", exc)
            return 2
        # Send the human-readable preamble to stderr so stdout stays pure
        # JSON when --format json is used.
        print(
            f"[--canonical] Checking canonical dir for build={args.build} "
            f"source={args.annotation_source}: {out}",
            file=sys.stderr,
        )
        status = get_status(out)
    else:
        status = get_status(args.output_dir)

    if args.format == "json":
        payload = {
            "output_dir": str(status.output_dir),
            "manifest_present": status.manifest_present,
            "build": status.build,
            "annotation_source": status.annotation_source,
            "manifest_updated_at": status.manifest_updated_at,
            "ready": status.ready,
            "missing": status.missing,
            "stale": status.stale,
            "artifacts": {
                name: {
                    "path": str(a.path) if a.path else None,
                    "exists": a.exists,
                    "stale": a.stale,
                    "stale_reason": a.stale_reason,
                }
                for name, a in status.artifacts.items()
            },
        }
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
    else:
        print(status.summary())

    return 0 if status.ready else 1


def _cmd_validate(args: argparse.Namespace) -> int:
    result = step_validate(
        build=args.build,
        annotation_source=args.annotation_source,
        output_dir=args.output_dir,
        verbosity=args.verbosity,
    )
    if result.success:
        print(f"validate: ok ({result.rows} rows)")
        return 0
    print(f"validate: FAILED — {result.error}", file=sys.stderr)
    return 1


def _cmd_list_builds(args: argparse.Namespace) -> int:
    from agentic_spliceai.splice_engine.resources.model_resources import (
        get_model_info,
        list_available_models,
    )

    names = list_available_models()
    if args.format == "json":
        payload = {n: get_model_info(n) for n in names}
        json.dump(payload, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    print()
    print(f"{'Name':<26} {'Build':<10} {'Annotation':<12} Notes")
    print("-" * 78)
    for n in names:
        info = get_model_info(n)
        print(
            f"{n:<26} "
            f"{str(info.get('training_build', '')):<10} "
            f"{str(info.get('annotation_source', '')):<12} "
            f"{(info.get('notes') or '').splitlines()[0][:50]}"
        )
    print()
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guard_overwrite(out: Path) -> None:
    """Warn (once) when the output dir already contains files and
    ``--force`` was not supplied. Non-blocking: individual steps still
    decide per-artifact.
    """
    logger.info(
        "Output dir %s already exists and is non-empty. Existing artifacts "
        "will be preserved (use --force to regenerate).",
        out,
    )


def _log_level(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


if __name__ == "__main__":
    raise SystemExit(main())
