"""Command-line interface for the base-layer application.

Subcommands
-----------

- ``list-predictors`` — show all registered predictors
- ``predict``         — run a predictor over genes or chromosomes
- ``evaluate``        — run predict + benchmark against annotations

Entry point: ``agentic-spliceai-base`` (see ``pyproject.toml``).

Examples
--------

    agentic-spliceai-base list-predictors

    agentic-spliceai-base predict \\
        --predictor openspliceai --genes BRCA1 TP53 \\
        --output-dir output/app_base_layer/brca1_tp53

    agentic-spliceai-base evaluate \\
        --predictor openspliceai --chromosomes 21 22 \\
        --output-dir output/app_base_layer/chr21_22 \\
        --track --tracking-project agentic-spliceai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .evaluator import evaluate_predictor
from .registry import list_predictors
from .runner import run_prediction

logger = logging.getLogger(__name__)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=_log_level(args.verbosity),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.command == "list-predictors":
        return _cmd_list_predictors(args)
    if args.command == "predict":
        return _cmd_predict(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)

    parser.print_help()
    return 2


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentic-spliceai-base",
        description=(
            "Base-layer application: pluggable splice-site prediction. "
            "Packaged version of examples/base_layer/ workflows."
        ),
    )
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1,
        help="0 = silent, 1 = info (default), 2 = debug.",
    )

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # list-predictors ------------------------------------------------------
    p_list = sub.add_parser(
        "list-predictors",
        help="List registered base predictors.",
    )
    p_list.add_argument(
        "--format", choices=("table", "json"), default="table",
        help="Output format (default: table).",
    )

    # predict --------------------------------------------------------------
    p_predict = sub.add_parser(
        "predict",
        help="Run a predictor over genes or chromosomes.",
    )
    _add_common_prediction_args(p_predict)

    # evaluate -------------------------------------------------------------
    p_eval = sub.add_parser(
        "evaluate",
        help="Predict and benchmark against annotated splice sites.",
    )
    _add_common_prediction_args(p_eval)
    p_eval.add_argument(
        "--track", action="store_true",
        help="Log metrics and artifacts to Weights & Biases.",
    )
    p_eval.add_argument(
        "--tracking-project", default=None,
        help="W&B project (falls back to $WANDB_PROJECT, then a default).",
    )
    p_eval.add_argument(
        "--tracking-tags", nargs="*", default=None,
        help="Optional W&B run tags (e.g., chr21 baseline).",
    )

    return parser


def _add_common_prediction_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--predictor", required=True,
        help="Registered predictor name (see list-predictors).",
    )
    group = sub.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--genes", nargs="+",
        help="Gene symbols to predict (e.g., BRCA1 TP53).",
    )
    group.add_argument(
        "--chromosomes", nargs="+",
        help="Chromosome identifiers (with or without 'chr' prefix).",
    )
    sub.add_argument(
        "--threshold", type=float, default=0.5,
        help="Splice-site probability threshold (default: 0.5).",
    )
    sub.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to persist predictions and manifest.",
    )
    sub.add_argument(
        "--skip-preflight", action="store_true",
        help=(
            "Skip the data-preparation readiness check. The pre-flight "
            "validates that gene_features + splice_sites exist for the "
            "predictor's build/annotation_source and warns otherwise."
        ),
    )
    sub.add_argument(
        "--strict-preflight", action="store_true",
        help=(
            "Refuse to run when pre-flight reports missing artifacts. "
            "Default is to warn and proceed."
        ),
    )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def _cmd_list_predictors(args: argparse.Namespace) -> int:
    entries = list_predictors()

    if args.format == "json":
        json.dump(entries, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return 0

    # Table format
    if not entries:
        print("No predictors registered.")
        return 0

    print()
    print(f"{'Predictor':<20} {'Build':<10} {'Annotation':<12} Notes")
    print("-" * 70)
    for e in entries:
        print(
            f"{str(e.get('name', '')):<20} "
            f"{str(e.get('training_build', '')):<10} "
            f"{str(e.get('annotation_source', '')):<12} "
            f"{str(e.get('notes', ''))[:70]}"
        )
    print()
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    if _run_preflight(args) == "abort":
        return 2
    result = run_prediction(
        predictor_name=args.predictor,
        genes=args.genes,
        chromosomes=args.chromosomes,
        threshold=args.threshold,
        output_dir=args.output_dir,
        verbosity=args.verbosity,
    )
    if not result.success:
        logger.error("Prediction failed: %s", result.error)
        return 1
    if args.verbosity >= 1:
        print(result)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    if _run_preflight(args) == "abort":
        return 2
    result = evaluate_predictor(
        predictor_name=args.predictor,
        genes=args.genes,
        chromosomes=args.chromosomes,
        threshold=args.threshold,
        output_dir=args.output_dir,
        track=bool(args.track),
        tracking_project=args.tracking_project,
        tracking_tags=args.tracking_tags,
        verbosity=args.verbosity,
    )
    if not result.success:
        logger.error("Evaluation failed: %s", result.error)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Pre-flight: data-preparation readiness check
# ---------------------------------------------------------------------------


def _run_preflight(args: argparse.Namespace) -> str:
    """Check data-preparation readiness for the chosen predictor's build.

    Returns ``"abort"`` if ``--strict-preflight`` is set and artifacts are
    missing; otherwise returns ``"ok"`` (with warnings printed for the
    user).

    The check is read-only and safe to call repeatedly. It is a hint to
    the user, not a barrier: base predictors lazy-load the needed
    annotations at run-time, so missing derived artifacts do not always
    block prediction — but evaluation results are meaningless without
    ``splice_sites_enhanced.tsv``.
    """
    if getattr(args, "skip_preflight", False):
        return "ok"

    # Look up predictor config (training_build + annotation_source).
    try:
        from .registry import get_predictor

        predictor = get_predictor(args.predictor)
        build = predictor.training_build
        source = predictor.annotation_source
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Pre-flight: could not resolve predictor metadata (%s). Skipping.",
            exc,
        )
        return "ok"

    # Resolve the canonical data-prep dir and inspect it.
    try:
        from agentic_spliceai.applications.data_preparation import (
            get_status,
            resolve_canonical_output_dir,
        )

        canonical = resolve_canonical_output_dir(
            build=build, annotation_source=source,
        )
        status = get_status(canonical)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Pre-flight: data-preparation status check failed (%s). "
            "Proceeding anyway — base predictors can lazy-load annotations.",
            exc,
        )
        return "ok"

    if status.ready:
        if args.verbosity >= 1:
            logger.info(
                "Pre-flight: data-preparation ready at %s (build=%s, source=%s)",
                status.output_dir, build, source,
            )
        return "ok"

    # Report what's missing / stale.
    _print_preflight_warning(
        predictor_name=args.predictor,
        build=build, source=source,
        status=status,
    )

    if getattr(args, "strict_preflight", False):
        logger.error(
            "Pre-flight failed and --strict-preflight is set. Aborting."
        )
        return "abort"
    return "ok"


def _print_preflight_warning(
    *, predictor_name: str, build: str, source: str, status,
) -> None:
    msg_lines = [
        "",
        "⚠  Pre-flight warning — data preparation is incomplete.",
        f"   Predictor:      {predictor_name} (build={build}, source={source})",
        f"   Canonical dir:  {status.output_dir}",
    ]
    if status.missing:
        msg_lines.append(f"   Missing:        {status.missing}")
    if status.stale:
        msg_lines.append(f"   Stale:          {status.stale}")
    msg_lines += [
        "",
        "   Base predictions may still work (annotations lazy-load), but",
        "   evaluation requires splice_sites_enhanced.tsv. To gap-fill:",
        f"     agentic-spliceai-ingest prepare --inplace \\",
        f"         --build {build} --annotation-source {source}",
        "",
        "   Bypass with --skip-preflight, fail-fast with --strict-preflight.",
        "",
    ]
    # Use print (not logger) so warnings land on stdout for humans.
    print("\n".join(msg_lines))


def _log_level(verbosity: int) -> int:
    if verbosity <= 0:
        return logging.WARNING
    if verbosity == 1:
        return logging.INFO
    return logging.DEBUG


if __name__ == "__main__":
    raise SystemExit(main())
