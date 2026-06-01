"""CLI for the artifact registry.

Invoke as ``python -m agentic_spliceai.registry <subcommand> [...]``::

    python -m agentic_spliceai.registry build       # regenerate REGISTRY.md
    python -m agentic_spliceai.registry validate    # CI-style validation
    python -m agentic_spliceai.registry add <path>  # create a starter MANIFEST
    python -m agentic_spliceai.registry list --tag demo:ui_integration
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .builder import build_registry
from .discovery import default_output_root, load_all_manifests
from .manifest import (
    ALLOWED_STATUSES,
    MANIFEST_FILENAME,
    Manifest,
    ManifestError,
    starter_manifest,
)
from .validator import has_errors, validate

logger = logging.getLogger(__name__)


def _cmd_build(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    registry_path = Path(args.registry_path).resolve() if args.registry_path else None
    written = build_registry(output_root, registry_path)
    print(f"Wrote: {written}")
    n = sum(1 for _ in load_all_manifests(output_root))
    print(f"  {n} artifacts indexed")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    settings_yaml = Path(args.settings_yaml).resolve() if args.settings_yaml else None
    issues = validate(output_root, settings_yaml)

    if not issues:
        print("OK — no issues.")
        return 0

    for issue in issues:
        print(str(issue))
    n_err = sum(1 for i in issues if i.severity == "error")
    n_warn = sum(1 for i in issues if i.severity == "warning")
    print(f"\n{n_err} error(s), {n_warn} warning(s).")
    return 1 if has_errors(issues) else 0


def _cmd_add(args: argparse.Namespace) -> int:
    artifact_dir = Path(args.artifact_dir).resolve()
    if not artifact_dir.is_dir():
        print(f"ERROR: not a directory: {artifact_dir}", file=sys.stderr)
        return 2

    manifest_path = artifact_dir / MANIFEST_FILENAME
    if manifest_path.exists() and not args.force:
        print(f"ERROR: {manifest_path} already exists (use --force to overwrite)", file=sys.stderr)
        return 2

    manifest = starter_manifest(
        path=artifact_dir,
        status=args.status,
        produced_by=args.produced_by or [],
        notes=args.notes or "",
        tags=args.tag or [],
    )
    manifest_path.write_text(manifest.to_yaml())
    print(f"Wrote starter manifest: {manifest_path}")
    print("Edit it to fill in `notes`, `referenced_by`, etc., then run:")
    print("  python -m agentic_spliceai.registry build")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root).resolve()
    manifests = load_all_manifests(output_root)

    if args.tag:
        manifests = [m for m in manifests if args.tag in m.tags]
    if args.status:
        manifests = [m for m in manifests if m.status == args.status]
    if args.topic:
        manifests = [m for m in manifests if m.topic == args.topic]

    if not manifests:
        print("(no artifacts match)")
        return 0

    # Sort by topic then artifact name for stable output.
    manifests.sort(key=lambda m: (m.topic, m.name))
    print(f"{'TOPIC':<22} {'NAME':<40} {'STATUS':<14} TAGS")
    print("-" * 90)
    for m in manifests:
        tags = " ".join(m.tags) if m.tags else "—"
        print(f"{m.topic:<22} {m.name:<40} {m.status:<14} {tags}")
    print(f"\n{len(manifests)} artifact(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m agentic_spliceai.registry",
        description="Manage the output/ artifact registry (MANIFEST.yaml-driven).",
    )
    p.add_argument(
        "--output-root",
        default="output",
        help="Root directory containing artifact subdirs. Default: ./output",
    )
    p.add_argument("-v", "--verbose", action="store_true")

    sub = p.add_subparsers(dest="subcommand", required=True)

    # build
    pb = sub.add_parser("build", help="Regenerate output/REGISTRY.md from manifests.")
    pb.add_argument(
        "--registry-path",
        default=None,
        help="Where to write REGISTRY.md. Default: <output-root>/REGISTRY.md",
    )
    pb.set_defaults(func=_cmd_build)

    # validate
    pv = sub.add_parser("validate", help="Validate manifests + cross-check settings.yaml.")
    pv.add_argument(
        "--settings-yaml",
        default="src/agentic_spliceai/splice_engine/config/settings.yaml",
        help="settings.yaml path for cross-checking status:active models. Pass empty string to skip.",
    )
    pv.set_defaults(func=_cmd_validate)

    # add
    pa = sub.add_parser("add", help="Write a starter MANIFEST.yaml for a new artifact.")
    pa.add_argument("artifact_dir", help="Path to the artifact directory.")
    pa.add_argument(
        "--status",
        default="active",
        choices=list(ALLOWED_STATUSES),
        help="Artifact status. Default: active.",
    )
    pa.add_argument(
        "--produced-by",
        action="append",
        help="Command or script that produced this artifact. Repeat for multi-step pipelines.",
    )
    pa.add_argument("--notes", default="", help="Free-form one-paragraph description.")
    pa.add_argument("--tag", action="append", help="Tag (repeatable), e.g. --tag demo:ui_integration.")
    pa.add_argument("--force", action="store_true", help="Overwrite an existing MANIFEST.")
    pa.set_defaults(func=_cmd_add)

    # list
    pl = sub.add_parser("list", help="Print a filtered listing of artifacts.")
    pl.add_argument("--tag", help="Filter to artifacts carrying this tag.")
    pl.add_argument(
        "--status",
        choices=list(ALLOWED_STATUSES),
        help="Filter to this status.",
    )
    pl.add_argument("--topic", help="Filter to artifacts in this topic dir.")
    pl.set_defaults(func=_cmd_list)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # Normalize the empty-string-skip for settings_yaml on validate.
    if getattr(args, "settings_yaml", None) == "":
        args.settings_yaml = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
