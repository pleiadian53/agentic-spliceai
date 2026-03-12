#!/usr/bin/env python3
"""
Stage Data — upload datasets and model weights to a running cluster.

Transfers local directories to a running SkyPilot cluster's network volume,
preserving the directory structure. Handles rsync + symlinks in one command
so you don't have to remember the remote paths.

The script:
  1. Resolves the cluster name (auto-detect or explicit)
  2. Rsyncs local data to the network volume on the pod
  3. Creates symlinks so scripts find data at expected paths

Usage:
    # Stage a dataset (e.g., Ensembl GRCh37 reference data)
    python examples/foundation_models/ops_stage_data.py data/ensembl/GRCh37

    # Stage model weights
    python examples/foundation_models/ops_stage_data.py data/models/spliceai

    # Stage multiple paths at once
    python examples/foundation_models/ops_stage_data.py \
        data/ensembl/GRCh37 \
        data/models/spliceai

    # Stage to a specific cluster (if multiple running)
    python examples/foundation_models/ops_stage_data.py \
        --cluster sky-d5c6-pleiadian53 \
        data/mane/GRCh38

    # Use model name shorthand for weights (resolves via resource manager)
    python examples/foundation_models/ops_stage_data.py --weights spliceai
    python examples/foundation_models/ops_stage_data.py --weights openspliceai

    # Combine: dataset + weights in one go
    python examples/foundation_models/ops_stage_data.py \
        data/ensembl/GRCh37 --weights spliceai

    # Dry run — show what would be transferred without doing it
    python examples/foundation_models/ops_stage_data.py --dry-run data/ensembl/GRCh37

    # Override volume mount point (default: /runpod-volume)
    python examples/foundation_models/ops_stage_data.py \
        --volume-mount /workspace data/my_dataset
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_VOLUME_MOUNT = "/runpod-volume"
_DEFAULT_DATA_PREFIX = "data"


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with output visible to the user."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        logger.error("Command failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)
    return result


def _run_capture(cmd: list[str]) -> str:
    """Run a command and capture stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.stdout


def _get_running_clusters() -> list[dict]:
    """Parse `sky status` to get running clusters."""
    output = _run_capture(["sky", "status"])
    clusters = []
    in_table = False

    for line in output.splitlines():
        if "NAME" in line and "STATUS" in line:
            in_table = True
            continue
        if line.startswith("-") or line.startswith("=") or not line.strip():
            if in_table and not line.strip():
                break
            continue
        if in_table:
            cols = line.split()
            if len(cols) >= 2:
                clusters.append({
                    "name": cols[0],
                    "status": cols[3] if len(cols) > 3 else "UNKNOWN",
                })
    return clusters


def _find_cluster(explicit: str | None) -> str:
    """Resolve cluster name: explicit, or auto-detect from running clusters."""
    if explicit:
        return explicit

    clusters = _get_running_clusters()
    up = [c for c in clusters if c["status"] == "UP"]

    if not up:
        logger.error("No running clusters found. Provision one first:")
        logger.error("  python examples/foundation_models/ops_provision_cluster.py")
        sys.exit(1)

    if len(up) == 1:
        name = up[0]["name"]
        logger.info("Auto-detected cluster: %s", name)
        return name

    # Multiple clusters — ask user to pick
    print("\nMultiple clusters running:")
    for i, c in enumerate(up, 1):
        print(f"  {i}. {c['name']}")
    choice = input("\nEnter number (or specify --cluster): ").strip()
    idx = int(choice) - 1
    return up[idx]["name"]


def _resolve_weights_path(model_name: str) -> Path:
    """Resolve model weights directory via the resource manager."""
    from agentic_spliceai.splice_engine.resources import get_model_resources

    resources = get_model_resources(model_name)
    registry = resources.get_registry()
    weights_dir = registry.get_model_weights_dir(model_name)

    if not weights_dir.exists():
        logger.error("Model weights not found: %s", weights_dir)
        sys.exit(1)

    files = list(weights_dir.glob("*"))
    if not files:
        logger.error("Model weights directory is empty: %s", weights_dir)
        sys.exit(1)

    logger.info("Resolved %s weights: %s (%d files)", model_name, weights_dir, len(files))
    return weights_dir


def _dir_size_mb(path: Path) -> float:
    """Get approximate directory size in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def stage_path(
    cluster: str,
    local_path: Path,
    volume_mount: str,
    data_prefix: str,
    dry_run: bool = False,
) -> None:
    """Stage a single local directory to the cluster's network volume.

    The remote layout mirrors the local layout:
        local:  data/ensembl/GRCh37/
        volume: /runpod-volume/data/ensembl/GRCh37/
        symlink: ~/sky_workdir/data/ensembl/GRCh37 -> volume path
    """
    # Compute the subpath relative to data_prefix
    # e.g., local_path = "data/ensembl/GRCh37" -> subpath = "ensembl/GRCh37"
    local_str = str(local_path)
    if local_str.startswith(f"{data_prefix}/"):
        subpath = local_str[len(data_prefix) + 1:]
    else:
        subpath = local_path.name

    volume_dest = f"{volume_mount}/{data_prefix}/{subpath}"
    workdir_link = f"~/sky_workdir/{data_prefix}/{subpath}"
    size_mb = _dir_size_mb(local_path)

    print(f"\n  Source:  {local_path.resolve()}")
    print(f"  Volume:  {volume_dest}/")
    print(f"  Link:    {workdir_link} -> {volume_dest}")
    print(f"  Size:    {size_mb:.0f} MB")

    if dry_run:
        print("  [DRY RUN] Skipping transfer.")
        return

    # Step 1: rsync to volume
    # -L: dereference symlinks (copy actual files, not symlink pointers)
    # --no-owner --no-group: cloud volumes (RunPod) don't allow chown
    _run([
        "rsync", "-Pavz", "-L", "--no-owner", "--no-group",
        str(local_path.resolve()) + "/",
        f"{cluster}:{volume_dest}/",
    ])

    # Step 2: create symlink in sky_workdir so scripts find data at expected path
    # Remove existing directory first — ln -sfn won't replace a real directory,
    # it creates the link *inside* it instead. SkyPilot workdir sync often
    # creates empty directory stubs that block symlink creation.
    link_parent = str(Path(workdir_link).parent)
    ssh_cmd = (
        f"mkdir -p {link_parent} && "
        f"[ -d {workdir_link} ] && [ ! -L {workdir_link} ] && rm -rf {workdir_link} || true && "
        f"ln -sfn {volume_dest} {workdir_link}"
    )
    _run(["ssh", cluster, ssh_cmd])

    print(f"  Staged successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage data and model weights to a running GPU cluster.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stage reference data
  %(prog)s data/ensembl/GRCh37

  # Stage model weights by name
  %(prog)s --weights spliceai
  %(prog)s --weights openspliceai

  # Stage everything for a SpliceAI run
  %(prog)s data/ensembl/GRCh37 --weights spliceai

  # Stage everything for an OpenSpliceAI run
  %(prog)s data/mane/GRCh38 --weights openspliceai
""",
    )
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="Local data directories to stage (e.g., data/ensembl/GRCh37)",
    )
    parser.add_argument(
        "--weights", type=str, action="append", default=[], metavar="MODEL",
        help="Stage model weights by name (e.g., spliceai, openspliceai). "
             "Can be specified multiple times.",
    )
    parser.add_argument(
        "--cluster", type=str, default=None,
        help="Target cluster name (auto-detects if only one running)",
    )
    parser.add_argument(
        "--volume-mount", type=str, default=_DEFAULT_VOLUME_MOUNT,
        help=f"Volume mount point on the pod (default: {_DEFAULT_VOLUME_MOUNT})",
    )
    parser.add_argument(
        "--data-prefix", type=str, default=_DEFAULT_DATA_PREFIX,
        help=f"Data prefix directory (default: {_DEFAULT_DATA_PREFIX})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be transferred without doing it",
    )

    args = parser.parse_args()

    # Collect all paths to stage
    all_paths: list[Path] = []

    for p in args.paths:
        if not p.exists():
            logger.error("Path not found: %s", p)
            sys.exit(1)
        all_paths.append(p)

    for model_name in args.weights:
        weights_path = _resolve_weights_path(model_name)
        all_paths.append(weights_path)

    if not all_paths:
        parser.print_help()
        print("\nError: specify at least one path or --weights MODEL")
        sys.exit(1)

    # Resolve cluster
    cluster = _find_cluster(args.cluster)

    print()
    print("=" * 60)
    print(f"Staging {len(all_paths)} path(s) to cluster: {cluster}")
    print("=" * 60)

    for path in all_paths:
        stage_path(
            cluster=cluster,
            local_path=path,
            volume_mount=args.volume_mount,
            data_prefix=args.data_prefix,
            dry_run=args.dry_run,
        )

    print()
    print("=" * 60)
    if args.dry_run:
        print("Dry run complete. Remove --dry-run to transfer.")
    else:
        print("All data staged successfully!")
        print()
        print(f"  SSH:  ssh {cluster}")
        print(f"  Verify:  ls {args.volume_mount}/{args.data_prefix}/")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
