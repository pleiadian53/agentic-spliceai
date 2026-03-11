#!/usr/bin/env python3
"""
Provision a GPU Cluster — acquire a pod, set up the environment, keep it alive.

Creates a reusable cluster for interactive SSH work and iterative experiment runs.
The cluster stays alive until you explicitly tear it down.

Optionally stages reference data to the network volume on first use. Subsequent
provisions validate that data exists and warn if missing.

Workflow:
    1. First time:  python examples/foundation_models/ops_provision_cluster.py --stage-data
    2. After that:  python examples/foundation_models/ops_provision_cluster.py
    3. SSH in:      ssh <cluster-name>
    4. Run jobs:    python examples/foundation_models/ops_run_pipeline.py --execute \\
                        --cluster <cluster-name> --no-teardown \\
                        -- python your_script.py --args
    5. Tear down:   sky down <cluster-name> -y

Usage:
    # Provision with defaults (A40, RunPod, network volume)
    python examples/foundation_models/ops_provision_cluster.py

    # First time: provision AND upload local data to network volume
    python examples/foundation_models/ops_provision_cluster.py --stage-data

    # Stage a different dataset (e.g., Ensembl GRCh37)
    python examples/foundation_models/ops_provision_cluster.py --stage-data \\
        --data-path ensembl/GRCh37

    # Provision with specific GPU
    python examples/foundation_models/ops_provision_cluster.py --gpu a100

    # Show running clusters
    python examples/foundation_models/ops_provision_cluster.py --status

    # Tear down a cluster
    python examples/foundation_models/ops_provision_cluster.py --down

    # Tear down ALL clusters
    python examples/foundation_models/ops_provision_cluster.py --down-all
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "foundation_models/configs/gpu_config.yaml"


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
    headers = []

    for line in output.splitlines():
        # Detect header row
        if "NAME" in line and "STATUS" in line:
            headers = line.split()
            in_table = True
            continue
        # Skip separator lines
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
                    "raw": line.strip(),
                })
    return clusters


def cmd_status() -> None:
    """Show running clusters."""
    clusters = _get_running_clusters()
    if not clusters:
        print("\nNo running clusters found.")
        print("  Provision one: python examples/foundation_models/ops_provision_cluster.py")
        print()
        return

    print()
    print("Running Clusters")
    print("-" * 60)
    for c in clusters:
        print(f"  {c['name']:40s} {c['status']}")
    print()
    print("SSH into a cluster:")
    print(f"  ssh {clusters[0]['name']}")
    print()
    print("Run a job on an existing cluster:")
    print(f"  python examples/foundation_models/ops_run_pipeline.py --execute \\")
    print(f"      --cluster {clusters[0]['name']} --no-teardown \\")
    print(f"      -- python your_script.py --args")
    print()


def cmd_down(cluster_name: str | None = None) -> None:
    """Tear down a specific cluster or prompt for selection."""
    clusters = _get_running_clusters()
    if not clusters:
        print("No running clusters to tear down.")
        return

    if cluster_name:
        _run(["sky", "down", cluster_name, "-y"])
        return

    if len(clusters) == 1:
        name = clusters[0]["name"]
        print(f"Tearing down: {name}")
        _run(["sky", "down", name, "-y"])
    else:
        print("Multiple clusters running:")
        for i, c in enumerate(clusters, 1):
            print(f"  {i}. {c['name']} ({c['status']})")
        choice = input("\nEnter number to tear down (or 'all'): ").strip()
        if choice.lower() == "all":
            for c in clusters:
                _run(["sky", "down", c["name"], "-y"], check=False)
        else:
            idx = int(choice) - 1
            _run(["sky", "down", clusters[idx]["name"], "-y"])


def cmd_down_all() -> None:
    """Tear down all clusters."""
    _run(["sky", "down", "-a", "-y"])


def cmd_provision(args: argparse.Namespace) -> None:
    """Provision a new cluster, optionally staging data to the volume."""
    from foundation_models.gpu_runner import GPU_SPECS, InfraConfig, _write_config

    infra = InfraConfig.from_yaml(args.config)
    if args.gpu:
        infra.gpu = args.gpu
    if args.model:
        infra.model = args.model
    if args.data_path:
        infra.data_path = args.data_path

    gpu = GPU_SPECS[infra.gpu]
    stage_data = args.stage_data
    job_name = "fm-workspace"

    # Validate local data exists if staging
    local_data = Path(infra.local_data_dir)
    if stage_data and not local_data.exists():
        logger.error("Local data not found: %s", local_data)
        logger.error(
            "Ensure data exists at '%s' before staging.\n"
            "  Generate with: agentic-spliceai-prepare --output %s",
            infra.local_data_dir, infra.local_data_dir,
        )
        sys.exit(1)

    # Check if a cluster is already running
    clusters = _get_running_clusters()
    existing = [c for c in clusters if c["status"] == "UP"]
    if existing:
        print()
        print("Existing cluster(s) found:")
        for c in existing:
            print(f"  {c['name']} ({c['status']})")
        print()
        print(f"SSH:  ssh {existing[0]['name']}")
        print()
        print("To run a job:")
        print(f"  python examples/foundation_models/ops_run_pipeline.py --execute \\")
        print(f"      --cluster {existing[0]['name']} --no-teardown \\")
        print(f"      -- python your_script.py --args")
        print()
        answer = input("Provision another cluster? [y/N] ").strip().lower()
        if answer != "y":
            return

    # Build setup: install packages, use volume pip cache if available
    setup_lines = ["set -e"]
    if infra.use_volume:
        pip_cache = f"{infra.volume_mount}/pip-cache"
        setup_lines.append(f"export PIP_CACHE_DIR={pip_cache}")
        setup_lines.append(f"mkdir -p {pip_cache}")
    setup_lines.extend([
        "pip install -e .",
        "pip install -e ./foundation_models",
    ])
    # Model dependency: conditional install (skip if already cached)
    model_pip = infra.model_pip
    if model_pip:
        for pkg in model_pip.split():
            setup_lines.append(f"pip show {pkg} >/dev/null 2>&1 || pip install {pkg}")
    if infra.extra_setup:
        setup_lines.append(infra.extra_setup)

    # Data symlink + validation
    data_local = infra.local_data_dir
    data_parent = str(Path(data_local).parent)
    prefix = infra.data_prefix

    run_lines = [
        "set -e",
        f"[ -L {prefix} ] && rm -f {prefix} || true",
        f"mkdir -p {data_parent}",
    ]

    # Stage data: upload local data to network volume via file_mounts
    if stage_data:
        run_lines.extend([
            "",
            "echo 'Staging reference data to network volume...'",
            f"mkdir -p {infra.volume_data_dir}",
            f"rsync -av --progress /tmp/upload-data/ {infra.volume_data_dir}/",
            "echo 'Data staged successfully.'",
            "",
        ])

    if infra.use_volume:
        run_lines.append(f"ln -sfn {infra.volume_data_dir} {data_local}")

    # Validate data exists on the volume (warn, don't fail)
    run_lines.extend([
        "",
        f"if [ ! -d {data_local} ] || [ -z \"$(ls -A {data_local}/ 2>/dev/null)\" ]; then",
        "  echo ''",
        "  echo 'WARNING: No data found at " + data_local + "/'",
        "  echo '  Stage data with: python examples/foundation_models/"
        "ops_provision_cluster.py --stage-data'",
        "  echo ''",
        "fi",
    ])

    run_lines.extend([
        f"mkdir -p {infra.output_remote}",
        "",
        "echo ''",
        "echo '============================================'",
        "echo 'Cluster provisioned and ready!'",
        "echo '============================================'",
        "echo ''",
        "echo 'GPU:'",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        "echo ''",
        f"echo 'Data:     {data_local}/'",
        f"ls {data_local}/ 2>/dev/null | head -5 || echo '  (empty)'",
        "echo ''",
        f"echo 'Output:   {infra.output_remote}/'",
        "echo ''",
        "echo 'Disk usage:'",
        f"du -sh {infra.volume_mount}/ 2>/dev/null || echo '  (no volume)'",
        "df -h /workspace/ 2>/dev/null | tail -1 || true",
        "echo ''",
    ])

    config: dict = {
        "name": job_name,
        "workdir": ".",
        "resources": {
            "accelerators": gpu["accelerator"],
            "cloud": infra.cloud,
            "image_id": infra.docker_image,
        },
        "setup": "\n".join(setup_lines),
        "run": "\n".join(run_lines),
    }

    # Forward HF_TOKEN from local environment if available.
    # Enables authenticated HuggingFace downloads (faster, no rate limits).
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        # Try loading from .env file in project root
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    hf_token = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if hf_token:
        config["envs"] = {"HF_TOKEN": hf_token}
        logger.info("HF_TOKEN found — will be available on cluster")

    if infra.use_volume:
        config["volumes"] = {infra.volume_mount: infra.volume_name}

    # If staging, upload local data via file_mounts
    if stage_data:
        config.setdefault("file_mounts", {})
        config["file_mounts"]["/tmp/upload-data"] = str(local_data)

    yaml_path = _write_config(config, job_name)

    print()
    print("=" * 60)
    print(f"Provisioning Cluster: {gpu['label']}")
    print(f"  Cloud:   {infra.cloud}")
    print(f"  GPU:     {gpu['label']} (${gpu['hourly_rate']:.2f}/hr)")
    print(f"  Model:   {infra.resolved_model}" + (f" (pip: {infra.model_pip})" if infra.model_pip else " (no deps)"))
    print(f"  Volume:  {infra.volume_name}" if infra.use_volume else "  Volume:  none")
    print(f"  Data:    {infra.local_data_dir}")
    if stage_data:
        print(f"  Staging: {local_data.resolve()} -> {infra.volume_data_dir}")
    print(f"  Config:  {yaml_path}")
    print("=" * 60)
    print()

    # SkyPilot keeps cluster alive by default (no --down flag).
    # The cluster persists until explicit `sky down`.
    _run(["sky", "launch", str(yaml_path), "-y"])

    # Find the cluster name
    clusters = _get_running_clusters()
    cluster_name = None
    for c in clusters:
        if job_name in c["name"]:
            cluster_name = c["name"]
            break
    if not cluster_name and clusters:
        cluster_name = clusters[-1]["name"]

    print()
    print("=" * 60)
    print("Cluster Ready!")
    print("=" * 60)
    print()
    if cluster_name:
        print(f"  Cluster:  {cluster_name}")
        print()
        print(f"  SSH:      ssh {cluster_name}")
        print()
        print("  Run a job:")
        print(f"    python examples/foundation_models/ops_run_pipeline.py --execute \\")
        print(f"        --cluster {cluster_name} --no-teardown \\")
        print(f"        -- python your_script.py --args")
        print()
        print("  Tear down when done:")
        print(f"    sky down {cluster_name} -y")
        print(f"    # or: python examples/foundation_models/ops_provision_cluster.py --down")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision and manage GPU clusters for interactive work.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true",
                        help="Show running clusters")
    action.add_argument("--down", nargs="?", const="__auto__", metavar="CLUSTER",
                        help="Tear down a cluster (auto-selects if only one)")
    action.add_argument("--down-all", action="store_true",
                        help="Tear down ALL clusters")

    parser.add_argument("--gpu", type=str, default=None, choices=["a40", "a100", "h100"],
                        help="GPU type (default: from gpu_config.yaml)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dependency profile from gpu_config.yaml "
                             "(e.g., evo2, hyenadna; 'none' to skip model deps)")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG,
                        help=f"Config file (default: {_DEFAULT_CONFIG})")
    parser.add_argument("--stage-data", action="store_true",
                        help="Upload local reference data to network volume during provisioning")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Dataset subpath, e.g. 'mane/GRCh38', 'ensembl/GRCh37' "
                             "(default: from gpu_config.yaml)")

    args = parser.parse_args()

    if args.status:
        cmd_status()
    elif args.down_all:
        cmd_down_all()
    elif args.down is not None:
        cluster = args.down if args.down != "__auto__" else None
        cmd_down(cluster)
    else:
        cmd_provision(args)


if __name__ == "__main__":
    main()
