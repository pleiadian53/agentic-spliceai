"""Generic GPU task runner for SkyPilot.

Separates infrastructure (GPU, cloud, volumes) from task commands.
The runner builds SkyPilot YAML configs and handles launch/download/teardown.

Usage from Python::

    from foundation_models.gpu_runner import InfraConfig, build_skypilot_config, launch

    infra = InfraConfig.from_yaml("foundation_models/configs/gpu_config.yaml")
    config = build_skypilot_config(infra, run_command="python my_script.py --arg val")
    launch(config, output_local=Path("./output/my_run/"))

Usage from CLI: see ``examples/foundation_models/05_run_pipeline.py``.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU specs and pricing
# ---------------------------------------------------------------------------

GPU_SPECS = {
    "a40": {
        "accelerator": "A40:1",
        "vram_gb": 48,
        "hourly_rate": 0.39,
        "hardware_profile": "a40-48gb",
        "label": "NVIDIA A40 48 GB",
    },
    "a100": {
        "accelerator": "A100-80GB:1",
        "vram_gb": 80,
        "hourly_rate": 1.64,
        "hardware_profile": "a100-80gb",
        "label": "NVIDIA A100 80 GB",
    },
    "h100": {
        "accelerator": "H100-80GB:1",
        "vram_gb": 80,
        "hourly_rate": 3.29,
        "hardware_profile": "h100-80gb",
        "label": "NVIDIA H100 80 GB",
    },
}

# ---------------------------------------------------------------------------
# Infrastructure config
# ---------------------------------------------------------------------------

_DEFAULT_DOCKER_IMAGE = "docker:nvcr.io/nvidia/pytorch:25.02-py3"
_DEFAULT_VOLUME_NAME = "AI lab extension"
_DEFAULT_VOLUME_MOUNT = "/runpod-volume"
_DEFAULT_OUTPUT_REMOTE = "/workspace/output"


@dataclass
class InfraConfig:
    """Infrastructure settings for remote GPU jobs.

    Data paths are composed from two fields:
      - ``data_prefix``: local root directory (default ``"data"``). This is the
        top-level directory that may be a symlink on the local machine.
      - ``data_path``: dataset-specific subpath (default ``"mane/GRCh38"``).

    Together they form the full local path ``{data_prefix}/{data_path}`` and the
    volume path ``{volume_mount}/{data_prefix}/{data_path}``.
    """

    gpu: str = "a40"
    cloud: str = "runpod"
    docker_image: str = _DEFAULT_DOCKER_IMAGE
    use_volume: bool = False
    volume_name: str = _DEFAULT_VOLUME_NAME
    volume_mount: str = _DEFAULT_VOLUME_MOUNT
    data_prefix: str = "data"
    data_path: str = "mane/GRCh38"
    extra_setup: str = ""
    extra_file_mounts: dict[str, str] = field(default_factory=dict)
    output_remote: str = _DEFAULT_OUTPUT_REMOTE

    @property
    def local_data_dir(self) -> str:
        """Full local path: ``{data_prefix}/{data_path}``."""
        return f"{self.data_prefix}/{self.data_path}"

    @property
    def volume_data_dir(self) -> str:
        """Full volume path: ``{volume_mount}/{data_prefix}/{data_path}``."""
        return f"{self.volume_mount}/{self.data_prefix}/{self.data_path}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> InfraConfig:
        """Load infrastructure config from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config file not found: %s — using defaults", path)
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        return cls(
            gpu=data.get("gpu", cls.gpu),
            cloud=data.get("cloud", cls.cloud),
            docker_image=data.get("docker_image", cls.docker_image),
            use_volume=data.get("use_volume", cls.use_volume),
            volume_name=data.get("volume_name", cls.volume_name),
            volume_mount=data.get("volume_mount", cls.volume_mount),
            data_prefix=data.get("data_prefix", cls.data_prefix),
            data_path=data.get("data_path", cls.data_path),
            extra_setup=data.get("extra_setup", cls.extra_setup),
            extra_file_mounts=data.get("extra_file_mounts", None) or {},
            output_remote=data.get("output_remote", cls.output_remote),
        )

    def apply_overrides(self, **kwargs: object) -> None:
        """Apply CLI overrides (only non-None values)."""
        for key, value in kwargs.items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)


# ---------------------------------------------------------------------------
# SkyPilot config builder
# ---------------------------------------------------------------------------


def _derive_job_name(run_command: str) -> str:
    """Derive a job name from the script path in the run command."""
    # Extract script name from "python path/to/script.py ..."
    match = re.search(r"python\s+\S*?(\w+)\.py", run_command)
    if match:
        name = match.group(1)
        # Strip numeric prefix (e.g., "03_embedding_extraction" -> "embedding-extraction")
        name = re.sub(r"^\d+_?", "", name)
        name = name.replace("_", "-")
        return f"fm-{name}"
    return "fm-job"


def build_skypilot_config(
    infra: InfraConfig,
    run_command: str,
    job_name: Optional[str] = None,
) -> dict:
    """Build a complete SkyPilot config dict from infrastructure + run command.

    Args:
        infra: Infrastructure configuration.
        run_command: The shell command to execute on the remote pod.
        job_name: Job name (auto-derived from run_command if omitted).

    Returns:
        A dict ready to be dumped as SkyPilot YAML.
    """
    if not job_name:
        job_name = _derive_job_name(run_command)

    gpu = GPU_SPECS[infra.gpu]

    # Base setup: install project packages
    setup_lines = [
        "set -e",
        "pip install -e .",
        "pip install -e ./foundation_models",
    ]
    if infra.extra_setup:
        setup_lines.append(infra.extra_setup)

    config: dict = {
        "name": job_name,
        "workdir": ".",
        "resources": {
            "accelerators": gpu["accelerator"],
            "cloud": infra.cloud,
            "image_id": infra.docker_image,
        },
        "setup": "\n".join(setup_lines),
    }

    # Data source: network volume (fast) or file_mounts upload (slow)
    # Scripts expect data at {data_prefix}/{data_path} relative to CWD.
    # The local project may have data_prefix as a symlink which SkyPilot's
    # workdir sync copies as a broken symlink. Remove before mkdir.
    data_local = infra.local_data_dir  # e.g. "data/mane/GRCh38"
    data_parent = str(Path(data_local).parent)  # e.g. "data/mane"
    prefix = infra.data_prefix  # e.g. "data"

    run_lines = ["set -e", f"[ -L {prefix} ] && rm -f {prefix}", f"mkdir -p {data_parent}"]

    if infra.use_volume:
        config["volumes"] = {infra.volume_mount: infra.volume_name}
        run_lines.append(f"ln -sfn {infra.volume_data_dir} {data_local}")
        run_lines.append("echo 'Using network volume data:'")
        run_lines.append(f"ls {data_local}/ | head -5")
    else:
        file_mounts = {"/workspace/data": f"./{data_local}"}
        run_lines.append(f"ln -sfn /workspace/data {data_local}")
        config["file_mounts"] = file_mounts

    # Extra file mounts (e.g., embeddings for training)
    if infra.extra_file_mounts:
        if "file_mounts" not in config:
            config["file_mounts"] = {}
        config["file_mounts"].update(infra.extra_file_mounts)

    # Output directory
    run_lines.append(f"mkdir -p {infra.output_remote}")
    run_lines.append("")

    # User's task command
    run_lines.append(run_command)
    run_lines.append("")

    # Completion banner
    run_lines.extend([
        'echo ""',
        'echo "============================================"',
        'echo "DONE — download results before tearing down:"',
        f'echo "  rsync -Pavz {job_name}:{infra.output_remote}/ ./output/"',
        f'echo "  sky down {job_name} -y"',
        'echo "============================================"',
    ])

    config["run"] = "\n".join(run_lines)
    return config


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def _find_cluster_name(requested_name: str) -> str:
    """Find the actual SkyPilot cluster name matching our requested job name.

    SkyPilot may prefix/hash the name we put in the YAML config.
    Falls back to the requested name if no match is found.
    """
    result = subprocess.run(
        ["sky", "status", "--refresh"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        logger.warning("sky status failed — using requested name: %s", requested_name)
        return requested_name

    # Parse sky status output: look for a cluster name containing our requested name
    for line in result.stdout.splitlines():
        # sky status table lines have cluster name as first column
        columns = line.split()
        if columns and requested_name in columns[0]:
            return columns[0]

    # Exact match not found — try the requested name directly
    return requested_name


def _write_config(config: dict, job_name: str) -> Path:
    """Write SkyPilot config to generated/ directory."""
    config_dir = Path("foundation_models/configs/skypilot/generated")
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / f"{job_name}.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote SkyPilot config: %s", yaml_path)
    return yaml_path


def print_dry_run(config: dict, infra: InfraConfig, output_dir: Path) -> None:
    """Print the generated SkyPilot config and commands."""
    job_name = config["name"]
    gpu = GPU_SPECS[infra.gpu]

    print()
    print("=" * 70)
    print("GPU Task Runner — Dry Run")
    print("=" * 70)
    print()

    print("Generated SkyPilot Config")
    print("-" * 40)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    print("Commands to Execute")
    print("-" * 40)
    print(f"  # 1. Launch")
    print(f"  sky launch <config>.yaml -y")
    print()
    print(f"  # 2. Download results")
    print(f"  rsync -Pavz {job_name}:{infra.output_remote}/ {output_dir}/")
    print()
    print(f"  # 3. Tear down")
    print(f"  sky down {job_name} -y")
    print()

    print("Cost Estimate")
    print("-" * 40)
    rate = gpu["hourly_rate"]
    print(f"  GPU:   {gpu['label']} (${rate:.2f}/hr)")
    print()
    print("To execute, re-run with --execute")
    print()


def launch(
    config: dict,
    output_local: Path,
    infra: InfraConfig,
) -> None:
    """Launch a SkyPilot job, download results, and tear down."""
    job_name = config["name"]
    gpu = GPU_SPECS[infra.gpu]

    yaml_path = _write_config(config, job_name)
    output_local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Launch
    print()
    print("=" * 70)
    print(f"Launching: {job_name} ({gpu['label']})")
    print("=" * 70)
    print()

    result = subprocess.run(["sky", "launch", str(yaml_path), "-y"], check=False)
    if result.returncode != 0:
        logger.error("sky launch failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    # Find the actual cluster name (SkyPilot may prefix/hash our requested name)
    cluster_name = _find_cluster_name(job_name)

    # Step 2: Download results
    logger.info("Downloading results...")
    rsync_result = subprocess.run(
        ["rsync", "-Pavz", f"{cluster_name}:{infra.output_remote}/", str(output_local) + "/"],
        check=False,
    )
    if rsync_result.returncode != 0:
        logger.error("rsync failed — pod may still be running. Download manually:")
        logger.error("  rsync -Pavz %s:%s/ %s/", cluster_name, infra.output_remote, output_local)
        sys.exit(rsync_result.returncode)

    # Step 3: Tear down
    logger.info("Tearing down pod (cluster: %s)...", cluster_name)
    subprocess.run(["sky", "down", cluster_name, "-y"], check=False)

    elapsed = time.time() - t0
    rate = gpu["hourly_rate"]
    hours = elapsed / 3600

    print()
    print("=" * 70)
    print("Job Complete")
    print("=" * 70)
    print(f"  Output:     {output_local}")
    print(f"  GPU:        {gpu['label']}")
    print(f"  Duration:   {elapsed / 60:.1f} min")
    print(f"  Est. cost:  ${rate * hours:.2f} ({rate:.2f}/hr x {hours:.2f} hr)")
    print()


def stage_data(infra: InfraConfig) -> None:
    """One-time: upload reference data to the network volume."""
    local_data = Path(infra.local_data_dir)
    if not local_data.exists():
        logger.error("Local data not found: %s", local_data)
        logger.error("Ensure data exists at '%s' before staging.", infra.local_data_dir)
        sys.exit(1)

    job_name = "stage-data"
    gpu = GPU_SPECS[infra.gpu]

    config = {
        "name": job_name,
        "resources": {
            "accelerators": gpu["accelerator"],
            "cloud": infra.cloud,
            "image_id": infra.docker_image,
        },
        "volumes": {
            infra.volume_mount: infra.volume_name,
        },
        "file_mounts": {
            "/tmp/upload-data": str(local_data),
        },
        "run": (
            f"set -e\n"
            f"echo 'Staging reference data to network volume...'\n"
            f"mkdir -p {infra.volume_data_dir}\n"
            f"rsync -av --progress /tmp/upload-data/ {infra.volume_data_dir}/\n"
            f"echo ''\n"
            f"echo 'Data staged. Contents:'\n"
            f"ls -lh {infra.volume_data_dir}/\n"
            f"du -sh {infra.volume_data_dir}\n"
            f"echo ''\n"
            f"echo '============================================'\n"
            f"echo 'DONE — tear down staging pod:'\n"
            f"echo '  sky down {job_name} -y'\n"
            f"echo '============================================'"
        ),
    }

    yaml_path = _write_config(config, job_name)

    print()
    print("=" * 70)
    print("Staging Reference Data to Network Volume")
    print("=" * 70)
    print(f"  Volume:  {infra.volume_name}")
    print(f"  Target:  {infra.volume_data_dir}")
    print(f"  Source:  {local_data.resolve()} (~10 GB)")
    print()
    print("This uploads data once. Future runs use --use-volume to skip re-upload.")
    print()

    result = subprocess.run(["sky", "launch", str(yaml_path), "-y"], check=False)
    if result.returncode != 0:
        logger.error("sky launch failed (exit code %d)", result.returncode)
        sys.exit(result.returncode)

    # Find the actual cluster name (SkyPilot may prefix/hash our requested name)
    cluster_name = _find_cluster_name(job_name)

    logger.info("Tearing down staging pod (cluster: %s)...", cluster_name)
    subprocess.run(["sky", "down", cluster_name, "-y"], check=False)

    print()
    print("Data staged successfully. Future runs:")
    print("  python examples/foundation_models/05_run_pipeline.py --execute \\")
    print("      --use-volume -- python your_script.py --your-args")
    print()
