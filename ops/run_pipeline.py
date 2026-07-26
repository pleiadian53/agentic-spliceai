#!/usr/bin/env python3
"""
GPU Task Runner — launch any script on a remote GPU via SkyPilot.

Infrastructure (GPU, cloud, volumes) is configured in ops/configs/gpu_config.yaml.
The task command is passed through after '--'.

Modes:
  (default)    Dry-run: print the generated SkyPilot config and commands
  --execute    Launch the job on a remote GPU pod
  --stage-data One-time: upload reference data to the network volume
  --local-only Run the '-- <cmd>' on THIS machine (no pod) — smoke-test locally

Usage:
    # Dry-run — see what would happen (default, no cost)
    python ops/run_pipeline.py \\
        -- python examples/meta_layer/13_evaluate_m3_novel.py \\
             --output /workspace/output/

    # Execute — launch on a remote GPU
    python ops/run_pipeline.py --execute \\
        -- python examples/meta_layer/13_evaluate_m3_novel.py \\
             --output /workspace/output/

    # Override infrastructure for one run
    python ops/run_pipeline.py --execute --gpu a100 \\
        -- python your_script.py --args

    # Reuse an existing cluster (skip provisioning + setup, saves ~3 min)
    python ops/run_pipeline.py --execute \\
        --cluster aspliceai-workspace --no-teardown \\
        -- python your_script.py --args

    # Stage data to the network volume (one-time)
    python ops/run_pipeline.py --stage-data
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Allow both `python ops/run_pipeline.py` and `python -m ops.run_pipeline` to
# resolve `from ops.gpu_runner import ...` by putting the project root (the
# parent of this ops/ package) on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = "ops/configs/gpu_config.yaml"


def _parse_args() -> tuple[argparse.Namespace, str, list[str]]:
    """Parse CLI args, splitting on '--' for the task command.

    Returns ``(args, task_command, task_argv)`` where ``task_command`` is the
    space-joined string (used to build the remote shell command) and
    ``task_argv`` is the original token list (used to exec locally without a
    shell, preserving the caller's quoting).
    """
    argv = sys.argv[1:]
    if "--" in argv:
        split_idx = argv.index("--")
        runner_args = argv[:split_idx]
        task_args = argv[split_idx + 1:]
    else:
        runner_args = argv
        task_args = []

    task_command = " ".join(task_args)

    parser = argparse.ArgumentParser(
        description="GPU Task Runner — launch any script on a remote GPU via SkyPilot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run (default)\n"
            "  %(prog)s -- python your_script.py --args\n\n"
            "  # Execute on a remote GPU\n"
            "  %(prog)s --execute -- python your_script.py --args\n\n"
            "  # Stage data to the network volume\n"
            "  %(prog)s --stage-data\n"
        ),
    )

    # Execution mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="Launch the job on a remote GPU pod")
    mode.add_argument("--stage-data", action="store_true",
                      help="One-time: upload reference data to the network volume")
    mode.add_argument("--local-only", action="store_true",
                      help="Run the '-- <cmd>' on THIS machine (no pod, no SkyPilot) "
                           "— handy for smoke-testing before spending pod time")

    # Infrastructure overrides (override gpu_config.yaml)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU type: a40, a100, h100, ... (overrides config)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dependency profile from gpu_config.yaml "
                             "(e.g., evo2, hyenadna; 'none' to skip model deps)")
    parser.add_argument("--cloud", type=str, default=None,
                        help="Cloud provider (overrides config)")
    parser.add_argument("--use-volume", action="store_true", default=None,
                        help="Read data from the network volume")
    parser.add_argument("--no-volume", action="store_true",
                        help="Upload data via file_mounts (ignore config)")
    parser.add_argument("--extra-setup", type=str, default=None,
                        help='Extra setup commands (e.g., "pip install -e ./foundation_models")')
    parser.add_argument("--extra-file-mounts", type=str, nargs="+", default=None,
                        help='Additional file mounts as "remote=local" pairs')
    parser.add_argument("--data-prefix", type=str, default=None,
                        help="Local data root directory (overrides config)")
    parser.add_argument("--data-path", type=str, default=None,
                        help='Dataset subpath, e.g. "mane/GRCh38" (overrides config)')

    # Cluster reuse
    parser.add_argument("--cluster", type=str, default=None,
                        help="Reuse an existing cluster (skip provisioning + setup)")
    parser.add_argument("--no-teardown", action="store_true",
                        help="Keep the cluster alive after the job (for iterative runs)")

    # Job settings
    parser.add_argument("--job-name", type=str, default=None,
                        help="Job name (auto-derived from the script name if omitted)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Local output directory for downloaded results")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG,
                        help=f"Path to gpu_config.yaml (default: {_DEFAULT_CONFIG})")

    args = parser.parse_args(runner_args)

    # Resolve --no-volume → use_volume=False
    if args.no_volume:
        args.use_volume = False

    return args, task_command, task_args


def _build_output_dir(args: argparse.Namespace, task_command: str) -> Path:
    """Auto-generate an output directory from the task command or use an explicit path."""
    if args.output_dir:
        return Path(args.output_dir)
    from ops.gpu_runner import _derive_job_name
    name = args.job_name or _derive_job_name(task_command) or "aspliceai-job"
    return Path("output") / "gpu_runs" / name


def _run_local_only(task_argv: list[str]) -> None:
    """Run the task command on this machine (no SkyPilot, no pod).

    A generic escape hatch: run the exact same command locally — e.g. to
    smoke-test on a 16 GB M1 before provisioning a pod. Executes the original
    argv list directly (no shell) so the caller's quoting is preserved, streams
    output, and exits with the child's return code.
    """
    if not task_argv:
        print("ERROR: --local-only needs a command after '--'. Example:")
        print("  python ops/run_pipeline.py --local-only -- python your_script.py --args")
        sys.exit(1)

    logger.info("Running locally (no pod): %s", " ".join(task_argv))
    result = subprocess.run(task_argv)
    sys.exit(result.returncode)


def main() -> None:
    args, task_command, task_argv = _parse_args()

    # Local escape hatch: run the same command here, no cloud involved.
    if args.local_only:
        _run_local_only(task_argv)
        return

    from ops.gpu_runner import (
        InfraConfig,
        build_skypilot_config,
        launch,
        print_dry_run,
        stage_data,
    )

    infra = InfraConfig.from_yaml(args.config)

    # Apply CLI overrides
    overrides: dict = {}
    if args.gpu is not None:
        overrides["gpu"] = args.gpu
    if args.model is not None:
        overrides["model"] = args.model
    if args.cloud is not None:
        overrides["cloud"] = args.cloud
    if args.use_volume is not None:
        overrides["use_volume"] = args.use_volume
    if args.extra_setup is not None:
        overrides["extra_setup"] = args.extra_setup
    if args.extra_file_mounts:
        mounts = {}
        for pair in args.extra_file_mounts:
            remote, local = pair.split("=", 1)
            mounts[remote] = local
        overrides["extra_file_mounts"] = mounts
    if args.data_prefix is not None:
        overrides["data_prefix"] = args.data_prefix
    if args.data_path is not None:
        overrides["data_path"] = args.data_path
    infra.apply_overrides(**overrides)

    # Stage data mode
    if args.stage_data:
        stage_data(infra)
        return

    # Build config from infrastructure + task command
    if not task_command:
        print("ERROR: No task command provided. Use '--' followed by the command.")
        print()
        print("Example:")
        print("  python ops/run_pipeline.py --execute \\")
        print("      -- python your_script.py --args")
        sys.exit(1)

    output_dir = _build_output_dir(args, task_command)
    config = build_skypilot_config(infra, task_command, job_name=args.job_name)

    if args.execute:
        launch(
            config, output_local=output_dir, infra=infra,
            cluster=args.cluster, teardown=not args.no_teardown,
        )
    else:
        print_dry_run(config, infra, output_dir)


if __name__ == "__main__":
    main()
