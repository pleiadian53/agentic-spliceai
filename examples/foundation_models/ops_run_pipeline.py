#!/usr/bin/env python3
"""
GPU Task Runner — Launch any script on a remote GPU via SkyPilot.

Infrastructure (GPU, cloud, volumes) is configured in gpu_config.yaml.
The task command is passed through after '--'.

Modes:
  (default)    Dry-run: print generated SkyPilot config and commands
  --execute    Launch the job on a remote GPU pod
  --stage-data One-time: upload reference data to network volume
  --local-only Run synthetic training pipeline locally (no cloud)

Usage:
    # Dry-run — see what would happen (default, no cost)
    python examples/foundation_models/ops_run_pipeline.py \\
        -- python examples/foundation_models/02_embedding_extraction.py \\
             --genes BRCA1 TP53 --model evo2 --model-size 7b \\
             --output /workspace/output/

    # Execute — launch on remote GPU
    python examples/foundation_models/ops_run_pipeline.py --execute \\
        -- python examples/foundation_models/02_embedding_extraction.py \\
             --genes BRCA1 TP53 --model evo2 --model-size 7b \\
             --output /workspace/output/

    # Override infrastructure for one run
    python examples/foundation_models/ops_run_pipeline.py --execute --gpu a100 \\
        -- python examples/foundation_models/03_train_and_evaluate.py \\
             --embeddings /workspace/embeddings/embeddings.h5 \\
             --output /workspace/output/ --architecture mlp --epochs 50

    # Reuse an existing cluster (skip provisioning + setup, saves ~3 min)
    python examples/foundation_models/ops_run_pipeline.py --execute \\
        --cluster sky-c0ec-pleiadian53 --no-teardown \\
        -- python examples/foundation_models/04_extract_and_train.py \\
             --all-genes --output /workspace/output/

    # Stage data to network volume (one-time)
    python examples/foundation_models/ops_run_pipeline.py --stage-data

    # Local synthetic pipeline (no cloud, no GPU)
    python examples/foundation_models/ops_run_pipeline.py --local-only \\
        --output-dir /tmp/fm_pipeline/
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

# Default config file path (relative to project root)
_DEFAULT_CONFIG = "foundation_models/configs/gpu_config.yaml"


def _parse_args() -> tuple[argparse.Namespace, str]:
    """Parse CLI args, splitting on '--' for the task command."""
    # Split argv on '--'
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
            "  # Execute on remote GPU\n"
            "  %(prog)s --execute -- python your_script.py --args\n\n"
            "  # Stage data to network volume\n"
            "  %(prog)s --stage-data\n"
        ),
    )

    # Execution mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="Launch the job on a remote GPU pod")
    mode.add_argument("--stage-data", action="store_true",
                      help="One-time: upload reference data to network volume")
    mode.add_argument("--local-only", action="store_true",
                      help="Run synthetic training pipeline locally")

    # Infrastructure overrides (override gpu_config.yaml)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU type: a40, a100, h100 (overrides config)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model dependency profile from gpu_config.yaml "
                             "(e.g., evo2, hyenadna; 'none' to skip model deps)")
    parser.add_argument("--cloud", type=str, default=None,
                        help="Cloud provider (overrides config)")
    parser.add_argument("--use-volume", action="store_true", default=None,
                        help="Read data from network volume")
    parser.add_argument("--no-volume", action="store_true",
                        help="Upload data via file_mounts (ignore config)")
    parser.add_argument("--extra-setup", type=str, default=None,
                        help='Extra setup commands (e.g., "pip install evo2")')
    parser.add_argument("--extra-file-mounts", type=str, nargs="+", default=None,
                        help='Additional file mounts as "remote=local" pairs')
    parser.add_argument("--data-prefix", type=str, default=None,
                        help='Local data root directory (overrides config)')
    parser.add_argument("--data-path", type=str, default=None,
                        help='Dataset subpath, e.g. "mane/GRCh38" (overrides config)')

    # Cluster reuse
    parser.add_argument("--cluster", type=str, default=None,
                        help="Reuse an existing cluster (skip provisioning + setup)")
    parser.add_argument("--no-teardown", action="store_true",
                        help="Keep the cluster alive after the job (for iterative runs)")

    # Job settings
    parser.add_argument("--job-name", type=str, default=None,
                        help="Job name (auto-derived from script name if omitted)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Local output directory for downloaded results")
    parser.add_argument("--config", type=str, default=_DEFAULT_CONFIG,
                        help=f"Path to gpu_config.yaml (default: {_DEFAULT_CONFIG})")

    args = parser.parse_args(runner_args)

    # Resolve --no-volume → use_volume=False
    if args.no_volume:
        args.use_volume = False

    return args, task_command


def _build_output_dir(args: argparse.Namespace, task_command: str) -> Path:
    """Auto-generate output directory from the task command or use explicit path."""
    if args.output_dir:
        return Path(args.output_dir)
    # Default: output/gpu_runs/<job_name>/
    from foundation_models.gpu_runner import _derive_job_name
    name = args.job_name or _derive_job_name(task_command) or "fm-job"
    return Path("output") / "gpu_runs" / name


def main() -> None:
    args, task_command = _parse_args()

    # Local-only mode doesn't need the runner
    if args.local_only:
        _run_local_only(args)
        return

    # Load infrastructure config
    from foundation_models.gpu_runner import (
        InfraConfig,
        build_skypilot_config,
        launch,
        print_dry_run,
        stage_data,
    )

    infra = InfraConfig.from_yaml(args.config)

    # Apply CLI overrides
    overrides = {}
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
        print("  python examples/foundation_models/ops_run_pipeline.py --execute \\")
        print("      -- python examples/foundation_models/02_embedding_extraction.py \\")
        print("           --genes BRCA1 TP53 --model evo2 --model-size 7b \\")
        print("           --output /workspace/output/")
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


def _run_local_only(args: argparse.Namespace) -> None:
    """Run synthetic training pipeline locally."""
    output_dir = Path(args.output_dir or "/tmp/fm_pipeline/")

    print()
    print("=" * 70)
    print("Foundation Model Pipeline — Local Mode (Synthetic Data)")
    print("=" * 70)
    print()

    cmd = [
        sys.executable, "examples/foundation_models/01_synthetic_pipeline.py",
        "--output", str(output_dir),
        "--n-genes", "5",
    ]
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error("Local pipeline failed with exit code %d", result.returncode)
        sys.exit(result.returncode)

    print()
    print("Local pipeline complete. Artifacts in:", output_dir)
    print()


if __name__ == "__main__":
    main()
