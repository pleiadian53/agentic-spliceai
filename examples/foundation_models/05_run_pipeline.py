#!/usr/bin/env python3
"""
Example: End-to-End Foundation Model Pipeline Orchestrator

Combines all 4 workflow steps into a single parameterized script:
  1. Resource check
  2. Extract embeddings (remote GPU via SkyPilot)
  3. Download results
  4. Train and evaluate classifier

Three execution modes:
  --dry-run   (default) Print resource check + generated SkyPilot config + commands
  --local-only          Run full pipeline with synthetic data (no GPU, no cloud)
  --execute             Actually launch remote jobs via SkyPilot CLI

Usage:
    # Plan mode — see what would happen (default, no cloud cost)
    python examples/foundation_models/05_run_pipeline.py \\
        --model-size 7b --gpu a40 --genes BRCA1 TP53

    # More genes for realistic training
    python examples/foundation_models/05_run_pipeline.py \\
        --model-size 7b --gpu a40 \\
        --genes BRCA1 TP53 BRCA2 RB1 CFTR EGFR PTEN MLH1 MSH2 APC

    # Local mode — synthetic data, always works
    python examples/foundation_models/05_run_pipeline.py \\
        --local-only --output-dir /tmp/fm_pipeline/

    # Execute mode — launches real SkyPilot jobs (costs money!)
    python examples/foundation_models/05_run_pipeline.py \\
        --execute --model-size 7b --gpu a40 \\
        --genes BRCA1 TP53 BRCA2 RB1 CFTR EGFR PTEN MLH1 MSH2 APC
"""

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
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


def _build_output_dir(args: argparse.Namespace) -> Path:
    """Auto-generate output directory from parameters."""
    if args.output_dir:
        return Path(args.output_dir)

    n_genes = len(args.genes)
    name = f"evo2_{args.model_size}_{n_genes}genes"
    return Path("output") / "fm_pipelines" / name


def _generate_skypilot_config(args: argparse.Namespace, output_dir: Path) -> dict:
    """Generate a SkyPilot job config dict from CLI args."""
    gpu = GPU_SPECS[args.gpu]
    gene_args = " ".join(args.genes)
    n_genes = len(args.genes)
    job_name = f"fm-{args.model_size}-{n_genes}genes"

    config = {
        "name": job_name,
        "workdir": ".",
        "resources": {
            "accelerators": gpu["accelerator"],
            "cloud": args.cloud,
        },
        "file_mounts": {
            "/workspace/data": "./data/mane/GRCh38",
        },
        "setup": (
            "set -e\n"
            "pip install -e .\n"
            "pip install -e ./foundation_models\n"
            "pip install evo2"
        ),
        "run": (
            f"set -e\n"
            f"mkdir -p /workspace/output\n"
            f"python examples/foundation_models/03_embedding_extraction.py \\\n"
            f"  --genes {gene_args} \\\n"
            f"  --model evo2 --model-size {args.model_size} \\\n"
            f"  --output /workspace/output/\n"
            f"\n"
            f'echo ""\n'
            f'echo "============================================"\n'
            f'echo "DONE — download results before tearing down:"\n'
            f'echo "  rsync -Pavz {job_name}:/workspace/output/ {output_dir}/embeddings/"\n'
            f'echo "  sky down {job_name} -y"\n'
            f'echo "============================================"'
        ),
    }
    return config


def _run_dry_run(args: argparse.Namespace, output_dir: Path) -> None:
    """Print resource check, generated config, and commands."""
    from foundation_models.utils.resources import (
        estimate_embedding_extraction,
        print_feasibility_report,
    )

    gpu = GPU_SPECS[args.gpu]

    print()
    print("=" * 70)
    print("Foundation Model Pipeline — Dry Run")
    print("=" * 70)
    print()

    # Resource check
    print("Step 1: Resource Check")
    print("-" * 40)
    print_feasibility_report(hardware=gpu["hardware_profile"])

    result = estimate_embedding_extraction(
        model_size=args.model_size,
        n_genes=len(args.genes),
        hardware=gpu["hardware_profile"],
    )
    if not result["feasible"]:
        print("WARNING: Embedding extraction is NOT feasible on this GPU!")
        for note in result["notes"]:
            print(f"  {note}")
        print()

    # Generated config
    config = _generate_skypilot_config(args, output_dir)
    print("Step 2: Generated SkyPilot Config")
    print("-" * 40)

    import yaml
    yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
    print(yaml_str)

    # Commands
    job_name = config["name"]
    print("Step 3: Commands to Execute")
    print("-" * 40)
    print(f"  # 1. Launch embedding extraction")
    print(f"  sky launch <config>.yaml -y")
    print()
    print(f"  # 2. Download embeddings to local")
    print(f"  rsync -Pavz {job_name}:/workspace/output/ {output_dir}/embeddings/")
    print()
    print(f"  # 3. Tear down pod")
    print(f"  sky down {job_name} -y")
    print()
    print(f"  # 4. Train classifier locally")
    print(f"  python examples/foundation_models/04_train_and_evaluate.py \\")
    print(f"      --embeddings {output_dir}/embeddings/embeddings.h5 \\")
    print(f"      --labels {output_dir}/embeddings/embeddings.labels.npz \\")
    print(f"      --output {output_dir}/checkpoints/ \\")
    print(f"      --architecture {args.architecture} --window-size {args.window_size} \\")
    print(f"      --epochs {args.epochs} --patience {args.patience}")
    print()

    # Cost estimate
    n_genes = len(args.genes)
    print("Cost Estimate")
    print("-" * 40)
    rate = gpu["hourly_rate"]
    # Rough estimate: ~3 min per gene on A40 with Evo2 7B
    est_hours = max(0.1, n_genes * 3 / 60)
    est_cost = rate * est_hours
    print(f"  GPU:            {gpu['label']} (${rate:.2f}/hr)")
    print(f"  Est. duration:  ~{est_hours * 60:.0f} min (for {n_genes} genes)")
    print(f"  Est. cost:      ~${est_cost:.2f}")
    print()
    print("To execute, re-run with --execute (costs money!)")
    print()


def _run_local_only(args: argparse.Namespace, output_dir: Path) -> None:
    """Run full pipeline with synthetic data locally."""
    print()
    print("=" * 70)
    print("Foundation Model Pipeline — Local Mode (Synthetic Data)")
    print("=" * 70)
    print()

    # Delegate to the synthetic training pipeline script
    cmd = [
        sys.executable, "examples/foundation_models/02_synthetic_training_pipeline.py",
        "--output", str(output_dir),
        "--architecture", args.architecture,
        "--window-size", str(args.window_size),
        "--epochs", str(args.epochs),
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


def _run_execute(args: argparse.Namespace, output_dir: Path) -> None:
    """Execute the full remote pipeline via SkyPilot CLI."""
    import yaml

    gpu = GPU_SPECS[args.gpu]

    # Resource check
    from foundation_models.utils.resources import estimate_embedding_extraction

    result = estimate_embedding_extraction(
        model_size=args.model_size,
        n_genes=len(args.genes),
        hardware=gpu["hardware_profile"],
    )
    if not result["feasible"]:
        print("RESOURCE CHECK FAILED")
        for note in result["notes"]:
            print(f"  {note}")
        sys.exit(1)

    # Generate and write YAML
    config = _generate_skypilot_config(args, output_dir)
    job_name = config["name"]

    config_dir = Path("foundation_models/configs/skypilot/generated")
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / f"{job_name}.yaml"

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Wrote SkyPilot config: %s", yaml_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Step 1: Launch
    print()
    print("=" * 70)
    print(f"Launching: {job_name} ({gpu['label']})")
    print("=" * 70)
    print()

    launch_result = subprocess.run(
        ["sky", "launch", str(yaml_path), "-y"],
        check=False,
    )
    if launch_result.returncode != 0:
        logger.error("sky launch failed (exit code %d)", launch_result.returncode)
        sys.exit(launch_result.returncode)

    # Step 2: Rsync
    emb_dir = output_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading results...")

    rsync_result = subprocess.run(
        ["rsync", "-Pavz", f"{job_name}:/workspace/output/", str(emb_dir) + "/"],
        check=False,
    )
    if rsync_result.returncode != 0:
        logger.error("rsync failed — pod may still be running. Download manually:")
        logger.error("  rsync -Pavz %s:/workspace/output/ %s/", job_name, emb_dir)
        # Don't tear down if rsync failed
        sys.exit(rsync_result.returncode)

    # Step 3: Tear down
    logger.info("Tearing down pod...")
    subprocess.run(["sky", "down", job_name, "-y"], check=False)

    elapsed_remote = time.time() - t0

    # Step 4: Train locally (if requested)
    if args.train_local:
        print()
        print("=" * 70)
        print("Training Classifier Locally")
        print("=" * 70)
        print()

        train_cmd = [
            sys.executable, "examples/foundation_models/04_train_and_evaluate.py",
            "--embeddings", str(emb_dir / "embeddings.h5"),
            "--labels", str(emb_dir / "embeddings.labels.npz"),
            "--output", str(output_dir / "checkpoints"),
            "--architecture", args.architecture,
            "--window-size", str(args.window_size),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
        ]
        subprocess.run(train_cmd, check=False)

    elapsed_total = time.time() - t0

    # Cost summary
    rate = gpu["hourly_rate"]
    remote_hours = elapsed_remote / 3600
    est_cost = rate * remote_hours

    print()
    print("=" * 70)
    print("Pipeline Complete")
    print("=" * 70)
    print(f"  Output:         {output_dir}")
    print(f"  GPU:            {gpu['label']}")
    print(f"  Remote time:    {elapsed_remote / 60:.1f} min")
    print(f"  Total time:     {elapsed_total / 60:.1f} min")
    print(f"  Est. GPU cost:  ${est_cost:.2f} ({rate:.2f}/hr x {remote_hours:.2f} hr)")
    print()
    print("  Note: For exact billing, check the RunPod dashboard.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end foundation model pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  --dry-run    (default) Print config and commands\n"
            "  --local-only Run with synthetic data, no cloud\n"
            "  --execute    Launch real SkyPilot jobs\n"
        ),
    )

    # Execution mode
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True,
                      help="Print config and commands (default)")
    mode.add_argument("--local-only", action="store_true",
                      help="Run with synthetic data locally")
    mode.add_argument("--execute", action="store_true",
                      help="Execute full remote pipeline via SkyPilot")

    # Model
    parser.add_argument("--model-size", type=str, default="7b",
                        choices=["7b", "40b"], help="Evo2 model size")

    # Compute
    parser.add_argument("--gpu", type=str, default="a40",
                        choices=list(GPU_SPECS.keys()),
                        help="GPU type (default: a40)")
    parser.add_argument("--cloud", type=str, default="runpod",
                        help="Cloud provider (default: runpod)")

    # Data
    parser.add_argument("--genes", type=str, nargs="+",
                        default=["BRCA1", "TP53", "BRCA2", "RB1", "CFTR",
                                 "EGFR", "PTEN", "MLH1", "MSH2", "APC"],
                        help="Gene symbols to process (default: 10 clinically important genes)")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if omitted)")

    # Training
    parser.add_argument("--architecture", type=str, default="mlp",
                        choices=["linear", "mlp", "cnn", "lstm"],
                        help="Classifier architecture")
    parser.add_argument("--window-size", type=int, default=1024,
                        help="Training window size (bp)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--train-local", action="store_true",
                        help="Train classifier locally after downloading embeddings")

    args = parser.parse_args()

    # Resolve mutually exclusive flags
    if args.execute:
        args.dry_run = False
    if args.local_only:
        args.dry_run = False

    output_dir = _build_output_dir(args)

    if args.local_only:
        _run_local_only(args, output_dir)
    elif args.execute:
        _run_execute(args, output_dir)
    else:
        _run_dry_run(args, output_dir)


if __name__ == "__main__":
    main()
