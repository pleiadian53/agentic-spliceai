#!/usr/bin/env python3
"""
Compute Resource Check — detect hardware and estimate task feasibility.

Detects the local compute environment (GPU type/VRAM, system RAM, disk space)
and estimates whether specific foundation model tasks can run on this hardware.
Returns recommended batch sizes and chunk sizes.

Dual interface:
  - CLI:          prints a human-readable report
  - Programmatic: ``check_compute()`` returns a ``ComputeReport`` dataclass

Usage:
    # Auto-detect and print report
    python examples/foundation_models/ops_compute_check.py

    # Check feasibility for a specific task
    python examples/foundation_models/ops_compute_check.py --task sparse_exon_classifier

    # Check with a specific output path (for disk space estimation)
    python examples/foundation_models/ops_compute_check.py --output-path /workspace/output/

    # Check for 40b model
    python examples/foundation_models/ops_compute_check.py --model-size 40b

Programmatic usage from other scripts::

    from ops_compute_check import check_compute

    report = check_compute(output_path="/workspace/output/sparse/")
    task = report.tasks["sparse_exon_classifier"]
    if not task.feasible:
        print("Insufficient resources:", task.notes)
    else:
        batch_size = task.recommended_batch_size
        chunk_size = task.recommended_chunk_size
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaskFeasibility:
    """Resource feasibility assessment for a single task."""

    feasible: bool
    required_vram_gb: float
    required_disk_gb: float
    recommended_batch_size: int
    recommended_chunk_size: int
    notes: list[str] = field(default_factory=list)


@dataclass
class ComputeReport:
    """Hardware detection and task feasibility report."""

    device: str              # "cuda", "mps", "cpu"
    device_name: str         # "NVIDIA A40", "Apple M1", etc.
    vram_gb: float           # GPU VRAM (0.0 for CPU/MPS)
    ram_gb: float            # System RAM
    disk_free_gb: float      # Free disk at output path
    tasks: dict[str, TaskFeasibility] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def _detect_cuda() -> tuple[str, float]:
    """Detect CUDA GPU name and VRAM in GB."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        line = result.stdout.strip().splitlines()[0]
        name, mem_mib = line.split(",", 1)
        vram_gb = float(mem_mib.strip()) / 1024
        return name.strip(), vram_gb
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return "", 0.0


def _detect_system_ram() -> float:
    """Detect total system RAM in GB."""
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            pass
    # Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except FileNotFoundError:
        pass
    # Fallback
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        return 0.0


def _detect_apple_chip() -> str:
    """Detect Apple Silicon chip name (M1, M2, etc.)."""
    if platform.system() != "Darwin":
        return ""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Apple Silicon (unknown)"


def detect_hardware(output_path: str = ".") -> ComputeReport:
    """Detect local compute environment.

    Args:
        output_path: Path to check for free disk space.

    Returns:
        ComputeReport with device, RAM, VRAM, and disk info.
    """
    ram_gb = _detect_system_ram()

    disk = shutil.disk_usage(output_path)
    disk_free_gb = disk.free / (1024 ** 3)

    # Try CUDA first
    gpu_name, vram_gb = _detect_cuda()
    if gpu_name:
        return ComputeReport(
            device="cuda",
            device_name=gpu_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            disk_free_gb=disk_free_gb,
        )

    # Try MPS (Apple Silicon)
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip_name = _detect_apple_chip()
        return ComputeReport(
            device="mps",
            device_name=chip_name or "Apple Silicon",
            vram_gb=0.0,  # MPS shares system RAM
            ram_gb=ram_gb,
            disk_free_gb=disk_free_gb,
        )

    # CPU-only
    return ComputeReport(
        device="cpu",
        device_name=platform.processor() or "Unknown CPU",
        vram_gb=0.0,
        ram_gb=ram_gb,
        disk_free_gb=disk_free_gb,
    )


# ---------------------------------------------------------------------------
# Task estimators
# ---------------------------------------------------------------------------

# Evo2 model memory requirements (approximate, in GB)
_EVO2_VRAM = {"7b": 14.0, "40b": 40.0}
_EVO2_ACTIVATION_PER_1K = {"7b": 4.6, "40b": 12.0}  # GB per 1K tokens


def estimate_sparse_exon_classifier(
    report: ComputeReport,
    model_size: str = "7b",
    n_positions: int = 1500,
    context_size: int = 8192,
) -> TaskFeasibility:
    """Estimate feasibility for Phase A: sparse exon classifier.

    The sparse classifier extracts final-position embeddings from both strands
    for N randomly sampled positions. Storage is minimal (~47 MB for 1500 positions).

    Resource bottleneck: GPU VRAM for Evo2 model + activation memory for one
    context window at a time.
    """
    model_vram = _EVO2_VRAM.get(model_size, 14.0)
    activation_vram = _EVO2_ACTIVATION_PER_1K[model_size] * (context_size / 1000)
    required_vram = model_vram + activation_vram

    # Storage: embeddings + model checkpoint + metadata
    hidden_dim = 4096 if model_size == "7b" else 8192
    emb_bytes = n_positions * 2 * hidden_dim * 4  # float32, 2 strands
    required_disk = max(0.5, emb_bytes / (1024 ** 3) * 3)  # 3x safety margin

    notes: list[str] = []
    feasible = True

    # VRAM check
    if report.device == "cuda":
        if report.vram_gb < required_vram:
            feasible = False
            notes.append(
                f"Evo2 {model_size} needs ~{required_vram:.0f} GB VRAM "
                f"(model: {model_vram:.0f} + activations: {activation_vram:.1f}), "
                f"available: {report.vram_gb:.0f} GB"
            )
        else:
            notes.append(
                f"VRAM OK: {report.vram_gb:.0f} GB available, "
                f"~{required_vram:.0f} GB needed"
            )
    elif report.device == "mps":
        feasible = False
        notes.append("Evo2 requires CUDA — MPS not supported for inference")
    else:
        feasible = False
        notes.append("Evo2 requires CUDA GPU")

    # Disk check
    if report.disk_free_gb < required_disk:
        feasible = False
        notes.append(
            f"Disk: {report.disk_free_gb:.1f} GB free, "
            f"need ~{required_disk:.1f} GB"
        )

    # Recommended chunk size based on VRAM headroom
    if report.device == "cuda" and report.vram_gb >= required_vram:
        headroom = report.vram_gb - model_vram
        if headroom >= 30:
            chunk_size = 8192
        elif headroom >= 15:
            chunk_size = 4096
        else:
            chunk_size = 2048
    else:
        chunk_size = context_size

    return TaskFeasibility(
        feasible=feasible,
        required_vram_gb=required_vram,
        required_disk_gb=required_disk,
        recommended_batch_size=1,
        recommended_chunk_size=chunk_size,
        notes=notes,
    )


def estimate_dense_predictor(
    report: ComputeReport,
    model_size: str = "7b",
) -> TaskFeasibility:
    """Estimate feasibility for Phase B: dense per-chromosome splice site predictor.

    Dense extraction produces per-nucleotide embeddings for entire chromosomes.
    Storage is the main bottleneck (~50 GB per chromosome uncompressed).
    """
    model_vram = _EVO2_VRAM.get(model_size, 14.0)
    required_vram = model_vram + 8.0  # model + chunked inference headroom
    required_disk = 60.0  # GB per chromosome (with margin)

    notes: list[str] = []
    feasible = True

    if report.device != "cuda":
        feasible = False
        notes.append("Evo2 requires CUDA GPU")
    elif report.vram_gb < required_vram:
        feasible = False
        notes.append(
            f"Evo2 {model_size} dense needs ~{required_vram:.0f} GB VRAM, "
            f"available: {report.vram_gb:.0f} GB"
        )
    else:
        notes.append(f"VRAM OK: {report.vram_gb:.0f} GB available")

    if report.disk_free_gb < required_disk:
        feasible = False
        notes.append(
            f"Disk: {report.disk_free_gb:.1f} GB free, "
            f"need ~{required_disk:.0f} GB per chromosome"
        )
    else:
        notes.append(f"Disk OK: {report.disk_free_gb:.1f} GB free")

    notes.append(
        "Streaming mode recommended: extract one chrom -> train -> delete -> next"
    )

    # Chunk size for dense extraction
    if report.device == "cuda":
        headroom = report.vram_gb - model_vram
        chunk_size = 8192 if headroom >= 30 else 4096
    else:
        chunk_size = 4096

    return TaskFeasibility(
        feasible=feasible,
        required_vram_gb=required_vram,
        required_disk_gb=required_disk,
        recommended_batch_size=1,
        recommended_chunk_size=chunk_size,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Registry of task estimators (extensible — add new tasks here)
_TASK_ESTIMATORS: dict[str, Callable[..., TaskFeasibility]] = {
    "sparse_exon_classifier": estimate_sparse_exon_classifier,
    "dense_predictor": estimate_dense_predictor,
}


def check_compute(
    output_path: str = ".",
    model_size: str = "7b",
    tasks: list[str] | None = None,
) -> ComputeReport:
    """Detect hardware and estimate feasibility for foundation model tasks.

    Args:
        output_path: Path to check for free disk space.
        model_size: Evo2 model variant ('7b' or '40b').
        tasks: Task names to estimate. If None, estimates all registered tasks.

    Returns:
        ComputeReport with hardware info and per-task feasibility.
    """
    path = Path(output_path)
    check_path = str(path if path.exists() else Path("."))

    report = detect_hardware(output_path=check_path)

    task_names = tasks or list(_TASK_ESTIMATORS.keys())
    for name in task_names:
        estimator = _TASK_ESTIMATORS.get(name)
        if estimator:
            report.tasks[name] = estimator(report, model_size=model_size)

    return report


def print_report(report: ComputeReport) -> None:
    """Print a human-readable compute report."""
    print()
    print("=" * 70)
    print("Compute Resource Report")
    print("=" * 70)
    print()
    print(f"  Device:     {report.device_name} ({report.device})")
    if report.vram_gb > 0:
        print(f"  VRAM:       {report.vram_gb:.1f} GB")
    print(f"  RAM:        {report.ram_gb:.1f} GB")
    print(f"  Disk free:  {report.disk_free_gb:.1f} GB")
    print()

    if not report.tasks:
        print("  No tasks estimated.")
        print()
        return

    print("-" * 70)
    print(f"  {'Task':<30s} {'Feasible':<10s} {'VRAM':<12s} {'Disk':<12s}")
    print("-" * 70)

    for name, task in report.tasks.items():
        status = "YES" if task.feasible else "NO"
        vram_str = f"{task.required_vram_gb:.0f} GB"
        disk_str = f"{task.required_disk_gb:.0f} GB"
        print(f"  {name:<30s} {status:<10s} {vram_str:<12s} {disk_str:<12s}")

    print()

    for name, task in report.tasks.items():
        print(f"  {name}:")
        for note in task.notes:
            print(f"    - {note}")
        if task.feasible:
            print(f"    - Recommended batch_size={task.recommended_batch_size}, "
                  f"chunk_size={task.recommended_chunk_size}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect compute environment and estimate task feasibility.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, default=None,
        choices=list(_TASK_ESTIMATORS.keys()),
        help="Estimate a specific task only (default: all tasks).",
    )
    parser.add_argument(
        "--model-size", type=str, default="7b", choices=["7b", "40b"],
        help="Evo2 model size for estimation (default: 7b).",
    )
    parser.add_argument(
        "--output-path", type=str, default=".",
        help="Path to check for free disk space (default: current directory).",
    )
    args = parser.parse_args()

    tasks = [args.task] if args.task else None
    report = check_compute(
        output_path=args.output_path,
        model_size=args.model_size,
        tasks=tasks,
    )
    print_report(report)


if __name__ == "__main__":
    main()
