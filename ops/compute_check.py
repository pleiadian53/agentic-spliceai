#!/usr/bin/env python3
"""
Compute Resource Check — detect the local (or pod) hardware environment.

Reports the compute device (CUDA GPU / Apple MPS / CPU), VRAM, system RAM, and
free disk at a given path. Useful both locally (is this laptop enough, or do I
need a pod?) and on a freshly provisioned pod (did I get the GPU I asked for?).

Dual interface:
  - CLI:          prints a human-readable report
  - Programmatic: ``check_compute()`` returns a ``ComputeReport`` dataclass

Usage:
    # Auto-detect and print a report
    python ops/compute_check.py

    # Check free disk at a specific output path
    python ops/compute_check.py --output-path /workspace/output/

    # Also print the foundation-model hardware-requirements reference table
    python ops/compute_check.py --fm-requirements

Programmatic usage from other scripts::

    from ops.compute_check import check_compute

    report = check_compute(output_path="/workspace/output/")
    if report.device != "cuda":
        print("No GPU detected — provision a pod with ops/provision_cluster.py")
"""

import argparse
import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ComputeReport:
    """Hardware detection report."""

    device: str              # "cuda", "mps", "cpu"
    device_name: str         # "NVIDIA A40", "Apple M1", etc.
    vram_gb: float           # GPU VRAM (0.0 for CPU/MPS)
    ram_gb: float            # System RAM
    disk_free_gb: float      # Free disk at the checked path


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
    """Detect the compute environment.

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
# Foundation-model hardware requirements (static reference)
# ---------------------------------------------------------------------------

# Approximate, informational only — single-GPU inference unless noted. VRAM
# figures are ballpark and vary with context length, batch size, and precision
# (bf16/fp16 vs fp32). Use these to decide which GPU to provision, not as a
# hard contract. No project imports; safe to read anywhere.
FM_HARDWARE: dict[str, dict] = {
    "evo2-7b": {
        "min_vram_gb": 40,
        "recommended_gpu": "A100 / A40",
        "notes": "FP8 autocast patch needed on <8.9 compute (A40/A100); bf16 fallback",
    },
    "evo2-40b": {
        "min_vram_gb": 80,
        "recommended_gpu": "H100 / A100-80G",
        "notes": "multi-GPU for training; ~80 GB+ even for inference",
    },
    "alphagenome": {
        "min_vram_gb": 16,
        "recommended_gpu": "A40 / A100",
        "notes": "1 Mb context; non-commercial license (approx)",
    },
    "splicebert": {
        "min_vram_gb": 8,
        "recommended_gpu": "T4 / L4 or better",
        "notes": "BERT-scale pre-mRNA model; light, CPU-feasible for small batches",
    },
    "hyenadna": {
        "min_vram_gb": 16,
        "recommended_gpu": "A40 / A100",
        "notes": "long context (up to 1M) drives VRAM; small at short context (approx)",
    },
    "pangolin": {
        "min_vram_gb": 4,
        "recommended_gpu": "any GPU (CPU-feasible)",
        "notes": "small splice CNN; runs on modest hardware",
    },
    "dnabert-2": {
        "min_vram_gb": 8,
        "recommended_gpu": "T4 / L4 / A40",
        "notes": "~117M params; loads from HuggingFace",
    },
    "nucleotide-transformer": {
        "min_vram_gb": 24,
        "recommended_gpu": "A40 / A100",
        "notes": "500M–2.5B variants; VRAM scales with model size (approx)",
    },
}


def print_fm_requirements(report: ComputeReport | None = None) -> None:
    """Print the static foundation-model hardware-requirements reference table.

    If a ``report`` is given, annotate each row with whether the detected VRAM
    meets the model's approximate minimum (CUDA only; MPS/CPU shown as 'n/a').
    """
    detected_vram = report.vram_gb if (report and report.device == "cuda") else None

    print()
    print("=" * 78)
    print("Genomic Foundation-Model Hardware Requirements (approximate reference)")
    print("=" * 78)
    print()
    header = f"  {'Model':<24s} {'Min VRAM':>9s}  {'Recommended GPU':<22s} {'Fits?':<6s}"
    print(header)
    print("  " + "-" * 74)
    for name, spec in FM_HARDWARE.items():
        min_vram = spec["min_vram_gb"]
        if detected_vram is None:
            fits = "n/a"
        else:
            fits = "yes" if detected_vram >= min_vram else "no"
        print(f"  {name:<24s} {str(min_vram) + ' GB':>9s}  "
              f"{spec['recommended_gpu']:<22s} {fits:<6s}")
    print()
    for name, spec in FM_HARDWARE.items():
        print(f"  {name}: {spec['notes']}")
    print()
    print("  Figures are approximate; verify against the model card before launch.")
    print()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_compute(output_path: str = ".") -> ComputeReport:
    """Detect the compute environment.

    Args:
        output_path: Path to check for free disk space. Falls back to the
            current directory if the path doesn't exist.

    Returns:
        ComputeReport with hardware info.
    """
    path = Path(output_path)
    check_path = str(path if path.exists() else Path("."))
    return detect_hardware(output_path=check_path)


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

    if report.device == "cuda":
        print("  CUDA GPU detected — ready for GPU workloads.")
    elif report.device == "mps":
        print("  Apple MPS detected — fine for small/local work; provision a pod")
        print("  for genome-scale training/eval (python ops/provision_cluster.py).")
    else:
        print("  CPU-only — provision a pod for GPU workloads")
        print("  (python ops/provision_cluster.py).")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect the compute environment (GPU / MPS / CPU, RAM, disk).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-path", type=str, default=".",
        help="Path to check for free disk space (default: current directory).",
    )
    parser.add_argument(
        "--fm-requirements", action="store_true",
        help="Also print a reference table of genomic foundation-model hardware "
             "requirements (static/approximate).",
    )
    args = parser.parse_args()

    report = check_compute(output_path=args.output_path)
    print_report(report)
    if args.fm_requirements:
        print_fm_requirements(report)


if __name__ == "__main__":
    main()
