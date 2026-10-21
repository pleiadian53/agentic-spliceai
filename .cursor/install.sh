#!/usr/bin/env bash
# Cloud Agent install step for Agentic-SpliceAI.
#
# Idempotent, non-interactive bootstrap of the development environment:
#   1. system build toolchain + headers needed to compile the bioinformatics
#      wheels (pysam, mappy, pyBigWig) and to create a venv,
#   2. an isolated virtualenv at .venv (system Python is PEP 668 managed),
#   3. a CPU-only PyTorch build (Cloud Agent VMs have no GPU) that the meta
#      layer imports at module load time,
#   4. the package itself in editable mode with the dev + bioinformatics extras.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- 1. System dependencies -------------------------------------------------
# python3-venv/-dev give us ensurepip + Python.h; the compression/curl -dev
# libraries are required by the htslib-backed genomics wheels. Only touch apt
# when something is actually missing so re-runs are fast.
need_apt=0
python3 -c "import ensurepip" >/dev/null 2>&1 || need_apt=1
[ -e /usr/include/python3.12/Python.h ] || need_apt=1
if [ "$need_apt" -eq 1 ]; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3.12-venv python3.12-dev build-essential \
    zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev
fi

# --- 2. Virtualenv ----------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# --- 3. CPU-only PyTorch ----------------------------------------------------
# Pin matches environment.yml. The CPU index avoids pulling the large CUDA
# wheels that would never be used on a GPU-less VM.
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu

# --- 4. Project (editable) + extras ----------------------------------------
python -m pip install -e ".[dev,bio,conservation]"

echo "Agentic-SpliceAI environment ready."
