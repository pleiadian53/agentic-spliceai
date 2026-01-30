# Pod Environment Setup Guide

Complete guide for setting up the Python environment on a new RunPods instance after syncing your code via rsync.

---

## 🎯 Overview

After syncing your code to `/workspace/agentic-spliceai/`, you need to:

1. Install Miniforge (includes mamba package manager)
2. Create the conda environment
3. Install PyTorch with CUDA support
4. Install the project in editable mode

**Estimated time**: 10-15 minutes

---

## ⚡ Quick Setup (Copy-Paste)

SSH to your pod and run these commands:

```bash
# 1. Install Miniforge
cd /workspace
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc

# 2. Create environment (choose minimal or full)
cd /workspace/agentic-spliceai

# Option A: Minimal environment (recommended for RunPods)
mamba env create -f runpods.example/environment-runpods-minimal.yml

# Option B: Full environment (if you need all features)
mamba env create -f environment.yml

# 3. Activate environment
mamba activate agenticspliceai

# 4. Install PyTorch with CUDA support (for GPU training)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install project in editable mode
pip install -e .

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

✅ **Done!** Your environment is ready for training.

---

## 📋 Step-by-Step Instructions

### Prerequisites

- ✅ Pod is running
- ✅ You can SSH to the pod
- ✅ Code synced to `/workspace/agentic-spliceai/` via rsync
- ✅ Internet connectivity on pod

### Step 1: Install Miniforge

**Why Miniforge?**
- Includes `mamba` (faster than conda)
- Free, open-source
- Works well with GPUs
- Standard for scientific computing

```bash
# SSH to your pod
ssh <your-pod-ssh-alias>

# Navigate to workspace
cd /workspace

# Download Miniforge installer
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh

# Install (non-interactive, installs to /workspace/miniforge3)
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
```

**Explanation of flags**:
- `-b`: Batch mode (no prompts)
- `-p /workspace/miniforge3`: Install to persistent workspace

**Initialize shell**:
```bash
# Initialize bash shell (adds to ~/.bashrc)
/workspace/miniforge3/bin/conda init bash

# Reload shell configuration
source ~/.bashrc
```

**Verify installation**:
```bash
which mamba
# Should show: /workspace/miniforge3/bin/mamba

mamba --version
# Should show: mamba x.x.x, conda x.x.x
```

✅ **Miniforge installed!**

---

### Step 2: Choose Your Environment

You have two environment options:

#### Option A: Minimal Environment (Recommended)

**Use this if**:
- Training models on GPU
- Want faster setup (~5-7 minutes)
- Don't need all features (Jupyter, web API, etc.)

```bash
cd /workspace/agentic-spliceai
mamba env create -f runpods.example/environment-runpods-minimal.yml
```

**What's included**:
- Python 3.10
- Core ML: PyTorch, scikit-learn, XGBoost
- Bioinformatics: biopython, pybedtools, pysam
- Visualization: matplotlib, plotly
- Utilities: tqdm, rich, wandb

**What's NOT included**:
- TensorFlow/Keras (add if needed: `pip install tensorflow`)
- Jupyter notebooks (add if needed: `pip install jupyter`)
- FastAPI web framework

#### Option B: Full Environment

**Use this if**:
- Need Jupyter notebooks
- Need web API (FastAPI)
- Need both TensorFlow and PyTorch
- Want all features

```bash
cd /workspace/agentic-spliceai
mamba env create -f environment.yml
```

**What's included**:
- Everything from minimal
- TensorFlow + Keras
- Jupyter + notebooks
- FastAPI + web tools
- Additional ML tools (LightGBM, CatBoost, etc.)

**Note**: Takes longer to install (~10-15 minutes)

---

### Step 3: Activate Environment

```bash
mamba activate agenticspliceai
```

You should see your prompt change to:
```bash
(agenticspliceai) root@runpods:~$
```

**To make it activate automatically**:
```bash
echo 'mamba activate agenticspliceai' >> ~/.bashrc
```

---

### Step 4: Install PyTorch with CUDA

**IMPORTANT**: PyTorch from conda doesn't always match your GPU. Install from PyTorch directly:

```bash
# For CUDA 12.1 (common on RunPods)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For different CUDA versions**:
```bash
# Check your CUDA version first
nvidia-smi

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify PyTorch + GPU**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected output**:
```
PyTorch: 2.x.x+cu121
CUDA available: True
GPU count: 1
GPU name: NVIDIA A40
```

✅ **PyTorch configured for GPU!**

---

### Step 5: Install Project

Install the project in **editable mode** (changes to code reflect immediately):

```bash
cd /workspace/agentic-spliceai
pip install -e .
```

**What this does**:
- Installs project as a package
- Editable mode (`-e`): changes reflect without reinstall
- Makes imports work: `from agentic_spliceai import ...`

**Verify installation**:
```bash
python -c "import agentic_spliceai; print('Project installed successfully!')"
```

---

### Step 6: Optional Dependencies

#### For SpliceAI Base Model
```bash
pip install tensorflow==2.17.0 spliceai==1.3.1
```

#### For Jupyter Notebooks
```bash
pip install jupyter ipykernel notebook
```

#### For Experiment Tracking
```bash
# Weights & Biases (already in minimal env)
wandb login

# TensorBoard
pip install tensorboard
```

#### For Additional Genomics Tools
```bash
pip install pysam pyfaidx gffutils
```

---

## 🔧 Troubleshooting

### Problem: `mamba: command not found`

**Cause**: Shell not initialized or `~/.bashrc` not sourced

**Solution**:
```bash
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc
```

### Problem: `CUDA not available` even with GPU

**Possible causes**:

1. **Wrong PyTorch build**
   ```bash
   # Check PyTorch build
   python -c "import torch; print(torch.__version__)"
   # Should show: 2.x.x+cu121 (or cu118, cu124)
   # If it shows: 2.x.x+cpu, you installed CPU-only version
   
   # Fix: Reinstall with CUDA
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

2. **CUDA driver mismatch**
   ```bash
   nvidia-smi
   # Check CUDA version at top right
   # Install matching PyTorch version
   ```

3. **Environment issue**
   ```bash
   # Check environment
   echo $CUDA_VISIBLE_DEVICES
   # Should be empty or show GPU IDs (0,1,2...)
   
   # If it shows -1 or wrong values, unset it
   unset CUDA_VISIBLE_DEVICES
   ```

### Problem: Environment creation fails

**Common issues**:

1. **Dependency conflicts**
   ```bash
   # Try with --force-reinstall
   mamba env create -f environment.yml --force
   
   # Or remove and retry
   mamba env remove -n agenticspliceai
   mamba env create -f runpods.example/environment-runpods-minimal.yml
   ```

2. **Network timeout**
   ```bash
   # Increase timeout
   conda config --set remote_read_timeout_secs 600
   
   # Retry
   mamba env create -f runpods.example/environment-runpods-minimal.yml
   ```

3. **Disk space**
   ```bash
   df -h /workspace
   # Ensure you have at least 5GB free
   
   # Clean conda cache if needed
   mamba clean --all -y
   ```

### Problem: Import errors after installation

```bash
# Verify environment is activated
conda env list
# Should show * next to agenticspliceai

# Reinstall project
cd /workspace/agentic-spliceai
pip install -e . --force-reinstall

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
# Should include /workspace/agentic-spliceai
```

### Problem: Slow environment creation

**Solutions**:
```bash
# 1. Use minimal environment (faster)
mamba env create -f runpods.example/environment-runpods-minimal.yml

# 2. Use mamba instead of conda (10x faster)
mamba env create -f environment.yml

# 3. Use libmamba solver with conda
conda config --set solver libmamba
conda env create -f environment.yml
```

---

## 📊 Environment Comparison

| Aspect | Minimal | Full |
|--------|---------|------|
| **Install time** | 5-7 minutes | 10-15 minutes |
| **Disk space** | ~3 GB | ~5 GB |
| **Python version** | 3.10 | 3.11 |
| **PyTorch** | ✅ (via pip) | ✅ (via conda) |
| **TensorFlow** | ❌ (add if needed) | ✅ |
| **Jupyter** | ❌ (add if needed) | ✅ |
| **FastAPI** | ❌ | ✅ |
| **Bioinformatics** | ✅ | ✅ |
| **Best for** | Training only | Full development |

---

## 🚀 Quick Environment Tests

After setup, verify everything works:

### Test 1: Python Environment
```bash
python -c "import sys; print(f'Python: {sys.version}')"
```

### Test 2: PyTorch + GPU
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

### Test 3: NumPy + Scientific Stack
```bash
python -c "import numpy as np; import pandas as pd; import sklearn; print('Scientific stack: OK')"
```

### Test 4: Bioinformatics Tools
```bash
python -c "import Bio; import pybedtools; import pysam; print('Bioinformatics tools: OK')"
```

### Test 5: Project Installation
```bash
python -c "import agentic_spliceai; print('Project: OK')"
```

### Test 6: Run Verification Script
```bash
cd /workspace/agentic-spliceai
python verify_setup.py
```

**All tests pass? You're ready to train! 🎉**

---

## 💡 Best Practices

### 1. Use Minimal Environment for RunPods
- Faster setup
- Less disk usage
- Sufficient for training

### 2. Install PyTorch Separately
- Better CUDA compatibility
- More control over version
- Avoid conda PyTorch issues

### 3. Keep Environment Files Updated
- Commit environment changes to git (locally)
- Sync to pod via rsync
- Recreate environment if major changes

### 4. Monitor Disk Space
```bash
# Check space before creating environment
df -h /workspace

# Clean conda cache after setup
mamba clean --all -y
```

### 5. Document Custom Installations
If you install additional packages:
```bash
# Save environment state
mamba env export > environment-snapshot.yml
```

---

## 🔄 Updating Environment

### Add New Package
```bash
# Activate environment
mamba activate agenticspliceai

# Install package
pip install new-package
# or
mamba install -c conda-forge new-package

# Optional: Update environment file locally
# (on your local machine, add to environment.yml or requirements.txt)
```

### Recreate Environment
```bash
# Remove old environment
mamba env remove -n agenticspliceai

# Sync latest code from local
# (on local machine: sync-to-pod)

# Create fresh environment
cd /workspace/agentic-spliceai
mamba env create -f runpods.example/environment-runpods-minimal.yml
mamba activate agenticspliceai
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## 📝 Summary Checklist

After setup is complete, verify:

- [ ] Miniforge installed (`which mamba` works)
- [ ] Environment created (`mamba env list` shows agenticspliceai)
- [ ] Environment activated (prompt shows `(agenticspliceai)`)
- [ ] PyTorch with CUDA installed (`torch.cuda.is_available()` returns True)
- [ ] Project installed (`import agentic_spliceai` works)
- [ ] GPU detected (`nvidia-smi` shows your GPU)
- [ ] All tests pass (`python verify_setup.py` succeeds)

**Estimated total time**: 10-15 minutes  
**Frequency**: Once per new pod  
**Result**: Ready for GPU training 🚀

---

## 🔗 Related Documentation

- **Pod SSH Setup**: `runpod_ssh_manager.sh` - Connecting to pod
- **Code Sync**: `RSYNC_QUICK_REFERENCE.md` - Syncing code/results
- **GitHub Setup** (if needed): `GITHUB_SSH_SETUP.md` - Git on pod
- **Local Development**: `LOCAL_DEVELOPMENT_WORKFLOW.md` - Recommended workflow

---

## 🎯 Next Steps

Once environment is set up:

1. **Verify data is available**:
   ```bash
   ls -lh /workspace/data/
   ```

2. **Test training script**:
   ```bash
   cd /workspace/agentic-spliceai
   python train.py --help
   ```

3. **Start training**:
   ```bash
   # In tmux (recommended - survives disconnection)
   tmux new -s training
   python train.py --config configs/your_config.yaml
   # Ctrl+B, D to detach
   ```

4. **Monitor training**:
   ```bash
   # Reattach to tmux
   tmux attach -t training
   
   # Or check outputs
   tail -f output/latest/train.log
   ```

5. **Download results** (from local machine):
   ```bash
   sync-from-pod
   ```

---

**Created**: January 28, 2026  
**Version**: 1.0.0  
**Applies to**: All RunPods instances with NVIDIA GPUs
