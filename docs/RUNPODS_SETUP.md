# RunPods Setup Guide

**For**: Setting up GPU instances on RunPods for agentic-spliceai training  
**Audience**: First-time RunPods users  
**Time**: ~15 minutes

---

## 🎯 Overview

RunPods provides on-demand GPU compute for training. This guide shows you how to:
1. Set up SSH access to RunPods instances
2. Configure your environment on the pod
3. Transfer data and start training

---

## 📦 Prerequisites

### On Your Local Machine

- ✅ SSH key (`~/.ssh/id_ed25519` or `~/.ssh/id_rsa`)
- ✅ agentic-spliceai repository cloned
- ✅ Bash shell (macOS, Linux, WSL)

### RunPods Account

- ✅ RunPods account created
- ✅ Payment method added
- ✅ SSH public key uploaded to RunPods

---

## 🚀 Quick Start

### Step 1: Set Up RunPods Scripts (One Time)

```bash
cd ~/work/agentic-spliceai

# Copy example templates
cp -r runpods.example runpods

# Make scripts executable
chmod +x runpods/scripts/*.sh
```

**Note**: The `runpods/` directory is NOT tracked in git (it's in `.gitignore`). This is intentional - it contains your personal configuration.

### Step 2: Acquire RunPods Instance

1. Go to [runpods.io](https://runpods.io)
2. Click **Deploy**
3. Select GPU (e.g., A40 48GB, H100 80GB)
4. Use template: **PyTorch** or **Fast Stable Diffusion**
5. Click **Deploy On-Demand** or **Deploy Spot**
6. Wait for pod to start (~1-2 minutes)

### Step 3: Get Connection Info

From RunPods dashboard:

1. Click **Connect** on your pod
2. Select **SSH over exposed TCP**
3. Copy the connection command:
   ```
   ssh root@ssh.runpods.io -p 12345 -i ~/.ssh/id_ed25519
   ```
4. Extract:
   - **Hostname**: `ssh.runpods.io`
   - **Port**: `12345`

### Step 4: Configure SSH Access (On Your Machine)

```bash
cd ~/work/agentic-spliceai/runpods/scripts
./runpod_ssh_manager.sh add agentic-spliceai
```

**Enter when prompted**:
- Hostname: `ssh.runpods.io`
- Port: `12345`
- Nickname: `a40-48gb` (or whatever helps you remember)
- SSH Key: Press Enter for default

**Result**: SSH config entry created

### Step 5: Test Connection

```bash
ssh runpod-agentic-spliceai-a40-48gb
```

**Expected**: You're now connected to the pod! 🎉

### Step 6: Setup Environment (On Pod)

Now that you're SSH'd into the pod:

```bash
# Install Miniforge
cd /workspace
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
/workspace/miniforge3/bin/conda init bash
source ~/.bashrc

# Clone repository
cd /workspace
git clone https://github.com/YOUR-USERNAME/agentic-spliceai.git
cd agentic-spliceai

# Create environment
mamba env create -f runpods/environment-runpods-minimal.yml
mamba activate agenticspliceai

# Install PyTorch with CUDA
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install additional packages
pip install transformers einops accelerate safetensors

# Install agentic-spliceai
pip install -e .

# Verify
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from agentic_spliceai.splice_engine.base_layer import BaseModelRunner; print('✅ OK')"
```

### Step 7: Transfer Data (From Local Machine)

Open a **new terminal** on your local machine:

```bash
# Transfer genomic data
rsync -avzP ~/work/agentic-spliceai/data/ \
  runpod-agentic-spliceai-a40-48gb:/workspace/data/

# This can take 10-30 minutes depending on data size
```

### Step 8: Start Training (On Pod)

Back in your SSH session:

```bash
# Use tmux (survives disconnection)
tmux new -s training

# Activate environment
cd /workspace/agentic-spliceai
mamba activate agenticspliceai

# Run training
python train.py --config configs/meta_layer_training.yaml

# Detach from tmux: Ctrl-B, then D
# Reattach later: tmux attach -t training
```

---

## 📋 Execution Model

### LOCAL (Your Machine)

These scripts/commands run on **your local machine**:

| Script | Purpose |
|--------|---------|
| `runpod_ssh_manager.sh` | Configure SSH access |
| `quick_pod_setup.sh` | Automated setup |
| `rsync` commands | Transfer data to/from pod |

**Location**: `~/work/agentic-spliceai/runpods/scripts/`  
**Modifies**: `~/.ssh/config` on your machine

### POD (RunPods Instance)

These run **ON the pod** (after SSH'ing):

- Installing Miniforge
- Cloning repository
- Creating conda environment
- Installing packages
- Running training scripts

---

## 💡 Tips

### Use tmux Always

```bash
# Start session
ssh runpod-agentic-spliceai-a40-48gb -t "tmux new -s work || tmux attach -t work"

# Why? Training continues even if SSH drops
```

### Monitor GPU

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or from local machine
ssh runpod-agentic-spliceai-a40-48gb "nvidia-smi"
```

### Check Costs

- RunPods dashboard shows $/hour
- Set up billing alerts
- **Terminate when not training** (pay only for compute time)

---

## 🔒 Privacy & Security

### Why runpods/ is NOT in Git

The `runpods/` directory contains:
- ❌ User-specific paths (`~/work/...`)
- ❌ SSH configuration history
- ❌ Personal workflow customizations
- ❌ Potentially sensitive information

### What IS Shared

- ✅ `runpods.example/` - Templates you can copy
- ✅ `docs/RUNPODS_SETUP.md` - This guide
- ✅ `environment-runpods-minimal.yml` - Public conda env template

---

## ❓ FAQ

### Q: Where do the scripts run?

**A**: Scripts in `runpods/scripts/` run **LOCALLY** (your machine). They configure SSH access **TO** pods. They do NOT run on the pod itself.

### Q: What about ~/work/scripts/runpod_manager.sh?

**A**: That's from an old design. **Ignore those references** in documentation. All scripts are now self-contained in `runpods/scripts/`.

### Q: Can I share my runpods/ directory?

**A**: **NO** - it contains personal configuration. Share the `runpods.example/` directory instead, which is tracked in git.

### Q: What if my workspace is not ~/work/?

**A**: The scripts work from any location. Just `cd` to your project and use relative paths.

---

## 🔄 Updating Your Setup

If we update the RunPods scripts:

```bash
cd ~/work/agentic-spliceai

# Get latest from git
git pull

# Update your runpods/ from example
cp -r runpods.example/scripts/* runpods/scripts/
cp runpods.example/environment-runpods-minimal.yml runpods/
```

---

## 📚 Additional Resources

- **Complete Workflow**: `runpods.example/AGENTIC_SPLICEAI_QUICK_START.md`
- **Customization**: `runpods.example/CUSTOMIZATION_NOTES.md`
- **RunPods Docs**: [docs.runpods.io](https://docs.runpods.io)

---

**Created**: January 28, 2026  
**Status**: Production-ready for agentic-spliceai
