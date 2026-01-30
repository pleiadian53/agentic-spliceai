# Local Development Workflow (Recommended)

**⚡ This is the fastest, simplest way to work with RunPods for most users.**

Develop code locally (where git is already set up), then sync to pod only for training. No GitHub SSH setup on pod needed!

---

## 🎯 Why This Approach?

### Advantages Over Pod-Based Development

✅ **Faster** - No SSH latency, instant feedback  
✅ **Familiar** - Use your IDE, debugger, git setup  
✅ **Cost-effective** - Don't pay for pod while coding  
✅ **Simpler** - No git/SSH config on pod  
✅ **Safer** - Code versioned locally before pod training  

### When to Use This

- ✅ **Model development** - Iterating on architecture, configs
- ✅ **Code changes** - Bug fixes, features, refactoring
- ✅ **Testing** - Unit tests, integration tests
- ✅ **Documentation** - README updates, docstrings

### When You Might Need Pod Git Instead

- ❌ Making code changes mid-training (rare)
- ❌ Collaborative training with live updates (advanced)
- ❌ Auto-committing results from experiments (specialized)

**For 95% of users**: This local development workflow is better.

---

## ⚡ Quick Start (5 Minutes)

### Step 1: Install rsync on Pod (One-Time Setup)

SSH to your pod and install rsync:

```bash
# On pod
ssh runpod-agentic-spliceai

# Install rsync (Ubuntu/Debian)
apt-get update && apt-get install -y rsync

# Verify installation
rsync --version
# Should show: rsync  version 3.x.x  protocol version 31
```

**Alternative for other distros**:
```bash
# CentOS/RHEL
yum install -y rsync

# Alpine
apk add rsync
```

### Step 2: Initial Code Sync

```bash
# On LOCAL machine
rsync -avz --exclude='data/' --exclude='.git/' --exclude='__pycache__/' \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

### Step 3: Setup Environment (On Pod)

```bash
# On pod
cd /workspace/agentic-spliceai
mamba env create -f environment.yml
mamba activate agenticspliceai
pip install -e .
```

**Done!** You're ready to develop locally and train on pod.

---

## 🔄 Daily Development Cycle

### Workflow Overview

```
┌─────────────┐
│   LOCAL     │  1. Edit code, test, commit
│ Development │  2. Push to GitHub
└──────┬──────┘
       │ rsync (fast, ~seconds)
       ▼
┌─────────────┐
│     POD     │  3. Train with GPU
│  Training   │  4. Download results
└─────────────┘
       │ rsync results back
       ▼
┌─────────────┐
│   LOCAL     │  5. Analyze, iterate
│  Analysis   │
└─────────────┘
```

### Step-by-Step Example

#### 1. Develop Locally

```bash
# On LOCAL machine
cd ~/work/agentic-spliceai

# Edit code
vim src/agentic_spliceai/models/splicing_model.py

# Test locally (if you have test data)
pytest tests/unit/

# Commit changes
git add .
git commit -m "Improve attention mechanism in splicing model"
git push origin main
```

#### 2. Sync Code to Pod

```bash
# On LOCAL machine
rsync -avz --exclude='data/' --exclude='.git/' --exclude='__pycache__/' \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/

# Output shows what was updated:
# sending incremental file list
# src/agentic_spliceai/models/splicing_model.py
# sent 12,543 bytes  received 42 bytes  8,390.00 bytes/sec
```

**Explanation of excludes**:
- `--exclude='data/'` - Don't sync large data files (upload separately)
- `--exclude='.git/'` - Don't sync git history (saves time/space)
- `--exclude='__pycache__/'` - Don't sync Python bytecode

#### 3. Train on Pod

```bash
# On pod (in existing SSH session or new one)
ssh runpod-agentic-spliceai
cd /workspace/agentic-spliceai
mamba activate agenticspliceai

# Start training
python train.py --config configs/full_chromosome.yaml

# Or in tmux (so you can disconnect)
tmux new -s training
python train.py --config configs/full_chromosome.yaml
# Ctrl+B, D to detach
```

#### 4. Sync Results Back to Local

```bash
# On LOCAL machine (while training or after completion)

# Option A: Sync entire output directory
rsync -avzP runpod-agentic-spliceai:/workspace/agentic-spliceai/output/ \
  ~/work/agentic-spliceai/output/

# Option B: Sync specific run
rsync -avzP runpod-agentic-spliceai:/workspace/agentic-spliceai/output/20260128_150235/ \
  ~/work/agentic-spliceai/output/20260128_150235/

# Option C: Download just checkpoints
rsync -avzP runpod-agentic-spliceai:/workspace/agentic-spliceai/checkpoints/ \
  ~/work/agentic-spliceai/checkpoints/
```

**Note**: `-P` flag shows progress and allows resuming interrupted transfers (useful for large files).

#### 5. Analyze Locally

```bash
# On LOCAL machine
cd ~/work/agentic-spliceai

# Analyze results with your local tools
jupyter notebook notebooks/analyze_results.ipynb

# Iterate and repeat!
```

---

## 📁 Data Management

### Initial Data Upload (One-Time, Large Transfer)

```bash
# Upload training data to pod (can take hours for large datasets)
rsync -avzP ~/work/agentic-spliceai/data/ \
  runpod-agentic-spliceai:/workspace/data/

# Or upload specific subdirectories
rsync -avzP ~/work/agentic-spliceai/data/ensembl/ \
  runpod-agentic-spliceai:/workspace/data/ensembl/
```

**Tip**: Use `tmux` locally so transfer continues if you close terminal:
```bash
tmux new -s upload
rsync -avzP ~/work/agentic-spliceai/data/ runpod-agentic-spliceai:/workspace/data/
# Ctrl+B, D to detach
```

### Where Should Data Live?

**Recommended structure on pod**:
```
/workspace/
├── agentic-spliceai/          # Code (synced frequently)
│   ├── src/
│   ├── configs/
│   └── output/                # Training outputs
└── data/                      # Data (uploaded once, shared across runs)
    ├── ensembl/
    ├── mane/
    └── spliceai_analysis/
```

**Why separate `data/` from code**:
- Data is large, uploaded once
- Code changes frequently, fast to sync
- Multiple projects can share same data directory

### Configuring Data Path

Update your code to find data:

```python
# In your config or code
DATA_DIR = os.environ.get('DATA_DIR', '/workspace/data')
```

Or use symlink:
```bash
# On pod
ln -s /workspace/data ~/work/agentic-spliceai/data
```

---

## 🚀 Advanced Tips

### Create an Alias for Quick Sync

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Sync code to agentic-spliceai pod
alias sync-agentic='rsync -avz --exclude="data/" --exclude=".git/" --exclude="__pycache__/" --exclude="*.pyc" --exclude=".pytest_cache/" --exclude=".ruff_cache/" ~/work/agentic-spliceai/ runpod-agentic-spliceai:/workspace/agentic-spliceai/'

# Sync results back
alias sync-results='rsync -avzP runpod-agentic-spliceai:/workspace/agentic-spliceai/output/ ~/work/agentic-spliceai/output/'
```

**Usage**:
```bash
# Quick sync (takes seconds)
sync-agentic

# Download results
sync-results
```

### Watch for Changes and Auto-Sync

For active development, auto-sync on file changes:

```bash
# Install fswatch (macOS)
brew install fswatch

# Auto-sync on changes
fswatch -o ~/work/agentic-spliceai/src | while read; do
  echo "Changes detected, syncing..."
  sync-agentic
done
```

**Linux alternative** using `inotifywait`:
```bash
# Install inotify-tools
sudo apt-get install inotify-tools

# Auto-sync
while inotifywait -r -e modify ~/work/agentic-spliceai/src; do
  sync-agentic
done
```

### Dry-Run Before Syncing

Preview what will be transferred:

```bash
# Show what would be synced (no actual transfer)
rsync -avzn --exclude='data/' --exclude='.git/' \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

The `-n` flag means "dry-run" (no changes made).

### Exclude Additional Files

Create an exclude file for complex patterns:

```bash
# Create exclude file
cat > ~/rsync-exclude.txt <<'EOF'
data/
.git/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.DS_Store
*.log
output/*/checkpoints/*.pt  # Exclude large checkpoint files
EOF

# Use exclude file
rsync -avz --exclude-from=~/rsync-exclude.txt \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

### Sync Only Specific Directories

```bash
# Sync only source code
rsync -avz ~/work/agentic-spliceai/src/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/src/

# Sync only configs
rsync -avz ~/work/agentic-spliceai/configs/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/configs/
```

---

## 🔄 Bidirectional Sync Strategy

### When to Sync Each Direction

**Local → Pod** (frequent):
- Code changes
- Config updates
- New scripts

**Pod → Local** (after training):
- Model checkpoints
- Training logs
- Output files
- Metrics/plots

### Avoiding Conflicts

**Rule of thumb**: Keep clear ownership:
- **Code lives locally** - Always edit locally, sync to pod
- **Outputs live on pod** - Generate on pod, sync to local
- **Never edit same file in both places**

### What If You Made Changes on Pod?

If you accidentally edited code on pod:

```bash
# 1. Sync pod changes to a temporary location
rsync -avzP runpod-agentic-spliceai:/workspace/agentic-spliceai/src/ \
  ~/work/temp-pod-changes/

# 2. Review differences
diff -r ~/work/agentic-spliceai/src/ ~/work/temp-pod-changes/

# 3. Manually merge important changes
cp ~/work/temp-pod-changes/important_file.py ~/work/agentic-spliceai/src/

# 4. Commit locally
cd ~/work/agentic-spliceai
git add src/important_file.py
git commit -m "Merge changes from pod"

# 5. Re-sync to pod
rsync -avz --exclude='data/' --exclude='.git/' \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

**Better approach**: Avoid editing on pod. Use it only for training.

---

## 📊 Typical File Sizes and Transfer Times

| What | Size | Upload Time | Download Time |
|------|------|-------------|---------------|
| **Code changes** | 1-10 MB | 1-5 seconds | 1-5 seconds |
| **Full codebase** | ~50 MB | 5-30 seconds | 5-30 seconds |
| **Small dataset** | 1-5 GB | 5-30 minutes | 5-30 minutes |
| **Medium dataset** | 10-50 GB | 1-5 hours | 1-5 hours |
| **Large dataset** | 100+ GB | 5-24 hours | 5-24 hours |
| **Model checkpoint** | 100-500 MB | 30 seconds - 2 min | 30 seconds - 2 min |
| **Training outputs** | 1-10 GB | 5-30 minutes | 5-30 minutes |

**Optimization tips**:
- Code syncs are fast (seconds)
- Data uploads are slow (do once)
- Compress large files: `tar -czf data.tar.gz data/` before transfer

---

## 🆚 Comparison: rsync vs Git on Pod

| Aspect | rsync (Local Dev) | Git on Pod |
|--------|-------------------|------------|
| **Setup time** | 1 minute | 5-10 minutes |
| **Pod setup** | Install rsync only | SSH keys + git config |
| **Code sync** | 1-5 seconds | `git pull` (~5-30 seconds) |
| **Works offline** | Yes (local dev) | No (needs GitHub) |
| **Version control** | Yes (local git) | Yes (pod git) |
| **Pod cost** | Low (coding free) | Higher (pod running) |
| **Best for** | Most users | Advanced workflows |

---

## 🚨 Troubleshooting

### Problem: `rsync: command not found` on Pod

**Solution**: Install rsync on pod:
```bash
ssh runpod-agentic-spliceai
apt-get update && apt-get install -y rsync
```

### Problem: Transfer Very Slow

**Causes & Solutions**:

1. **Large `.git/` directory being synced**
   ```bash
   # Always exclude .git/
   rsync -avz --exclude='.git/' ...
   ```

2. **Transferring data files unnecessarily**
   ```bash
   # Exclude data directory
   rsync -avz --exclude='data/' ...
   ```

3. **Many small files**
   ```bash
   # Compress during transfer
   rsync -avz --compress-level=9 ...
   ```

### Problem: Permission Denied

```bash
# Ensure you can SSH to pod
ssh runpod-agentic-spliceai

# Check pod path exists
ssh runpod-agentic-spliceai "ls -la /workspace/"

# Create directory if needed
ssh runpod-agentic-spliceai "mkdir -p /workspace/agentic-spliceai"
```

### Problem: Files Out of Sync

**Verify sync**:
```bash
# Check what's on pod
ssh runpod-agentic-spliceai "ls -R /workspace/agentic-spliceai/src/"

# Compare file checksums
rsync -avzn --checksum ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

**Force full re-sync**:
```bash
# Delete and re-sync (be careful!)
ssh runpod-agentic-spliceai "rm -rf /workspace/agentic-spliceai"
rsync -avz --exclude='data/' --exclude='.git/' \
  ~/work/agentic-spliceai/ \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/
```

### Problem: Can't Download Large Results

```bash
# Use compression and resume support
rsync -avzP --compress-level=9 \
  runpod-agentic-spliceai:/workspace/agentic-spliceai/output/ \
  ~/work/agentic-spliceai/output/

# If interrupted, just re-run same command (will resume)
```

---

## 📋 Complete Setup Checklist

### One-Time Pod Setup

- [ ] Install rsync on pod
- [ ] Upload data to pod (if needed)
- [ ] Sync initial codebase
- [ ] Create conda environment
- [ ] Test training script

### Per-Session Workflow

- [ ] Edit code locally
- [ ] Test locally (if possible)
- [ ] Commit changes to git
- [ ] Sync code to pod (`rsync`)
- [ ] SSH to pod
- [ ] Start training
- [ ] Sync results back
- [ ] Analyze locally

---

## 🎯 Summary

**Key Points**:

1. ✅ **Develop locally** - Faster, familiar, free
2. ✅ **Sync with rsync** - Fast (seconds), simple
3. ✅ **Train on pod** - Use GPU only when needed
4. ✅ **Download results** - Analyze locally
5. ✅ **One-time data upload** - Don't re-sync large files

**Time Investment**:
- Setup: 5 minutes (install rsync)
- Per sync: 1-5 seconds
- Per training session: ~30 seconds overhead

**Cost Savings**:
- Don't pay for pod while developing
- Only use GPU for actual training

**This is the recommended workflow for 95% of users!**

---

## 🔗 Related Documentation

- **Pod SSH Setup**: `../README.md` - Connecting to pod
- **GitHub SSH** (if needed): `GITHUB_SSH_SETUP.md` - Advanced workflow
- **Quick Reference**: `../scripts/runpod_ssh_manager.sh` - SSH management

---

**Created**: January 28, 2026  
**Version**: 1.0.0  
**Recommended**: Yes (default workflow)
