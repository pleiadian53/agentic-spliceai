# rsync Quick Reference for This Pod

Quick commands for syncing code and results between your local machine and this RunPods instance.

---

## 📤 Sync Code FROM Local TO Pod

**Standard sync** (run on your **local machine**):
```bash
rsync -avz --exclude='data/' --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/' \
  ~/work/agentic-spliceai/ \
  <POD_SSH_ALIAS>:/workspace/agentic-spliceai/
```

**Sync only source code**:
```bash
rsync -avz ~/work/agentic-spliceai/src/ \
  <POD_SSH_ALIAS>:/workspace/agentic-spliceai/src/
```

**Sync only configs**:
```bash
rsync -avz ~/work/agentic-spliceai/configs/ \
  <POD_SSH_ALIAS>:/workspace/agentic-spliceai/configs/
```

**Dry run** (see what would be transferred without actually doing it):
```bash
rsync -avzn --exclude='data/' --exclude='.git/' --exclude='__pycache__/' \
  ~/work/agentic-spliceai/ \
  <POD_SSH_ALIAS>:/workspace/agentic-spliceai/
```

---

## 📥 Sync Results FROM Pod TO Local

**Download training outputs** (run on your **local machine**):
```bash
rsync -avzP <POD_SSH_ALIAS>:/workspace/agentic-spliceai/output/ \
  ~/work/agentic-spliceai/output/
```

**Download specific run**:
```bash
rsync -avzP <POD_SSH_ALIAS>:/workspace/agentic-spliceai/output/20260128_150235/ \
  ~/work/agentic-spliceai/output/20260128_150235/
```

**Download checkpoints only**:
```bash
rsync -avzP <POD_SSH_ALIAS>:/workspace/agentic-spliceai/checkpoints/ \
  ~/work/agentic-spliceai/checkpoints/
```

---

## 🚀 Recommended Aliases

Add to your `~/.bashrc` or `~/.zshrc` **on your local machine**:

```bash
# Replace <POD_SSH_ALIAS> with your actual SSH alias (e.g., runpod-agentic-spliceai)
export POD_SSH="<POD_SSH_ALIAS>"

# Sync code to pod
alias sync-to-pod='rsync -avz --exclude="data/" --exclude=".git/" --exclude="__pycache__/" --exclude="*.pyc" --exclude=".pytest_cache/" --exclude=".ruff_cache/" ~/work/agentic-spliceai/ $POD_SSH:/workspace/agentic-spliceai/'

# Sync results from pod
alias sync-from-pod='rsync -avzP $POD_SSH:/workspace/agentic-spliceai/output/ ~/work/agentic-spliceai/output/'

# Dry run to see what would sync
alias sync-check='rsync -avzn --exclude="data/" --exclude=".git/" --exclude="__pycache__/" ~/work/agentic-spliceai/ $POD_SSH:/workspace/agentic-spliceai/'
```

**Usage** (after adding aliases):
```bash
sync-to-pod      # Quick sync code to pod
sync-from-pod    # Download results from pod
sync-check       # Preview what would sync
```

---

## ⚡ Typical Workflow

### 1. Develop Locally
```bash
# On local machine
cd ~/work/agentic-spliceai
vim src/agentic_spliceai/models/splicing_model.py
git add .
git commit -m "Update model architecture"
git push
```

### 2. Sync to Pod
```bash
# On local machine (takes 1-5 seconds)
sync-to-pod
```

### 3. Train on Pod
```bash
# SSH to pod
ssh <POD_SSH_ALIAS>
cd /workspace/agentic-spliceai
mamba activate agenticspliceai
python train.py --config configs/full_chromosome.yaml
```

### 4. Download Results
```bash
# On local machine (while or after training)
sync-from-pod
```

### 5. Analyze Locally
```bash
# On local machine
cd ~/work/agentic-spliceai
jupyter notebook notebooks/analyze_results.ipynb
```

---

## 📊 rsync Flags Explained

| Flag | Meaning |
|------|---------|
| `-a` | Archive mode (preserves permissions, timestamps, etc.) |
| `-v` | Verbose (show file names being transferred) |
| `-z` | Compress during transfer |
| `-P` | Show progress + allow resuming interrupted transfers |
| `-n` | Dry run (no actual changes) |
| `--exclude='pattern'` | Skip files/dirs matching pattern |

---

## 💡 Tips

1. **Always sync code TO pod before training** - Ensures latest changes
2. **Use `-P` for large downloads** - Shows progress, allows resume
3. **Exclude large directories** - Don't sync `data/`, `.git/`, cache files
4. **Dry run first if unsure** - Use `-n` flag to preview
5. **Pod is temporary** - No need to commit/push from pod, do it locally

---

## 🔧 Current Pod Info

**Project location**: `/workspace/agentic-spliceai/`
**rsync version**: `3.2.7`
**Protocol**: `31`

---

## ❓ Troubleshooting

**Slow transfers?**
- Make sure you're excluding `.git/` and `data/`
- Use `--compress-level=9` for better compression

**Permission denied?**
- Check SSH access: `ssh <POD_SSH_ALIAS>`
- Verify path exists: `ssh <POD_SSH_ALIAS> "ls /workspace/"`

**Want to sync entire directory structure?**
```bash
# Include empty directories and permissions
rsync -avz --delete --exclude='data/' --exclude='.git/' \
  ~/work/agentic-spliceai/ \
  <POD_SSH_ALIAS>:/workspace/agentic-spliceai/
```

---

**Created**: January 28, 2026  
**Pod**: RunPods ephemeral instance  
**Workflow**: Local development → rsync → Pod training → rsync results back
