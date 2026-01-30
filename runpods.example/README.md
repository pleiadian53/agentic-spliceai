# RunPods Setup for agentic-spliceai

**This is an EXAMPLE** - Copy and customize for your setup.

---

## ⚠️ Important: This Directory is NOT Tracked

The `runpods/` directory contains user-specific configuration and is excluded from git via `.gitignore`.

This `runpods.example/` directory provides templates you can copy.

---

## 🚀 Quick Setup

### Step 1: Copy Example Directory

```bash
cd ~/work/agentic-spliceai
cp -r runpods.example runpods
```

### Step 2: Customize for Your Setup

Edit `runpods/` files to match your environment:

1. **`quick_pod_setup.sh`** - Verify repository URL
2. **`environment-runpods-minimal.yml`** - Adjust if needed
3. **Documentation** - Update paths if your workspace is not `~/work/`

### Step 3: Use Scripts

```bash
cd runpods/scripts
./runpod_ssh_manager.sh add agentic-spliceai
```

---

## 📋 What's Included

### Scripts (Self-Contained)

All scripts work **within the project** - no external dependencies:

```
runpods/scripts/
├── runpod_ssh_manager.sh      # SSH config manager
├── quick_pod_setup.sh          # Automated setup
└── test_runpod_manager.sh      # Test suite
```

These run **LOCALLY** (on your machine), not on the pod.

### Environment Configuration

```
runpods/environment-runpods-minimal.yml  # Minimal conda env for pods
```

### Documentation

```
runpods/docs/                            # Setup guides
runpods/docs/GITHUB_SSH_SETUP.md        # GitHub SSH access (essential!)
runpods/AGENTIC_SPLICEAI_QUICK_START.md # Complete workflow
runpods/CUSTOMIZATION_NOTES.md          # How to customize
```

---

## 🔧 Execution Model

### LOCAL Scripts (Your Machine)

These run on **your local machine**:

| Script | Purpose | Modifies |
|--------|---------|----------|
| `runpod_ssh_manager.sh` | Configure SSH | `~/.ssh/config` (local) |
| `quick_pod_setup.sh` | Automated setup | `~/.ssh/config` (local) |
| `test_runpod_manager.sh` | Test scripts | Nothing |

**Why local?**
- Configure SSH access **TO** pods
- Manage SSH keys on your machine
- Store connection history locally

### POD Scripts (RunPods Instance)

These run **ON the pod**:

```bash
# After SSH'ing to pod
ssh runpod-agentic-spliceai-a40-48gb

# 1. Setup GitHub SSH (for cloning private repos and pushing)
# See: runpods.example/docs/GITHUB_SSH_SETUP.md for detailed guide
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub  # Add this to https://github.com/settings/ssh/new

# 2. Configure SSH and Git
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF

git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

# 3. Clone and setup environment
cd /workspace
git clone git@github.com:pleiadian53/agentic-spliceai.git
cd agentic-spliceai
mamba env create -f runpods/environment-runpods-minimal.yml
mamba activate agenticspliceai
```

---

## 🔒 Why Not Track runpods/?

### Privacy Reasons

1. **User-specific paths**: References to `~/work/`, your projects
2. **SSH history**: JSON files with connection details
3. **Personal workflow**: Customizations for your setup
4. **Local configuration**: Not universally applicable

### What Should Be Shared?

✅ **Templates** (`runpods.example/`) - Share structure  
✅ **Documentation** (public docs) - How to set up  
❌ **Your runpods/** - Personal configuration

---

## 📚 Documentation

### For First-Time Setup

1. **Copy example**: `cp -r runpods.example runpods`
2. **Read**: `runpods/AGENTIC_SPLICEAI_QUICK_START.md`
3. **Customize**: Update repository URL if needed
4. **Test**: `cd runpods/scripts && ./test_runpod_manager.sh`

### Key Files to Customize

| File | What to Change |
|------|----------------|
| `scripts/quick_pod_setup.sh` | Repository URL (line ~166) |
| `environment-runpods-minimal.yml` | Env name if different |
| `docs/` | Paths if workspace not `~/work/` |

---

## 🎯 Typical Workflow

### 1. Initial Setup (Once)

```bash
# On local machine
cd ~/work/agentic-spliceai
cp -r runpods.example runpods
cd runpods/scripts
```

### 2. Configure Pod Access (Per Pod)

```bash
# On local machine
./runpod_ssh_manager.sh add agentic-spliceai

# Enter pod details when prompted:
# - Hostname: ssh.runpods.io
# - Port: 12345
# - Nickname: a40-48gb
```

### 3. Connect and Setup (On Pod)

```bash
# On local machine
ssh runpod-agentic-spliceai-a40-48gb

# Now on pod - setup GitHub SSH access first (see docs/GITHUB_SSH_SETUP.md)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""
cat ~/.ssh/id_ed25519_github.pub  # Add to GitHub: https://github.com/settings/ssh/new

# Configure SSH and Git
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF

git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

# Clone and setup
cd /workspace
git clone git@github.com:pleiadian53/agentic-spliceai.git
cd agentic-spliceai
mamba env create -f runpods/environment-runpods-minimal.yml
mamba activate agenticspliceai
pip install -e .
```

### 4. Transfer Data

```bash
# On local machine (new terminal)
rsync -avzP ~/work/agentic-spliceai/data/ \
  runpod-agentic-spliceai-a40-48gb:/workspace/data/
```

---

## ❓ FAQ

### Q: Why cp -r instead of git?

**A**: The `runpods/` directory is in `.gitignore` so it's not tracked. You copy the example and customize it for your setup.

### Q: What if I have a different workspace structure?

**A**: Edit the documentation files to reflect your paths. The scripts are self-contained and work within the project.

### Q: Can I share my runpods/ setup?

**A**: Only share if you've removed all personal information (paths, SSH history, etc.). Better to share customization tips in wiki/docs.

### Q: Do I need ~/work/scripts/?

**A**: **NO!** That's from the old design. All scripts are now self-contained in `runpods/scripts/`. Documentation may reference old paths - those are outdated.

---

## 🔄 Migration from Old Setup

If you have an existing `~/work/scripts/` setup:

```bash
# Old way (universal scripts)
~/work/scripts/runpod_manager.sh add project

# New way (project-contained)
cd ~/work/agentic-spliceai/runpods/scripts
./runpod_ssh_manager.sh add agentic-spliceai
```

**Benefits of new way**:
- ✅ Self-contained per project
- ✅ Version controlled (via example/)
- ✅ No external dependencies
- ✅ Easier to share/document

---

## 📝 Contributing

If you improve the RunPods setup:

1. Update `runpods.example/` (tracked)
2. **Don't** commit your `runpods/` (ignored)
3. Document changes in `runpods.example/README.md`
4. Share via PR to the example directory

---

**Last Updated**: January 28, 2026  
**Status**: Template - Copy and customize for your use
