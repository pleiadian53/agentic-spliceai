# GitHub SSH Setup for RunPods

Complete guide for setting up SSH access to GitHub on a new RunPods instance, enabling you to clone private repositories and push changes.

---

## 🎯 Why SSH Instead of HTTPS?

### HTTPS Limitations
```bash
# HTTPS clone - read-only for public repos
git clone https://github.com/pleiadian53/agentic-spliceai.git
# ❌ Can't push without Personal Access Token (PAT)
# ❌ PAT needs manual entry or credential storage
# ❌ PATs expire and need rotation
```

### SSH Benefits
```bash
# SSH clone - full read/write access
git clone git@github.com:pleiadian53/agentic-spliceai.git
# ✅ Push/pull without passwords
# ✅ Key-based authentication (more secure)
# ✅ One-time setup per pod
# ✅ Standard practice for development
```

---

## ⚡ Quick Setup (5 Minutes)

### Prerequisites
- ✅ RunPods instance running
- ✅ SSH access to pod configured (via `runpod_ssh_manager.sh`)
- ✅ GitHub account

### Step 1: Generate SSH Key on Pod

SSH into your pod and run:

```bash
# Generate Ed25519 key (recommended - smaller, more secure than RSA)
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""

# Alternative: Add a comment to identify the key (optional)
ssh-keygen -t ed25519 -C "pleiadian53-runpod" -f ~/.ssh/id_ed25519_github -N ""
```

**Explanation**:
- `-t ed25519`: Key type (modern, secure)
- `-f ~/.ssh/id_ed25519_github`: File location
- `-N ""`: No passphrase (convenient for pods, which are ephemeral)
- `-C "comment"`: Optional label to identify the key

**Output**:
```
Generating public/private ed25519 key pair.
Your identification has been saved in /root/.ssh/id_ed25519_github
Your public key has been saved in /root/.ssh/id_ed25519_github.pub
```

### Step 2: Display Public Key

```bash
cat ~/.ssh/id_ed25519_github.pub
```

**Copy the entire output** (it looks like):
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJqJw... pleiadian53-runpod
```

### Step 3: Add Key to GitHub

1. **Go to**: https://github.com/settings/ssh/new
2. **Title**: "RunPods A40 - Jan 2026" (or any descriptive name)
3. **Key**: Paste the public key from Step 2
4. **Click**: "Add SSH key"

**Tip**: Use descriptive titles like "RunPods-A40-2026-01-27" so you can track and remove old keys later.

### Step 4: Configure SSH

Create SSH config to use the new key for GitHub:

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF
```

**Explanation**:
- `Host github.com`: Applies to all github.com connections
- `IdentityFile`: Which SSH key to use
- `StrictHostKeyChecking no`: Auto-accept GitHub's host key (safe for GitHub)

### Step 5: Test SSH Connection

```bash
ssh -T git@github.com
```

**Expected output**:
```
Hi pleiadian53! You've successfully authenticated, but GitHub does not provide shell access.
```

✅ If you see this, SSH is working!

### Step 6: Configure Git Identity

**Required for making commits**:

```bash
# Use your GitHub username
git config --global user.name "pleiadian53"

# Use GitHub's noreply email (recommended for privacy)
git config --global user.email "pleiadian53@users.noreply.github.com"
```

**About the Email**:
- `pleiadian53@users.noreply.github.com` is GitHub's **privacy email**
- Keeps your real email private while still letting you commit
- Format: `<username>@users.noreply.github.com`
- **Perfectly valid and recommended!**

**Verify settings**:
```bash
git config --global --list | grep user
```

### Step 7: Clone Repository

```bash
cd /workspace
git clone git@github.com:pleiadian53/agentic-spliceai.git
cd agentic-spliceai
```

✅ **Done!** You can now push and pull freely.

---

## 🔄 Testing Push Access

```bash
# Make a small test change
echo "# Pod test" >> README.md

# Stage and commit
git add README.md
git commit -m "Test commit from pod"

# Push
git push origin main
```

If this works, you have full read/write access! 🎉

---

## 🚨 Troubleshooting

### Problem: "Permission denied (publickey)"

```bash
# 1. Verify key exists
ls -la ~/.ssh/id_ed25519_github*

# 2. Verify SSH config
cat ~/.ssh/config | grep -A 4 "Host github.com"

# 3. Test with verbose output
ssh -vT git@github.com
```

**Common causes**:
- Key not added to GitHub account
- Wrong IdentityFile path in SSH config
- SSH config syntax error

### Problem: "Could not resolve hostname github.com"

```bash
# Test internet connectivity
ping -c 3 github.com
```

**Fix**: Check pod's network connection.

### Problem: "Author identity unknown"

```
*** Please tell me who you are.
Run: git config --global user.name "Your Name"
     git config --global user.email "you@example.com"
```

**Fix**: Run Step 6 (Configure Git Identity).

### Problem: "Host key verification failed"

```bash
# Remove old host key and try again
ssh-keygen -R github.com
ssh -T git@github.com
```

### Problem: Can clone but can't push

```
remote: Permission to pleiadian53/agentic-spliceai.git denied
```

**Causes**:
- SSH key not added to GitHub
- Using HTTPS URL instead of SSH URL

**Fix**:
```bash
# Check current remote URL
git remote -v

# If it shows https://, change to SSH:
git remote set-url origin git@github.com:pleiadian53/agentic-spliceai.git
```

---

## 📋 Complete Setup Script

For automated setup, create this script on your **pod**:

```bash
#!/bin/bash
# File: ~/setup_github_ssh.sh

set -e  # Exit on error

echo "🔑 Setting up GitHub SSH access..."

# 1. Generate key
if [ ! -f ~/.ssh/id_ed25519_github ]; then
    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""
    echo "✓ SSH key generated"
else
    echo "✓ SSH key already exists"
fi

# 2. Configure SSH
mkdir -p ~/.ssh
cat > ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  StrictHostKeyChecking no
EOF
echo "✓ SSH config updated"

# 3. Configure git
git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"
echo "✓ Git identity configured"

# 4. Display public key
echo ""
echo "📋 Add this public key to GitHub (https://github.com/settings/ssh/new):"
echo "─────────────────────────────────────────────────────────────────────"
cat ~/.ssh/id_ed25519_github.pub
echo "─────────────────────────────────────────────────────────────────────"
echo ""
echo "After adding the key to GitHub, test with: ssh -T git@github.com"
```

**Usage**:
```bash
# On pod
curl -o ~/setup_github_ssh.sh https://YOUR_GIST_URL/setup_github_ssh.sh
chmod +x ~/setup_github_ssh.sh
./setup_github_ssh.sh
```

---

## 🔐 Security Best Practices

### For Ephemeral Pods (Recommended)

Since RunPods instances are temporary:

1. **No passphrase on keys** - Convenient, acceptable risk for temporary compute
2. **Descriptive key titles** - Easy to identify and remove later
3. **Remove old keys** - Clean up at https://github.com/settings/keys after releasing pod

### For Long-Running Pods (Higher Security)

If you keep a pod for weeks/months:

```bash
# Generate key WITH passphrase
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github

# Add to ssh-agent (so you don't re-enter passphrase constantly)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519_github
```

### Key Rotation

**After releasing a pod**:
1. Go to: https://github.com/settings/keys
2. Find the key (e.g., "RunPods-A40-2026-01-27")
3. Click "Delete"

This ensures old pods can't access your repos.

---

## 🎯 Multi-Project Workflow

### Same SSH Key for Multiple Repos

The SSH key grants access to **all repos** your GitHub account can access:

```bash
cd /workspace

# Clone multiple projects with same key
git clone git@github.com:pleiadian53/agentic-spliceai.git
git clone git@github.com:pleiadian53/meta-spliceai.git
git clone git@github.com:pleiadian53/genai-lab.git
```

No additional setup needed! 🎉

### Per-Project Git Config (Optional)

If you want different names/emails per project:

```bash
# Global (default for all repos)
git config --global user.name "pleiadian53"
git config --global user.email "pleiadian53@users.noreply.github.com"

# Project-specific (overrides global)
cd /workspace/agentic-spliceai
git config user.name "Agentic AI Bot"
git config user.email "bot@example.com"
```

---

## 📊 Comparison: SSH vs HTTPS vs GitHub CLI

| Feature | SSH | HTTPS + PAT | GitHub CLI (`gh`) |
|---------|-----|-------------|-------------------|
| **Setup Time** | 5 min | 2 min | 3 min |
| **Authentication** | Key-based | Token | OAuth |
| **Push/Pull** | ✅ Seamless | ✅ With token | ✅ Seamless |
| **Token Expiration** | ❌ Never | ⚠️ Yes (rotate) | ⚠️ Yes |
| **Standard Practice** | ✅ Yes | ⚠️ Acceptable | ❌ Less common |
| **Works Offline** | ✅ Yes | ✅ Yes | ❌ No |
| **Recommended** | ✅ **Yes** | ⚠️ If SSH blocked | ❌ Not needed |

**Verdict**: SSH is the standard for development environments.

---

## 🔗 Related Documentation

- **Pod SSH Setup**: `START_HERE.md` - How to connect to the pod itself
- **Pod Environment**: `AGENTIC_SPLICEAI_QUICK_START.md` - Complete pod setup
- **Quick Setup Script**: `scripts/quick_pod_setup.sh` - Automated pod configuration

---

## 📝 Summary Checklist

Once setup is complete, verify:

- [ ] SSH key generated on pod
- [ ] Public key added to GitHub
- [ ] SSH config created
- [ ] `ssh -T git@github.com` succeeds
- [ ] Git identity configured
- [ ] Can clone via SSH
- [ ] Can push commits

**Estimated time**: 5 minutes  
**Frequency**: Once per new pod  
**Result**: Full Git push/pull access 🚀

---

## ❓ FAQ

### Q: Do I need `gh` CLI?

**A**: No. `gh` is convenient for creating PRs and managing issues, but **not required** for basic Git operations (clone, push, pull). SSH alone is sufficient.

### Q: Can I reuse my local machine's SSH key?

**A**: Not recommended. Each pod should have its own key:
- Easier to track and revoke
- Follows principle of least privilege
- No need to expose your main dev key

### Q: What email should I use for commits?

**A**: Use GitHub's noreply email: `<username>@users.noreply.github.com`
- Keeps your real email private
- Still attributes commits to your account
- Perfectly valid and recommended!

### Q: How do I copy/paste the public key?

**On pod via SSH**:
```bash
# Display key, then copy from terminal
cat ~/.ssh/id_ed25519_github.pub
```

**Alternative** (if you have `xclip` or `pbcopy`):
```bash
# Linux
cat ~/.ssh/id_ed25519_github.pub | xclip -selection clipboard

# macOS (if accessing pod from Mac terminal)
ssh runpod-agentic-spliceai "cat ~/.ssh/id_ed25519_github.pub" | pbcopy
```

### Q: Can I use the same key across multiple pods?

**A**: Yes, but not recommended:
- You'd need to copy the private key between pods (security risk)
- Hard to track which pod has which key
- Better: Generate a new key per pod (takes 1 minute)

### Q: What if I lose the private key?

**A**: No problem:
1. Delete the old public key from GitHub: https://github.com/settings/keys
2. Generate a new key pair on the new pod
3. Add the new public key to GitHub

Private keys are **never** recoverable - that's by design for security.

---

**Created**: January 28, 2026  
**Version**: 1.0.0  
**Applies to**: All RunPods instances
