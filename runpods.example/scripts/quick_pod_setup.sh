#!/bin/bash
# Quick Pod Setup - Automate the entire pod setup process
# Usage: ./quick_pod_setup.sh [project_name]

set -e

PROJECT=${1:-"agentic-spliceai"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_MANAGER="$SCRIPT_DIR/runpod_ssh_manager.sh"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Helper functions
log_header() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} ${CYAN}$1${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

log_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Banner
clear
cat << 'EOF'
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        RunPods Quick Setup for Agentic-SpliceAI               ║
║        Automated Pod Configuration & Connection               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF

echo ""
echo -e "${CYAN}Project:${NC} $PROJECT"
echo ""

# Check prerequisites
log_header "Checking Prerequisites"

log_step "1" "Checking SSH manager"
if [ ! -f "$SSH_MANAGER" ]; then
    log_error "SSH manager not found at: $SSH_MANAGER"
    exit 1
fi
log_success "SSH manager found"

log_step "2" "Checking SSH key"
SSH_KEY="$HOME/.ssh/id_ed25519"
if [ ! -f "$SSH_KEY" ]; then
    SSH_KEY="$HOME/.ssh/id_rsa"
    if [ ! -f "$SSH_KEY" ]; then
        log_warn "No SSH key found. You may need to generate one."
        read -p "Generate SSH key now? (y/n): " gen_key
        if [ "$gen_key" = "y" ]; then
            ssh-keygen -t ed25519 -C "runpods-$PROJECT" -f "$HOME/.ssh/id_ed25519"
            SSH_KEY="$HOME/.ssh/id_ed25519"
        else
            log_error "SSH key required for pod access"
            exit 1
        fi
    fi
fi
log_success "SSH key: $SSH_KEY"

# Add SSH config
log_header "Configure SSH Connection"

log_info "Adding SSH configuration for $PROJECT pod..."
echo ""

"$SSH_MANAGER" add "$PROJECT"

if [ $? -ne 0 ]; then
    log_error "Failed to add SSH configuration"
    exit 1
fi

# Extract the host alias that was just created
HOST_ALIAS=$(grep -A1 "# RunPods: $PROJECT" ~/.ssh/config | tail -1 | awk '{print $2}')

if [ -z "$HOST_ALIAS" ]; then
    log_error "Could not determine host alias. Check SSH config manually."
    exit 1
fi

log_success "SSH config added: $HOST_ALIAS"

# Test connection
log_header "Testing Connection"

log_step "3" "Connecting to pod..."

if ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$HOST_ALIAS" "echo 'Connection test successful'" >/dev/null 2>&1; then
    log_success "Connection successful!"
else
    log_error "Connection failed. Please check:"
    echo "  1. Pod is running in RunPods dashboard"
    echo "  2. Hostname and port are correct"
    echo "  3. SSH key is uploaded to RunPods"
    exit 1
fi

# Get pod info
log_header "Pod Information"

log_info "Gathering pod details..."
echo ""

ssh "$HOST_ALIAS" "
echo '=== System Information ==='
echo 'Hostname:' \$(hostname)
echo 'Kernel:' \$(uname -r)
echo ''
echo '=== GPU Information ==='
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ''
echo '=== Disk Space ==='
df -h /workspace | tail -1 | awk '{print \"Available:\", \$4, \"out of\", \$2}'
echo ''
echo '=== Memory ==='
free -h | grep Mem | awk '{print \"Available:\", \$7, \"out of\", \$2}'
" 2>/dev/null || log_warn "Could not retrieve all pod information"

# Setup checklist
log_header "Setup Checklist"

cat << EOF
Next steps to complete pod setup:

□ Install Miniforge
  ssh $HOST_ALIAS
  cd /workspace
  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p /workspace/miniforge3
  /workspace/miniforge3/bin/conda init bash
  source ~/.bashrc

□ Clone repository
  cd /workspace
  git clone https://github.com/pleiadian53/agentic-spliceai.git
  cd agentic-spliceai

□ Create environment
  mamba env create -f environment-runpods-minimal.yml
  mamba activate agenticspliceai
  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers einops
  pip install -e .

□ Transfer data
  # From local machine:
  rsync -avzP ~/work/$PROJECT/data/ $HOST_ALIAS:/workspace/data/

□ Start training
  ssh $HOST_ALIAS -t "tmux new -s training"
  cd /workspace/agentic-spliceai
  mamba activate agenticspliceai
  # Run your training commands

EOF

# Quick actions menu
log_header "Quick Actions"

cat << EOF
Available commands:

1. Connect to pod:
   ssh $HOST_ALIAS

2. Connect with tmux:
   ssh $HOST_ALIAS -t "tmux new -s main || tmux attach -t main"

3. Transfer data:
   rsync -avzP ~/work/$PROJECT/data/ $HOST_ALIAS:/workspace/data/

4. Execute remote command:
   ssh $HOST_ALIAS "YOUR_COMMAND"

5. Port forwarding (Jupyter):
   ssh -L 8888:localhost:8888 $HOST_ALIAS

6. Port forwarding (MLflow):
   ssh -L 5000:localhost:5000 $HOST_ALIAS

7. View pod details:
   $SSH_MANAGER list

8. Remove pod config:
   $SSH_MANAGER remove

EOF

# Interactive actions
echo ""
read -p "$(echo -e ${CYAN}What would you like to do?${NC}) [connect/transfer/help/exit]: " action

case $action in
    connect|c)
        log_info "Connecting to pod..."
        ssh "$HOST_ALIAS"
        ;;
    
    transfer|t)
        log_info "Starting data transfer..."
        read -p "Source directory [~/work/$PROJECT/data/]: " src_dir
        src_dir=${src_dir:-~/work/$PROJECT/data/}
        
        read -p "Destination directory [/workspace/data/]: " dst_dir
        dst_dir=${dst_dir:-/workspace/data/}
        
        log_info "Transferring: $src_dir -> $HOST_ALIAS:$dst_dir"
        rsync -avzP "$src_dir" "$HOST_ALIAS:$dst_dir"
        log_success "Transfer complete!"
        ;;
    
    help|h)
        log_info "Documentation available at:"
        echo "  - $SCRIPT_DIR/README.md"
        echo "  - $SCRIPT_DIR/RUNPOD_QUICK_REFERENCE.md"
        echo "  - $SCRIPT_DIR/RUNPOD_SSH_MANAGER_GUIDE.md"
        ;;
    
    exit|e|quit|q)
        log_info "Setup complete!"
        ;;
    
    *)
        log_info "No action selected. Setup complete!"
        ;;
esac

# Final summary
log_header "Setup Summary"

cat << EOF
✓ SSH configuration added
✓ Connection tested and working
✓ Pod information retrieved

Host alias: $HOST_ALIAS
Quick connect: ssh $HOST_ALIAS

For complete setup guide, see:
  $SCRIPT_DIR/../../meta_spliceai/splice_engine/meta_layer/docs/setup/RUNPODS_COMPLETE_SETUP.md

Happy training! 🚀

EOF

exit 0
