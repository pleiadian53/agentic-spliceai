#!/bin/bash
# RunPods SSH Configuration Manager
# Purpose: Manage SSH configs for different RunPods instances across projects
# Usage: ./runpod_ssh_manager.sh [add|remove|list|backup|restore]

set -e

# Configuration
SSH_CONFIG="$HOME/.ssh/config"
BACKUP_DIR="$HOME/.ssh/config_backups"
HISTORY_FILE="$HOME/.ssh/runpod_history.json"
SCRIPT_VERSION="1.0.0"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${BLUE}===${NC} $1 ${BLUE}===${NC}\n"
}

# Ensure required directories exist
init_dirs() {
    mkdir -p "$BACKUP_DIR"
    
    # Initialize history file if it doesn't exist
    if [ ! -f "$HISTORY_FILE" ]; then
        echo '{"pods": []}' > "$HISTORY_FILE"
    fi
}

# Backup SSH config
backup_config() {
    local timestamp=$(date +"%Y%m%d_%H%M%S")
    local backup_file="${BACKUP_DIR}/config.${timestamp}"
    
    if [ -f "$SSH_CONFIG" ]; then
        cp "$SSH_CONFIG" "$backup_file"
        log_info "Backed up SSH config to: $backup_file"
        return 0
    else
        log_warn "SSH config file doesn't exist yet"
        return 1
    fi
}

# List all backups
list_backups() {
    log_header "Available SSH Config Backups"
    
    if [ ! -d "$BACKUP_DIR" ]; then
        log_info "No backups found"
        return
    fi
    
    local backups=($(ls -t "$BACKUP_DIR"/config.* 2>/dev/null || true))
    
    if [ ${#backups[@]} -eq 0 ]; then
        log_info "No backups found"
        return
    fi
    
    echo "Index | Timestamp           | Size   | Age"
    echo "------|---------------------|--------|------------------"
    
    local idx=1
    for backup in "${backups[@]}"; do
        local filename=$(basename "$backup")
        local timestamp=${filename#config.}
        local size=$(du -h "$backup" | cut -f1)
        local age=$(find "$backup" -mtime +0 -printf "%Td days ago\n" 2>/dev/null || echo "Today")
        
        printf "%-5s | %-19s | %-6s | %s\n" "$idx" "$timestamp" "$size" "$age"
        ((idx++))
    done
    
    echo ""
}

# Restore from backup
restore_backup() {
    list_backups
    
    if [ ! -d "$BACKUP_DIR" ] || [ -z "$(ls -A "$BACKUP_DIR" 2>/dev/null)" ]; then
        log_error "No backups available to restore"
        exit 1
    fi
    
    read -p "Enter backup index to restore (or 'cancel'): " choice
    
    if [ "$choice" = "cancel" ]; then
        log_info "Restore cancelled"
        return
    fi
    
    local backups=($(ls -t "$BACKUP_DIR"/config.* 2>/dev/null))
    local selected_backup="${backups[$((choice-1))]}"
    
    if [ -z "$selected_backup" ] || [ ! -f "$selected_backup" ]; then
        log_error "Invalid backup selection"
        exit 1
    fi
    
    # Backup current config before restoring
    backup_config
    
    cp "$selected_backup" "$SSH_CONFIG"
    log_info "Restored SSH config from: $(basename "$selected_backup")"
}

# Get RunPods SSH details from user
get_pod_details() {
    local project_name=$1
    
    echo ""
    log_info "Enter RunPods SSH connection details"
    echo "You can find these in RunPods dashboard under 'SSH over exposed TCP'"
    echo ""
    
    read -p "Pod Hostname/IP: " hostname
    read -p "Pod Port: " port
    read -p "Pod Nickname (e.g., a40-50gb): " nickname
    read -p "SSH Key Path [~/.ssh/id_ed25519]: " ssh_key
    
    # Set defaults
    ssh_key=${ssh_key:-~/.ssh/id_ed25519}
    
    # Validate inputs
    if [ -z "$hostname" ] || [ -z "$port" ]; then
        log_error "Hostname and port are required!"
        exit 1
    fi
    
    # Create host alias
    local host_alias="runpod-${project_name}"
    if [ -n "$nickname" ]; then
        host_alias="${host_alias}-${nickname}"
    fi
    
    echo ""
    log_info "Configuration Summary:"
    echo "  Host Alias: $host_alias"
    echo "  Hostname:   $hostname"
    echo "  Port:       $port"
    echo "  SSH Key:    $ssh_key"
    echo ""
    
    read -p "Proceed with this configuration? (y/n): " confirm
    if [ "$confirm" != "y" ]; then
        log_info "Configuration cancelled"
        exit 0
    fi
    
    # Return values via array (simulate return multiple values)
    echo "$host_alias|$hostname|$port|$ssh_key|$nickname"
}

# Add or update pod config
add_pod_config() {
    local project_name=$1
    
    if [ -z "$project_name" ]; then
        read -p "Project name (e.g., agentic-spliceai, genai-lab): " project_name
    fi
    
    if [ -z "$project_name" ]; then
        log_error "Project name is required!"
        exit 1
    fi
    
    # Get pod details
    local details=$(get_pod_details "$project_name")
    
    IFS='|' read -r host_alias hostname port ssh_key nickname <<< "$details"
    
    # Backup current config
    backup_config
    
    # Check if host already exists
    if grep -q "^Host $host_alias$" "$SSH_CONFIG" 2>/dev/null; then
        log_warn "Host '$host_alias' already exists in SSH config"
        read -p "Update existing entry? (y/n): " update_confirm
        
        if [ "$update_confirm" = "y" ]; then
            # Remove old entry
            remove_pod_config_internal "$host_alias" "silent"
        else
            log_info "Keeping existing configuration"
            exit 0
        fi
    fi
    
    # Generate SSH config block
    local config_block="
# RunPods: $project_name ($nickname)
# Added: $(date '+%Y-%m-%d %H:%M:%S')
Host $host_alias
    HostName $hostname
    Port $port
    User root
    IdentityFile $ssh_key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 60
    ServerAliveCountMax 5
    # Connection timeout settings
    ConnectTimeout 10
    # Compression for faster transfers
    Compression yes
"
    
    # Append to SSH config
    echo "$config_block" >> "$SSH_CONFIG"
    
    # Record in history
    record_pod_history "$project_name" "$host_alias" "$hostname" "$port" "$nickname"
    
    log_info "Successfully added SSH config for: $host_alias"
    echo ""
    log_info "You can now connect with: ssh $host_alias"
    
    # Test connection
    echo ""
    read -p "Test connection now? (y/n): " test_confirm
    if [ "$test_confirm" = "y" ]; then
        log_info "Testing SSH connection..."
        ssh -o ConnectTimeout=5 "$host_alias" "echo 'Connection successful!'" || \
            log_error "Connection test failed. Check your pod details."
    fi
}

# Remove pod config
remove_pod_config_internal() {
    local host_alias=$1
    local mode=${2:-"interactive"}
    
    if [ ! -f "$SSH_CONFIG" ]; then
        log_error "SSH config file not found"
        return 1
    fi
    
    if ! grep -q "^Host $host_alias$" "$SSH_CONFIG"; then
        if [ "$mode" != "silent" ]; then
            log_error "Host '$host_alias' not found in SSH config"
        fi
        return 1
    fi
    
    # Backup before removing
    if [ "$mode" != "silent" ]; then
        backup_config
    fi
    
    # Remove the host block (including comments and blank lines)
    # This uses awk to remove from "Host hostN" to the next "Host" or EOF
    awk -v host="$host_alias" '
        /^Host / { if (p) print ""; p=0 }
        /^Host '"$host_alias"'$/ { p=1; next }
        !p { print }
    ' "$SSH_CONFIG" > "${SSH_CONFIG}.tmp"
    
    mv "${SSH_CONFIG}.tmp" "$SSH_CONFIG"
    
    if [ "$mode" != "silent" ]; then
        log_info "Removed SSH config for: $host_alias"
    fi
}

remove_pod_config() {
    list_pods
    
    read -p "Enter host alias to remove (or 'cancel'): " host_alias
    
    if [ "$host_alias" = "cancel" ]; then
        log_info "Removal cancelled"
        return
    fi
    
    remove_pod_config_internal "$host_alias"
}

# List all RunPods entries
list_pods() {
    log_header "Current RunPods SSH Configurations"
    
    if [ ! -f "$SSH_CONFIG" ]; then
        log_info "No SSH config file found"
        return
    fi
    
    # Extract RunPods entries
    awk '
        /^# RunPods:/ { 
            project=$3; 
            getline; added=$3" "$4;
            getline; host=$2;
            getline; hostname=$2;
            getline; port=$2;
            printf "%-30s %-25s %s:%s\n", host, project, hostname, port
        }
    ' "$SSH_CONFIG"
    
    echo ""
}

# Record pod in history
record_pod_history() {
    local project=$1
    local host_alias=$2
    local hostname=$3
    local port=$4
    local nickname=$5
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Simple JSON append (not using jq for portability)
    # Note: This is a simplified version. For production, consider using jq
    python3 -c "
import json
import sys
from pathlib import Path

history_file = Path('$HISTORY_FILE')
try:
    with open(history_file, 'r') as f:
        data = json.load(f)
except:
    data = {'pods': []}

pod_entry = {
    'project': '$project',
    'host_alias': '$host_alias',
    'hostname': '$hostname',
    'port': '$port',
    'nickname': '$nickname',
    'added': '$timestamp',
    'status': 'active'
}

# Check if pod already exists and update
found = False
for i, pod in enumerate(data['pods']):
    if pod['host_alias'] == '$host_alias':
        data['pods'][i] = pod_entry
        found = True
        break

if not found:
    data['pods'].append(pod_entry)

with open(history_file, 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || log_warn "Could not update history (python3 required)"
}

# Show pod history
show_history() {
    log_header "RunPods History"
    
    if [ ! -f "$HISTORY_FILE" ]; then
        log_info "No history available"
        return
    fi
    
    python3 -c "
import json
from pathlib import Path

history_file = Path('$HISTORY_FILE')
try:
    with open(history_file, 'r') as f:
        data = json.load(f)
    
    if not data.get('pods'):
        print('No pods in history')
        return
    
    print(f\"{'Project':<20} {'Host Alias':<30} {'Added':<20} {'Status':<10}\")
    print('-' * 82)
    
    for pod in sorted(data['pods'], key=lambda x: x['added'], reverse=True):
        project = pod.get('project', 'N/A')
        host_alias = pod.get('host_alias', 'N/A')
        added = pod.get('added', 'N/A')
        status = pod.get('status', 'N/A')
        print(f\"{project:<20} {host_alias:<30} {added:<20} {status:<10}\")
except Exception as e:
    print(f'Error reading history: {e}')
" 2>/dev/null || log_warn "Could not read history (python3 required)"
    
    echo ""
}

# Generate quick setup script for a project
generate_setup_script() {
    local project=$1
    
    if [ -z "$project" ]; then
        read -p "Project name: " project
    fi
    
    local script_path="./runpod_quick_setup_${project}.sh"
    
    cat > "$script_path" << 'EOFSCRIPT'
#!/bin/bash
# Quick RunPods Setup Script
# Auto-generated by runpod_ssh_manager.sh

set -e

PROJECT_NAME="PROJECT_PLACEHOLDER"

echo "=== RunPods Quick Setup for $PROJECT_NAME ==="
echo ""

# Add SSH config
~/work/agentic-spliceai/runpods/scripts/runpod_ssh_manager.sh add "$PROJECT_NAME"

# Get the newly created host alias
HOST_ALIAS=$(grep "runpod-${PROJECT_NAME}" ~/.ssh/config | head -1 | awk '{print $2}')

echo ""
echo "=== Testing Connection ==="
ssh -o ConnectTimeout=10 "$HOST_ALIAS" "hostname && nvidia-smi --query-gpu=name --format=csv,noheader"

echo ""
echo "=== Setup Complete ==="
echo "You can now SSH with: ssh $HOST_ALIAS"
EOFSCRIPT
    
    sed -i '' "s/PROJECT_PLACEHOLDER/$project/g" "$script_path" 2>/dev/null || \
        sed -i "s/PROJECT_PLACEHOLDER/$project/g" "$script_path"
    
    chmod +x "$script_path"
    
    log_info "Generated quick setup script: $script_path"
}

# Main menu
show_menu() {
    log_header "RunPods SSH Configuration Manager v${SCRIPT_VERSION}"
    echo "1) Add/Update pod configuration"
    echo "2) Remove pod configuration"
    echo "3) List current pods"
    echo "4) Show history"
    echo "5) List backups"
    echo "6) Restore from backup"
    echo "7) Generate quick setup script"
    echo "8) Exit"
    echo ""
    read -p "Select option: " choice
    
    case $choice in
        1) add_pod_config ;;
        2) remove_pod_config ;;
        3) list_pods ;;
        4) show_history ;;
        5) list_backups ;;
        6) restore_backup ;;
        7) 
            read -p "Project name: " proj
            generate_setup_script "$proj"
            ;;
        8) exit 0 ;;
        *) log_error "Invalid option"; show_menu ;;
    esac
}

# Main
main() {
    init_dirs
    
    local command=${1:-"menu"}
    
    case $command in
        add)
            shift
            add_pod_config "$@"
            ;;
        remove)
            remove_pod_config
            ;;
        list)
            list_pods
            ;;
        history)
            show_history
            ;;
        backups)
            list_backups
            ;;
        restore)
            restore_backup
            ;;
        generate)
            shift
            generate_setup_script "$@"
            ;;
        menu)
            show_menu
            ;;
        *)
            echo "Usage: $0 [add|remove|list|history|backups|restore|generate|menu]"
            echo ""
            echo "Commands:"
            echo "  add [project]  - Add or update pod configuration"
            echo "  remove         - Remove pod configuration"
            echo "  list           - List current pod configurations"
            echo "  history        - Show pod history"
            echo "  backups        - List config backups"
            echo "  restore        - Restore from backup"
            echo "  generate       - Generate quick setup script"
            echo "  menu           - Show interactive menu (default)"
            exit 1
            ;;
    esac
}

main "$@"
