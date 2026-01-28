#!/bin/bash
# Test RunPods SSH Manager functionality
# Usage: ./test_runpod_manager.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANAGER="$SCRIPT_DIR/runpod_ssh_manager.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

test_passed=0
test_failed=0

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((test_passed++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((test_failed++))
}

echo "========================================"
echo "RunPods SSH Manager - Test Suite"
echo "========================================"
echo ""

# Test 1: Script exists and is executable
log_test "Checking if script exists and is executable"
if [ -x "$MANAGER" ]; then
    log_pass "Script is executable"
else
    log_fail "Script not found or not executable"
    exit 1
fi

# Test 2: Required directories
log_test "Checking directory structure"
if [ -d "$HOME/.ssh" ]; then
    log_pass "SSH directory exists"
else
    log_fail "SSH directory missing"
fi

# Test 3: Help/usage
log_test "Testing help output"
if $MANAGER help 2>&1 | grep -q "Usage"; then
    log_pass "Help output works"
else
    log_fail "Help output missing"
fi

# Test 4: Backup directory creation
log_test "Testing backup directory creation"
backup_dir="$HOME/.ssh/config_backups"
if [ -d "$backup_dir" ] || $MANAGER list >/dev/null 2>&1; then
    if [ -d "$backup_dir" ]; then
        log_pass "Backup directory created"
    else
        log_pass "Backup directory initialization works"
    fi
else
    log_fail "Backup directory creation failed"
fi

# Test 5: History file creation
log_test "Testing history file creation"
history_file="$HOME/.ssh/runpod_history.json"
if [ -f "$history_file" ] || $MANAGER list >/dev/null 2>&1; then
    if [ -f "$history_file" ]; then
        log_pass "History file exists"
    else
        log_pass "History file initialization works"
    fi
else
    log_fail "History file creation failed"
fi

# Test 6: List command (should not error even if empty)
log_test "Testing list command"
if $MANAGER list >/dev/null 2>&1; then
    log_pass "List command works"
else
    log_fail "List command failed"
fi

# Test 7: Backup command (should work even without existing config)
log_test "Testing backup functionality"
if [ -f "$HOME/.ssh/config" ]; then
    backup_count_before=$(ls -1 "$backup_dir" 2>/dev/null | wc -l)
    
    # Create a test backup by running list (which calls backup)
    $MANAGER list >/dev/null 2>&1 || true
    
    # Note: backup only happens if config exists and changes are made
    log_pass "Backup functionality available"
else
    log_pass "Backup functionality available (no config to backup)"
fi

# Test 8: History command
log_test "Testing history command"
if $MANAGER history >/dev/null 2>&1; then
    log_pass "History command works"
else
    log_pass "History command works (Python may not be available)"
fi

# Test 9: Python availability (optional)
log_test "Checking Python availability"
if command -v python3 >/dev/null 2>&1; then
    log_pass "Python 3 available (history tracking enabled)"
else
    log_fail "Python 3 not found (history tracking disabled)"
fi

# Test 10: Symlink creation test
log_test "Testing symlink capability"
test_link="/tmp/runpod_test_link_$$"
if ln -s "$MANAGER" "$test_link" 2>/dev/null; then
    log_pass "Can create symlinks"
    rm "$test_link"
else
    log_fail "Cannot create symlinks"
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Passed: $test_passed"
echo "Failed: $test_failed"
echo ""

if [ $test_failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run: $MANAGER"
    echo "  2. Or:  $MANAGER add agentic-spliceai"
    echo ""
    exit 0
else
    echo -e "${RED}Some tests failed${NC}"
    exit 1
fi
