#!/usr/bin/env bash
# sync_obsidian.sh
#
# Sync agentic-spliceai/docs/ → Obsidian vault (combio-lab/agentic-spliceai/docs/)
#
# Usage:
#   ./scripts/sync_obsidian.sh          # live sync
#   ./scripts/sync_obsidian.sh --dry    # preview changes without touching vault
#
# Notes:
#   - JS/CSS assets are excluded (not useful in Obsidian)
#   - --delete removes files in the vault that no longer exist in docs/
#   - Run this after any reorganization or new docs/ additions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC="${PROJECT_ROOT}/docs/"
VAULT_PROJECT="iCloud~md~obsidian/Documents/PleiadianLab/Projects/combio-lab/agentic-spliceai"
DST="${HOME}/Library/Mobile Documents/${VAULT_PROJECT}/docs/"

DRY_RUN=""
if [[ "${1:-}" == "--dry" ]]; then
    DRY_RUN="-n"
    echo "==> DRY RUN — no files will be changed"
fi

echo "==> Source : ${SRC}"
echo "==> Target : ${DST}"
echo ""

rsync -av${DRY_RUN} \
    --delete \
    --exclude='*.js' \
    --exclude='*.css' \
    "${SRC}" \
    "${DST}"

echo ""
echo "==> Sync complete."
