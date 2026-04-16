#!/bin/bash
# Pod bootstrap — make the working directory look like the persistent volume.
#
# Why this exists:
#   The repo uses cwd-relative paths in several places (e.g.,
#   `Registry.data_root='data'` in `splice_engine/resources/registry.py`,
#   relative `output/...` paths in benchmark scripts). On a fresh pod, the
#   data and outputs live on the persistent network volume at
#   /runpod-volume/, not under the cwd ~/sky_workdir/. Without symlinks,
#   relative paths silently miss the data and code falls back to broken
#   states (uniform priors, missing checkpoints, etc.).
#
#   This script idempotently creates the standard symlinks and verifies
#   that critical artifacts are reachable. Source it (or invoke it as the
#   first step) from every pod-runnable ops script.
#
# Usage (from inside another ops script):
#   source ~/sky_workdir/examples/meta_layer/ops_bootstrap_pod.sh
#
# Usage (standalone, sanity check):
#   bash examples/meta_layer/ops_bootstrap_pod.sh
#
# Override defaults via env vars before sourcing:
#   PERSISTENT_ROOT=/some/other/volume bash ops_bootstrap_pod.sh
#   REQUIRE_PATHS="data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa other/path"

set -u

# ── Configuration ───────────────────────────────────────────────────────
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/runpod-volume}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$HOME/sky_workdir}"

# Subdirectories under PERSISTENT_ROOT that get linked into RUNTIME_ROOT.
# Add to this list (space-separated) if you have other long-lived dirs.
LINK_DIRS="${LINK_DIRS:-data output}"

# Optional pre-staged caches at the volume root (not under data/output)
# that scripts reference by absolute path. Listed for the sanity check.
EXTRA_PATHS="${EXTRA_PATHS:-bigwig_cache}"

# Required paths to verify exist after symlinks. Override or extend.
# Paths can be absolute (/runpod-volume/...) or relative to RUNTIME_ROOT.
REQUIRE_PATHS="${REQUIRE_PATHS:-data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/models/openspliceai data/GRCh38/junction_data/junctions_gtex_v8.parquet}"

# ── Helpers ─────────────────────────────────────────────────────────────
_ts() { date +'%Y-%m-%d %H:%M:%S'; }
_log() { echo "[$(_ts)] [bootstrap] $*"; }

# ── Volume sanity ───────────────────────────────────────────────────────
if [ ! -d "$PERSISTENT_ROOT" ]; then
    _log "FATAL persistent root $PERSISTENT_ROOT not present — is the volume mounted?"
    return 1 2>/dev/null || exit 1
fi
mkdir -p "$RUNTIME_ROOT"

# ── Symlink loop ────────────────────────────────────────────────────────
for sub in $LINK_DIRS; do
    src="$PERSISTENT_ROOT/$sub"
    dst="$RUNTIME_ROOT/$sub"
    mkdir -p "$src"  # ensure source dir exists so the link isn't dangling

    if [ -L "$dst" ]; then
        # Already a symlink — verify target
        existing="$(readlink "$dst")"
        if [ "$existing" = "$src" ]; then
            _log "OK   $dst -> $src (already linked)"
        else
            _log "WARN $dst points to $existing (expected $src) — re-linking"
            rm "$dst"
            ln -s "$src" "$dst"
            _log "OK   $dst -> $src"
        fi
    elif [ -e "$dst" ]; then
        # Real directory or file in the way — refuse to clobber silently
        _log "FAIL $dst exists and is NOT a symlink. Refusing to overwrite."
        _log "     Move or remove it manually, then re-run bootstrap."
        return 1 2>/dev/null || exit 1
    else
        ln -s "$src" "$dst"
        _log "OK   $dst -> $src (created)"
    fi
done

# ── Sanity check: required paths reachable ──────────────────────────────
_log "Checking required paths…"
SANITY_OK=true
for p in $REQUIRE_PATHS; do
    # Resolve relative paths against RUNTIME_ROOT
    case "$p" in
        /*) full="$p" ;;
        *)  full="$RUNTIME_ROOT/$p" ;;
    esac
    if [ -e "$full" ]; then
        _log "  OK   $p"
    else
        _log "  FAIL $p (not found at $full)"
        SANITY_OK=false
    fi
done

# ── Optional extras (informational only) ────────────────────────────────
for p in $EXTRA_PATHS; do
    full="$PERSISTENT_ROOT/$p"
    if [ -e "$full" ]; then
        size="$(du -sh "$full" 2>/dev/null | cut -f1)"
        _log "  INFO $full ($size)"
    else
        _log "  INFO $full (absent — features may degrade)"
    fi
done

if [ "$SANITY_OK" != "true" ]; then
    _log "Bootstrap completed with WARNINGS — required paths missing."
    return 1 2>/dev/null || exit 1
fi

_log "Bootstrap OK — runtime $RUNTIME_ROOT mirrors persistent $PERSISTENT_ROOT."

# Allow the file to be sourced without exiting the parent shell
return 0 2>/dev/null || exit 0
