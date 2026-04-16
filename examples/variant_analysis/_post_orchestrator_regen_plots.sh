#!/bin/bash
# Wait for an orchestrator PID to finish, then regenerate ClinVar plots
# from the saved delta_scores.json (using the now-fixed generate_plots()).
# Idempotent: skips dirs that already have pr_curve.png.
#
# Usage on pod:
#   nohup bash _post_orchestrator_regen_plots.sh <orchestrator_pid> \
#     > /runpod-volume/output/m4_benchmarks/post_orchestrator.log 2>&1 &

set +e

ORCH_PID="${1:-}"
BASE_OUT="${BASE_OUT:-/runpod-volume/output/m4_benchmarks}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] [post-orch] $*"; }

if [ -n "$ORCH_PID" ]; then
    log "waiting for orchestrator PID $ORCH_PID"
    while kill -0 "$ORCH_PID" 2>/dev/null; do sleep 30; done
    log "orchestrator finished"
fi

cd "${RUNTIME_ROOT:-$HOME/sky_workdir}"

for run in clinvar_m1s_v2 clinvar_m2s_v2; do
    DIR="$BASE_OUT/$run"
    if [ ! -f "$DIR/delta_scores.json" ]; then
        log "skip $run (no delta_scores.json)"
        continue
    fi
    if [ -f "$DIR/pr_curve.png" ]; then
        log "skip $run (plots already present)"
        continue
    fi
    log "regenerating plots for $run"
    python - <<PY 2>&1 | sed "s/^/  [$run] /"
import json, sys
from pathlib import Path
sys.path.insert(0, "examples/variant_analysis")
from importlib import import_module
mod = import_module("03_clinvar_benchmark")
results = json.load(open("$DIR/delta_scores.json"))
mod.generate_plots(results, Path("$DIR"))
print("plots OK")
PY
done

log "done."
