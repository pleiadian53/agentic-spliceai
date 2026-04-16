#!/bin/bash
# M4 benchmark sweep: ClinVar + MutSpliceDB × {M1-S v2, M2-S v2}
#
# Designed to run unattended on the pod for ~7-11 hours. Pre-flight
# checks catch >90% of silent failures (missing files, broken
# checkpoint, code bugs) in ~2 minutes before committing GPU hours.
# Per-run isolation: a failure in one run does NOT kill the others.
#
# Usage (on the pod):
#   nohup bash examples/variant_analysis/ops_m4_benchmarks_pod.sh \
#     > /runpod-volume/output/m4_benchmarks/orchestrator.log 2>&1 &
#
# Outputs go to /runpod-volume/output/m4_benchmarks/<benchmark>_<checkpoint>/

set -u  # unset variable = fatal; do NOT use set -e (we want per-run isolation)

# ── Bootstrap (data symlinks + path sanity) ──────────────────────────────
# Idempotent: safe to source on every run. Verifies runtime cwd mirrors the
# persistent volume so cwd-relative paths (Registry.data_root='data', etc.)
# resolve correctly. See examples/meta_layer/ops_bootstrap_pod.sh.
REQUIRE_PATHS="data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa data/models/openspliceai data/clinvar/clinvar_splice_snvs.parquet data/mutsplicedb/splice_sites_induced.tsv data/GRCh38/junction_data/junctions_gtex_v8.parquet" \
    source ~/sky_workdir/examples/meta_layer/ops_bootstrap_pod.sh || {
        echo "[FATAL] bootstrap failed — aborting before pre-flight"
        exit 1
    }

# ── Configuration ────────────────────────────────────────────────────────
CHECKPOINTS=(
    "m1s_v2:/runpod-volume/output/meta_layer/m1s_v2_logit_blend/best.pt"
    "m2s_v2:/runpod-volume/output/meta_layer/m2s_v2/best.pt"
)
FASTA="/runpod-volume/data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
# Two ClinVar parquets, both produced by 02_clinvar_download.py:
#   - CLINVAR_ALL: no splice filter → mostly non-splicing pathogenic
#     variants (missense, nonsense, synonymous). Retained as a
#     *reference/negative-control* run to show why the filter matters.
#   - CLINVAR_SPLICE: splice-filtered (proximity + splice MC). This is
#     the population the model is actually designed to score.
CLINVAR_ALL="/runpod-volume/data/clinvar/clinvar_all_snvs.parquet"
CLINVAR_SPLICE="/runpod-volume/data/clinvar/clinvar_splice_snvs.parquet"
# SpliceAI-style scoring radii for max_delta reduction. ±50 bp matches
# OpenSpliceAI's default dist_var. ±100 bp is commonly used in
# cryptic-site activation studies. Both are kept side-by-side so the
# effect of scoring-window choice is visible per benchmark.
SCORE_RADII=(${SCORE_RADII:-50 100})
# Radius used for the pre-flight smoke test (one value only; the real
# sweep iterates over SCORE_RADII).
SCORE_RADIUS_SMOKE="${SCORE_RADII[0]}"
# 04_mutsplicedb_benchmark.py loads this TSV (parser-flattened sites);
# the MutSpliceDB BRP CSV is the upstream source.
MUTSPLICEDB="/runpod-volume/data/mutsplicedb/splice_sites_induced.tsv"
# Pre-staged bigWig cache (~32 GB) for conservation/epigenetic/chromatin
# features. Pod has no internet; without this, those channels fall back
# to zeros and the meta layer scores degrade.
BIGWIG_CACHE="/runpod-volume/bigwig_cache"
BASE_OUT="/runpod-volume/output/m4_benchmarks"
mkdir -p "$BASE_OUT"

cd ~/sky_workdir

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
section() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

# ── Pre-flight checks ────────────────────────────────────────────────────
section "PRE-FLIGHT CHECKS"

PREFLIGHT_OK=true
check_file() {
    if [ -f "$1" ]; then
        log "  OK   $1 ($(du -h "$1" | cut -f1))"
    else
        log "  FAIL $1 (missing)"
        PREFLIGHT_OK=false
    fi
}

log "Data files:"
check_file "$FASTA"
check_file "$CLINVAR_ALL"
check_file "$CLINVAR_SPLICE"
check_file "$MUTSPLICEDB"

log "Checkpoints:"
for cp in "${CHECKPOINTS[@]}"; do
    check_file "${cp#*:}"
done

log "GPU:"
if nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    log "  OK   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
    log "  FAIL nvidia-smi unavailable"
    PREFLIGHT_OK=false
fi

log "Python imports:"
python -c "
import torch
import agentic_spliceai
from agentic_spliceai.splice_engine.meta_layer.inference.variant_runner import VariantRunner
print('  OK   torch', torch.__version__, 'cuda available:', torch.cuda.is_available())
print('  OK   agentic_spliceai', agentic_spliceai.__file__)
print('  OK   VariantRunner._to_genomic_alleles:', hasattr(VariantRunner, '_to_genomic_alleles'))
" || PREFLIGHT_OK=false

log "Smoke test (5-variant MutSpliceDB run per checkpoint):"
for cp in "${CHECKPOINTS[@]}"; do
    name="${cp%:*}"
    path="${cp#*:}"
    smoke_log="$BASE_OUT/.smoke_${name}.log"
    log "  Running smoke test for $name..."
    python -u examples/variant_analysis/04_mutsplicedb_benchmark.py \
        --checkpoint "$path" \
        --fasta "$FASTA" \
        --mutsplicedb "$MUTSPLICEDB" \
        --bigwig-cache "$BIGWIG_CACHE" \
        --score-radius "$SCORE_RADIUS_SMOKE" \
        --max-variants 5 \
        --device cuda \
        --output-dir "$BASE_OUT/.smoke_${name}" \
        > "$smoke_log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && grep -q "Variants scored:" "$smoke_log"; then
        log "    OK   smoke ($name) — see $smoke_log"
    else
        log "    FAIL smoke ($name) exit=$rc — see $smoke_log"
        PREFLIGHT_OK=false
    fi
done

if [ "$PREFLIGHT_OK" != "true" ]; then
    section "PRE-FLIGHT FAILED — aborting full run"
    exit 1
fi

section "PRE-FLIGHT OK — launching full sweep"

# ── Per-run launcher (isolated, captures failures) ───────────────────────
RUN_RESULTS=()
run_benchmark() {
    local label="$1"; shift
    local out_dir="$BASE_OUT/$label"
    local log_file="$out_dir/run.log"
    mkdir -p "$out_dir"

    section "RUN: $label  →  $out_dir"
    log "Command: $*"

    "$@" > "$log_file" 2>&1
    local rc=$?

    if [ $rc -eq 0 ]; then
        log "RESULT: $label = OK"
        RUN_RESULTS+=("OK   $label")
    else
        log "RESULT: $label = FAIL (exit=$rc)"
        log "        last 20 lines of $log_file:"
        tail -20 "$log_file" | sed 's/^/        | /'
        RUN_RESULTS+=("FAIL $label (exit=$rc)")
    fi
    return 0  # always return 0 so set -u doesn't kill us; we tracked rc above
}

# ── Sweep: checkpoints × benchmarks × score radii ───────────────────────
# Sequential (avoids GPU contention). Output dirs suffixed with _r<radius>
# so results across radii coexist. Total runs:
#   2 checkpoints × 3 benchmarks × len(SCORE_RADII) radii
for radius in "${SCORE_RADII[@]}"; do
    section "SCORE RADIUS = ±${radius} bp"

    for cp in "${CHECKPOINTS[@]}"; do
        name="${cp%:*}"
        path="${cp#*:}"

        # MutSpliceDB (full = all 434 variants)
        run_benchmark "mutsplicedb_${name}_r${radius}" \
            python -u examples/variant_analysis/04_mutsplicedb_benchmark.py \
                --checkpoint "$path" \
                --fasta "$FASTA" \
                --mutsplicedb "$MUTSPLICEDB" \
                --bigwig-cache "$BIGWIG_CACHE" \
                --score-radius "$radius" \
                --device cuda \
                --output-dir "$BASE_OUT/mutsplicedb_${name}_r${radius}"

        # ClinVar (splice-filtered) — the benchmark the model is designed for
        run_benchmark "clinvar_splice_${name}_r${radius}" \
            python -u examples/variant_analysis/03_clinvar_benchmark.py \
                --checkpoint "$path" \
                --fasta "$FASTA" \
                --clinvar "$CLINVAR_SPLICE" \
                --bigwig-cache "$BIGWIG_CACHE" \
                --score-radius "$radius" \
                --device cuda \
                --output-dir "$BASE_OUT/clinvar_splice_${name}_r${radius}"

        # ClinVar (unfiltered) — reference/negative control. ~89% of
        # pathogenic variants here are non-splicing (missense, nonsense,
        # synonymous) and should score near the noise floor; kept to
        # make the filter's effect visible side-by-side.
        run_benchmark "clinvar_all_${name}_r${radius}" \
            python -u examples/variant_analysis/03_clinvar_benchmark.py \
                --checkpoint "$path" \
                --fasta "$FASTA" \
                --clinvar "$CLINVAR_ALL" \
                --bigwig-cache "$BIGWIG_CACHE" \
                --score-radius "$radius" \
                --device cuda \
                --output-dir "$BASE_OUT/clinvar_all_${name}_r${radius}"
    done
done

# ── Summary ──────────────────────────────────────────────────────────────
section "SWEEP COMPLETE — $(date)"
log "Per-run results:"
for r in "${RUN_RESULTS[@]}"; do
    log "  $r"
done

# Cleanup smoke test artifacts
rm -rf "$BASE_OUT"/.smoke_* 2>/dev/null

# Exit code reflects whether ANY run failed
for r in "${RUN_RESULTS[@]}"; do
    [[ "$r" == FAIL* ]] && exit 1
done
exit 0
