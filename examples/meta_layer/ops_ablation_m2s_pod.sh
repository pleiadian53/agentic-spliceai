#!/bin/bash
# M2-S ablation study: evaluate with each modality group zeroed out.
#
# Uses the Ensembl test gene cache (M2-S's target distribution) to
# measure each modality's contribution to alternative site detection.
# Compare with ops_ablation_m1s_pod.sh (M1-S on MANE test cache) to
# see whether modality importance shifts between canonical and
# alternative splice site prediction.
#
# Prerequisites:
#   1. M2-S checkpoint: output/meta_layer/m2s_v4_cleanannot/{best,config}.pt
#   2. Ensembl test cache built on the cleanannot corpus, as produced by
#      ops_eval_tissue_pod.sh:
#      /runpod-volume/output/meta_layer/gene_cache_ensembl_cleanannot/test/
#
# Paths are overridable so a differently-named checkpoint or cache can be
# pointed at without editing the script:
#   CHECKPOINT=... CACHE_DIR=... bash ops_ablation_m2s_pod.sh
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_ablation_m2s_pod.sh \
#       > /runpod-volume/output/m2s_ablation/ablation.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
CHECKPOINT="${CHECKPOINT:-$WORKDIR/output/meta_layer/m2s_v4_cleanannot/best.pt}"
CACHE_DIR="${CACHE_DIR:-/runpod-volume/output/meta_layer/gene_cache_ensembl_cleanannot/test}"
OUTPUT_DIR="${OUTPUT_DIR:-/runpod-volume/output/m2s_ablation}"

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "M2-S Ablation Study (Ensembl test cache) — $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Cache:      $CACHE_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# Verify prerequisites before burning GPU time on a run that cannot finish
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    echo "  Stage it to the volume, or set CHECKPOINT=<path>."
    exit 1
fi
CACHE_COUNT=$(find "$CACHE_DIR" -name "*.npz" 2>/dev/null | wc -l)
echo "  Gene cache: $CACHE_COUNT genes"
if [ "$CACHE_COUNT" -lt 100 ]; then
    echo "ERROR: cache too small or missing at $CACHE_DIR"
    echo "  Build it with ops_eval_tissue_pod.sh (~3.4 h, needs the bigWig cache),"
    echo "  or set CACHE_DIR=<path> to an existing Ensembl test cache."
    exit 1
fi

run_ablation() {
    local label="$1"
    shift
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Ablation: $label — $(date)"
    echo "────────────────────────────────────────────────────────────"
    python -u examples/meta_layer/08_evaluate_sequence_model.py \
        --checkpoint "$CHECKPOINT" \
        --cache-dir "$CACHE_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --device cuda \
        "$@"
}

# 1. Full model (baseline — all 9 multimodal channels)
run_ablation "full_model"

# 2. No multimodal features (sequence + base scores only)
run_ablation "no_multimodal" --zero-channels all

# 3. No conservation (PhyloP + PhastCons)
run_ablation "no_conservation" --zero-channels phylop_score phastcons_score

# 4. No junction support
run_ablation "no_junction" --zero-channels junction_log1p junction_has_support

# 5. No epigenetic (histone marks)
run_ablation "no_epigenetic" --zero-channels h3k36me3_max h3k4me3_max

# 6. No chromatin accessibility (ATAC + DNase)
run_ablation "no_chromatin" --zero-channels atac_max dnase_max

# 7. No RBP binding
run_ablation "no_rbp" --zero-channels rbp_n_bound

echo ""
echo "============================================================"
echo "M2-S ablation complete — $(date)"
echo "Results: $OUTPUT_DIR/eval_ablation_*.json"
echo "============================================================"
echo ""
echo "Summary of all ablation results:"
# The full-model baseline writes eval_results.json (08 names the file after the
# zeroed channels), so glob both or the run everything is compared against is
# silently missing from the table.
for f in "$OUTPUT_DIR"/eval_results.json "$OUTPUT_DIR"/eval_ablation_*.json; do
    [ -f "$f" ] || continue
    label=$(basename "$f" .json | sed 's/eval_ablation_/zeroed:/; s/eval_results/full_model/')
    pr_auc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"meta_model\"][\"macro_pr_auc\"]:.4f}')" 2>/dev/null || echo "N/A")
    fn_red=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"fn_reduction_pct\"]:+.1f}%')" 2>/dev/null || echo "N/A")
    echo "  $label: PR-AUC=$pr_auc  FN_red=$fn_red"
done
