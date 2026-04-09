#!/bin/bash
# Run M1-S ablation study: evaluate with each modality group zeroed out.
#
# Reuses the existing gene cache from ops_eval_m1s_pod.sh — no cache rebuild.
# Each ablation run is inference-only (~15-20 min on A40).
#
# Prerequisites:
#   1. M1-S gene cache already built: /runpod-volume/output/meta_layer/gene_cache_mane/test/
#   2. M1-S checkpoint: output/meta_layer/m1s/best.pt + config.pt
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_ablation_m1s_pod.sh \
#       > /runpod-volume/output/m1s_eval/ablation.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
CHECKPOINT=$WORKDIR/output/meta_layer/m1s/best.pt
CACHE_DIR=/runpod-volume/output/meta_layer/gene_cache_mane/test
OUTPUT_DIR=/runpod-volume/output/m1s_eval

cd "$WORKDIR"

echo "============================================================"
echo "M1-S Ablation Study — $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Cache:      $CACHE_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# Verify cache exists
CACHE_COUNT=$(ls "$CACHE_DIR"/*.npz 2>/dev/null | wc -l)
echo "  Gene cache: $CACHE_COUNT genes"
if [ "$CACHE_COUNT" -lt 100 ]; then
    echo "ERROR: Cache too small. Run ops_eval_m1s_pod.sh first."
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

# 1. No multimodal features (sequence + base scores only)
run_ablation "no_multimodal" --zero-channels all

# 2. No conservation (PhyloP + PhastCons)
run_ablation "no_conservation" --zero-channels phylop_score phastcons_score

# 3. No epigenetic (histone marks)
run_ablation "no_epigenetic" --zero-channels h3k36me3_max h3k4me3_max

# 4. No chromatin accessibility (ATAC + DNase)
run_ablation "no_chromatin" --zero-channels atac_max dnase_max

# 5. No junction support
run_ablation "no_junction" --zero-channels junction_log1p junction_has_support

# 6. No RBP binding
run_ablation "no_rbp" --zero-channels rbp_n_bound

echo ""
echo "============================================================"
echo "Ablation study complete — $(date)"
echo "Results: $OUTPUT_DIR/eval_ablation_*.json"
echo "============================================================"
echo ""
echo "Summary of all ablation results:"
for f in "$OUTPUT_DIR"/eval_ablation_*.json; do
    label=$(basename "$f" .json | sed 's/eval_ablation_//')
    pr_auc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"meta_model\"][\"macro_pr_auc\"]:.4f}')" 2>/dev/null || echo "N/A")
    fn_red=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"fn_reduction_pct\"]:+.1f}%')" 2>/dev/null || echo "N/A")
    echo "  $label: PR-AUC=$pr_auc  FN_red=$fn_red"
done
