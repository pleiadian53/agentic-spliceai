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
#   1. M2-S checkpoint: output/meta_layer/m2c/best.pt + config.pt
#   2. Ensembl test cache: /runpod-volume/output/meta_layer/gene_cache_ensembl/test/
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_ablation_m2s_pod.sh \
#       > /runpod-volume/output/m2s_ablation/ablation.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
CHECKPOINT=$WORKDIR/output/meta_layer/m2c/best.pt
CACHE_DIR=/runpod-volume/output/meta_layer/gene_cache_ensembl/test
OUTPUT_DIR=/runpod-volume/output/m2s_ablation

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "M2-S Ablation Study (Ensembl test cache) — $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  Cache:      $CACHE_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# Verify cache exists
CACHE_COUNT=$(find "$CACHE_DIR" -name "*.npz" | wc -l)
echo "  Gene cache: $CACHE_COUNT genes"
if [ "$CACHE_COUNT" -lt 100 ]; then
    echo "ERROR: Cache too small. Build Ensembl test cache first."
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
for f in "$OUTPUT_DIR"/eval_ablation_*.json; do
    label=$(basename "$f" .json | sed 's/eval_ablation_//')
    pr_auc=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"meta_model\"][\"macro_pr_auc\"]:.4f}')" 2>/dev/null || echo "N/A")
    fn_red=$(python -c "import json; d=json.load(open('$f')); print(f'{d[\"fn_reduction_pct\"]:+.1f}%')" 2>/dev/null || echo "N/A")
    echo "  $label: PR-AUC=$pr_auc  FN_red=$fn_red"
done
