#!/bin/bash
# Launch M1-S training on GPU pod (A40).
#
# Prerequisites:
#   1. Data staged: predictions, junction, eCLIP parquets
#   2. Conservation bigWigs cached: /runpod-volume/bigwig_cache/*.bw
#   3. Package installed: pip install -e .
#
# Usage:
#   ssh sky-cb9e-pleiadian53
#   cd ~/sky_workdir
#   mkdir -p /runpod-volume/output/meta_layer
#   nohup bash examples/meta_layer/ops_train_m1s_pod.sh > /runpod-volume/output/meta_layer/m1s_train.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
OUTPUT_DIR=/runpod-volume/output/meta_layer/m1s
BIGWIG_CACHE=/runpod-volume/bigwig_cache

cd "$WORKDIR"

echo "============================================================"
echo "M1-S Training — $(date)"
echo "  Device:      cuda"
echo "  BigWig cache: $BIGWIG_CACHE"
echo "  Output:      $OUTPUT_DIR"
echo "============================================================"

# Verify data is staged
echo "Checking data..."
ls data/mane/GRCh38/openspliceai_eval/precomputed/predictions_chr1.parquet >/dev/null
ls data/mane/GRCh38/junction_data/junctions_gtex_v8.parquet >/dev/null
ls data/mane/GRCh38/rbp_data/eclip_peaks.parquet >/dev/null
ls "$BIGWIG_CACHE"/hg38.phyloP100way.bw >/dev/null
echo "  All data present."

python -u examples/meta_layer/07_train_sequence_model.py \
    --mode m1 \
    --device cuda \
    --epochs 50 \
    --batch-size 8 \
    --accumulation-steps 2 \
    --lr 1e-3 \
    --hidden-dim 32 \
    --activation gelu \
    --samples-per-epoch 100000 \
    --patience 10 \
    --bigwig-cache "$BIGWIG_CACHE" \
    --cache-dir /runpod-volume/output/meta_layer/gene_cache \
    --use-shards \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "Training complete — $(date)"
echo "Results: $OUTPUT_DIR"
echo "============================================================"
