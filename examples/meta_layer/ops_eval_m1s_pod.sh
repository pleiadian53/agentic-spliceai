#!/bin/bash
# Evaluate M1-S on the SpliceAI test split (chr1,3,5,7,9).
#
# Builds gene cache (9 multimodal channels), then runs streaming
# evaluation comparing meta model vs base model.
#
# Prerequisites (on pod):
#   1. Package:       pip install -e ".[bio,conservation]"   (includes pyBigWig)
#   2. MANE data:     data/mane/GRCh38/splice_sites_enhanced.tsv
#   3. FASTA:         data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
#   4. Base scores:   data/mane/GRCh38/openspliceai_eval/precomputed/predictions_chr*.parquet
#   5. BigWig cache:  /runpod-volume/bigwig_cache/*.bw  (22 files, ~17 GB)
#   6. M1-S model:    output/meta_layer/m1s/best.pt + config.pt
#   7. Junction data: data/GRCh38/junction_data/junctions_gtex_v8.parquet
#   8. eCLIP data:    data/mane/GRCh38/rbp_data/eclip_peaks.parquet
#
# Data staging (run from LOCAL machine):
#   # M1-S checkpoint (if not already on pod)
#   rsync -Pavz output/meta_layer/m1s/{best.pt,config.pt} \
#       <cluster>:~/sky_workdir/output/meta_layer/m1s/
#
#   # Updated source code
#   rsync -Pavz --include='*/' --include='*.py' --exclude='*' \
#       src/agentic_spliceai/ <cluster>:~/sky_workdir/src/agentic_spliceai/
#   rsync -Pavz --relative examples/meta_layer/08_evaluate_sequence_model.py \
#       <cluster>:~/sky_workdir/
#
# BigWig cache (first time only, ~17 GB):
#   BIGWIG_CACHE=/runpod-volume/bigwig_cache bash scripts/data/download_bigwig_cache.sh
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_eval_m1s_pod.sh \
#       > /runpod-volume/output/m1s_eval/eval.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
CHECKPOINT=$WORKDIR/output/meta_layer/m1s/best.pt
BIGWIG_CACHE=/runpod-volume/bigwig_cache
OUTPUT_DIR=/runpod-volume/output/m1s_eval

cd "$WORKDIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "M1-S Evaluation — $(date)"
echo "  Checkpoint: $CHECKPOINT"
echo "  BigWig:     $BIGWIG_CACHE"
echo "  Output:     $OUTPUT_DIR"
echo "============================================================"

# Verify data is staged
echo "Checking prerequisites..."
ls "$CHECKPOINT" >/dev/null
ls "$WORKDIR/output/meta_layer/m1s/config.pt" >/dev/null
ls data/mane/GRCh38/splice_sites_enhanced.tsv >/dev/null
ls data/mane/GRCh38/openspliceai_eval/precomputed/predictions_chr1.parquet >/dev/null
ls data/GRCh38/junction_data/junctions_gtex_v8.parquet >/dev/null
ls data/mane/GRCh38/rbp_data/eclip_peaks.parquet >/dev/null
ls "$BIGWIG_CACHE"/hg38.phyloP100way.bw >/dev/null
python -c "import pyBigWig" 2>/dev/null || { echo "ERROR: pyBigWig not installed"; exit 1; }
echo "  All prerequisites present."

python -u examples/meta_layer/08_evaluate_sequence_model.py \
    --checkpoint "$CHECKPOINT" \
    --build-cache \
    --bigwig-cache "$BIGWIG_CACHE" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "============================================================"
echo "M1-S evaluation complete — $(date)"
echo "Results: $OUTPUT_DIR/eval_results.json"
echo "============================================================"
