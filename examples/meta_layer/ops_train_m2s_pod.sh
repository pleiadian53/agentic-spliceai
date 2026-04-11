#!/bin/bash
# Train M2-S: Meta-layer on Ensembl labels (expanded training data).
# (Formerly ops_train_m2c_pod.sh)
#
# M2-S uses the same architecture as M1-S but trains on Ensembl splice
# sites instead of MANE.  This exposes the model to alternative splice
# sites during training, testing whether broader label coverage improves
# OOD generalization beyond what the logit-space blend achieves alone.
#
# The base model (OpenSpliceAI) is NOT retrained — only the meta-layer.
#
# Prerequisites (on pod):
#   1. Package installed:   pip install -e .
#   2. FASTA:               data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
#   3. Ensembl GTF:         data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf
#   4. Ensembl splices:     data/ensembl/GRCh38/splice_sites_enhanced.tsv
#   5. MANE base scores:    data/mane/GRCh38/openspliceai_eval/precomputed/
#   6. Ensembl base scores: /runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed/
#   7. BigWig cache:        /runpod-volume/bigwig_cache/*.bw
#   8. Junction data:       data/mane/GRCh38/junction_data/junctions_gtex_v8.parquet
#   9. eCLIP data:          data/mane/GRCh38/rbp_data/eclip_peaks.parquet
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_train_m2c_pod.sh \
#       > /runpod-volume/output/meta_layer/m2c_train.log 2>&1 &

set -e

WORKDIR=~/sky_workdir
OUTPUT_DIR=/runpod-volume/output/meta_layer/m2c
CACHE_DIR=/runpod-volume/output/meta_layer/gene_cache_ensembl
BIGWIG_CACHE=/runpod-volume/bigwig_cache
ENSEMBL_SCORES=/runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed

cd "$WORKDIR"

echo "============================================================"
echo "M2c Training (Ensembl labels) — $(date)"
echo "  Device:       cuda"
echo "  Annotation:   Ensembl GRCh38.112"
echo "  Cache:        $CACHE_DIR"
echo "  BigWig:       $BIGWIG_CACHE"
echo "  Output:       $OUTPUT_DIR"
echo "============================================================"

# Verify data is staged
echo "Checking prerequisites..."
ls data/ensembl/GRCh38/splice_sites_enhanced.tsv >/dev/null
ls data/ensembl/GRCh38/Homo_sapiens.GRCh38.112.gtf >/dev/null
ls "$ENSEMBL_SCORES"/predictions_chr1.parquet >/dev/null
ls data/mane/GRCh38/junction_data/junctions_gtex_v8.parquet >/dev/null
ls "$BIGWIG_CACHE"/hg38.phyloP100way.bw >/dev/null
echo "  All prerequisites present."

python -u examples/meta_layer/07_train_sequence_model.py \
    --mode m1 \
    --annotation-source ensembl \
    --device cuda \
    --epochs 50 \
    --lr 1e-3 \
    --hidden-dim 32 \
    --activation gelu \
    --samples-per-epoch 100000 \
    --patience 10 \
    --bigwig-cache "$BIGWIG_CACHE" \
    --base-scores-dir "$ENSEMBL_SCORES" \
    --cache-dir "$CACHE_DIR" \
    --use-shards \
    --output-dir "$OUTPUT_DIR"
    # batch-size and accumulation-steps auto-detected from device:
    #   cuda → batch_size=16, accumulation_steps=1 (eff=16)

echo "============================================================"
echo "M2c training complete — $(date)"
echo "Results: $OUTPUT_DIR"
echo "============================================================"
