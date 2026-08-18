#!/bin/bash
# Train M3-S with the disease-anchor fold (SF3B1/ENCODE-KD positivized, TDP-43 masked).
#
# Reproduces the promoted M3-v1 recipe EXACTLY, changing one thing: disease
# anchors are folded into the label set instead of falling through to class-2
# negatives. Selected sources (default SF3B1 + ENCODE-KD) become up-weighted
# positives; TDP-43 (STMN2/UNC13A) is masked so it stays an honest eval probe.
#
# Anti-circularity: training builds train-chromosome genes only, so the 171
# held-out SF3B1 acceptors on chr1/3/5/7/9 (= D2) are never rasterized. The one
# run answers three questions: does folding SF3B1 in improve held-out SF3B1
# ranking (D2 vs baseline M3-v1)? does D1 hold (no general-recall regression)?
# does SF3B1 learning transfer to the masked TDP-43 sites (STMN2/UNC13A probe)?
#
# The base model (OpenSpliceAI) is NOT retrained — only the meta-layer.
#
# Prerequisites (on pod):
#   1. Package installed:  pip install -e .
#   2. MANE FASTA + GTF + splice_sites_enhanced.tsv (default m3 = MANE)
#   3. MANE base scores:   data/mane/GRCh38/openspliceai_eval/precomputed/
#   4. BigWig cache:       /runpod-volume/bigwig_cache/*.bw
#   5. Junction + eCLIP data (dense channels)
#   6. M3 labels:          data/mane/GRCh38/m3_labels/{positives_pooled,
#                          annotation_mask,disease_anchors}.parquet
#
# Usage:
#   ssh <cluster>
#   cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_train_m3_pod.sh \
#       > /runpod-volume/output/meta_layer/m3s_anchorpos_train.log 2>&1 &
#
# Override the fold policy via env (defaults give the CV run above):
#   DISEASE_ANCHORS=positivize ANCHOR_WEIGHT=5 \
#   ANCHOR_SOURCES=sf3b1_cryptic,encode_kd_cryptic bash ops_train_m3_pod.sh

set -e

WORKDIR=~/sky_workdir
OUTPUT_DIR=${OUTPUT_DIR:-output/meta_layer/m3s_anchorpos}
BIGWIG_CACHE=${BIGWIG_CACHE:-/runpod-volume/bigwig_cache}
DISEASE_ANCHORS=${DISEASE_ANCHORS:-positivize}
ANCHOR_WEIGHT=${ANCHOR_WEIGHT:-5}
# Budget policy: anchors occupy this fraction of positive-loss mass (per-anchor
# weight derived from it, overriding ANCHOR_WEIGHT). ~5% is the policy-consistent
# first run; a flat 5x would be only ~1.2%. Set empty to fall back to ANCHOR_WEIGHT.
ANCHOR_BUDGET_FRAC=${ANCHOR_BUDGET_FRAC:-0.05}
ANCHOR_SOURCES=${ANCHOR_SOURCES:-sf3b1_cryptic,encode_kd_cryptic}

cd "$WORKDIR"

echo "============================================================"
echo "M3-S + disease-anchor fold — $(date)"
echo "  Device:        cuda"
echo "  Anchors:       $DISEASE_ANCHORS (budget=${ANCHOR_BUDGET_FRAC:-off} w=$ANCHOR_WEIGHT, sources=$ANCHOR_SOURCES)"
echo "  BigWig:        $BIGWIG_CACHE"
echo "  Output:        $OUTPUT_DIR"
echo "  Baseline diff: M3-v1 recipe + --disease-anchors (one variable)"
echo "============================================================"

# Verify labels are staged
echo "Checking prerequisites..."
ls data/mane/GRCh38/m3_labels/positives_pooled.parquet >/dev/null
ls data/mane/GRCh38/m3_labels/annotation_mask.parquet >/dev/null
ls data/mane/GRCh38/m3_labels/disease_anchors.parquet >/dev/null
ls "$BIGWIG_CACHE"/hg38.phyloP100way.bw >/dev/null
echo "  All prerequisites present."

python -u examples/meta_layer/07_train_sequence_model.py \
    --mode m3 \
    --device cuda \
    --epochs 50 \
    --patience 10 \
    --samples-per-epoch 100000 \
    --remove-paralogs \
    --confirmed-weight 2.0 \
    --disease-anchors "$DISEASE_ANCHORS" \
    --anchor-weight "$ANCHOR_WEIGHT" \
    ${ANCHOR_BUDGET_FRAC:+--anchor-budget-frac "$ANCHOR_BUDGET_FRAC"} \
    --anchor-positivize-sources "$ANCHOR_SOURCES" \
    --bigwig-cache "$BIGWIG_CACHE" \
    --use-shards \
    --output-dir "$OUTPUT_DIR"

echo "============================================================"
echo "M3-anchor training complete — $(date)"
echo "Results: $OUTPUT_DIR"
echo "Next: eval via ops_eval_m3_novel_pod.sh (--mode build-cache on pod),"
echo "      rsync cache back, then 13_evaluate_m3_novel.py --mode eval locally."
echo "      M3-anchor is already registered in DEFAULT_MODELS."
echo "============================================================"
