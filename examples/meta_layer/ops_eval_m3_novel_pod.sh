#!/bin/bash
# Build the M3 Phase D / Tier 0 evaluation feature cache on a pod.
#
# This is the ONLY pod-dependent step of the M3 novel-splice-site eval: it builds
# the shared 9-channel per-gene .npz cache, which needs the bigWig cache on the
# network volume. Scoring + metrics run locally afterward
# (13_evaluate_m3_novel.py --mode eval), so the pod is billed only for the build.
#
# Prereqs on the volume (/runpod-volume), all kept after Phase C:
#   bigwig_cache/*.bw
#   data/mane/GRCh38/openspliceai_eval/precomputed/predictions_chr{1,3,5,7,9}.parquet
#     (base scores — the build HARD-FAILS if any test chrom is missing, so a
#      missing file can't silently become a uniform 1/3 prior)
#   data/mane/GRCh38/rbp_data/eclip_peaks_neuronal.parquet   (RBP channel)
#   data/GRCh38/junction_data/junctions_gtex_v8.parquet      (junction channel)
#   data/mane/GRCh38/splice_sites_enhanced.tsv + MANE GTF + reference FASTA
#
# Stage the gene list from LOCAL first (it lives under output/, gitignored, so a
# workdir sync will NOT carry it):
#   rsync -avz output/meta_layer/m3_eval_d1/eval_genes.txt \
#       <cluster>:~/sky_workdir/output/meta_layer/m3_eval_d1/
#
# Usage (on pod):
#   ssh <cluster>; cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_eval_m3_novel_pod.sh \
#       > /runpod-volume/output/meta_layer/m3_eval_d1/build_cache.log 2>&1 &

set -e
WORKDIR=~/sky_workdir
cd "$WORKDIR"

# Symlink /runpod-volume -> data/, output/ so cwd-relative paths hit the volume.
source examples/meta_layer/ops_bootstrap_pod.sh

GENE_LIST=output/meta_layer/m3_eval_d1/eval_genes.txt
CACHE_DIR=output/meta_layer/m3_eval_d1/gene_cache   # -> /runpod-volume/output/... (persists, rsync-able)
BIGWIG_CACHE=/runpod-volume/bigwig_cache

echo "============================================================"
echo "M3 Phase D — feature cache build — $(date)"
ls "$GENE_LIST" >/dev/null
echo "  Genes:   $(wc -l < "$GENE_LIST")"
echo "  Cache:   $CACHE_DIR"
echo "  BigWig:  $BIGWIG_CACHE"
echo "============================================================"

# --base-scores-dir defaults to the MANE openspliceai precomputed dir (what M3 v1
# trained with) via the registry; the script's own preflight verifies test-chrom
# base scores exist. If they are only on the Ensembl path on this volume, re-run
# with: --base-scores-dir /runpod-volume/data/ensembl/GRCh38/openspliceai_eval/precomputed
python -u examples/meta_layer/13_evaluate_m3_novel.py --mode build-cache \
    --gene-list  "$GENE_LIST" \
    --cache-dir  "$CACHE_DIR" \
    --bigwig-cache "$BIGWIG_CACHE" \
    --device cuda

echo "============================================================"
echo "M3 Phase D cache build complete — $(date)"
echo "  rsync $CACHE_DIR back to LOCAL, then: 13_evaluate_m3_novel.py --mode eval"
echo "============================================================"
