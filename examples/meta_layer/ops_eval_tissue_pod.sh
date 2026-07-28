#!/usr/bin/env bash
# Tissue-stratified alternative-site evaluation on a GPU pod (RunPod).
#
# Produces base + M2-S per-tissue recall over the held-out chromosomes:
#   1) the M2-S alternative-site eval (09) dumps per-site outcomes
#      (chrom, position, splice_type, base_detected, meta_detected);
#   2) the tissue eval (16) stratifies those by GTEx tissue of junction support.
#
# Needs the pod's Ensembl base scores + bigWig cache on the volume (for the eval
# cache) and the M2-S checkpoint. Assumes ops_bootstrap_pod.sh has symlinked
# data/ and output/ to the volume.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
source examples/meta_layer/ops_bootstrap_pod.sh

CHROMS="chr1 chr3 chr5 chr7 chr9"
OUT="output/meta_layer/m2s_v4_cleanannot_alt_eval"
CACHE="output/meta_layer/gene_cache_ensembl_cleanannot"
BASE_SCORES="data/ensembl/GRCh38/openspliceai_eval/precomputed"
BIGWIG_CACHE="${BIGWIG_CACHE:-data/cache/bigwig}"

# Reuse the alt-site eval cache if already built (>100 npz), else build it.
BUILD_FLAG="--build-cache"
if [ -d "$CACHE/test" ] && [ "$(ls "$CACHE"/test/*.npz 2>/dev/null | wc -l)" -gt 100 ]; then
  BUILD_FLAG=""
  echo "Reusing existing eval cache at $CACHE/test"
fi

# 1) M2-S alternative-site eval, dumping per-site outcomes.
python -u examples/meta_layer/09_evaluate_alternative_sites.py \
    --checkpoint output/meta_layer/m2s_v4_cleanannot/best.pt \
    --annotation-source ensembl \
    --base-scores-dir "$BASE_SCORES" \
    --cache-dir "$CACHE" $BUILD_FLAG \
    --bigwig-cache "$BIGWIG_CACHE" \
    --device cuda \
    --output-dir "$OUT" \
    --dump-site-outcomes "$OUT/site_outcomes.parquet"

# 2) Tissue-stratified base + meta recall (both from the dump).
python -u examples/meta_layer/16_evaluate_tissue_stratified.py \
    --models base,meta --tissues dnase5 --test-chroms $CHROMS \
    --meta-outcomes "$OUT/site_outcomes.parquet" \
    --output-dir "$OUT"

echo "Done -> $OUT/tissue_stratified.json (base + M2-S per tissue)."
echo "Pull it back to output/meta_layer/m2s_v4_cleanannot_alt_eval/ to populate the dashboard tissue panel."
