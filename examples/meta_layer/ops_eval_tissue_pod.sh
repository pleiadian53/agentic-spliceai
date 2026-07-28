#!/usr/bin/env bash
# Tissue-stratified alternative-site evaluation on a GPU pod (RunPod).
#
# Locally we can only compute the tissue *landscape* (alt-site counts per tissue),
# because the held-out base scores and the dense-feature eval cache live on the
# volume, not on a laptop. This runner produces the held-out numbers there.
#
# Assumes ops_bootstrap_pod.sh has symlinked data/ and output/ to the volume.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root
source examples/meta_layer/ops_bootstrap_pod.sh

CHROMS="chr1 chr3 chr5 chr7 chr9"
OUT="output/meta_layer/m2s_v4_cleanannot_alt_eval"

# ---------------------------------------------------------------------------
# 1) Held-out BASE per-tissue recall.
#    Needs Ensembl test-chrom base scores on the volume
#    (data/ensembl/GRCh38/openspliceai_eval/precomputed/). Pass --base-scores if
#    they are stored as something other than predictions.tsv.
# ---------------------------------------------------------------------------
python -u examples/meta_layer/16_evaluate_tissue_stratified.py \
    --models base --tissues dnase5 --test-chroms $CHROMS \
    --output-dir "$OUT"

# ---------------------------------------------------------------------------
# 2) META (M2-S) per-tissue recall.
#    Requires per-site meta outcomes from the M2-S neural eval. Contract for
#    16's --meta-outcomes: a parquet with columns
#        chrom, position, splice_type, meta_detected (bool)
#    covering the alternative sites on $CHROMS. Generate it by dumping per-site
#    argmax outcomes from the alternative-site eval (09), then:
#
#    python -u examples/meta_layer/16_evaluate_tissue_stratified.py \
#        --models base,meta --tissues dnase5 --test-chroms $CHROMS \
#        --meta-outcomes "$OUT/site_outcomes.parquet" --output-dir "$OUT"
#
# The tissue panel in the Bio Lab UI (/metrics) reads $OUT/tissue_stratified.json
# and fills in the Meta bars automatically once meta_recall is present.
echo "Base per-tissue done -> $OUT/tissue_stratified.json"
echo "Meta step: generate site_outcomes.parquet from the M2-S eval, then re-run with --meta-outcomes."
