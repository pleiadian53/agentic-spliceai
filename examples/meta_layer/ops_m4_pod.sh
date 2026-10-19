#!/bin/bash
# M4 regulator arm on a pod: event-site features (all modalities) → held-out-regulator benchmark.
#
# Stage 1 (features): examples/data_preparation/m4/03_build_site_features.py for every
#   long / short / partner site of kd_events.parquet (41,127 sites), with ALL modalities.
#   The epigenetic + chromatin-accessibility bigWigs live only on the volume, which is why
#   this runs here. Checkpointed per chromosome (--resume), so a restart loses one chrom.
# Stage 2 (benchmark): examples/meta_layer/17_m4_fewshot_benchmark.py for each setting.
#   TabPFN methods use TabPFN v2 (free, Apache-2.0-based license, no token) on CUDA.
#
# Prerequisites (on the volume, checked below before any compute):
#   data/mane/GRCh38/{FASTA, openspliceai_eval/precomputed/, m4_labels/kd_events.parquet,
#                     m4_labels/regulator_site_features.parquet, rbp_data/eclip_peaks.parquet}
#   data/ensembl/Homo_sapiens.GRCh38.112.gtf   (Ensembl-only genes, scored on demand)
#   data/models/openspliceai, GTEx junctions, $BIGWIG_CACHE (22 tracks of full_stack.yaml)
#
# Usage:
#   ssh <cluster>; cd ~/sky_workdir
#   nohup bash examples/meta_layer/ops_m4_pod.sh > /runpod-volume/output/meta_layer/m4_pod.log 2>&1 &
#
# Env overrides:
#   STAGES="features benchmark"   SETTINGS="S3 S2"   FEATURE_SET=compact
#   BIGWIG_CACHE=/runpod-volume/bigwig_cache   BENCH_ARGS="--regulators 40"

set -eo pipefail

WORKDIR=~/sky_workdir
BIGWIG_CACHE=${BIGWIG_CACHE:-/runpod-volume/bigwig_cache}
STAGES=${STAGES:-"features benchmark"}
SETTINGS=${SETTINGS:-"S3 S2"}
FEATURE_SET=${FEATURE_SET:-compact}
BENCH_ARGS=${BENCH_ARGS:-}

cd "$WORKDIR"
M4=data/mane/GRCh38/m4_labels
REQUIRE_PATHS="data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
data/mane/GRCh38/openspliceai_eval/precomputed \
data/models/openspliceai \
data/GRCh38/junction_data/junctions_gtex_v8.parquet \
data/mane/GRCh38/rbp_data/eclip_peaks.parquet \
data/ensembl/Homo_sapiens.GRCh38.112.gtf \
$M4/kd_events.parquet $M4/regulator_site_features.parquet $BIGWIG_CACHE" \
    source examples/meta_layer/ops_bootstrap_pod.sh

echo "============================================================"
echo "M4 pod run — $(date)"
echo "  stages=$STAGES settings=$SETTINGS features=$FEATURE_SET"
echo "  bigWig cache: $BIGWIG_CACHE"
echo "  TabPFN: v2 (default; override with BENCH_ARGS=\"--tabpfn-version ...\")"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The stock image Python is PEP-668 externally managed (same setting as ops/gpu_runner.py setup).
python -c "import tabpfn" 2>/dev/null || PIP_BREAK_SYSTEM_PACKAGES=1 pip install -q "tabpfn==9.2.0"

# Preflight: every bigWig the full_stack config uses must be cached locally; otherwise the
# modality silently streams from ENCODE/UCSC (slow, and a failed fetch is a failed run).
python - "$BIGWIG_CACHE" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "examples/features")
from config_loader import load_workflow_config
from agentic_spliceai.splice_engine.features.modalities import chrom_access as C
from agentic_spliceai.splice_engine.features.modalities import conservation as V
from agentic_spliceai.splice_engine.features.modalities import epigenetic as E

cache = Path(sys.argv[1])
cfg, _, _ = load_workflow_config(config_name="full_stack")
epi, acc = cfg.modality_configs["epigenetic"], cfg.modality_configs["chrom_access"]
want = [E.EPIGENETIC_TRACK_REGISTRY["GRCh38"][c][m]["filename"]
        for c in epi.cell_lines for m in epi.marks]
want += [C.ATAC_TRACK_REGISTRY["GRCh38"][c]["filename"] for c in acc.cell_lines]
want += [C.DNASE_TRACK_REGISTRY["GRCh38"][c]["filename"] for c in acc.dnase_cell_lines]
want += [V.TRACK_REGISTRY["GRCh38"][t]["filename"] for t in ("phylop", "phastcons")]
missing = [f for f in want if not (cache / f).exists()]
print(f"bigWig preflight: {len(want) - len(missing)}/{len(want)} tracks cached in {cache}")
if missing:
    sys.exit(f"MISSING (would stream remotely): {missing}")
PY

if [[ " $STAGES " == *" features "* ]]; then
    python -u examples/data_preparation/m4/03_build_site_features.py \
        --bigwig-cache "$BIGWIG_CACHE" --resume
fi

if [[ " $STAGES " == *" benchmark "* ]]; then
    for s in $SETTINGS; do
        python -u examples/meta_layer/17_m4_fewshot_benchmark.py \
            --setting "$s" --feature-set "$FEATURE_SET" --tabpfn-workers 8 $BENCH_ARGS
    done
fi

echo "============================================================"
echo "M4 pod run complete — $(date)"
echo "Pull back (from the laptop):"
echo "  rsync -avz -L --no-owner --no-group --no-perms <cluster>:/runpod-volume/$M4/site_features.parquet $M4/"
echo "  rsync -avz -L --no-owner --no-group --no-perms <cluster>:/runpod-volume/output/meta_layer/m4_fewshot/ output/meta_layer/m4_fewshot/"
echo "  rsync -avz -L --no-owner --no-group --no-perms <cluster>:/runpod-volume/output/meta_layer/m4_labels/ output/meta_layer/m4_labels/"
echo "============================================================"
