# BigWig Data Sources and Caching

## Overview

Several modalities in the multimodal feature engineering pipeline depend on
remote BigWig files hosted by UCSC and ENCODE:

| Modality | Source | Tracks | Total size |
|----------|--------|--------|------------|
| Conservation | UCSC Genome Browser | PhyloP, PhastCons (100-way) | ~14.7 GB |
| Epigenetic | ENCODE S3 | H3K36me3, H3K4me3 (5 cell lines each) | ~3.5 GB |
| Chromatin accessibility | ENCODE S3 | ATAC-seq (5 cell lines), DNase-seq (5 tissues) | ~13 GB |

These files are queried at per-nucleotide resolution by:
- **Position-level modalities** (`conservation.py`, `epigenetic.py`, `chrom_access.py`)
  — batch queries per chromosome for sampled positions
- **Dense feature extractor** (`dense_feature_extractor.py`) — window-level
  queries for sequence-level meta models (M*-S)

## Local Caching

Remote streaming works but is slow (~2 min/gene for conservation alone).
For iterative development or GPU pod training, cache files locally.

**Default cache path**: `data/cache/bigwig/` (configured in `settings.yaml`
under `cache.bigwig_dir`).

**Download script**:
```bash
# Local (default path from settings.yaml)
bash scripts/data/download_bigwig_cache.sh

# GPU pod (custom path)
BIGWIG_CACHE=/runpod-volume/bigwig_cache bash scripts/data/download_bigwig_cache.sh
```

The script is **idempotent** — it skips files that already exist and are
larger than 1 MB (catches truncated downloads). Safe to run repeatedly.

**Resolution chain** in `DenseFeatureExtractor`:
1. Explicit `--bigwig-cache` CLI argument (highest priority)
2. `settings.yaml` → `cache.bigwig_dir` (if directory exists)
3. Remote URL streaming (fallback)

For conservation tracks, `_get_bigwig_handle()` checks the cache by the
track's `filename` field from the registry. For ENCODE tracks,
`_query_bigwig_url()` checks by the URL's basename (e.g., `ENCFF163NTH.bigWig`).

## ENCODE URL Stability

**ENCODE periodically re-processes and re-uploads data files.** When this
happens:

1. The **S3 path changes** (the UUID subdirectory in the URL is regenerated)
2. The **accession may stay the same** (same data, new path) or **change**
   (new processing, new accession supersedes old one)
3. The old S3 URL returns `NoSuchKey` (HTTP 404 from S3, not a standard 404)

This means **hardcoded S3 URLs go stale** over time. We discovered this on
2026-04-03 when 12 of 20 ENCODE bigWig downloads failed with `NoSuchKey`.

### How to detect stale URLs

```bash
# Check if a URL still works (expect 200, not 404)
curl -sI "https://encode-public.s3.amazonaws.com/2020/10/28/.../ENCFF355OWW.bigWig" | head -1
# HTTP/1.1 404 Not Found  ← stale

# Or check via ENCODE API (more reliable)
curl -s "https://www.encodeproject.org/files/ENCFF355OWW/?format=json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('status:', d.get('status'))
print('url:', d.get('cloud_metadata', {}).get('url', 'NO URL'))
"
```

### How to update stale URLs

1. Find the **experiment ID** from the registry (e.g., `ENCSR000AMB`)
2. Query ENCODE API for the current released file:
   ```bash
   curl -s "https://www.encodeproject.org/search/?type=File\
   &dataset=/experiments/ENCSR000AMB/\
   &file_format=bigWig\
   &output_type=fold+change+over+control\
   &assembly=GRCh38\
   &status=released\
   &format=json&limit=1" | python3 -c "
   import sys, json
   d = json.load(sys.stdin)
   f = d['@graph'][0]
   print('accession:', f['accession'])
   print('url:', f.get('cloud_metadata', {}).get('url'))
   "
   ```
3. Update the URL (and accession/filename if changed) in the registry:
   - `epigenetic.py` → `EPIGENETIC_TRACK_REGISTRY`
   - `chrom_access.py` → `ATAC_TRACK_REGISTRY` / `DNASE_TRACK_REGISTRY`
4. Update `scripts/data/download_bigwig_cache.sh` to match

**Important**: The download script URLs and the Python registry URLs must
stay in sync. The download script has comments noting the sync date.

### Output types by assay

Different assays use different `output_type` values on ENCODE:

| Assay | output_type | Notes |
|-------|------------|-------|
| ChIP-seq (H3K36me3, H3K4me3) | `fold change over control` | Normalized signal |
| ATAC-seq | `fold change over control` | Normalized signal |
| DNase-seq | `read-depth normalized signal` | Different normalization |

Using the wrong `output_type` returns no results from the API.

## UCSC URL Stability

UCSC conservation track URLs (`hgdownload.soe.ucsc.edu`) are **highly stable**
— they haven't changed since 2015 (file dates on the server). These are
unlikely to break.

## Incident Log

### 2026-04-03: ENCODE S3 URL migration

**Symptom**: 12 of 20 ENCODE bigWig downloads returned `NoSuchKey` from S3.
pyBigWig silently returned NULL handles, causing all ENCODE channels to
fall back to zeros during training.

**Root cause**: ENCODE had re-processed and re-uploaded files under new S3
paths. Some accessions were superseded entirely (new accession numbers).

**Fix**: Queried ENCODE REST API for current URLs per experiment ID. Updated
4 accessions (2 in `epigenetic.py`, 2 in `chrom_access.py`) and 14 S3 URLs
across the registries and download script.

**Files updated**:
- `src/.../features/modalities/epigenetic.py` (EPIGENETIC_TRACK_REGISTRY)
- `src/.../features/modalities/chrom_access.py` (ATAC/DNASE_TRACK_REGISTRY)
- `scripts/data/download_bigwig_cache.sh`

**Prevention**: Before a major training run, verify URLs with:
```bash
BIGWIG_CACHE=/tmp/test bash scripts/data/download_bigwig_cache.sh
```
If any downloads fail, follow the update procedure above.
