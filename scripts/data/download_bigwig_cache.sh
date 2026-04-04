#!/bin/bash
# Download BigWig files for the dense feature extractor.
#
# Downloads conservation (UCSC PhyloP/PhastCons) and ENCODE (histone marks,
# chromatin accessibility) bigWig tracks to a local cache directory.  Once
# cached, the DenseFeatureExtractor reads from local files instead of
# streaming remotely — essential for GPU pods and faster on any system.
#
# Default cache: data/cache/bigwig/ (auto-resolved from settings.yaml)
# Override:      BIGWIG_CACHE=/path/to/cache bash scripts/data/download_bigwig_cache.sh
#
# Total size: ~17 GB (conservation ~14.7 GB + ENCODE ~2.5 GB)
#
# Usage:
#   bash scripts/data/download_bigwig_cache.sh              # default path
#   BIGWIG_CACHE=/runpod-volume/bigwig_cache bash scripts/data/download_bigwig_cache.sh  # pod

CACHE_DIR="${BIGWIG_CACHE:-data/cache/bigwig}"
mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"

echo "BigWig cache directory: $(pwd)"
echo ""

download() {
  local url="$1"
  local min_size="${2:-1000000}"  # minimum valid size in bytes (default 1 MB)
  local fname
  fname=$(basename "$url")

  # Skip if file exists AND is large enough (catches truncated downloads)
  if [ -f "$fname" ]; then
    local size
    size=$(stat -c%s "$fname" 2>/dev/null || stat -f%z "$fname" 2>/dev/null || echo 0)
    if [ "$size" -gt "$min_size" ]; then
      echo "  [skip] $fname ($(numfmt --to=iec "$size" 2>/dev/null || echo "${size} bytes"))"
      return 0
    else
      echo "  [incomplete] $fname (${size} bytes < min ${min_size}) — re-downloading"
      rm -f "$fname"
    fi
  fi

  echo "  [download] $fname ..."
  # Prefer curl (macOS built-in, no homebrew dependency issues)
  if command -v curl >/dev/null 2>&1; then
    if ! curl -sfL --retry 3 --retry-delay 5 -o "$fname" "$url"; then
      echo "  [FAILED] $fname — giving up"
      rm -f "$fname"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if ! wget -q --timeout=60 --tries=3 "$url" -O "$fname"; then
      echo "  [FAILED] $fname — giving up"
      rm -f "$fname"
    fi
  else
    echo "  [ERROR] Neither curl nor wget available"
    return 1
  fi
}

# ── Conservation (UCSC, GRCh38 100-way alignment) ────────────────────
echo "=== Conservation (PhyloP + PhastCons) ==="
download "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
download "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw"

# ── Epigenetic: H3K36me3 (ENCODE, 5 cell lines) ─────────────────────
# URLs synced from EPIGENETIC_TRACK_REGISTRY in epigenetic.py (2026-04-03)
echo ""
echo "=== H3K36me3 (exon body mark, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2020/10/29/2d536994-e28f-4f50-b5d4-c0441e0e059f/ENCFF163NTH.bigWig"  # K562
download "https://encode-public.s3.amazonaws.com/2020/10/29/7bf3c0a1-df2b-40fd-950f-9aa735aca81a/ENCFF312MUY.bigWig"  # GM12878
download "https://encode-public.s3.amazonaws.com/2020/11/19/89e1cb7b-0019-46c0-86d7-ed7b7910a445/ENCFF488THD.bigWig"  # H1
download "https://encode-public.s3.amazonaws.com/2022/06/14/cbf630ce-5691-4d19-8649-15accfd22fd7/ENCFF247LOP.bigWig"  # HepG2
download "https://encode-public.s3.amazonaws.com/2020/10/29/9e0a8b55-8146-4ec1-8804-ca5e50e80ef2/ENCFF049JNX.bigWig"  # keratinocyte

# ── Epigenetic: H3K4me3 (ENCODE, 5 cell lines) ──────────────────────
echo ""
echo "=== H3K4me3 (promoter mark, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2016/08/12/e59a8bb2-02ee-451d-994d-732db7c523a0/ENCFF814IYI.bigWig"  # K562
download "https://encode-public.s3.amazonaws.com/2016/08/23/6b4cd8db-2a4c-4023-97c4-1232aaa3ef59/ENCFF776DPQ.bigWig"  # GM12878
download "https://encode-public.s3.amazonaws.com/2020/10/19/14b72686-b8a9-48f1-81b2-5c57d157dc30/ENCFF602IAY.bigWig"  # H1
download "https://encode-public.s3.amazonaws.com/2020/09/30/2ca06a67-3a26-48f4-8354-fc4613c73b67/ENCFF284FVP.bigWig"  # HepG2
download "https://encode-public.s3.amazonaws.com/2020/10/20/72767546-aa79-4c57-8c72-68c380e41387/ENCFF632PUY.bigWig"  # keratinocyte

# ── ATAC-seq (ENCODE, 5 cell lines) ─────────────────────────────────
# URLs synced from ATAC_TRACK_REGISTRY in chrom_access.py (2026-04-03)
echo ""
echo "=== ATAC-seq (chromatin accessibility, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2021/02/24/b1a5f637-dba2-4272-b09e-86316d037135/ENCFF754EAC.bigWig"  # K562
download "https://encode-public.s3.amazonaws.com/2021/02/24/4c49a77a-edb2-4766-8328-d44ebe7446eb/ENCFF487LOB.bigWig"  # GM12878
download "https://encode-public.s3.amazonaws.com/2021/02/24/bfb21d9d-a2eb-4ce1-a536-236ccc97eef8/ENCFF664EJT.bigWig"  # HepG2
download "https://encode-public.s3.amazonaws.com/2021/03/16/0b01a7f6-3cf2-4f6e-b49a-675703bf3776/ENCFF872SDF.bigWig"  # A549
download "https://encode-public.s3.amazonaws.com/2021/02/24/4dd00198-47b5-4ac6-9ac1-f11e2804800c/ENCFF282RNO.bigWig"  # IMR-90

# ── DNase-seq (ENCODE, 5 primary tissues) ────────────────────────────
# URLs synced from DNASE_TRACK_REGISTRY in chrom_access.py (2026-04-03)
echo ""
echo "=== DNase-seq (chromatin accessibility, 5 primary tissues) ==="
download "https://encode-public.s3.amazonaws.com/2020/11/22/a8f1a670-279c-4096-bd41-dfea58797f8c/ENCFF243QQE.bigWig"  # brain_cortex
download "https://encode-public.s3.amazonaws.com/2025/08/21/8bdf58c3-fe1f-4c6b-984d-3cc854907be7/ENCFF534MZU.bigWig"  # heart
download "https://encode-public.s3.amazonaws.com/2025/10/01/a199b4b3-6547-4c98-b6ff-191254c80b0d/ENCFF958NZO.bigWig"  # lung
download "https://encode-public.s3.amazonaws.com/2025/10/01/fb9a13cf-3cf6-4565-937b-93eb4a184a0a/ENCFF952ARY.bigWig"  # muscle
download "https://encode-public.s3.amazonaws.com/2025/08/14/5940c739-e7f5-4d12-80a9-6eba2c4bc169/ENCFF008VTZ.bigWig"  # liver

echo ""
echo "=== Summary ==="
echo "Total files: $(ls *.bigWig *.bw 2>/dev/null | wc -l)"
du -sh .
