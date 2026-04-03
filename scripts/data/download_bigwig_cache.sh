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
  local fname
  fname=$(basename "$url")
  if [ -f "$fname" ]; then
    echo "  [skip] $fname (already exists)"
    return 0
  fi
  echo "  [download] $fname ..."
  if ! wget -q --timeout=60 --tries=3 "$url" -O "$fname"; then
    echo "  [FAILED] $fname — retrying..."
    rm -f "$fname"
    wget -q --timeout=120 --tries=2 "$url" -O "$fname" || {
      echo "  [FAILED] $fname — giving up"
      rm -f "$fname"
    }
  fi
}

# ── Conservation (UCSC, GRCh38 100-way alignment) ────────────────────
echo "=== Conservation (PhyloP + PhastCons) ==="
download "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phyloP100way/hg38.phyloP100way.bw"
download "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/phastCons100way/hg38.phastCons100way.bw"

# ── Epigenetic: H3K36me3 (ENCODE, 5 cell lines) ─────────────────────
echo ""
echo "=== H3K36me3 (exon body mark, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2020/10/29/2d536994-e28f-4f50-b5d4-c0441e0e059f/ENCFF163NTH.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/10/29/7bf3c0a1-df2b-40fd-950f-9aa735aca81a/ENCFF312MUY.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/10/28/fe8ce5ee-abae-4fb1-be29-c3c91f33af36/ENCFF355OWW.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/11/12/ce9ce618-4139-4e05-94d5-deea57fb9d0b/ENCFF429WMO.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/10/28/7b8a6dd2-7c5e-4d90-9f16-65a3f89b3a10/ENCFF914VIV.bigWig"

# ── Epigenetic: H3K4me3 (ENCODE, 5 cell lines) ──────────────────────
echo ""
echo "=== H3K4me3 (promoter mark, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2016/08/12/e59a8bb2-02ee-451d-994d-732db7c523a0/ENCFF814IYI.bigWig"
download "https://encode-public.s3.amazonaws.com/2016/08/23/6b4cd8db-2a4c-4023-97c4-1232aaa3ef59/ENCFF776DPQ.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/07/15/ab8c4277-74cc-4497-8e2e-2e2d6b60d0a1/ENCFF928HKQ.bigWig"
download "https://encode-public.s3.amazonaws.com/2016/08/23/3b7f15dc-2f5b-4e2b-b500-99adbe2bcc11/ENCFF561SFW.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/07/15/f6a6a24e-a792-4f53-928a-1a67fe84ee5b/ENCFF516IMF.bigWig"

# ── ATAC-seq (ENCODE, 5 cell lines) ─────────────────────────────────
echo ""
echo "=== ATAC-seq (chromatin accessibility, 5 cell lines) ==="
download "https://encode-public.s3.amazonaws.com/2021/02/24/b1a5f637-dba2-4272-b09e-86316d037135/ENCFF754EAC.bigWig"
download "https://encode-public.s3.amazonaws.com/2021/02/24/d22e144b-e283-481e-ac8c-5f21d5e1adf7/ENCFF706FIE.bigWig"
download "https://encode-public.s3.amazonaws.com/2021/06/14/4d919406-02df-4d33-b9d0-a2d8bb2f09d1/ENCFF972WKP.bigWig"
download "https://encode-public.s3.amazonaws.com/2021/05/27/b18f81f9-faba-4b09-a9b3-2d1f8b31e76f/ENCFF534HOB.bigWig"
download "https://encode-public.s3.amazonaws.com/2021/06/14/80e5ee87-dc3f-4c57-8b44-cb24d7d7a3e3/ENCFF802XTN.bigWig"

# ── DNase-seq (ENCODE, 5 primary tissues) ────────────────────────────
echo ""
echo "=== DNase-seq (chromatin accessibility, 5 primary tissues) ==="
download "https://encode-public.s3.amazonaws.com/2020/11/22/a8f1a670-279c-4096-bd41-dfea58797f8c/ENCFF243QQE.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/11/23/58b77d8f-2a28-46f1-81f1-28e4dea29a35/ENCFF750CPR.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/11/22/69fdda47-e9b7-4ad7-b21e-84a6de99bb39/ENCFF759OLD.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/11/23/79a94f84-49f1-4158-a9f0-67e2c8a7ae41/ENCFF975SCJ.bigWig"
download "https://encode-public.s3.amazonaws.com/2020/11/22/e98eecc8-3a2a-4a2f-bc0b-80c76dcfc8e1/ENCFF741JEH.bigWig"

echo ""
echo "=== Summary ==="
echo "Total files: $(ls *.bigWig *.bw 2>/dev/null | wc -l)"
du -sh .
