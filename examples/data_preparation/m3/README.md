# M3 data preparation — novel splice-site label workflow

Run-ordered data workflow that produces the **M3 training label set** (novel
splice-site discovery). Completing every step here is a **precondition** to
training the M3 meta-model (Phase C, under `examples/meta_layer/`).

This directory holds *data preparation* only. Model training / evaluation /
monitoring of the meta-models lives under `examples/meta_layer/`. The design
and findings docs live under
[`examples/meta_layer/docs/M3/`](../../meta_layer/docs/M3/).

> Run each step after `mamba activate agentic-spliceai`.

## MLOps position

```
curate → INGEST/PROCESS (this dir) → VALIDATE → train (meta_layer) → eval → monitor → deploy
                                         ▲
                            splice-site strand validator (precondition gate)
```

Nothing downstream should start until the steps below complete and the
annotation passes the strand-validation gate (donor→GT / acceptor→AG, ~0.98
**both strands**).

## Preconditions

1. **Strand-correct annotation TSVs.** GENCODE/RefSeq/MANE/Ensembl
   `splice_sites_enhanced.tsv` must be the strand-fixed versions (regenerated
   2026-05-25; minus-strand GT/AG ~0.98). Validate before running.
2. **External downloads** (one-time, into the shared `data/` tree):
   - SpliceVault `SpliceVault_data_GRCh38.tsv.gz` → `data/splicevault/GRCh38/` (step 03)
   - RefSeq GTF `hg38.ncbiRefSeq.gtf` → `data/refseq/GRCh38/` (annotation)
   - SF3B1 Figshare `file_s03` → `data/sf3b1_deboever/` (step 05)
   - ENCODE-KD SpliceTools rMATS A3SS/A5SS → `data/encode_kd_splicetools/1_RBP_kd/` (step 06)

## Run order

| # | Script | Produces | Notes |
|---|---|---|---|
| 0 | `../../meta_layer/11_junction_coverage_audit.py` | `output/.../q3_novel_junction_sides.parquet` | **prerequisite** — GTEx novel junction sides (kept in meta_layer; general junction analysis) |
| 01 | `01_cross_annotation_audit.py` | `A1_survivors_final.parquet` | q3 sides absent from GENCODE ∪ RefSeq-curated |
| 02 | `02_build_positive_pool.py` | `positives_gtex_novel.parquet` | depth/tissue + GT/AG filter (GTEx-novel arm) |
| 03 | `03_ingest_splicevault.py` | `positives_splicevault.parquet` | the bulk positive source (~154K novel cryptic sites) |
| 04 | `04_merge_positives.py` | `positives_pooled.parquet` | GTEx-novel ∪ SpliceVault, annotation-clean |
| 05 | `05_ingest_sf3b1_anchors.py` | `anchors/anchors_sf3b1.parquet` | held-out disease anchor (hg19→GRCh38 liftover) |
| 06 | `06_ingest_encode_kd_anchors.py` | `anchors/anchors_encode_kd.parquet` | held-out disease anchor (rMATS KD-gained) |
| 07 | `07_ingest_tdp43_anchors.py` | `anchors/anchors_tdp43.parquet` | held-out disease anchor (ALS STMN2/UNC13A) |
| 08 | `08_finalize_anchors.py` | `disease_anchors.parquet` + decontaminated pool | combines anchors; anti-joins them out of training |
| 09 | `09_build_negatives.py` | `negatives.parquet` + `annotation_mask.parquet` | hard/easy negatives + loss ignore-mask |

All outputs land in `data/mane/GRCh38/m3_labels/`.

## Final label set (consumed by Phase C training)

| Artifact | Count | Role |
|---|---:|---|
| `positives_pooled.parquet` | 154,113 | novel sites (donor/acceptor) — SpliceVault-dominated |
| `negatives.parquet` | 308,000 | 154K hard (canonical-dinuc non-sites) + 154K easy |
| `annotation_mask.parquet` | 825,746 | annotated sites → **loss ignore-index** |
| `disease_anchors.parquet` | 6,351 | held-out (Phase D2 generalization); anti-joined out of training |

**Framing (settled 2026-05-26):** M3 is a splice-site **recognizer** + post-filter.
Annotated sites are masked in the loss (a novel and an annotated donor are
sequence-identical); novelty is exact set-subtraction at inference. See
[`../../meta_layer/docs/M3/m3_design.md`](../../meta_layer/docs/M3/m3_design.md) §5.

## Next: Phase C (training, a pod job)

Extract multimodal features (sequence + base scores + conservation/epigenetic/
RBP via bigWig) at the labeled positions and train `MetaSpliceModel` (M3,
junction modality dropped from inputs). bigWig streaming = pod, not local. The
positive/negative/mask label set above is the entry point.

## Related
- Design + findings: [`../../meta_layer/docs/M3/`](../../meta_layer/docs/M3/) (README, m3_design, m3_prerequisites, label_audit_A1)
