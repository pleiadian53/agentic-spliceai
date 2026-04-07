# M1-S Evaluation Tutorial

Guide to all evaluation modes in `08_evaluate_sequence_model.py`.

---

## Quick Reference

| Use Case | Key Flags | Typical Runtime |
|----------|-----------|-----------------|
| Standard test evaluation | `--checkpoint --cache-dir` | 15-20 min (A40) |
| Build cache + evaluate | `--checkpoint --build-cache` | 2-3 hrs (first run) |
| Quick sanity check | `--checkpoint --cache-dir --max-genes 5` | <1 min |
| Channel ablation | `--zero-channels junction_log1p junction_has_support` | 15-20 min |
| Threshold sweep | `--sweep-thresholds` | +1 min (post-eval) |
| Temperature calibration | `--calibrate-temperature --val-cache-dir` | +10 min (pre-eval) |
| Fixed temperature | `--temperature 1.5` | Same as standard |
| FASTA inference | `--fasta sequences.fa` | Depends on input |

---

## Case 1: Standard Test Evaluation

Evaluate M1-S on the SpliceAI test split (chr1, 3, 5, 7, 9) using a
pre-built gene cache.

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --device cuda
```

**Output**: `eval_results.json` with PR-AUC, accuracy, FN/FP counts,
top-k metrics, and comparison between meta model and base model.

**Prerequisites**: Gene cache must already exist. See Case 2 to build it.

---

## Case 2: Build Cache + Evaluate

Build the gene cache from scratch (FASTA + base scores + multimodal
features) and then evaluate. This is the full pipeline for first-time
setup on a new pod.

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --build-cache \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --device cuda
```

**Prerequisites**:
- FASTA reference genome (auto-resolved from registry)
- Base model prediction parquets (auto-resolved, or `--base-scores-dir`)
- Splice site annotations (`splice_sites_enhanced.tsv`)
- BigWig files for conservation/epigenetic features
- Libraries: `pyBigWig`, `pyfaidx`, `scikit-learn` (checked by preflight)

**Memory**: Streaming architecture keeps peak RAM < 10 GB for 5,500 genes.
The script runs a preflight check before starting expensive cache builds.

---

## Case 3: Custom Chromosome / Gene Evaluation

Evaluate on specific chromosomes or genes.

```bash
# Specific chromosomes
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --test-chroms chr1 chr3

# Specific genes
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --genes BRCA1 TP53 CFTR

# Quick sanity check (first N genes)
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --max-genes 3 --device cpu
```

---

## Case 4: Channel Ablation Study

Zero out specific multimodal channels at inference time to measure their
contribution. Reuses the existing gene cache — no rebuild needed.

```bash
# Remove junction support channels
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --zero-channels junction_log1p junction_has_support \
    --device cuda

# Remove all multimodal features (sequence + base scores only)
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --zero-channels all \
    --device cuda
```

**Available channels**: `phylop_score`, `phastcons_score`, `h3k36me3_max`,
`h3k4me3_max`, `atac_max`, `dnase_max`, `junction_log1p`,
`junction_has_support`, `rbp_n_bound`.

**Output**: `eval_ablation_<channels>.json`.

See `examples/meta_layer/ops_ablation_m1s_pod.sh` for the full ablation
battery and `results/m1s_ablation_study.md` for analysis.

---

## Case 5: Threshold Sweep

Find the optimal operating point on the precision-recall curve. Useful
for tuning the FP/FN trade-off.

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --sweep-thresholds \
    --device cuda
```

Reports for each splice type (donor, acceptor):
- Precision, recall, F1 at each threshold
- Best F1 threshold
- Best precision at >= 95% recall
- Comparison with base model at default threshold (0.5)

---

## Case 6: Temperature Scaling (Calibration)

Learn optimal temperature T from validation genes, then evaluate with
calibrated probabilities. Temperature scaling adjusts the model's
confidence without changing ranking (PR-AUC preserved).

### Step 1: Build validation gene cache

```bash
# Build cache for SpliceAI validation chromosomes (chr2,4,6,8,10)
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --build-cache \
    --test-chroms chr2 chr4 chr6 chr8 chr10 \
    --cache-dir output/meta_layer/m1s/gene_cache/val \
    --bigwig-cache /runpod-volume/bigwig_cache \
    --device cuda
```

### Step 2: Calibrate and evaluate

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --calibrate-temperature \
    --val-cache-dir output/meta_layer/m1s/gene_cache/val \
    --sweep-thresholds \
    --device cuda
```

### Alternative: Fixed temperature

If you already know the optimal T from a previous calibration:

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --temperature 1.5 \
    --device cuda
```

**How it works**:
1. Collects raw logits from validation genes (not test genes)
2. Optimizes `T` to minimize negative log-likelihood
3. At test time: `calibrated = alpha * softmax(logits / T) + (1-alpha) * softmax(base_scores)`
4. Higher T → softer predictions (less confident, fewer FPs)
5. Lower T → sharper predictions (more confident, fewer FNs)

**Output**: `eval_results_T<value>.json` with calibration metadata.

---

## Case 7: FASTA Inference

Run splice site prediction on arbitrary DNA sequences without gene
annotations, base model scores, or multimodal features. Uses uniform
1/3 base-score prior and zero multimodal features.

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --fasta /path/to/sequences.fa \
    --output-format parquet \
    --device cpu
```

**Output**: `fasta_predictions.{tsv,parquet}` with columns:
`sequence_id`, `position`, `donor_prob`, `acceptor_prob`, `neither_prob`.

**Caveats**:
- Predictions are degraded without multimodal features and real base scores
- Sequences < 5,401 bp use single-window padding (less accurate at edges)
- This mode is for exploratory use — not for benchmarking

---

## Ensembl Annotation Source

For evaluating on Ensembl gene annotations instead of MANE:

```bash
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --annotation-source ensembl \
    --build-cache \
    --base-scores-dir /data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --device cuda
```

For alternative site evaluation (Ensembl \ MANE), use
`09_evaluate_alternative_sites.py` instead.

---

## Common Workflows

### First-time pod setup

```bash
# 1. Provision pod and stage data (see ops_eval_m1s_pod.sh)
# 2. Build cache + evaluate
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --build-cache --device cuda

# 3. Run ablation study
bash examples/meta_layer/ops_ablation_m1s_pod.sh

# 4. Threshold sweep
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --sweep-thresholds --device cuda
```

### FP reduction investigation

```bash
# 1. Temperature calibration (requires val cache)
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --calibrate-temperature \
    --val-cache-dir output/meta_layer/m1s/gene_cache/val \
    --sweep-thresholds --device cuda

# 2. Compare: remove noisy epigenetic channels
python 08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/m1s/best.pt \
    --cache-dir output/meta_layer/m1s/gene_cache/test \
    --zero-channels h3k36me3_max h3k4me3_max \
    --sweep-thresholds --device cuda
```

---

## Output Files

| File | When Produced |
|------|---------------|
| `eval_results.json` | Standard evaluation |
| `eval_results_T<value>.json` | Temperature-scaled evaluation |
| `eval_ablation_<channels>.json` | Ablation study |
| `fasta_predictions.{tsv,parquet}` | FASTA inference |

All JSON results contain: `meta_model` metrics, `base_model` metrics,
`fn_reduction_pct`, `fp_reduction_pct`, `meta_topk`, `base_topk`,
model/checkpoint metadata, and optionally `threshold_sweep` and
`calibration` dicts.
