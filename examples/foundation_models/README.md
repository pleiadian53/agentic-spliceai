# Foundation Models Examples

**Sub-project**: `foundation_models/` — multi-model splice site prediction pipeline
**Models**: SpliceBERT, Evo2, HyenaDNA, DNABERT-2 (extensible via `BaseEmbeddingModel`)

---

## Learning Path

**New to foundation models?**
1. Start with `01_synthetic_pipeline.py` — full end-to-end workflow without GPU
2. Run `03_train_and_evaluate.py` — train a classifier from pre-extracted embeddings

**Ready for real embeddings?**
3. Extract embeddings with `02_embedding_extraction.py` (requires GPU)
4. Run `05_sparse_exon_classifier.py` — multi-model comparison on sparse exon data

**Genome-scale splice prediction (main pipeline):**
5. Use `07a_direct_shard_splice_predictor.py` — extract, train, calibrate, evaluate

**Fine-tuning foundation models:**
6. Use `08_foundation_model_finetuning.py` — unfreeze and fine-tune the embedding model

**GPU cluster operations:**
7. `ops_provision_cluster.py` — acquire and manage RunPod pods via SkyPilot
8. `ops_stage_data.py` — upload reference data to network volume
9. `ops_run_pipeline.py` — execute jobs on running clusters

---

## Available Examples

### 01_synthetic_pipeline.py
**End-to-End Pipeline with Synthetic Data**

Full workflow — generate, train, evaluate — without needing a GPU. Always works on any hardware.

```bash
python examples/foundation_models/01_synthetic_pipeline.py --output /tmp/fm_demo/
```

**Runtime**: < 30 seconds on CPU.

---

### 02_embedding_extraction.py
**Extract Foundation Model Embeddings (Requires GPU)**

Loads gene sequences and extracts per-nucleotide embeddings from any registered model.

```bash
python examples/foundation_models/02_embedding_extraction.py \
    --genes BRCA1 TP53 --output /tmp/fm_demo/embeddings/
```

---

### 03_train_and_evaluate.py
**Train and Evaluate SpliceClassifier**

Trains from pre-extracted embeddings with early stopping and full evaluation.

```bash
python examples/foundation_models/03_train_and_evaluate.py \
    --embeddings /tmp/fm_demo/embeddings.h5 \
    --labels /tmp/fm_demo/embeddings.labels.npz \
    --output /tmp/fm_demo/model/
```

---

### 04_extract_and_train.py
**Combined Extract + Train Pipeline**

Single-command workflow that extracts embeddings and trains a classifier.

---

### 05_sparse_exon_classifier.py
**Multi-Model Sparse Exon Classifier**

Reproduce the Evo2 paper's exon classification benchmark across multiple models.
Supports `--mock` for local testing without CUDA.

```bash
# Mock mode (no GPU)
python examples/foundation_models/05_sparse_exon_classifier.py --mock

# Real: compare SpliceBERT vs HyenaDNA on chr22
python examples/foundation_models/05_sparse_exon_classifier.py \
    --model splicebert --chromosomes 22
```

**Results (Phase A)**: Evo2 0.9937, SpliceBERT 0.8594, HyenaDNA 0.8242, DNABERT-2 0.6556 AUROC.

---

### 06_dense_splice_predictor.py
**Dense Splice Site Predictor (Deprecated)**

Earlier approach using per-gene HDF5 caching. Superseded by `07a` for genome-scale work.

---

### 07_genome_scale_splice_predictor.py
**Genome-Scale Splice Predictor (Original)**

Per-gene HDF5 caching with streaming DataLoader. Works for small chromosome subsets
but hits disk limits at genome scale. See `07a` for the production version.

---

### 07a_direct_shard_splice_predictor.py
**Production Genome-Scale Splice Site Predictor**

The main entry point for training splice site classifiers on foundation model embeddings.
Three-phase pipeline: extract embeddings to shards, train classifier, evaluate on held-out
chromosomes.

```bash
# Mock mode (local, no GPU)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --mock -o /tmp/test-07a/

# Full genome (GPU pod)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes all \
    --split spliceai \
    --batch-size 256 \
    -o /runpod-volume/output/splice_classifier/splicebert-full-genome/

# Evaluate a trained model (no retraining)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes all \
    --split spliceai \
    --eval-only \
    --checkpoint /path/to/model/ \
    -o /path/to/output/

# Resume after interruption (reuses existing shards)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes all \
    --resume \
    -o /path/to/output/

# Subset with balanced split (largest chroms to train)
python examples/foundation_models/07a_direct_shard_splice_predictor.py \
    --foundation-model splicebert \
    --chromosomes 3,22 \
    --split balanced \
    -o /tmp/test-subset/
```

**Key features**:
- Direct-to-shard extraction (no per-gene HDF5 files)
- Fork-safe HDF5 dataset with multi-worker DataLoader
- Incremental manifest saving (crash-recoverable)
- Post-hoc temperature calibration
- Per-chromosome test evaluation (memory-bounded)
- `--checkpoint` / `--eval-only` for loading trained models
- `--split balanced` for subset chromosome runs
- `--train-chromosomes` / `--test-chromosomes` for explicit control

**Latest results (SpliceBERT, full genome)**:
- 380K training windows, 93 shards, SpliceAI chromosome split
- Best val_AUPRC: 0.8904 (epoch 97/100)
- Calibration: ECE 0.0001 → 0.0000

---

### 08_foundation_model_finetuning.py
**Foundation Model Fine-Tuning**

Unfreezes the foundation model and fine-tunes end-to-end with the splice classifier head.
Supports warm-starting from a frozen-head checkpoint (`--from-frozen`).

```bash
# Mock mode
python examples/foundation_models/08_foundation_model_finetuning.py \
    --mock -o /tmp/test-finetune/

# Real: fine-tune SpliceBERT on chr22
python examples/foundation_models/08_foundation_model_finetuning.py \
    --foundation-model splicebert \
    --chromosomes 22 \
    --from-frozen /path/to/07a/model/best_model.pt \
    -o /path/to/output/
```

---

## Ops Scripts

### ops_provision_cluster.py
**Provision GPU Clusters**

Acquire a RunPod pod via SkyPilot, install packages, link the network volume.

```bash
# Provision with defaults (A40)
python examples/foundation_models/ops_provision_cluster.py

# First time: provision + upload data to volume
python examples/foundation_models/ops_provision_cluster.py --stage-data

# Show running clusters
python examples/foundation_models/ops_provision_cluster.py --status

# Tear down
python examples/foundation_models/ops_provision_cluster.py --down
```

### ops_stage_data.py
**Stage Data to Network Volume**

Direct rsync to a running pod — faster than SkyPilot file_mounts for iterative uploads.

```bash
python examples/foundation_models/ops_stage_data.py
python examples/foundation_models/ops_stage_data.py --weights spliceai
```

### ops_run_pipeline.py
**Execute Jobs on Clusters**

Run scripts on existing or new clusters.

```bash
python examples/foundation_models/ops_run_pipeline.py --execute \
    --cluster <name> --no-teardown \
    -- python your_script.py --args
```

---

## GPU Infrastructure

### Network Volume
- RunPod volume "AI lab extension" (500 GB, CA-MTL-1)
- Mounted at `/runpod-volume` on SkyPilot pods (NOT `/workspace`)
- Persists across pod teardown — shards, models, and data survive

### Key Paths on Pod
| Path | Content |
|------|---------|
| `~/sky_workdir/` | Project code (synced by SkyPilot) |
| `/runpod-volume/data/` | FASTA, GTF, splice sites |
| `/runpod-volume/output/` | Training output, shards, models |

### GPU Config
Edit `foundation_models/configs/gpu_config.yaml` for GPU type, model deps, and volume settings.

---

## Related

- **Training performance guide**: `docs/training-performance/gpu-utilization-guide.md`
- **SpliceBERT integration**: `foundation_models/foundation_models/splicebert/`
- **SpliceClassifier**: `foundation_models/foundation_models/classifiers/splice_classifier.py`
- **Data pipeline**: `foundation_models/foundation_models/data/` (datasets, sharding)
- **Model comparison**: `docs/model-comparison.md`
- **Session logs**: `dev/sessions/` (detailed implementation history)

---

**Last Updated**: March 26, 2026
