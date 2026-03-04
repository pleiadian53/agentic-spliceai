# Foundation Models Examples

**Sub-project**: `foundation_models/` (Evo2, Nucleotide Transformer, etc.)
**Purpose**: User-facing tutorials for embedding extraction, classifier training, and resource planning

---

## Learning Path

**New to foundation models?**
1. Start with `01_resource_check.py` — understand what your hardware can run
2. Run `02_synthetic_training_pipeline.py` — full end-to-end workflow without needing Evo2

**Ready for real embeddings?**
3. Extract embeddings with `03_embedding_extraction.py` (requires Evo2 + GPU)
4. Train and evaluate with `04_train_and_evaluate.py`

**Full pipeline orchestrator:**
5. Use `05_run_pipeline.py` to combine all steps with a single command

---

## Available Examples

### 01_resource_check.py
**Hardware Feasibility Check**

Detects current hardware and prints a feasibility table for all foundation model tasks.

```bash
# Auto-detect current hardware
python examples/foundation_models/01_resource_check.py

# Simulate a specific hardware target
python examples/foundation_models/01_resource_check.py --hardware a40-48gb
python examples/foundation_models/01_resource_check.py --hardware a100-80gb

# List available profiles
python examples/foundation_models/01_resource_check.py --list-hardware

# Detailed check for a specific task
python examples/foundation_models/01_resource_check.py --task embedding --model-size 40b
```

**Output**: Feasibility table showing which tasks can run (embedding extraction, classifier training, LoRA fine-tuning) with memory estimates.

---

### 02_synthetic_training_pipeline.py
**End-to-End Pipeline with Synthetic Data**

Full workflow — generate, train, evaluate — without needing Evo2 or a GPU. Always works on any hardware.

```bash
python examples/foundation_models/02_synthetic_training_pipeline.py --output /tmp/fm_demo/

# Custom parameters
python examples/foundation_models/02_synthetic_training_pipeline.py \
    --output /tmp/fm_demo/ --n-genes 10 --architecture cnn --epochs 30
```

**What it does**:
1. Generates synthetic embeddings with realistic exon/intron block structure
2. Trains an ExonClassifier with early stopping and LR scheduling
3. Evaluates and saves metrics JSON

**Output**: Embeddings HDF5, labels NPZ, model checkpoint, metrics JSON. AUROC ~0.5 is expected (random embeddings); real Evo2 embeddings should yield AUROC > 0.9.

**Runtime**: < 30 seconds on CPU.

---

### 03_embedding_extraction.py
**Extract Evo2 Embeddings (Requires GPU)**

Loads gene sequences, runs resource check, and extracts per-nucleotide Evo2 embeddings.

```bash
# Check resources first
python examples/foundation_models/01_resource_check.py --task embedding

# Extract embeddings
python examples/foundation_models/03_embedding_extraction.py \
    --genes BRCA1 TP53 --output /tmp/fm_demo/embeddings/

# Evo2 40B (requires A100 80GB+)
python examples/foundation_models/03_embedding_extraction.py \
    --genes BRCA1 --model-size 40b --output /tmp/fm_demo/embeddings/

# Remote GPU via SkyPilot
sky launch foundation_models/configs/skypilot/extract_embeddings_a40.yaml
```

**Prerequisites**: Prepared gene data (`agentic-spliceai-prepare --genes BRCA1 TP53`).

**Output**: HDF5 embeddings + `.labels.npz` per-nucleotide exon labels.

---

### 04_train_and_evaluate.py
**Train and Evaluate ExonClassifier**

Trains from pre-extracted embeddings with resource check, early stopping, and full evaluation.

```bash
# From synthetic data
python examples/foundation_models/04_train_and_evaluate.py \
    --embeddings /tmp/fm_demo/embeddings.h5 \
    --labels /tmp/fm_demo/embeddings.labels.npz \
    --output /tmp/fm_demo/model/

# With custom architecture
python examples/foundation_models/04_train_and_evaluate.py \
    --embeddings /tmp/fm_demo/embeddings.h5 \
    --labels /tmp/fm_demo/embeddings.labels.npz \
    --output /tmp/fm_demo/model/ \
    --architecture cnn --window-size 1024 --epochs 50

# Remote GPU via SkyPilot
sky launch foundation_models/configs/skypilot/train_classifier_a40.yaml
```

**Output**: Model checkpoint (`best_model.pt`), evaluation metrics JSON (AUROC, AUPRC, F1, precision, recall).

---

### 05_run_pipeline.py
**End-to-End Pipeline Orchestrator**

Combines all 4 steps (resource check, extract, download, train) into a single parameterized command. Three modes: dry-run (default), local-only, and execute.

```bash
# Dry-run — print config + commands (no cost)
python examples/foundation_models/05_run_pipeline.py \
    --model-size 7b --gpu a40 --chromosomes 22

# Local — synthetic data, always works
python examples/foundation_models/05_run_pipeline.py \
    --local-only --output-dir /tmp/fm_pipeline/

# Execute — launches SkyPilot jobs (costs money!)
python examples/foundation_models/05_run_pipeline.py \
    --execute --model-size 7b --gpu a40 --chromosomes 22 --train-local
```

**Knobs**: `--gpu` (a40/a100/h100), `--model-size` (7b/40b), `--chromosomes`, `--architecture`, `--window-size`, `--epochs`, `--cloud` (default: runpod)

**Cost tracking**: Estimates GPU cost from wall-clock time and hourly rates. Prints cost summary after execution.

**Output**: Auto-named directory (e.g., `output/fm_pipelines/evo2_7b_chr22/`) with embeddings, checkpoints, and metrics.

---

## Common Workflows

### Local: Synthetic Pipeline (No GPU Required)
```bash
# Full pipeline with synthetic data
python examples/foundation_models/02_synthetic_training_pipeline.py --output /tmp/fm_demo/
```

### Local: Real Embeddings (Requires GPU)
```bash
# 1. Prepare data
agentic-spliceai-prepare --genes BRCA1 TP53

# 2. Extract embeddings
python examples/foundation_models/03_embedding_extraction.py \
    --genes BRCA1 TP53 --output /tmp/fm_real/

# 3. Train and evaluate
python examples/foundation_models/04_train_and_evaluate.py \
    --embeddings /tmp/fm_real/embeddings.h5 \
    --labels /tmp/fm_real/embeddings.labels.npz \
    --output /tmp/fm_real/model/
```

### Remote GPU via SkyPilot
```bash
# Check which hardware you need
python examples/foundation_models/01_resource_check.py --hardware a40-48gb

# Launch embedding extraction on RunPod A40
sky launch foundation_models/configs/skypilot/extract_embeddings_a40.yaml

# Launch classifier training
sky launch foundation_models/configs/skypilot/train_classifier_a40.yaml

# Tear down when done
sky down extract-emb-7b
sky down train-classifier
```

---

## SkyPilot Configs

Pre-built YAML templates in `foundation_models/configs/skypilot/`:

| Config | GPU | Task |
|--------|-----|------|
| `extract_embeddings_a40.yaml` | A40 48GB | Evo2 7B embedding extraction |
| `extract_embeddings_a100.yaml` | A100 80GB | Evo2 40B embedding extraction |
| `train_classifier_a40.yaml` | A40 48GB | ExonClassifier training |

---

## Related

- **Notebook tutorial**: `../../notebooks/foundation_models/01_training_pipeline.ipynb`
- **Sub-project scripts**: `foundation_models/examples/` (developer-level 01-04)
- **Sub-project docs**: `foundation_models/docs/`
- **Resource estimator**: `foundation_models/foundation_models/utils/resources.py`
- **Base layer examples**: `../base_layer/`
- **Data preparation**: `../data_preparation/`

---

**Last Updated**: March 3, 2026
