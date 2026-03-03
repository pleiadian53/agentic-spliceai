# Base Layer Examples

**Phase**: 1-3 (Foundation + Data Preparation + Workflow Orchestration)
**Purpose**: Driver scripts for base model splice site prediction

---

## 📖 Documentation

- **[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)**: Comprehensive evaluation guide
  - ✅ Position count verification
  - 📊 Splice site detection statistics (TP/FP/FN/TN)
  - 📈 Performance metrics (F1, precision, recall, PR-AUC)
  - 🧬 Multi-gene and full-genome support

## 📋 Available Examples

### 01_phase1_prediction.py
**Phase 1: Single Gene Prediction**

Demonstrates the complete Phase 1 workflow for a single gene.

```bash
# From project root
cd /Users/pleiadian53/work/agentic-spliceai
python examples/base_layer/01_phase1_prediction.py --gene BRCA1
python examples/base_layer/01_phase1_prediction.py --gene TP53 --model openspliceai

# Or from examples/base_layer directory
cd examples/base_layer
python 01_phase1_prediction.py --gene BRCA1
```

**What it does**:
- Loads genomic resources (GTF, FASTA)
- Extracts gene data
- Loads OpenSpliceAI model (5 ensemble models)
- Generates splice site predictions
- **Highlights top predicted splice sites with scores**
- Shows detection statistics and score distributions

**Output**: Console output showing:
- Top 5 donor sites with positions and scores (e.g., `chr17:7,673,701 score=0.999`)
- Top 5 acceptor sites with positions and scores
- Total splice sites detected above threshold (e.g., 10 donors, 10 acceptors)
- Example true negatives to show strong discrimination
- Runtime: ~5-10s per gene

---

### 02_chromosome_prediction.py
**Phase 1: Multi-Gene Chromosome Prediction**

Demonstrates prediction on multiple genes from a chromosome.

```bash
# From project root
python examples/base_layer/02_chromosome_prediction.py --chromosome chr21 --genes 10
python examples/base_layer/02_chromosome_prediction.py --chromosome chr17 --genes 5 --model openspliceai
```

**What it does**:
- Loads gene annotations for specified chromosome
- Selects N genes for processing
- Runs predictions on all selected genes
- Reports per-gene and aggregate statistics

**Output**: Console output with:
- Per-gene position counts and processing status
- Aggregate statistics across all genes
- Runtime: ~30-120s depending on gene count

---

### 04_chunked_prediction.py
**Phase 3: Chunked Prediction Workflow (test mode)**

Demonstrates the chunked prediction pipeline with checkpointing and resume.

```bash
# Specific genes
python examples/base_layer/04_chunked_prediction.py --genes BRCA1 TP53 MYC EGFR --chunk-size 2

# Whole chromosome
python examples/base_layer/04_chunked_prediction.py --chromosomes chr22 --chunk-size 100

# Genes + chromosomes (union)
python examples/base_layer/04_chunked_prediction.py --chromosomes chr22 --genes BRCA1 --chunk-size 100

# Resume from previous run
python examples/base_layer/04_chunked_prediction.py --chromosomes chr22 --chunk-size 100 --resume --output-dir <prev_output>

# Refined usage

# Short name — resolves to output/chunking_prediction_test/
python examples/base_layer/04_chunked_prediction.py --chromosomes chr19 --genes TP53 --chunk-size 100 \
  --output-dir chunking_prediction_test

# Absolute/home path — used as-is
python examples/base_layer/04_chunked_prediction.py --chromosomes chr19 --genes TP53 --chunk-size 100 \
  --output-dir ~/work/agentic-spliceai/output/chunking_prediction_test

```

**What it does**:
- Flexible targeting: specific genes, whole chromosomes, or both
- Splits genes into configurable chunks
- Saves per-chunk checkpoints (TSV artifacts)
- Supports resume after interruption
- Tracks gene processing via manifest

**Output**: Console output plus structured artifacts in timestamped `output/` directory.

---

### 05_genome_precomputation.py
**Phase 3: Meta-Layer Precomputation (production mode)**

Demonstrates genome-scale precomputation for meta-layer training input.

```bash
python examples/base_layer/05_genome_precomputation.py --chromosomes chr22 --chunk-size 100
python examples/base_layer/05_genome_precomputation.py --chromosomes chr21 chr22 --chunk-size 500 --resume
```

**What it does**:
- Uses production mode: output routes to registry-managed path
- Path is annotation-source, build, and model-specific (e.g., `data/mane/GRCh38/openspliceai_eval/precomputed/`)
- Produces per-nucleotide raw scores for meta layer consumption
- Stable path (no timestamp) for cross-session resume

**Output**: Structured artifacts in `data/{source}/{build}/{model}_eval/precomputed/`

---

## 🎯 Learning Path

**New to base layer?**
1. Start with `01_phase1_prediction.py` (single gene)
2. Try `02_chromosome_prediction.py` (multiple genes)
3. Explore data preparation examples in `../data_preparation/`

**Ready for production?**
4. Try `04_chunked_prediction.py` (chunking + resume)
5. Run `05_genome_precomputation.py` (meta-layer training data)

---

## 🔗 Related Examples

- **Data Preparation**: `../data_preparation/` - Extract genes, sequences, splice sites
- **Integration Tests**: `../../tests/integration/base_layer/` - Automated tests
- **Notebooks**: `../../notebooks/base_layer/` - Educational notebooks

---

**Last Updated**: March 3, 2026
