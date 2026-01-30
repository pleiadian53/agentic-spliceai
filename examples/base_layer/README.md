# Base Layer Examples

**Phase**: 1-2 (Foundation + Data Preparation)  
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

## 🎯 Learning Path

**New to base layer?**
1. Start with `01_phase1_prediction.py` (single gene)
2. Try `02_chromosome_prediction.py` (multiple genes)
3. Explore data preparation examples in `../data_preparation/`

---

## 🔗 Related Examples

- **Data Preparation**: `../data_preparation/` - Extract genes, sequences, splice sites
- **Integration Tests**: `../../tests/integration/base_layer/` - Automated tests
- **Notebooks**: `../../notebooks/base_layer/` - Educational notebooks

---

**Last Updated**: January 30, 2026
