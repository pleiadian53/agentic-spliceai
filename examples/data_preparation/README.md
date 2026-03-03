# Data Preparation Examples

**Phase**: 2 (Data Preparation Module)  
**Purpose**: Driver scripts for genomic data extraction and preparation

---

## 📋 Available Examples

### 01_prepare_gene_data.py
**Gene Annotation & Sequence Extraction**

Extract gene annotations from GTF and sequences from FASTA.

```bash
python 01_prepare_gene_data.py --genes BRCA1 TP53 --output /tmp/gene_data/
python 01_prepare_gene_data.py --genes BRCA1 --annotation-source mane
```

**What it does**:
- Loads gene annotations from GTF
- Extracts DNA sequences from FASTA
- Saves to TSV files

**Output**: `genes.tsv`, `sequences.tsv`

---

### 02_prepare_splice_sites.py
**Splice Site Annotation Extraction**

Extract and annotate splice sites from GTF with metadata inference.

```bash
python 02_prepare_splice_sites.py --genes BRCA1 --output /tmp/splice_sites/
python 02_prepare_splice_sites.py --genes BRCA1 TP53 --annotation-source mane
```

**What it does**:
- Parses GTF for exon annotations
- Derives splice sites from exon boundaries
- Infers metadata (biotype, exon numbers, etc.)
- Filters to specified genes

**Output**: `splice_sites_enhanced.tsv`

---

### 03_full_data_pipeline.py
**Complete Data Preparation Pipeline**

Run all data preparation steps in one command. Supports both per-gene and genome-wide extraction.

```bash
# Per-gene extraction
python 03_full_data_pipeline.py --genes BRCA1 TP53 EGFR --output /tmp/full_pipeline/
python 03_full_data_pipeline.py --genes BRCA1 --skip-sequences

# Genome-wide extraction (omit --genes)
python 03_full_data_pipeline.py --output /tmp/genome_wide/ --skip-sequences
```

**What it does**:
- Extracts gene annotations (all genes if `--genes` omitted)
- Extracts DNA sequences (skip with `--skip-sequences`)
- Extracts splice site annotations
- Saves all data in organized format
- Creates summary JSON

**Output**: `genes.tsv`, `sequences.tsv`, `splice_sites_enhanced.tsv`, `preparation_summary.json`

**Equivalent CLI**: `agentic-spliceai-base-prepare --genes <genes> --output <output>`

---

### 04_generate_ground_truth.py
**Genome-Wide Ground Truth Generation**

Generate the `splice_sites_enhanced.tsv` file used as ground truth when evaluating base model predictions (SpliceAI, OpenSpliceAI, etc.).

```bash
# MANE (curated, ~370K sites, ~19K genes) — default
python 04_generate_ground_truth.py --output data/mane/GRCh38/

# Ensembl (comprehensive, ~2.8M sites, all transcripts)
python 04_generate_ground_truth.py --output data/ensembl/GRCh38/ --annotation-source ensembl

# Force re-extraction (overwrite cached file)
python 04_generate_ground_truth.py --output data/mane/GRCh38/ --force
```

**What it does**:
- Extracts ALL splice sites from GTF (no gene filtering)
- Produces the same `splice_sites_enhanced.tsv` used by evaluation scripts
- Shows per-chromosome breakdown

**Output**: `splice_sites_enhanced.tsv`

**Equivalent CLI**: `agentic-spliceai-prepare --output <dir> --splice-sites-only`

---

### validate_mane_metadata.py
**MANE Metadata Validation**

Validate MANE metadata inference against Ensembl ground truth.

```bash
python validate_mane_metadata.py
```

**What it does**:
- Extracts splice sites from Ensembl (ground truth)
- Extracts splice sites from MANE (with inference)
- Compares metadata consistency for 10 test genes
- Reports validation results

**Test Genes**: BRCA1, TP53, EGFR, MYC, KRAS, PTEN, APC, BRAF, PIK3CA, NRAS

**Output**: Console report with validation summary

---

## 🎯 Learning Path

**New to data preparation?**
1. Start with `01_prepare_gene_data.py` (genes + sequences)
2. Try `02_prepare_splice_sites.py` (splice site extraction)
3. Run `03_full_data_pipeline.py` (complete pipeline — per-gene or genome-wide)
4. Use `04_generate_ground_truth.py` (genome-wide ground truth for evaluation)
5. Validate with `validate_mane_metadata.py` (quality check)

---

## 📊 Common Workflows

### Prepare Data for Single Gene
```bash
python 01_prepare_gene_data.py --genes BRCA1 --output /tmp/brca1/
```

### Prepare Data for Multiple Genes
```bash
python 03_full_data_pipeline.py --genes BRCA1 TP53 EGFR --output /tmp/multi_gene/
```

### Extract Only Splice Sites
```bash
python 02_prepare_splice_sites.py --genes BRCA1 --output /tmp/splice_only/
```

### Generate Genome-Wide Ground Truth for Evaluation
```bash
python 04_generate_ground_truth.py --output data/mane/GRCh38/
```

### Validate MANE Inference
```bash
python validate_mane_metadata.py
```

---

## 🔗 Related Examples

- **Base Layer**: `../base_layer/` - Run predictions using prepared data
- **CLI Tool**: Use `agentic-spliceai-base-prepare` for production workflows
- **Integration Tests**: `../../tests/integration/base_layer/` - Automated tests
- **Notebooks**: `../../notebooks/data_preparation/` - Educational notebooks

---

**Last Updated**: March 2, 2026
