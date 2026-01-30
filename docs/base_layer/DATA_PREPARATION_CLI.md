# Data Preparation CLI Guide

Command-line interface for preparing genomic data for splice site prediction.

---

## Quick Start

```bash
# Install package (if not already installed)
pip install -e .

# Extract data for specific genes
agentic-spliceai-prepare --genes BRCA1 TP53 --output data/prepared/

# Extract data for chromosome
agentic-spliceai-prepare --chromosomes 21 --output data/prepared/
```

---

## Overview

The `agentic-spliceai-prepare` command extracts and prepares three types of genomic data:

1. **Gene annotations** - Gene metadata from GTF files
2. **Sequences** - DNA sequences from FASTA files
3. **Splice sites** - Donor and acceptor splice site positions

**Output formats**: TSV (default), Parquet, or both

---

## Basic Usage

### Extract Genes and Sequences

```bash
# Single gene
agentic-spliceai-prepare --genes BRCA1 --output data/prepared/

# Multiple genes
agentic-spliceai-prepare --genes BRCA1 TP53 UNC13A --output data/prepared/

# Entire chromosome
agentic-spliceai-prepare --chromosomes 21 --output data/prepared/

# Multiple chromosomes
agentic-spliceai-prepare --chromosomes 1 2 3 --output data/prepared/
```

### Output Files

By default, creates:
- `genes.tsv` - Gene annotations
- `sequences.tsv` - Gene sequences
- `splice_sites_enhanced.tsv` - Splice site annotations
- `preparation_summary.json` - Extraction summary

---

## Advanced Usage

### Extract Only Splice Sites

Useful for full genome splice site extraction:

```bash
agentic-spliceai-prepare --build GRCh38 \
  --output data/ensembl/GRCh38/ \
  --splice-sites-only
```

**Why?** Splice sites are expensive to extract (10-30 minutes for full genome), but once extracted, they can be reused indefinitely.

### Skip Sequence Extraction

Faster if you only need annotations:

```bash
agentic-spliceai-prepare --genes BRCA1 TP53 \
  --output data/prepared/ \
  --skip-sequences
```

### Force Re-extraction

Override cached files:

```bash
agentic-spliceai-prepare --genes BRCA1 \
  --output data/prepared/ \
  --force
```

### Parquet Output

For efficient storage and loading:

```bash
agentic-spliceai-prepare --genes BRCA1 TP53 \
  --output data/prepared/ \
  --format parquet
```

Or both formats:

```bash
agentic-spliceai-prepare --genes BRCA1 TP53 \
  --output data/prepared/ \
  --format both
```

### Custom GTF/FASTA Files

```bash
agentic-spliceai-prepare --genes BRCA1 \
  --gtf /path/to/custom/annotations.gtf \
  --fasta /path/to/custom/genome.fa \
  --output data/custom/
```

---

## Options Reference

### Target Selection

```
--genes GENE [GENE ...]       Gene symbols or IDs (e.g., BRCA1 TP53)
--chromosomes CHR [CHR ...]   Chromosomes (e.g., 21 22 X Y)
```

**Note**: Provide either `--genes` or `--chromosomes`, not both.

### Build and Source

```
--build BUILD                 Genome build (default: GRCh38)
                              Options: GRCh38, GRCh37, GRCh38_MANE

--annotation-source SOURCE    Annotation source (default: mane)
                              Options: mane, ensembl, gencode
```

### Custom Paths

```
--gtf PATH                    Custom GTF file (overrides build/source)
--fasta PATH                  Custom FASTA file (overrides build/source)
```

### Output Options

```
--output DIR, -o DIR          Output directory (required)
--force                       Force re-extraction even if files exist
--format FORMAT               Output format: tsv, parquet, both (default: tsv)
```

### Content Selection

```
--splice-sites-only           Extract only splice sites (skip genes/sequences)
--skip-sequences              Skip sequence extraction (annotations only)
--skip-splice-sites           Skip splice site extraction
```

### Verbosity

```
--verbosity {0,1,2}           Output verbosity (default: 1)
                              0: minimal, 1: normal, 2: detailed
```

---

## Output Formats

### genes.tsv

Gene annotations with columns:
- `seqname` - Chromosome (e.g., 'chr17')
- `gene_id` - Gene ID (e.g., 'ENSG00000012048')
- `gene_name` - Gene symbol (e.g., 'BRCA1')
- `start` - Start position (1-based)
- `end` - End position (1-based)
- `strand` - Strand ('+' or '-')
- `gene_biotype` - Gene type (e.g., 'protein_coding')

### sequences.tsv

Gene sequences with columns:
- `gene_id` - Gene identifier
- `gene_name` - Gene symbol
- `seqname` - Chromosome
- `start` - Start position
- `end` - End position
- `strand` - Strand
- `sequence` - DNA sequence (uppercase)

### splice_sites_enhanced.tsv

Splice site annotations with columns:
- `chrom` - Chromosome name
- `start` - Start position (BED interval)
- `end` - End position (BED interval)
- `position` - Exact splice site position (1-based)
- `strand` - Strand ('+' or '-')
- `site_type` - 'donor' or 'acceptor'
- `gene_id` - Gene identifier
- `transcript_id` - Transcript identifier
- `gene_name` - Gene symbol
- `gene_biotype` - Gene biotype
- `transcript_biotype` - Transcript biotype
- `exon_id` - Exon identifier
- `exon_number` - Exon number
- `exon_rank` - Exon rank

### preparation_summary.json

Summary of extraction with:
- Timestamp
- Input parameters (build, genes, chromosomes)
- Output file paths
- Statistics (gene counts, splice site counts)

---

## Common Workflows

### 1. Quick Gene Exploration

Extract data for a few genes:

```bash
agentic-spliceai-prepare --genes BRCA1 TP53 --output /tmp/quick_explore/
```

### 2. Prepare Training Data

Extract splice sites for specific genes:

```bash
agentic-spliceai-prepare --genes BRCA1 TP53 UNC13A \
  --output data/training/ \
  --format parquet
```

### 3. Full Genome Splice Sites

Extract and cache all splice sites once:

```bash
agentic-spliceai-prepare --build GRCh38 \
  --output data/ensembl/GRCh38/ \
  --splice-sites-only
```

This creates `data/ensembl/GRCh38/splice_sites_enhanced.tsv` which can be reused by all subsequent operations.

### 4. Chromosome-Level Processing

Process one chromosome at a time:

```bash
for chr in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 X Y; do
  agentic-spliceai-prepare --chromosomes $chr \
    --output data/chromosomes/chr${chr}/ \
    --format parquet
done
```

---

## Performance Tips

### Caching

- **Splice sites** are cached automatically. If you run the command twice with the same output directory, it will reuse the existing `splice_sites_enhanced.tsv` file.
- Use `--force` to override caching.

### Storage

- **TSV format**: Human-readable, larger file size
- **Parquet format**: Binary, smaller size, faster loading
- For large datasets, use Parquet

### Memory

- Splice site extraction for full genome requires ~2-4 GB RAM
- Chromosome-level extraction requires ~500 MB - 1 GB RAM
- Gene-level extraction requires minimal RAM

---

## Troubleshooting

### Issue: GTF file not found

**Solution**: Specify custom path with `--gtf`:

```bash
agentic-spliceai-prepare --genes BRCA1 \
  --gtf /path/to/annotations.gtf \
  --fasta /path/to/genome.fa \
  --output data/prepared/
```

### Issue: Slow extraction

**Solution**: Extract splice sites once for the entire build, then reuse:

```bash
# One-time: Extract all splice sites (~10-30 minutes)
agentic-spliceai-prepare --build GRCh38 \
  --output data/ensembl/GRCh38/ \
  --splice-sites-only

# Fast: Extract genes (reuses cached splice sites)
agentic-spliceai-prepare --genes BRCA1 \
  --output data/prepared/
```

### Issue: Out of memory

**Solution**: Process one chromosome at a time:

```bash
agentic-spliceai-prepare --chromosomes 21 \
  --output data/chr21/
```

---

## Integration with Python API

The CLI wraps the Python API. You can also use directly:

```python
from agentic_spliceai.splice_engine.base_layer.data import (
    prepare_gene_data,
    prepare_splice_site_annotations
)

# Extract genes and sequences
gene_df = prepare_gene_data(genes=['BRCA1', 'TP53'])

# Extract splice sites
result = prepare_splice_site_annotations(
    output_dir='data/prepared',
    genes=['BRCA1', 'TP53']
)
splice_df = result['splice_sites_df']
```

See the [Python API documentation](../src/agentic_spliceai/splice_engine/base_layer/data/README.md) for more details.

---

## See Also

- [Phase 2 Completion Report](../../dev/base_layer/PHASE2_COMPLETE.md) - Technical details
- [Base Layer Architecture](PROCESSING_ARCHITECTURE.md) - System architecture
- [Python API](../../src/agentic_spliceai/splice_engine/base_layer/data/README.md) - Python interface
