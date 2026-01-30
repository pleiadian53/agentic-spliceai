# Data Preparation Notebooks

**Phase**: 2  
**Purpose**: Educational notebooks for genomic data extraction and preparation

---

## 📋 Available Notebooks

### Coming Soon

**01_gene_extraction/** - Gene & Sequence Extraction
- GTF file format explained
- FASTA file format explained
- Extracting gene annotations
- Extracting DNA sequences
- Working with Polars DataFrames

**02_splice_site_extraction/** - Splice Site Annotation
- Understanding exon boundaries
- Deriving splice sites from GTF
- MANE vs Ensembl differences
- Metadata inference for MANE
- Quality validation

---

## 🎯 Learning Objectives

### After completing these notebooks, you will understand:

1. **Genomic File Formats**:
   - GTF (Gene Transfer Format) structure
   - FASTA sequence format
   - Chromosome naming conventions
   - Gene/transcript/exon relationships

2. **Data Extraction**:
   - How to parse GTF files
   - How to extract sequences from FASTA
   - How to filter by genes/chromosomes
   - How to handle missing data

3. **Splice Site Annotation**:
   - How splice sites relate to exons
   - Donor vs acceptor site positions
   - Metadata fields and their meaning
   - MANE-specific considerations

4. **Quality Validation**:
   - How to validate extracted data
   - Comparing MANE vs Ensembl
   - Checking metadata completeness
   - Identifying potential issues

---

## 🔗 See Also

- **Examples**: `../../examples/data_preparation/` - Quick driver scripts
- **CLI**: `agentic-spliceai-base-prepare` - Production tool
- **Tests**: `../../tests/integration/base_layer/` - Integration tests
- **Docs**: `../../docs/base_layer/` - Technical documentation

---

**Last Updated**: January 30, 2026  
**Status**: Placeholder - notebooks coming soon
