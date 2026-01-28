# Splice Sites Enhanced Annotation Summary

**Source:** MANE (Matched Annotation from NCBI and EBI) - GRCh38  
**File:** `data/splice_sites_enhanced.tsv` (symlinked to `/Users/pleiadian53/work/meta-spliceai/data/mane/GRCh38/splice_sites_enhanced.tsv`)  
**Total Records:** 369,918 splice sites (plus 1 header row = 369,919 total lines)

## Overview
This file contains annotated splice site positions from MANE transcripts, which provide a single representative transcript per gene. The dataset is perfectly balanced between donor and acceptor sites and includes both standard chromosomes and alternative/fix patch sequences from GRCh38.

## Column Descriptions

### 1. **chrom** (Chromosome)
- **Type:** String
- **Format:** Standard: `chr1`, `chr2`, ..., `chr22`, `chrX`, `chrY`; Alternative/Fix: `chr*_*` (e.g., `chr1_KN196472v1_fix`)
- **Description:** Chromosome identifier where the splice site is located
- **Example:** `chr1`, `chr19_GL949746v1_alt`, `chr6_KV766194v1_fix`
- **Coverage:** 65 unique chromosomes/contigs:
  - 24 standard chromosomes (chr1-22, chrX, chrY)
  - 41 alternative contigs and fix patches
  - 368,578 sites (99.6%) on standard chromosomes
  - 1,340 sites (0.4%) on alternative/fix patches

### 2. **start** (Region Start)
- **Type:** Integer (1-based)
- **Description:** Start coordinate of a 5bp window around the splice site (position - 2)
- **Example:** `65432`

### 3. **end** (Region End)
- **Type:** Integer (1-based)
- **Description:** End coordinate of a 5bp window around the splice site (position + 2)
- **Example:** `65436`
- **Note:** This defines a window of [start, end) or [position-2, position+2]

### 4. **position** (Splice Site Position)
- **Type:** Integer (1-based)
- **Description:** The exact genomic coordinate of the splice site
- **Example:** `65434`
- **Note:** This is always `start + 2` (the center of the 5bp window)

### 5. **strand** (Genomic Strand)
- **Type:** Character
- **Values:** `+` (forward/plus strand) or `-` (reverse/minus strand)
- **Description:** The DNA strand on which the gene is transcribed
- **Distribution:**
  - 185,846 sites (50.2%) on + strand
  - 184,072 sites (49.8%) on - strand
- **Examples:** 
  - `+` for OR4F5 gene
  - `-` for NOC2L gene

### 6. **site_type** (Splice Site Type)
- **Type:** String
- **Values:** `donor` or `acceptor`
- **Description:** Type of splice site
  - **donor:** 5' splice site (GT dinucleotide at exon-intron boundary)
  - **acceptor:** 3' splice site (AG dinucleotide at intron-exon boundary)
- **Distribution:** 
  - 184,959 donor sites
  - 184,959 acceptor sites (perfectly balanced)

### 7. **gene_id** (Gene Identifier)
- **Type:** String
- **Format:** `gene-{SYMBOL}`
- **Description:** NCBI/MANE gene identifier with the gene symbol
- **Examples:** 
  - `gene-OR4F5`
  - `gene-SAMD11`
  - `gene-NOC2L`
- **Coverage:** 18,200 unique genes

### 8. **transcript_id** (Transcript Identifier)
- **Type:** String
- **Format:** `rna-{RefSeq_ID}.{version}`
- **Description:** RefSeq mRNA transcript identifier from MANE
- **Examples:**
  - `rna-NM_001005484.2`
  - `rna-NM_001385641.1`
  - `rna-NM_015658.4`
- **Coverage:** 18,264 unique transcripts
- **Note:** MANE provides one representative transcript per gene, though the slight excess of transcripts (18,264 vs 18,200 genes) suggests some genes may have multiple MANE transcripts

### 9. **gene_name** (Gene Symbol)
- **Type:** String
- **Description:** HGNC gene symbol
- **Examples:** `OR4F5`, `SAMD11`, `NOC2L`, `AGRN`

### 10. **gene_biotype** (Gene Biotype)
- **Type:** String (empty)
- **Status:** Column present but not populated in this dataset

### 11. **transcript_biotype** (Transcript Biotype)
- **Type:** String (empty)
- **Status:** Column present but not populated in this dataset

### 12. **exon_id** (Exon Identifier)
- **Type:** String (empty)
- **Status:** Column present but not populated in this dataset

### 13. **exon_number** (Exon Number)
- **Type:** String (empty)
- **Status:** Column present but not populated in this dataset

### 14. **exon_rank** (Exon Rank/Position)
- **Type:** Integer
- **Description:** The positional rank of the exon containing this splice site within the transcript
- **Range:** 1 to 363
- **Statistics:**
  - Average: 11.64 exons per transcript
  - Maximum: 363 (represents transcripts with very high exon counts)
- **Interpretation:**
  - For donor sites: indicates which exon ends at this position
  - For acceptor sites: indicates which exon begins at this position
  - Exon rank of 1 indicates the first exon in the transcript
- **Note:** This field is populated for all 369,918 records

## Key Characteristics

1. **MANE Dataset:** Uses curated, representative transcripts rather than all possible isoforms
2. **Perfectly Balanced Data:** Exactly 184,959 donor sites and 184,959 acceptor sites
3. **Window-based:** Each splice site includes a 5bp genomic window (±2bp from the splice site)
4. **Strand-aware:** Nearly balanced strand distribution (50.2% + strand, 49.8% - strand)
5. **RefSeq-based:** Uses RefSeq transcript identifiers with version numbers
6. **Comprehensive Genomic Coverage:** Includes 65 chromosomes/contigs:
   - All standard chromosomes (chr1-22, chrX, chrY)
   - GRCh38 alternative contigs and fix patches
   - 99.6% of sites are on standard chromosomes
7. **Exon Context:** Includes exon rank information (average 11.64 exons per transcript, max 363)

## Usage Notes

- The 5bp window (start-end) can be useful for extracting sequence context around splice sites
- The `position` column gives the exact splice site coordinate for precise lookups
- This represents a more focused dataset compared to Ensembl annotations (which include all transcript isoforms)
- Version numbers in transcript IDs indicate the RefSeq annotation version
- The `exon_rank` field provides positional context for each splice site within its transcript
- Several columns (gene_biotype, transcript_biotype, exon_id, exon_number) are present but unpopulated
- Alternative contigs and fix patches represent sequence corrections and variants from the reference assembly

## Data Quality

- **Completeness:** All core fields populated (chrom, positions, strand, site_type, gene/transcript IDs, gene_name, exon_rank)
- **Balance:** Perfect 50/50 split between donor and acceptor sites
- **Gene Coverage:** 18,200 genes with 18,264 transcripts (single representative per gene)
- **Exon Distribution:** Wide range from single-exon to 363-exon transcripts

---

## Verification Status

**✅ Verified:** Schema and statistics confirmed accurate (Nov 18, 2025)
- All column descriptions match actual data structure
- Row counts and distributions verified
- Used in `chart_agent/examples/analyze_splice_sites.py` with successful test runs
