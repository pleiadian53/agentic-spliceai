# Ground Truth Generation for Custom Genome Builds

## Overview

`04_generate_ground_truth.py` extracts splice site annotations from any GTF
file and produces `splice_sites_enhanced.tsv` — the standard label format
used across the entire pipeline (base layer evaluation, meta-layer training,
FASTA inference validation).

The script works with any genome build that has a GTF annotation file with
exon features.  It is not limited to GRCh38 or human genomes.

---

## Standard Usage (Registry-Based)

For genome builds registered in the resource registry (MANE, Ensembl):

```bash
# MANE / GRCh38 — curated canonical transcripts (~370K sites, ~19K genes)
python 04_generate_ground_truth.py --output data/mane/GRCh38/

# Ensembl / GRCh38 — all transcripts (~2.8M sites, ~63K genes)
python 04_generate_ground_truth.py --output data/ensembl/GRCh38/

# The annotation source and build are inferred from the output path
```

## Custom GTF Usage (Any Genome Build)

For genome builds not in the registry, provide the GTF directly with `--gtf`:

### T2T-CHM13

The Telomere-to-Telomere CHM13 reference resolved centromeric regions,
segmental duplications, and acrocentric chromosomes missing from GRCh38.
NCBI provides RefSeq Curated annotations for CHM13v2.0.

```bash
# Download T2T-CHM13 annotation (RefSeq Curated)
# Source: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_009914755.1/
wget https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/009/914/755/GCF_009914755.1_T2T-CHM13v2.0/GCF_009914755.1_T2T-CHM13v2.0_genomic.gtf.gz
gunzip GCF_009914755.1_T2T-CHM13v2.0_genomic.gtf.gz

# Generate splice site ground truth
python 04_generate_ground_truth.py \
    --gtf GCF_009914755.1_T2T-CHM13v2.0_genomic.gtf \
    --output data/t2t_chm13/ \
    --build T2T-CHM13 \
    --annotation-source refseq
```

### Human Pangenome (HPRC)

Individual haplotype-resolved assemblies from the Human Pangenome Reference
Consortium.  Each assembly may have its own lifted-over or projected
gene annotations.

```bash
python 04_generate_ground_truth.py \
    --gtf /path/to/HG002_mat.gtf \
    --output data/pangenome/HG002_mat/ \
    --build pangenome \
    --annotation-source hprc
```

### Non-Human Organisms

Any organism with a GTF annotation works:

```bash
# Mouse (GRCm39, Ensembl)
python 04_generate_ground_truth.py \
    --gtf Mus_musculus.GRCm39.112.gtf \
    --output data/mouse/GRCm39/ \
    --build GRCm39 \
    --annotation-source ensembl

# Zebrafish
python 04_generate_ground_truth.py \
    --gtf Danio_rerio.GRCz11.112.gtf \
    --output data/zebrafish/GRCz11/ \
    --build GRCz11 \
    --annotation-source ensembl
```

---

## How It Works

The extraction pipeline is:

```
GTF file
  → parse exon features (gene_id, transcript_id, chrom, start, end, strand)
  → for each transcript: derive splice sites from exon boundaries
      - donor = exon end (5' splice site)
      - acceptor = exon start (3' splice site)
  → deduplicate by (chrom, position, splice_type, gene_id)
  → write splice_sites_enhanced.tsv
```

The core extraction function `extract_splice_sites()` makes no assumptions
about genome build, chromosome naming, or annotation provenance.  It only
requires that the GTF contains `exon` features with standard GTF attributes
(`gene_id`, `transcript_id`, `exon_number`).

### Output Format

The TSV has 14 columns matching the pipeline-wide standard:

| Column | Description | Example |
|--------|-------------|---------|
| chrom | Chromosome name (as in GTF) | chr1, 1, NC_060925.1 |
| start | Exon boundary start (BED-style) | 11873 |
| end | Exon boundary end | 11874 |
| position | Exact splice site position (1-based) | 11873 |
| strand | Strand | + or - |
| splice_type | Site type | donor or acceptor |
| gene_id | Gene identifier | ENSG00000223972 |
| transcript_id | Transcript identifier | ENST00000456328 |
| gene_name | Gene symbol | DDX11L1 |
| gene_biotype | Gene biotype | protein_coding |
| transcript_biotype | Transcript biotype | protein_coding |
| exon_id | Exon identifier | ENSE00002234944 |
| exon_number | Exon number in transcript | 2 |
| exon_rank | Exon rank | 2 |

### Chromosome Naming

Different genome builds use different naming conventions:

| Build | Convention | Examples |
|-------|-----------|----------|
| MANE (GRCh38) | chr prefix | chr1, chr2, chrX |
| Ensembl (GRCh38) | bare | 1, 2, X |
| T2T-CHM13 (RefSeq) | accession | NC_060925.1 |
| T2T-CHM13 (aliases) | chr prefix | chr1, chr2 |

The script preserves whatever naming the GTF uses.  Downstream tools that
compare across annotation sources (e.g., Eval-Ensembl-Alt site filtering) normalize
chromosome names at comparison time using `ensure_chrom_column()`.

---

## Downstream Usage

### Meta-Layer Training (M1-M3)

The `splice_sites_enhanced.tsv` serves as the label source for
`build_splice_labels()`, which converts per-gene positions into
`[L]` label arrays (0=donor, 1=acceptor, 2=neither):

```
07_train_sequence_model.py → build_gene_cache() → build_splice_labels()
                                                       ↑
                                          splice_sites_enhanced.tsv
```

### Eval-Ensembl-Alt (Ensembl \ MANE)

Two `splice_sites_enhanced.tsv` files (one MANE, one Ensembl) are loaded
simultaneously.  The set difference identifies alternative splice sites.
See `examples/meta_layer/docs/evaluation_hierarchy.md`.

### FASTA Inference Validation

When running FASTA inference on sequences from a custom build, generate
ground truth labels from that build's GTF, then compare predicted splice
positions against the known sites.  See
`examples/meta_layer/docs/cross_build_fasta_validation.md`.

---

## Related Scripts

- `02_prepare_splice_sites.py` — Older script; use `04_` instead
- `03_full_data_pipeline.py --skip-sequences` — Full pipeline variant
- `agentic-spliceai-prepare --splice-sites-only` — CLI equivalent
