# Reference Genome Foundations

## The Setup

You download a reference genome for your splice site prediction pipeline. Two files land in your directory: a FASTA and a GTF. You uncompress them, load the sequences, extract annotations. Then you pause.

The FASTA has 3.2 billion nucleotides for human. The GTF has 60,000 genes and 200,000+ transcripts. You notice something: the same genomic coordinate can appear in multiple GTF files from different sources. You check Ensembl. You check GENCODE. You check MANE. Same sequences. Different annotations. What's happening?

This document walks you through that tension and why it matters for AgenticSpliceAI.

## Two Layers: Sequence and Meaning

Think of a reference genome as two independent layers stacked on top of each other.

The FASTA is the coordinate system. It's a satellite image: 3.2 billion nucleotides from chromosome 1 to 22, plus mitochondrial DNA, arranged in a fixed order. Every position has an address: chr1:1000000. The sequence at that address is constant. This is the ground truth for the cell's DNA as we understand it.

The GTF (or GFF) is the map. It assigns meaning to those coordinates. "Nucleotides 1000000 to 1005000 on chromosome 1 are exon 2 of gene BRCA2." "1005001 to 1005200 are the intron between exons 2 and 3." The GTF doesn't contain sequences. It contains annotations: boundaries, strand direction, transcript IDs, gene names. The map is built by inference from experimental data (RNA-seq, proteomics, conservation) processed through algorithms, filtered by confidence thresholds, curated by human judgment.

The key insight: many FASTAs exist. Many GTFs exist. A FASTA defines the coordinate system. A GTF assigns meaning. Different GTFs can label the same coordinates differently, and they're all built on top of the same FASTA.

This is why you can have three different GTFs all pointing to the same GRCh38 FASTA and all be correct about different things simultaneously.

## How Many FASTAs Are There?

Very few. For humans, three main ones.

**GRCh37 (hg19).** Released 2009. Assembled from sequencing data collected over years. Contains gaps (N-runs) in repetitive regions like centromeres and some telomeres. Most human genetic studies from 2009–2013 were mapped to GRCh37. Position chr1:1000000 in GRCh37 is not necessarily the same biological location as chr1:1000000 in GRCh38. Conversion requires liftover — a computational alignment from one coordinate system to another, which has failure modes.

**GRCh38 (hg38).** Released 2013. Same basic strategy as GRCh37 — assembled from clone-by-clone sequencing plus high-throughput reads — but with better closure of gaps and more accurate structural resolution. It's the production standard. Most published pipelines, databases, and clinical tools use GRCh38 coordinates. When people say "reference genome" without qualification, they mean GRCh38.

**T2T-CHM13 (Telomere-to-Telomere, Complete Human Reference).** Released 2022. A gapless assembly of a single haploid genome using long-read (PacBio HiFi) and nanopore sequencing. Fills ~200 megabases of sequence that GRCh37 and GRCh38 left as gaps: centromeres, pericentromeric repeats, short arms of acrocentric chromosomes. Introduces ~1956 new genes, of which ~100 are predicted coding. T2T-CHM13 is not yet a production standard — too new, too few pipelines support it — but it's the most complete linear reference we have.

Each FASTA is a different coordinate system. You cannot mix coordinates between them without explicit conversion, and conversion is imperfect (some regions resist alignment). If you train a model on GRCh37 coordinates and apply it to GRCh38 data without liftover, you will make systematic errors in certain regions.

## Within a Genome Build: One FASTA, Many Annotations

Now fix the FASTA. Use GRCh38. Everything downstream is built on GRCh38 coordinates.

You can download annotations from multiple providers:

- **MANE (Matched Annotation from NCBI and EBI).** A curated subset focusing on protein-coding genes with strong evidence (GENCODE annotation + Ensembl annotation + RefSeq consensus). ~20,000 genes. High confidence, narrow scope. Designed for clinical and reference applications.

- **GENCODE.** A comprehensive manual annotation based on RNA-seq evidence and computational prediction. ~60,000 genes, including long non-coding RNAs, small RNAs, pseudogenes. Many transcripts per gene (some genes have 100+ isoforms). Different versions (v39, v40, v41, v47, etc.) reflect re-evaluation of the RNA-seq evidence, algorithm refinements, and filtering updates. Maintained by the ENCODE consortium.

- **Ensembl.** Another comprehensive annotation, built from similar sources but with different RNA-seq datasets, different algorithms for transcript assembly, different confidence filters. ~65,000 genes. Often aligned to GENCODE for comparison but not identical.

- **RefSeq (NCBI).** A conservative reference collection, historically curated by the National Center for Biotechnology Information. Smaller than GENCODE or Ensembl. Strongly used in clinical settings.

All of these point to the same GRCh38 FASTA. All of them assign different boundaries to the same genes. Why?

Because gene structure is inferred, not directly observed. An RNA-seq library gives you millions of sequenced fragments. A computational pipeline assembles those fragments into contigs. Another algorithm merges contigs into transcript models. Then filters: does the transcript have evidence in multiple samples? Does the boundary match conservation across species? Is the predicted protein in-frame and without premature stop codons? Different providers use different RNA-seq datasets (different tissues, different conditions, different cohorts), different assembly algorithms, and different thresholds. The result is three versions of "truth" for the same genome, all defensible.

This matters for AgenticSpliceAI because your base model (SpliceAI or OpenSpliceAI) operates on raw sequence. But the meta-model layers in annotation-dependent features. If you train on GENCODE v47 and deploy on MANE, you're working with different isoform sets, different exon boundaries, different definitions of what's a "canonical" isoform. This is annotation uncertainty. It's real noise in the training signal.

## Two Independent Axes of Versioning

Confusion often arises here, so let's be explicit.

**Axis 1: Assembly version.** GRCh37 → GRCh38 → T2T-CHM13. Different FASTA files. Coordinates don't transfer. This is rare. Assemblies are expensive and take 10+ years to become mature. GRCh38 will remain the standard for years more.

**Axis 2: Annotation version.** GENCODE v39, v40, v41, v47. MANE Release 1.0, Release 1.1. Ensembl release 109, 110, 111. Same FASTA. Different GTF. This is common. Annotations update annually or more often as RNA-seq datasets accumulate and algorithms improve. You can swap annotations without liftover — coordinates stay the same, only the interpretation changes.

The two axes are independent. A GRCh38 FASTA from 2013 works perfectly with GENCODE v47 from 2024, even though GRCh38 is a decade old, because the FASTA doesn't change, only the annotations on top of it do.

## Practical Distinctions Within a FASTA

When you download a GRCh38 FASTA from Ensembl or NCBI, you might see several variants:

**primary_assembly.fa.** Contains only the primary chromosomal DNA: chromosomes 1–22, X, Y, and mitochondrial DNA. No unplaced scaffolds, no patches for local assembly corrections, no haplotype alternatives. This is the standard for most analyses. Use this unless you have a specific reason not to.

**toplevel.fa.** The complete sequence: primary assembly plus unplaced/unlocalized scaffolds, patches, alternate loci. Much larger. Only needed if your pipeline handles scaffolds explicitly or if you're doing comprehensive assembly validation.

**analysis_set.fa.** A reduced set optimized for certain analyses. Sometimes excludes very repetitive regions (like pericentromeric sequences) to reduce noise in read mapping. Check the data provider's documentation — the definition varies.

Most RNA-seq pipelines and reference databases use primary_assembly. GRCh38 primary_assembly is 3.05 gigabases. GRCh37 was also defined this way, so they're roughly comparable despite different coordinates.

There's also the question of chromosome naming:

- Ensembl: `1`, `2`, ... `MT` (no "chr" prefix)
- NCBI/UCSC: `chr1`, `chr2`, ... `chrM` (with "chr" prefix)

Your pipeline needs to handle both or normalize to one. This is a common source of silent errors — coordinates look right but the coordinate system doesn't match.

And sequence masking:

- **Soft-masked:** Lowercase letters (a, c, g, t) for repetitive regions identified by algorithms like RepeatMasker. Uppercase (A, C, G, T) for unique sequence. You can use soft-masked sequences directly; lowercase and uppercase have the same biological meaning in DNA. Some tools ignore masking; some treat it as informative.

- **Hard-masked:** Ns for repetitive regions. Loses the actual sequence; you can't recover it from the hard-masked FASTA alone. Use this only if your pipeline can't handle ambiguity.

Most downloads default to soft-masked.

## Connecting to AgenticSpliceAI

Your base SpliceAI model operates on 10,000-bp windows of raw sequence. It learned patterns in splice sites from training data. The model is agnostic to annotation.

Your meta-model, if you're building one (for AgenticSpliceAI), depends on annotation features: isoform prevalence, transcript boundaries, exon-intron structure. These come from GTF. The same 10,000-bp window, annotated differently by GENCODE versus MANE, becomes a different training example.

Here's why that's interesting: you can fix the FASTA, vary the GTF, and measure the impact on model performance. What happens to your false positive rate when you swap GENCODE v47 for MANE? Does the meta-model overfit to GENCODE's specific isoform set? How much of your FP/FN problem comes from annotation uncertainty versus genuine model error?

This is directly relevant to the FP/FN correction work in the meta-model. Some of your false positives might not be false at all — they might be real splice sites that GENCODE doesn't annotate because it was trained on a different RNA-seq cohort or filtered more aggressively. Swapping annotations is a way to surface that.

## The Graph Future and Current Practice

The genomics field is slowly moving toward graph genomes: data structures where a node is a sequence segment and an edge is a variant path. Instead of "the reference is a linear string," the model is "the reference is a graph where each path from start to finish is a valid haplotype." The Pan Genome project has 47 phased, diploid assemblies (from 47 individuals across populations), annotated with GENCODE and Ensembl. A true reference would be a graph encoding all 47.

This is not here yet. For production work — AgenticSpliceAI included — the linear reference (FASTA + GTF) remains standard. But it's worth knowing the direction the field is moving.

## Reproducibility: Be Specific

Always specify your reference fully. Don't say "GRCh38." Say "GRCh38, primary_assembly.fa, GENCODE v47, Ensembl naming convention, soft-masked."

Example in methods:

> Genomic coordinates are aligned to GRCh38 primary_assembly release 94 (NCBI). Gene annotations are from GENCODE release 47 (comprehensive annotation, including lncRNA). Isoform prevalence is derived from GENCODE canonical transcripts only.

Or:

> SpliceAI base model predictions are made against GRCh38 primary_assembly. Meta-model features are extracted from MANE Release 1.1 to ensure annotation-minimal comparison.

This specificity matters because your colleague running the same code on a different annotation version will get different results, and both will be reproducible. The difference is real; it's not noise.

## Summary Points

1. A reference genome is two layers: FASTA (coordinate system, very few exist) and GTF/GFF (annotations, many versions per FASTA).

2. For humans: GRCh37 and GRCh38 are different coordinate systems (no direct position transfer). T2T-CHM13 is a newer, gapless assembly.

3. Within GRCh38: MANE, GENCODE, Ensembl, and RefSeq all label the same coordinates differently. This is not an error. It's annotation uncertainty stemming from different RNA-seq evidence and filtering.

4. Annotation versions (GENCODE v39 → v47) are independent of assembly versions (GRCh37 → GRCh38). You can update one without the other.

5. For AgenticSpliceAI, this matters because your meta-model learns on annotated data. Swapping annotations reveals how much of your model's uncertainty comes from the annotation layer versus the underlying sequence prediction.

6. When publishing or deploying, specify genome build, FASTA variant (primary_assembly, toplevel, etc.), annotation source, and annotation version. This enables reproducibility and helps others understand the sources of your results.
