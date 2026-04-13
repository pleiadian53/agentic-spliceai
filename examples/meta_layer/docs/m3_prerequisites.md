# M3 Prerequisites — Is Novel Splice-Site Discovery Worth Building?

A tutorial and research-workflow document for the decision to invest
in **M3** — the meta-layer variant targeted at discovering splice
sites that are **not in any annotation** (beyond Ensembl, beyond
GENCODE). M3 is architecturally similar to M1-S and M2-S but trained
with junction evidence as a *label* rather than a feature, on the
premise that junctions *themselves* can identify novel splice sites
the annotation has missed.

This document walks through:

1. What the audit found (experimental results)
2. Whether junction support is a reliable novelty signal
3. What other data sources exist and how to combine them
4. How to make novel sites reliable enough for clinical/drug-target use
5. A go/no-go recommendation for M3

Everything here is grounded in the audit outputs in
`output/meta_layer/junction_coverage_audit*`. Companion document
[junction_coverage_findings.md](junction_coverage_findings.md)
documents the strand asymmetry and biotype composition effects that
surfaced during the investigation.

---

## 1. Experimental Results

### 1.1 Setup

- **Script:** `examples/meta_layer/11_junction_coverage_audit.py`
- **Notebook (tutorial version):** `notebooks/meta_layer/junction_coverage_audit.ipynb`
- **Data inputs:**
  - `data/GRCh38/junction_data/junctions_gtex_v8.parquet` (353K
    GTEx v8 junctions, 54 tissues aggregated)
  - `data/ensembl/GRCh38/splice_sites_enhanced.tsv` (~2.83M sites)
  - `data/mane/GRCh38/splice_sites_enhanced.tsv` (~370K canonical)
- **Scope:** SpliceAI test chromosomes (chr1, 3, 5, 7, 9) for the
  notebook version; full genome for the script default.

The audit answers four questions:

| # | Question |
|---|---|
| Q1 | Coverage of annotated sites (chr1/3/5/7/9 + genome-wide) |
| Q2 | Canonical (MANE) vs alt-only breakdown |
| Q3 | Junction sides outside any Ensembl site (M3 ceiling) |
| Q4 | Biotype-filtered coverage (strip speculative transcripts) |

### 1.2 Headline numbers (genome-wide)

| Metric | Value |
|---|---:|
| Overall annotated-site coverage | **68.3%** |
| MANE canonical coverage | 94.6% |
| Alt-only (Ensembl \\ MANE) coverage | 42.3% |
| Canonical − alt-only gap | **52.3 pts** |
| Alt-only after biotype filter | 45.0% |
| Junction sides outside Ensembl (novel candidates) | **67,490** (9.6%) |
| Median depth at novel candidates | 4,000–7,000 reads |
| Median tissue breadth at novel candidates | 51 / 54 |

### 1.3 Key takeaways

**Takeaway 1: Junction evidence is strong for canonical biology,
thin for alternatives.** 95% of canonical (MANE) sites are junction-
supported; only 42–45% of alt-only sites are. Bulk GTEx RNA-seq
captures the "main" isoform of each gene well and doesn't catch most
annotated alternatives.

**Takeaway 2: Test-chromosome and genome-wide numbers are nearly
identical.** Within 1 pt across every metric (68.3% vs 69.0% overall,
94.6% vs 94.3% for MANE, etc). This validates the SpliceAI split as
representative for junction analysis and means M2-S/M3 PR-AUCs
computed on chr1/3/5/7/9 generalize to the full genome.

**Takeaway 3: The novel-junction pool is not garbage.** The 67,490
junction sides not in any Ensembl splice site have median depth
4K–7K reads and are expressed in 51/54 tissues. These are not
tissue-specific marginal hits. That's genuinely promising raw
material for novel discovery — but comes with caveats (see §2).

**Takeaway 4: Part of the alt-only gap is biotype-driven.** Filtering
retained_intron, pseudogene, NMD, and protein_coding_CDS_not_defined
sites lifts alt-only coverage by ~3 points. Modest, which means
most of the gap is *real* thin junction support for annotated
alternatives, not annotation artifacts. See
[junction_coverage_findings.md](junction_coverage_findings.md).

---

## 2. Is Junction Support a Reliable Novelty Signal?

The intuition for M3 is: "GTEx sees a junction at a position the
annotation doesn't list → novel splice site." The audit makes this
more complicated.

### 2.1 Reasons for optimism

- **67,490 novel junction sides is not noise.** Depth and breadth are
  comparable to the covered-alt subset.
- **Annotations are known to miss real splice sites.** Ensembl and
  GENCODE are curated conservatively; cell-type-specific and
  developmentally-regulated isoforms are routinely underrepresented.
  Tissue atlases like GTEx, recount3, and ENCODE long-read routinely
  discover novel junctions on well-annotated genes.
- **Tissue breadth as a filter.** The 51/54 median tissue support at
  novel candidates suggests these aren't spurious alignments — 
  spurious junctions rarely appear in many tissues independently.

### 2.2 Reasons for caution

- **Alignment artifacts.** STAR junction calls can arise from
  misalignment (e.g., repetitive regions, paralogs with high
  similarity). Even high-coverage junctions can be alignment
  artifacts that appear consistently across samples because all
  samples share the same aligner failure mode.
- **Genuine but non-functional junctions.** Noisy splicing is a
  documented phenomenon — the spliceosome sometimes uses cryptic
  sites at low rates; these produce real junction reads but don't
  produce functional isoforms. "Novel" by our definition ≠ "real
  alternative transcript."
- **Incomplete annotation, not novel.** A site absent from Ensembl
  may still be in another annotation (RefSeq, UCSC Known Genes,
  NCBI's Gene predictions), or in a recent Ensembl release the
  TSV hasn't been rebuilt from. Some of the 67K "novel" may simply
  reflect annotation lag.
- **Strand/coordinate bugs in aggregation.** The audit surfaced
  strand asymmetry that turned out to be biotype-driven, not a
  pipeline bug — but similar issues could lurk in how "novel" is
  counted. Every novel-junction count should be independently
  validated against at least one external annotation.

### 2.3 What "novel" needs to mean for M3 to be useful

For the 67K candidates to be worth training a model on as positive
labels, we need:

1. **Independent confirmation** that the coordinate isn't annotated
   in any major database (Ensembl + GENCODE + RefSeq + UCSC at
   minimum).
2. **Evidence of splicing regulation, not just alignment signal** —
   canonical GT/AG dinucleotides at the boundaries, evolutionary
   conservation of the junction, compatible reading frame (for
   coding regions).
3. **Tissue specificity or developmental pattern** — a novel
   junction that's only expressed in one condition is more biologically
   meaningful than one that appears ubiquitously at low levels.

Without these filters, M3 trained on raw novel junctions is likely
to learn to recognize "things that look like splice sites that were
accidentally missed from annotation" — useful, but a narrower scope
than what the name implies.

---

## 3. Alternative and Complementary Data Sources

GTEx v8 is one tissue atlas among many. To confidently call a novel
site, we'd want convergent evidence from multiple sources. Below is
a practical map of complementary datasets that extend GTEx's reach.

### 3.1 Broader RNA-seq coverage (more tissues, more conditions)

- **recount3** (recount.bio, 750K+ RNA-seq samples) aggregates GTEx,
  TCGA, SRA, and more. Significantly broader tissue and disease
  coverage than GTEx alone. Junctions are available in a uniform
  aggregated format. Good for **tissue-specific alt isoforms** and
  **disease-context-specific splicing**.
- **FANTOM6/CAGE-seq** complements RNA-seq by tagging transcription
  start sites; helps disambiguate real alternative promoters from
  alternative splicing.
- **Developmental / single-cell atlases** (e.g., BrainSpan, Tabula
  Muris, Tabula Sapiens). Developmental stages and rare cell types
  are grossly underrepresented in GTEx bulk samples. Many
  "novel-by-GTEx" junctions likely exist here.

### 3.2 Long-read sequencing (full-transcript resolution)

- **ENCODE4 long-read RNA-seq** (PacBio Iso-Seq, Oxford Nanopore).
  Reads span full transcripts, so entire isoform structures are
  observed directly — no assembly-from-junctions ambiguity. Critical
  for confirming that a novel junction participates in a coherent
  full transcript, not just fragmentary splicing.
- **GTEx v9+** (forthcoming) is expected to include long-read data
  from a subset of GTEx samples.

### 3.3 Disease and cancer RNA-seq

- **TCGA / ICGC** (tumor RNA-seq across ~30 cancer types). Cancer
  splicing is often aberrant, and the *difference* set (novel-in-
  tumor, not-in-normal) defines cancer-induced alternative splicing
  — a specific subclass of "novel" with real drug-target relevance.
  See [project_m4_cancer_induced_splicing.md] for the M4 planned scope.
- **GTEx × disease atlases** for splicing QTLs and tissue-specific
  aberrant splicing.

### 3.4 Functional/proteomic evidence

- **Proteogenomics (MS/MS proteomics)** — does the inferred novel
  isoform produce a detectable peptide? The strongest confirmation
  of translation. Datasets like CPTAC and the Human Protein Atlas
  include matched MS.
- **Ribosome profiling (Ribo-seq)** — does the novel junction
  appear in ribosome-protected fragments? Indicates functional
  translation.

### 3.5 Evolutionary conservation

- Sites evolutionarily conserved across mammals are more likely
  functional. The conservation modality is already in M2-S v2;
  using a stricter conservation threshold on novel-junction
  candidates is a cheap filter.

### 3.6 Deep-learning priors

- Foundation models (Evo2, Nucleotide Transformer) fine-tuned on
  splice site prediction can provide an independent sequence-based
  prior for whether a position "looks like" a functional splice
  site, regardless of whether it's annotated. Combining these with
  junction evidence reduces false positives from alignment
  artifacts.

### 3.7 Combining sources — practical stacking

For a practically useful novel-site call:

1. **Junction evidence across multiple cohorts** (GTEx + recount3 +
   disease-specific atlas): coverage in ≥ 2 independent cohorts.
2. **Coordinate consistency with long-read transcript catalogs**
   (ENCODE long-read, FLAIR assemblies).
3. **Sequence plausibility** via foundation model confidence above
   a threshold.
4. **Evolutionary conservation** (phyloP / phastCons elevated at
   boundary).
5. **Functional evidence** (peptide match for protein-coding
   candidates, Ribo-seq signal).

A site that passes 3–4 of these filters is a candidate worth
further investigation. A site that passes 5 is a strong candidate
for clinical follow-up.

---

## 4. Reliability for Drug Targets

Drug-target applications of novel splice sites impose much stricter
requirements than discovery alone. Short list of criteria:

### 4.1 Must haves

- **Reproducibility** across independent cohorts (not just GTEx).
- **Tissue specificity** matches the target disease's tissue.
- **Frame-preserving** for coding regions (most therapeutic
  interventions target coding proteins).
- **Differential between disease and healthy state** — if the
  isoform is present in both, targeting it may have off-target
  effects.

### 4.2 Desirable

- **Proteomic confirmation** — a peptide unique to the novel
  isoform is detected by MS/MS. Without this, the isoform may
  not be translated.
- **Structural rationale** — the novel isoform produces a protein
  whose structure supports a drug-binding pocket, allosteric
  site, etc. Computational structure prediction (AlphaFold) is
  cheap here.
- **Human genetic evidence** — variants near the novel splice
  site are associated with the target phenotype (e.g., via GWAS
  or rare-variant collapsing tests).

### 4.3 Caveats specific to GTEx-surfaced candidates

- **GTEx is normal tissue.** Disease-induced splicing is by
  definition not in GTEx. Cancer-induced, autoimmune-induced,
  and neurodegeneration-associated splicing need disease cohorts.
- **Healthy-tissue splicing as a drug target is rare.** Most
  splice-modulating therapeutics (antisense oligonucleotides
  targeting SMN2 in SMA, risdiplam) target *dysregulation*, not
  healthy alternative splicing.

### 4.4 A practical pipeline

For drug-target-grade novel site calls:

1. M3 surfaces candidate novel splice sites from GTEx + recount3.
2. Intersect with disease-specific cohort (TCGA for cancer, MSBB
   for Alzheimer's, etc.) to find disease-context-specific novel
   sites.
3. Filter by frame preservation + conservation.
4. Confirm via long-read and/or proteomics.
5. Target validation in cell lines and patient-derived systems.

M3 is step 1 of many. Its role is candidate generation, not
drug-target identification.

---

## 5. Is M3 Worth Developing?

### 5.1 The conservative case (maybe not yet)

- GTEx alone can't confirm over half of *annotated* alt isoforms.
  A model trained to surface sites beyond annotations is working
  in thinner territory.
- The 67,490 novel candidates are a pool, but without independent
  confirmation (§2.3) they're a mix of real novelty, annotation
  lag, and alignment artifacts.
- M2-S v2 already achieves PR-AUC 0.967 on alt sites. The
  marginal return on adding an M3 head (which would compete for
  capacity with M2-S) may be small.
- Development cost: training pipeline, label generation, evaluation
  strategy, agentic validation for biological sanity. Non-trivial.

### 5.2 The bullish case (yes, but scoped)

- Novel splice site discovery is scientifically important and
  directly feeds Phase 8 (isoform discovery) — the ultimate
  project goal.
- Junction evidence from GTEx is free, immediate, and already
  wired into the feature pipeline. Training M3 with junctions as
  labels is a small incremental lift over the existing M1-S/M2-S
  infrastructure.
- Combining M3's predictions with the stacking sources in §3.7
  could produce a genuinely high-precision novel-site catalog.
- Long-read data (ENCODE4, GTEx v9) is maturing and will soon
  make end-to-end transcript validation practical.

### 5.3 Recommendation

**Proceed with M3, but scope it as a candidate-generation model,
not a truth-finder.** Specifically:

1. **Train M3 on GTEx-novel sites that pass a minimum filter**:
   depth ≥ 100 reads, ≥ 5 tissues, not in any of Ensembl /
   GENCODE / RefSeq, boundary dinucleotides match GT/AG.
2. **Evaluate M3 on held-out long-read confirmations** (ENCODE4)
   as the positive-truth set. Don't evaluate on "sites not in
   Ensembl" alone — that's circular.
3. **Gate M3 outputs behind the agentic layer**: literature
   validation, structural plausibility, and cross-annotation
   cross-check are cheap at the prediction stage and expensive
   to bolt on later.
4. **Document a clear failure mode**: M3's output is a candidate
   list, not a truth claim. Downstream consumers (Phase 8,
   clinical) should apply the §3.7 stacking and §4 drug-target
   criteria as appropriate.

### 5.4 Milestones before committing to full M3 development

- [ ] Cross-check the 67,490 novel junction sides against GENCODE,
      RefSeq, UCSC Known Genes. How many are truly absent from *all*
      major annotations?
- [ ] Of the truly-novel set, what fraction have GT/AG boundary
      dinucleotides? Conservation signal?
- [ ] Spot-check 20 novel candidates manually: visualize in IGV,
      check GTEx V9 coverage, confirm at least 5 are well-supported.
- [ ] Draft the label-generation and training loss for M3 (is it a
      binary novel/not-novel head? per-position regression on
      junction depth?). Pick architecture before spending cycles
      on data.

Once those milestones are met, the M3 training pipeline should
follow the same pattern as M2-S: train on Ensembl-keyed cache,
evaluate on alt sites + novel sites, run ablation.

---

## Reproduction

```bash
# Regenerate all audit outputs (test chroms)
python examples/meta_layer/11_junction_coverage_audit.py

# Full genome
python examples/meta_layer/11_junction_coverage_audit.py --chroms all \
    --output-dir output/meta_layer/junction_coverage_audit_fullgenome

# Notebook version (same analysis, tutorial style)
jupyter notebook notebooks/meta_layer/junction_coverage_audit.ipynb
```

## Related

- [junction_coverage_findings.md](junction_coverage_findings.md) —
  detailed findings on strand asymmetry and biotype composition
- [meta_model_variants_m1_m4.md](meta_model_variants_m1_m4.md) —
  overall M1/M2/M3/M4 design doc
- [ood_generalization.md](ood_generalization.md) — why training
  labels matter for alt-site detection
- `notebooks/meta_layer/junction_coverage_audit.ipynb` — tutorial
  version of the analysis with inline commentary
