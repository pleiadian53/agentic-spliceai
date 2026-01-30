# Isoform Discovery: Context-Aware Splice Prediction

**Status**: Research & Planning Phase  
**Goal**: Discover novel isoforms induced by variants, disease, stress, and other external factors

---

## 🎯 Vision

**Problem**: Current annotations (MANE, Ensembl) capture known isoforms, but miss:
- Disease-specific isoforms
- Variant-induced alternative splicing
- Stress-response transcripts
- Tissue-specific rare isoforms
- Developmental stage-specific variants
- Environmental condition-specific splicing

**Solution**: Use Meta-SpliceAI's adaptive prediction to discover novel isoforms by:
1. **Meta Layer**: Detect context-dependent splice sites beyond canonical annotations
2. **Agentic Layer**: Validate predictions with literature and experimental evidence
3. **Integration**: Combine with RNA-seq, clinical variants, and disease databases

---

## 📊 Current State vs Future Vision

### Current: Canonical Isoform Prediction

```
Base Layer (MANE)
    ↓
Predict 1-2 canonical transcripts per gene
    ↓
Validated against known annotations
```

**Coverage**: ~44 splice sites per gene (BRCA1 example)

---

### Future: Context-Aware Isoform Discovery

```
Base Layer (MANE baseline)
    ↓
Meta Layer (adaptive refinement)
    ↓
Novel Splice Site Detection
    ↓
Isoform Reconstruction
    ↓
Agentic Validation (literature + experimental)
    ↓
Novel Isoform Catalog
```

**Potential Coverage**: 1,218+ splice sites per gene (Ensembl-level + novel discoveries)

---

## 🏗️ Proposed Architecture

### Layer 1: Meta Layer - Novel Splice Site Detection

**Goal**: Identify high-confidence splice sites beyond MANE canonical set

**Components**:

1. **Delta Score Analysis**
   - Compare meta-layer predictions to base-layer
   - High delta = context-dependent splice site
   - Flag sites with high confidence but not in MANE

2. **Context Clustering**
   - Group similar contexts (same variants, disease, tissue)
   - Identify context-specific splice patterns
   - Discover condition-dependent isoforms

3. **Isoform Reconstruction**
   - Assemble detected splice sites into full transcripts
   - Validate transcript structure (ORF, NMD rules)
   - Generate novel isoform annotations

4. **Confidence Scoring**
   - Multi-factor confidence score:
     - Meta-layer prediction strength
     - Conservation across species
     - RNA-seq evidence (if available)
     - Splice motif strength
     - Consistency with biological rules

**Output**:
```
Novel Splice Sites:
- Position: chr17:43,067,608 (donor)
- Confidence: 0.92
- Context: BRCA1-c.68_69del variant
- Evidence: High delta score (0.45), strong motif
- Status: Candidate for validation
```

---

### Layer 2: Agentic Layer - Evidence Aggregation

**Goal**: Validate novel isoforms with external evidence

**Components**:

1. **Literature Research Agent**
   - Query PubMed for reports of:
     - Novel isoforms in gene of interest
     - Disease-associated alternative splicing
     - Functional impact of splice variants
   - Extract evidence from papers
   - Build evidence graph

2. **Database Integration Agent**
   - **RNA-seq Data**:
     - GTEX (tissue-specific expression)
     - Cancer atlases (tumor-specific isoforms)
     - Disease cohorts
   - **Variant Databases**:
     - ClinVar (pathogenic splice variants)
     - gnomAD (population-level splice variants)
   - **Isoform Databases**:
     - GENCODE (comprehensive annotations)
     - RefSeq (curated transcripts)

3. **Validation Workflow Agent**
   - Propose validation experiments:
     - RT-PCR primers for novel junction
     - Minigene assay design
     - CRISPR screens for splice modifiers
   - Prioritize candidates by:
     - Clinical relevance
     - Functional impact
     - Validation feasibility

4. **Evidence Aggregation**
   - Combine predictions with external evidence
   - Score novel isoforms by evidence strength:
     - RNA-seq support: High (direct evidence)
     - Literature support: Medium (indirect evidence)
     - Prediction only: Low (needs validation)

**Output**:
```
Novel Isoform Report:
- Gene: BRCA1
- Novel Junction: chr17:43,067,608-43,082,575
- Context: Tumor samples, BRCA1-mutant
- Meta-Layer Confidence: 0.92
- RNA-seq Evidence: 15/120 tumor samples (12.5%)
- Literature: 3 papers report similar junction
- Validation Priority: HIGH (clinical relevance)
- Proposed Experiment: RT-PCR primers designed
```

---

## 🔬 Data Integration Strategy

### Primary Data Sources

1. **RNA-seq Data**
   - **GTEX**: Tissue-specific isoforms (53 tissues, 17,382 samples)
   - **TCGA**: Cancer-specific isoforms (33 cancer types)
   - **GTEx + TCGA junction files**: Direct evidence of novel splice junctions
   - **Custom cohorts**: Disease-specific RNA-seq

2. **Variant Data**
   - **ClinVar**: Pathogenic splice variants
   - **gnomAD**: Population-level splice variants
   - **COSMIC**: Cancer-associated mutations
   - **Personal genomes**: Individual-level predictions

3. **Epigenetic Data**
   - **ENCODE**: Histone marks, chromatin state
   - **Roadmap Epigenomics**: Tissue-specific epigenomes
   - **ChIP-seq**: Splicing factor binding

4. **Conservation Data**
   - **PhyloP**: Evolutionary conservation
   - **PhastCons**: Conserved elements
   - **Cross-species comparison**: Mouse, zebrafish homologs

---

### Data Processing Pipeline

```
1. Load Base Predictions (MANE canonical)
   ↓
2. Load Meta Predictions (context-aware)
   ↓
3. Compute Delta Scores
   ↓
4. Filter High-Confidence Novel Sites
   ↓
5. Cross-reference with RNA-seq Junctions
   ↓
6. Query Literature for Evidence
   ↓
7. Score and Rank Candidates
   ↓
8. Generate Discovery Reports
```

---

## 📋 Use Cases

### 1. Disease-Specific Isoforms

**Scenario**: Identify breast cancer-specific BRCA1 isoforms

**Workflow**:
```bash
# Predict on tumor samples with BRCA1 variants
agentic-spliceai meta predict \
  --gene BRCA1 \
  --variants tumor_variants.vcf \
  --context tumor \
  --discover-isoforms

# Validate with TCGA RNA-seq
agentic-spliceai isoform validate \
  --predictions brca1_novel_isoforms.tsv \
  --rnaseq-cohort TCGA-BRCA \
  --output validation_report.html
```

**Expected Output**:
- 10-20 high-confidence novel splice sites
- 3-5 novel isoforms with RNA-seq support
- Literature evidence for functional impact
- Clinical relevance scores

---

### 2. Variant-Induced Splicing Changes

**Scenario**: Patient has VUS (variant of uncertain significance) near splice site

**Workflow**:
```bash
# Predict impact of variant on splicing
agentic-spliceai variant predict \
  --vcf patient_vus.vcf \
  --gene-list candidate_genes.txt \
  --detect-novel-junctions

# Research evidence
agentic-spliceai agentic research \
  --novel-junctions detected_junctions.tsv \
  --query "pathogenic splicing impact" \
  --sources pubmed,clinvar,splicevardb
```

**Expected Output**:
- Novel splice junctions induced by variant
- Predicted functional impact (NMD, truncation, etc.)
- Literature evidence for similar variants
- Clinical interpretation (likely pathogenic, benign, etc.)

---

### 3. Tissue-Specific Isoform Discovery

**Scenario**: Discover brain-specific isoforms relevant to neurological disease

**Workflow**:
```bash
# Compare across tissues
agentic-spliceai isoform discover \
  --gene-list neurological_genes.txt \
  --tissues brain,cerebellum,cortex \
  --reference-tissues heart,liver,kidney \
  --rnaseq-source GTEX

# Validate with brain RNA-seq
agentic-spliceai isoform validate \
  --candidates brain_specific_isoforms.tsv \
  --rnaseq-cohort GTEX-Brain \
  --min-samples 5
```

**Expected Output**:
- Brain-specific isoforms not in MANE
- Expression patterns across brain regions
- Developmental trajectories
- Disease associations

---

## 🛠️ Implementation Roadmap

### Phase 5 (Meta Layer) - Foundation

**Add to existing meta-layer work**:
- ✅ Delta score computation (already exists in ValidatedDeltaPredictor)
- 🆕 Novel splice site detector
- 🆕 Confidence scoring module
- 🆕 Context clustering

**Deliverables**:
- Detect novel splice sites with high confidence
- Score sites by multiple evidence lines
- Group by context (variants, disease, tissue)

---

### Phase 8 (NEW) - Isoform Discovery & Validation

**Dedicated phase for isoform discovery**:

#### 8.1: Meta Layer Extensions
- Isoform reconstruction from splice sites
- Transcript structure validation (ORF, NMD)
- Novel isoform annotation format
- Integration with base predictions

#### 8.2: RNA-seq Integration
- GTEX junction file parser
- TCGA junction file parser
- Custom cohort support
- Junction-level validation

#### 8.3: Agentic Validation
- Isoform research agent (literature mining)
- Evidence aggregation (multi-source)
- Validation workflow generator
- Discovery report generation

#### 8.4: Clinical Applications
- Variant-induced isoform prediction
- Disease-specific isoform catalog
- VUS interpretation workflow
- Therapeutic target identification

**Estimated Timeline**: 4-6 weeks (after Phase 7)

---

## 📊 Success Metrics

### Discovery Metrics

- **Novel isoforms discovered**: Target 100+ across 50 genes
- **RNA-seq validation rate**: >70% with junction support
- **Literature confirmation**: >50% with supporting papers
- **Functional impact**: >30% affect protein function

### Validation Metrics

- **Precision**: % of predictions with RNA-seq support
- **Recall**: % of known rare isoforms detected
- **Clinical utility**: Impact on VUS interpretation
- **Novel discoveries**: Isoforms not in any database

---

## 🔬 Research Questions

### Biological Questions

1. **How common are context-dependent isoforms?**
   - Proportion of genes with disease-specific isoforms
   - Tissue-specificity of alternative splicing
   - Variant-induced splicing changes

2. **What contexts induce novel splicing?**
   - Stress response
   - Developmental stages
   - Disease progression
   - Therapeutic interventions

3. **Can we predict functional impact?**
   - NMD escape
   - Protein domain loss/gain
   - Clinical significance

### Technical Questions

1. **What confidence threshold for novel sites?**
   - Balance precision vs recall
   - Context-dependent thresholds
   - Multi-factor scoring

2. **How to reconstruct full isoforms?**
   - Splice site pairing
   - Exon inclusion/exclusion
   - ORF prediction

3. **How to validate computationally?**
   - RNA-seq evidence strength
   - Conservation requirements
   - Motif scoring

---

## 📚 Related Work

### Literature

1. **Disease-specific splicing**:
   - Cancer-specific isoforms (TCGA studies)
   - Neurological disease splicing (autism, ALS)
   - Cardiac disease isoforms (cardiomyopathy)

2. **Computational approaches**:
   - LeafCutter (differential splicing)
   - MAJIQ (local splice variation)
   - rMATS (alternative splicing events)
   - SUPPA2 (isoform quantification)

3. **Validation methods**:
   - RT-PCR validation
   - Long-read sequencing (PacBio, Nanopore)
   - Minigene assays

### Databases

- **GENCODE**: Comprehensive gene annotations
- **RefSeq**: Curated transcript sequences
- **APPRIS**: Principal isoform annotations
- **IsoformAtlas**: Tissue-specific isoforms
- **SpliceVault**: Alternative splicing database

---

## 💡 Key Innovations

### 1. Context-Aware Discovery

**Beyond static annotations**: Predict isoforms specific to:
- Individual genetic backgrounds
- Disease states
- Tissue/cell types
- Environmental conditions

### 2. Multi-Modal Evidence Integration

**Combine**:
- Base-layer predictions (canonical)
- Meta-layer predictions (adaptive)
- RNA-seq data (direct evidence)
- Literature (biological context)
- Conservation (evolutionary support)

### 3. Agentic Validation

**LLM-powered**:
- Hypothesis generation
- Evidence synthesis
- Experimental design
- Report generation

---

## 🎯 Next Steps

### Immediate (Research Phase)

1. **Literature review**:
   - Survey isoform discovery methods
   - Identify key use cases
   - Define success criteria

2. **Data inventory**:
   - Available RNA-seq datasets
   - Variant databases
   - Existing novel isoform catalogs

3. **Prototype design**:
   - Novel splice site detector
   - Confidence scoring
   - Validation workflow

### Short Term (Phase 5 Integration)

1. **Extend meta layer**:
   - Add delta score analysis
   - Implement confidence scoring
   - Create novel site detector

2. **Test on known cases**:
   - Disease-associated isoforms
   - Variant-induced changes
   - Tissue-specific examples

### Long Term (Phase 8)

1. **Full implementation**:
   - Isoform reconstruction
   - RNA-seq integration
   - Agentic validation
   - Clinical workflows

2. **Validation study**:
   - Experimental validation
   - Clinical cohorts
   - Publication

---

## 📁 Documentation Structure

```
docs/isoform_discovery/
├── README.md                          ← This file - Vision & overview
├── ARCHITECTURE.md                    ← Technical architecture
├── USE_CASES.md                       ← Detailed use cases
├── DATA_INTEGRATION.md                ← Data sources & formats
├── VALIDATION_METHODS.md              ← How to validate discoveries
└── RESEARCH_QUESTIONS.md              ← Open questions & experiments

dev/research/isoform_discovery/
├── brainstorming/                     ← Ideas, sketches
├── literature_review/                 ← Paper summaries
├── prototype/                         ← Experimental code
└── experiments/                       ← Pilot studies
```

---

## 🤝 Collaboration Opportunities

### Academic Partners
- Tissue-specific isoform experts
- Disease cohort access
- Experimental validation labs

### Clinical Partners
- VUS interpretation needs
- Patient cohorts
- Clinical validation

### Industry Partners
- RNA-seq datasets
- Therapeutic target discovery
- Diagnostic development

---

**Status**: 🔬 Research & Planning  
**Next**: Extend meta layer (Phase 5) with novel site detection  
**Long-term**: Dedicated isoform discovery phase (Phase 8)

**Questions? Ideas? Add to**: `dev/research/isoform_discovery/brainstorming/`
