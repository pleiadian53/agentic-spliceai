# RBP eCLIP Binding for Splice Site Prediction

**Date**: 2026-03-22
**Modality**: `rbp_eclip` (8th registered modality)
**Output**: 8 columns per position

---

## Why RBP Binding Matters for Splicing

Base-layer models (SpliceAI, OpenSpliceAI) predict splice sites from **DNA sequence alone**.
Conservation scores add evolutionary constraint. Epigenetic marks add tissue-specific chromatin
context. But none of these capture **active post-transcriptional regulation** -- the proteins
that directly bind RNA and control which exons are included or skipped.

RNA-binding proteins (RBPs) are the **direct effectors** of alternative splicing:

- **SR proteins** bind exonic splicing enhancers (ESEs), recruit U1/U2 snRNP to promote exon inclusion
- **hnRNPs** bind exonic/intronic splicing silencers (ESSs/ISSs) to promote exon skipping
- **Core regulators** like RBFOX2, PTBP1, U2AF1/U2AF2, ELAVL1, QKI, and MBNL1 drive
  tissue-specific and context-dependent splicing programs

RBP binding at a position means that position is under **active regulatory control** --
a qualitatively different signal from sequence potential (base scores), evolutionary
pressure (conservation), or chromatin state (epigenetic marks).

| Family | Examples | Mechanism | Effect on splicing |
|--------|----------|-----------|-------------------|
| **SR proteins** | SRSF1, SRSF3, SRSF7 | Bind ESEs, recruit U1/U2 snRNP | Promote exon inclusion |
| **hnRNPs** | HNRNPA1, HNRNPC, HNRNPK | Bind ESSs/ISSs | Promote exon skipping |
| **Tissue regulators** | RBFOX2, PTBP1, QKI, MBNL1 | Tissue-specific binding motifs | Context-dependent |
| **Core machinery** | U2AF1, U2AF2, SF3B4 | 3' splice site/branch point recognition | Required for splicing |

---

## What eCLIP Measures

Unlike ChIP-seq (which captures DNA-protein interactions in chromatin), eCLIP captures
**RNA-protein interactions** via UV crosslinking. The method provides position-level
evidence of where each RBP physically contacts RNA in living cells.

### The Measurement Pipeline

```
Wet lab (eCLIP)                     Computational (ENCODE pipeline)
──────────────                     ────────────────────────────────
UV crosslink RNA-protein            Align reads -> genome (STAR)
Immunoprecipitate with              Input-normalized: eCLIP / SMInput
  anti-RBP antibody                   = fold-enrichment signal
Reverse transcription               IDR (Irreproducible Discovery Rate)
  (truncates at crosslink site)       Replicate concordance filter
Size-matched input (SMInput)        Call peaks -> narrowPeak BED
Sequence (paired-end)               Write peaks with signalValue + pValue
```

### Interpreting the Values

- **signalValue**: Fold-enrichment over size-matched input (dimensionless, typically 2--50+)
- **pValue**: Statistical significance of enrichment (-log10 transform in narrowPeak files)
- **Peak intervals**: ~50--200bp (UV crosslink site +/- fragment size)
- **Zero**: No peak overlaps this position -- the RBP was not detected binding here

### Comparison to Other CLIP Methods

| Method | UV crosslink | Resolution | Input control | Used here |
|--------|-------------|-----------|---------------|-----------|
| **eCLIP** | Yes (254nm) | ~50--200bp | SMInput | **Yes** |
| CLIP-seq | Yes | ~50--200bp | No | No |
| iCLIP | Yes (individual nt) | ~1bp | No | No |
| RIP-seq | No | ~1000bp | Input | No |

eCLIP's size-matched input (SMInput) control is critical -- it accounts for RNA abundance
and non-specific binding, making the fold-enrichment signal interpretable as true
protein-RNA contact rather than background.

---

## The ENCODE eCLIP Dataset

The ENCODE consortium profiled **~150 RBPs** across K562 and HepG2 cell lines using eCLIP.
These are the same two cell lines used in the epigenetic modality's ENCODE ChIP-seq panel,
enabling direct multi-modal integration at the same positions.

Key properties:
- **IDR-filtered replicated peaks**: Biological replicate concordance filter removes
  >90% of irreproducible peaks while retaining >95% of the true signal
- **GRCh38 aligned**: Consistent with OpenSpliceAI and the rest of the feature pipeline
- **narrowPeak BED format**: Standard ENCODE output with signalValue and pValue per peak

### RBP Binding Near Splice Sites

```
5' Exon           Intron                         3' Exon
═══════╗──────────────────────────────────╔═══════
       GT                               AG
       |                                |
   ━━━ SRSF1 ━━━                   ━━━ PTBP1 ━━━
   (ESE: enhances                  (polypyrimidine
    exon inclusion)                 tract binding)

    ━━ HNRNPA1 ━━                     ━━ U2AF2 ━━
    (ISS: silences                   (3'SS recognition,
     this donor)                      essential for splicing)

               ━━━ RBFOX2 ━━━
               (YGCAYG motif,
                intronic enhancer)
```

Different RBPs bind at characteristic positions relative to splice sites. SR proteins
tend to bind exonic enhancers; hnRNPs bind silencer elements in exons or introns;
U2AF2 binds the polypyrimidine tract upstream of the acceptor AG. The meta-layer can
learn these spatial patterns from the overlap of eCLIP peaks with splice site positions.

---

## Data Landscape Beyond ENCODE (as of March 2026)

ENCODE's eCLIP program profiled ~150 RBPs but only in **K562 (leukemia) and HepG2
(liver carcinoma)** -- standardized reference cell lines, not chosen for tissue diversity.
This is the largest *coherent* CLIP dataset (same protocol, same lab, same cell types),
but it is not the only source of RBP binding data.

### Alternative and Complementary Data Sources

| Resource | Coverage | Strengths | Limitations |
|----------|----------|-----------|-------------|
| **ENCODE eCLIP** | ~150 RBPs, 2 cell lines | Gold standard: SMInput control, IDR filtering, uniform protocol | K562/HepG2 only |
| **POSTAR3** | 348 RBPs, 1,445 CLIP experiments, 7 species | Broadest coverage: aggregates data from many labs and cell types | Heterogeneous quality (different CLIP variants, no uniform control) |
| **RBPsuite 2.0** (2025) | Sequence-based prediction | Cell-type agnostic; works on any genome | Computational prediction, not experimental |
| **RBPWorld** (2025) | Cross-species RBP functional database | Disease associations, functional annotations | Not binding-site level |

**POSTAR3** is the most actionable near-term supplement. It curates CLIP-seq data from
published studies across many cell types and tissues -- not just ENCODE's K562/HepG2.
Trade-off: broader cell type coverage but noisier signal (different labs, different
CLIP variants, no uniform SMInput control). Our `aggregate_eclip_peaks.py` script
could be extended to ingest POSTAR3 data as a future enhancement.

### Emerging Methods (Not Yet at Scale)

- **Antibody-barcode eCLIP**: Multiplexes many RBPs in a single experiment using
  DNA-barcoded antibodies. Dramatically reduces cost per RBP, which could enable
  expansion to additional cell types. Published 2022 in Nature Methods.
- **MAPIT-seq** (2024--2025): Maps RBP-RNA interactions *in situ* on frozen tissue
  sections. Could eventually provide primary tissue RBP binding data, but not yet
  available at genome-wide scale for many RBPs.
- **seCLIP**: Single-end eCLIP removes the paired-end sequencing requirement, reducing
  cost ~2x. Used in some ENCODE Phase 4 experiments.

### ENCODE Phase 4 Status

ENCODE Phase 4 has expanded chromatin profiling (ChIP-seq, ATAC-seq) to 234 cell types,
but eCLIP profiling has **not** been expanded to new cell lines as of March 2026. The
high per-experiment cost (~$5K) and antibody requirements make scaling eCLIP to many
cell types more challenging than scaling histone ChIP-seq.

### Practical Implications

For our feature pipeline:
1. **Start with ENCODE eCLIP** (current implementation) -- highest quality, uniform protocol
2. **Evaluate POSTAR3** as a supplementary data source after the ENCODE-only baseline
   is validated -- gains tissue diversity at the cost of noisier signal
3. **Monitor antibody-barcode eCLIP** and MAPIT-seq publications for future multi-tissue
   RBP binding data

---

## Cancer Cell Line Bias

ENCODE's K562 (chronic myeloid leukemia) and HepG2 (hepatocellular carcinoma) are
**cancer cell lines**, not normal primary cells. This raises a valid concern: does
cancer-specific RBP dysregulation introduce bias into splice site prediction?

The answer depends on the prediction target.

### Constitutive Splice Sites: Minimal Risk

Core splicing machinery (U2AF1, U2AF2, SF3B4, PRPF8) binds at canonical GT/AG splice
sites in **all cell types** -- cancer or normal. These are the most conserved components
of splicing, and their binding positions are determined by sequence motifs, not cell state.

```
Core machinery binding is sequence-determined:

   ...exonAG|GTraagt...      U1 snRNP binds GT + downstream consensus
   ...yyyyyyyyAG|exon...     U2AF2 binds polypyrimidine tract (yyyy)
                             U2AF1 binds the AG dinucleotide

These interactions are invariant across cell types. The eCLIP signal at constitutive
sites from K562/HepG2 is representative of any human cell.
```

### Alternative Splice Sites: Moderate Risk, Manageable

Cancer cells have altered RBP expression profiles. For example:
- **SRSF1** is frequently overexpressed in tumors (proto-oncogenic)
- **hnRNP ratios** are shifted in many cancers
- **PTBP1/PTBP2** switching occurs during neural differentiation (and is disrupted
  in gliomas)

This means:
- **Binding positions are still correct**: SRSF1 binds its GAAGAA motif whether the
  cell is cancerous or not. What changes is *how much* binding (occupancy), not *where*.
- **Signal intensity may be inflated**: `rbp_max_signal` at cancer-upregulated RBP
  sites may be higher than in normal tissue.
- **Binary features are robust**: `rbp_has_splice_regulator` (is *any* regulator bound?)
  is less affected by expression-level differences than signal-magnitude features.

### Novel/Induced Splice Sites: Highest Risk

Cancer-specific neo-splice sites (aberrant splicing driven by oncogenic RBP
dysregulation) could get high RBP scores, creating false positives for "novel"
splice sites that are actually cancer artifacts. This is the scenario where
precision loss in the M3 model could occur.

### Built-In Mitigations

Several features in the current design hedge against cancer bias:

1. **`rbp_cell_line_breadth`**: Positions bound in *both* K562 AND HepG2
   (breadth=2) are more likely constitutive/general. Cancer-specific artifacts
   tend to be cell-line-specific (breadth=1). The meta-layer can learn to weight
   breadth=2 sites higher for canonical prediction.

2. **Multi-modal fusion**: The meta-layer does not rely on RBP alone. Cancer
   artifacts would typically show:
   - High RBP signal but **low conservation** (not evolutionarily constrained)
   - High RBP signal but **low junction breadth** (not seen across GTEx tissues)
   - The combination of conservation + junction + RBP is self-correcting

3. **Binary features are cell-type-robust**: `rbp_has_splice_regulator` captures
   whether a regulatory motif *exists* at the position, which is sequence-determined.

### Recommended Evaluation Checkpoints

After full-genome meta-layer training with RBP features:

1. **Precision ablation**: Compare precision on constitutive splice sites with and
   without the RBP modality. If precision drops, the cancer bias is leaking through.
2. **Breadth stratification**: Evaluate separately for breadth=2 (constitutive) vs
   breadth=1 (cell-type-specific) positions.
3. **Cancer RBP flag** (optional): If bias is measurable, add `rbp_n_cancer_dysregulated`
   -- count of RBPs known to be cancer-overexpressed (SRSF1, HNRNPA1, etc.) at each
   position. Let the meta-layer learn to discount these.
4. **POSTAR3 comparison**: Once POSTAR3 data is integrated, compare RBP binding at
   the same positions from cancer vs non-cancer CLIP experiments.

---

## Feature Schema (8 Columns)

RBP eCLIP features are **sparse**: most genomic positions have no overlapping eCLIP peaks.
Unbound positions receive 0.0 for all features, which is informative -- it means no RBP
was detected binding at that position in either cell line.

### Count-Based Features (0.0 for unbound positions)

| Column | Description | Range |
|--------|-------------|-------|
| `rbp_n_bound` | Number of unique RBPs with a peak overlapping this position | [0, ~150] |
| `rbp_n_sr_proteins` | Number of SR protein family members bound | [0, ~10] |
| `rbp_n_hnrnps` | Number of hnRNP family members bound | [0, ~10] |

### Signal Features (0.0 for unbound positions)

| Column | Description | Range |
|--------|-------------|-------|
| `rbp_max_signal` | Max fold-enrichment across all overlapping peaks | [0, ~100+] |
| `rbp_mean_signal` | Mean fold-enrichment across all overlapping peaks | [0, ~50] |
| `rbp_max_neg_log10_pvalue` | Max -log10(pValue) significance across peaks | [0, ~300] |

### Binary Features

| Column | Description | Values |
|--------|-------------|--------|
| `rbp_has_splice_regulator` | Any known splice regulator bound at this position | {0.0, 1.0} |

### Cell Line Features

| Column | Description | Range |
|--------|-------------|-------|
| `rbp_cell_line_breadth` | Number of cell lines with binding evidence | [0, 2] for K562+HepG2 |

---

## Feature Interpretation

### Position-Level Patterns

The combination of RBP eCLIP features with other modalities creates interpretable signatures:

- **High `rbp_n_bound` + `rbp_has_splice_regulator=1`**: Actively regulated splice site.
  Multiple RBPs compete or cooperate here -- likely an alternative splicing hotspot.
- **High base score + `rbp_n_bound=0` + high conservation**: Constitutive, sequence-dependent
  splice site. No regulatory protein binding needed -- the sequence motif is sufficient.
- **Moderate base score + SRSF1 binding + low conservation**: Enhancer-dependent exon.
  May be tissue-specific (included only where SR protein is expressed/active).
- **High `rbp_n_hnrnps` near donor**: Likely silenced or skipped exon. Multiple hnRNPs
  binding near a donor site suggests active repression of this exon's inclusion.

### Absence of Binding Is Informative

A position with `rbp_n_bound = 0` is not missing data -- it means no RBP was detected
binding there in either K562 or HepG2. This parallels how `junction_has_support = 0`
is informative in the junction modality. The meta-layer learns from both presence and
absence of binding, which is why unbound positions receive **0.0** (not NaN).

### Cell Line Breadth

- **breadth=2**: Constitutive binding -- the same RBP binds at this position in both
  K562 and HepG2. Suggests a regulatory interaction that is not cell-type-specific.
- **breadth=1**: Cell-type-specific binding -- the RBP binds in one cell line but not
  the other. Indicates tissue-regulated splicing at this position.
- **breadth=0**: No binding detected in either cell line.

---

## Data Preparation

### Aggregation Script

The `scripts/aggregate_eclip_peaks.py` script downloads IDR-filtered eCLIP narrowPeak
files from ENCODE, aggregates them across RBPs and cell lines, and writes a single
parquet file for the feature pipeline.

```bash
# Download from ENCODE (recommended for first run)
python scripts/aggregate_eclip_peaks.py \
    --output data/mane/GRCh38/rbp_data/ \
    --cache-dir data/mane/GRCh38/rbp_data/raw/

# From pre-downloaded narrowPeak BED files
python scripts/aggregate_eclip_peaks.py \
    --input-dir /path/to/beds/ \
    --output data/mane/GRCh38/rbp_data/
```

### Path Resolution

The `RbpEclipModality` resolves data via a 3-tier lookup:

1. **Explicit config**: `modality_configs.rbp_eclip.eclip_data_path` in YAML
2. **Resource registry**: Registered path for the current build
3. **Convention**: `{build_dir}/rbp_data/eclip_peaks.parquet`

Or specify explicitly in the YAML config:

```yaml
modality_configs:
  rbp_eclip:
    eclip_data_path: /path/to/eclip_peaks.parquet
```

---

## YAML Configuration

### In full_stack.yaml (M2: RBP eCLIP as feature)

```yaml
pipeline:
  modalities:
    - base_scores
    - annotation
    - sequence
    - genomic
    - conservation
    - epigenetic
    - junction
    - rbp_eclip          # RBP binding as input feature

modality_configs:
  rbp_eclip:
    aggregation: summarized        # cross-RBP summary stats
    cell_lines: [K562, HepG2]
    min_neg_log10_pvalue: 2.0      # -log10(0.01) — IDR-filtered peaks only
    batch_size: 5000               # positions per batch in interval overlap
```

### In isoform_discovery.yaml (lower threshold for novel site discovery)

```yaml
modality_configs:
  rbp_eclip:
    aggregation: summarized
    cell_lines: [K562, HepG2]
    min_neg_log10_pvalue: 2.0
    batch_size: 5000
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model` | `openspliceai` | Base model for build resolution |
| `eclip_data_path` | `null` | Explicit path to peaks parquet (overrides registry lookup) |
| `aggregation` | `summarized` | Aggregation mode: `summarized` (8 cross-RBP cols) |
| `cell_lines` | `[K562, HepG2]` | Cell lines to include from ENCODE eCLIP |
| `min_neg_log10_pvalue` | `2.0` | Minimum -log10(pValue) to include a peak |
| `batch_size` | `5000` | Positions per batch in interval overlap (controls memory) |

---

## Build Considerations

ENCODE eCLIP data is **GRCh38 only**. Like epigenetic marks, there are no ENCODE eCLIP
datasets aligned to GRCh37.

| Build | Conservation | Epigenetic | RBP eCLIP |
|-------|-------------|-----------|-----------|
| GRCh38 (OpenSpliceAI) | 100-way PhyloP/PhastCons | Full ENCODE panel | **Full ENCODE eCLIP** |
| GRCh37 (SpliceAI) | 46-way PhyloP/PhastCons | Not available | **Not available** |

For GRCh37/SpliceAI users, the `RbpEclipModality` should:
1. Gracefully degrade: fill all 8 columns with 0.0
2. Log a clear warning explaining the build limitation
3. The meta-layer handles missing modalities via feature masking (NaN-aware models)

This asymmetry further reinforces the preference for OpenSpliceAI (GRCh38) as the
primary base model for multimodal meta-layer training.

---

## Connection to Meta-Layer Models

| Meta Model | RBP eCLIP Role | Purpose |
|---|---|---|
| **M1**: Enhanced canonical | Not needed | Confirms known splice sites (GTF labels) |
| **M2**: Alternative site detector | **Feature** | Regulatory context for tissue-specific splicing |
| **M3**: Novel site predictor | **Feature** | Orthogonal to junction target (protein-RNA, not RNA-seq) |
| **M4**: Perturbation/induced | **Feature** | RBP KD/KO predictions vs observed splicing changes |

**No target leakage**: eCLIP measures protein-RNA UV crosslinking, not RNA-seq junction
counting. Using RBP binding as a feature in M3 (where junction support is the target)
is safe because the two assays measure fundamentally different things -- physical
protein contact vs. spliced RNA abundance.

---

## Integration with Existing Modalities

Adding rbp_eclip as the 8th modality brings the total feature count to 94:

```
base_scores:    43 columns
annotation:      3 columns
sequence:        3 columns
genomic:         4 columns
conservation:    9 columns
epigenetic:     12 columns
junction:       12 columns
rbp_eclip:       8 columns
────────────────────────────
Total:          94 feature columns
```

The RBP eCLIP modality is complementary to existing modalities:
- **vs. conservation**: Conservation reflects evolutionary constraint; RBP binding reflects
  active regulation in present-day cells
- **vs. epigenetic**: Histone marks reflect chromatin state (DNA-level); RBP binding
  reflects post-transcriptional regulation (RNA-level)
- **vs. junction**: Junctions are the outcome of splicing; RBP binding is a causal input
  to the splicing decision

---

## References

### eCLIP and ENCODE
- Van Nostrand et al. (2020) "A large-scale binding and functional map of human
  RNA-binding proteins" Nature 583:711-719
- Van Nostrand et al. (2016) "Robust transcriptome-wide discovery of RNA-binding
  protein binding sites with enhanced CLIP (eCLIP)" Nature Methods 13:508-514
- Van Nostrand et al. (2022) "Multiplexed transcriptome discovery of RNA-binding
  protein binding sites by antibody-barcode eCLIP" Nature Methods 20:1084-1092
- ENCODE Project Consortium (2020) "Expanded encyclopaedias of DNA elements in the
  human and mouse genomes" Nature 583:699-710
- ENCODE Project Consortium (2025) "An expanded registry of candidate cis-regulatory
  elements" Nature (ENCODE Phase 4, 234 cell types)

### RBP Databases and Tools
- Zhao et al. (2022) "POSTAR3: an updated platform for exploring post-transcriptional
  regulation coordinated by RNA-binding proteins" Nucleic Acids Research 50:D287-D297
- RBPsuite 2.0 (2025) "An updated RNA-protein binding site prediction suite"
  BMC Biology
- Liao et al. (2024) "Improved discovery of RNA-binding protein binding sites in
  eCLIP data using DEWSeq" Nucleic Acids Research 52:e1

### RBP Families and Splicing
- Long & Bhatt (2021) "Roles of the SR protein family in splicing regulation"
  Wiley Interdiscip Rev RNA 12:e1613
- Geuens et al. (2016) "The hnRNP family: insights into their role in health and
  disease" Human Genetics 135:851-867
- Fu & Ares (2014) "Context-dependent control of alternative splicing by RNA-binding
  proteins" Nature Reviews Genetics 15:689-701

### Cancer Splicing and Cell Line Bias
- El Marabti & Bhatt (2024) "Alternative splicing and related RNA binding proteins
  in human health and disease" Signal Transduction and Targeted Therapy
- Urbanski et al. (2018) "The cancer spliceome: Reprogramming of alternative splicing
  in cancer" Frontiers in Molecular Biosciences 5:80
- Agrawal et al. (2024) "A systematic identification of RBPs driving aberrant splicing
  in cancer" Biomedicines 12:2592
