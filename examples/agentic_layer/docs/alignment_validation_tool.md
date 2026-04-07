# Alignment-Based Splice Prediction Validation

## Motivation

The meta-layer's FASTA inference mode (`08_evaluate_sequence_model.py --fasta`)
produces splice site predictions on arbitrary sequences without ground truth
annotations.  To evaluate these predictions, we need to map them back to a
reference genome where splice sites are known.

This is a classic alignment use case (analogous to BLAST), but specialized for
splice site coordinate mapping.  It bridges the meta-layer (predictions) and
the agentic layer (automated evidence gathering and validation).

---

## Tool Design

### Input
- Predicted splice sites from FASTA inference (`fasta_predictions.tsv`)
  - Columns: `sequence_id, position, donor_prob, acceptor_prob, neither_prob`
- Target reference genome (default: GRCh38) with splice annotations

### Output
- Validation report: for each predicted splice site, concordance with known
  annotations at the aligned reference position
- Summary statistics: concordance rate, novel predictions, false positives

### Pipeline

```
Input FASTA          Meta-layer              Alignment             Annotation
sequences    --->    predictions    --->     to reference  --->   lookup at
                     [L, 3] probs           (minimap2)            mapped coords
                                                                      |
                                                                      v
                                                              Validation report
                                                              (concordance, novel
                                                               sites, FP rate)
```

### Step 1: Alignment

Use `minimap2` (preferred over BLAST for long sequences):

```bash
minimap2 -a --cs reference.fa query.fa > alignment.sam
```

For splice-aware alignment:
```bash
minimap2 -a --splice reference.fa query.fa > alignment.sam
```

The `--splice` flag is specifically designed for aligning mRNA/cDNA to genomes
and detects GT-AG/GC-AG intron boundaries.

### Step 2: Coordinate Mapping

For each predicted splice site at position `p` in the query sequence:
1. Parse the SAM/BAM alignment to get reference coordinates
2. Map query position `p` to reference position `p_ref` via CIGAR string
3. Handle gaps, insertions, deletions at splice boundaries

Key considerations:
- Splice site positions are typically at intron-exon boundaries
- The predicted position in the query may span an intron in the reference
- Multi-mapping sequences need disambiguation (use MAPQ filtering)

### Step 3: Annotation Lookup

At each mapped reference coordinate `p_ref`:
1. Look up known splice sites from MANE/Ensembl annotations
2. Check if the predicted splice type (donor/acceptor) matches
3. Classify as:
   - **Concordant**: predicted site matches a known annotation
   - **Novel**: predicted site has no known annotation (potential discovery)
   - **Discordant**: predicted donor where annotation says acceptor (or vice versa)
   - **False positive**: high-probability prediction at a non-splice position

### Step 4: Junction Evidence (Optional Tier 3)

If RNA-seq junction data is available for the reference:
1. Look up junction reads at `p_ref`
2. Novel predictions with junction support are higher-confidence discoveries
3. Novel predictions without junction support need further validation

---

## Use Cases

### 1. Cross-Build Validation (T2T-CHM13)

The T2T-CHM13 reference resolved many gaps in GRCh38, particularly in
centromeric regions, segmental duplications, and acrocentric chromosomes.
Genes in these regions may have altered splice site predictions.

```
T2T-CHM13 gene  --->  Meta-layer  --->  Align to GRCh38  --->  Compare
sequence               predictions       (minimap2)             with MANE/Ensembl
```

Expected findings:
- Most canonical splice sites should be concordant (conserved GT-AG boundaries)
- Structural variants may shift splice positions
- T2T-resolved regions may reveal previously hidden splice sites

### 2. Pangenome Analysis

The Human Pangenome Reference Consortium provides haplotype-resolved assemblies
for diverse populations.  Splice site variation across haplotypes is
largely unexplored.

```
Pangenome        --->  Meta-layer  --->  Align to GRCh38  --->  Population-level
haplotype              predictions       (minimap2)             splice variation
sequences                                                       analysis
```

### 3. Synthetic Biology

For designed gene constructs (codon-optimized, synthetic introns), the tool
validates that intended splice sites are predicted by the model.

### 4. Cross-Species Ortholog Analysis

Aligning orthologous genes from other species to human GRCh38:
- Do predicted splice sites align with human splice sites?
- Where do they diverge?  (Potential species-specific alternative splicing)

---

## Implementation Considerations

### As an Agentic Tool

This alignment validation step is a natural fit for the agentic layer:

1. **Autonomous execution**: Given a FASTA prediction file, the agent
   automatically selects the appropriate alignment tool and reference
2. **Multi-source evidence**: The agent can combine alignment results with
   junction data, conservation scores, and literature evidence
3. **Report generation**: The Nexus research agent can produce an
   interpretive report on novel splice site discoveries

### Dependencies

- `minimap2` (alignment; available via conda/pip)
- `pysam` (SAM/BAM parsing; already in project dependencies)
- `pyfaidx` (FASTA access; already in project dependencies)
- Reference genome + annotations (GRCh38 MANE/Ensembl; already available)

### Coordinate System Alignment

Critical detail: the meta-layer uses **gene-relative** coordinates (position 0
= gene start), while alignment produces **genome-absolute** coordinates.  The
mapping must account for:
- Gene strand (reverse complement for minus-strand genes)
- 0-based vs 1-based conventions (SAM is 1-based, internal model is 0-based)
- Chromosome naming (chr prefix present/absent)

Use `ensure_chrom_column()` from `splice_engine/data/schema.py` for chromosome
name normalization.

---

## Status

**Not yet implemented.**  This is a planned agentic-layer tool for Phase 7.

The meta-layer side (FASTA inference mode) is complete as of April 2026.
The evaluation hierarchy document (`examples/meta_layer/docs/evaluation_hierarchy.md`)
describes how this tool fits into the broader validation strategy.

---

## Related Documents

- [Evaluation hierarchy](../../meta_layer/docs/evaluation_hierarchy.md) — evidence tiers for prediction validation
- [Model variants M1-M4](../../../docs/meta_layer/methods/00_model_variants_m1_m4.md) — M3 uses junction as validation target
- [Nexus research agent](../../../src/nexus/) — multi-agent orchestration for evidence fusion
