# Negative Strand Genes: Coordinate Systems, Splice Prediction, and Variant Effects

A practical guide to the problems that arise when predicting splice sites and
variant effects on negative-strand genes. These issues are under-discussed in
the literature despite roughly half of all human genes being on the minus
strand.

---

## 1. Two Coordinate Systems

Every nucleotide in the genome has two addresses that must be kept
straight:

| System | Direction | Used by |
|--------|-----------|---------|
| **Genomic** (chromosome) | Always 5'&rarr;3' on the **plus strand**; positions increase from p-arm to q-arm | FASTA files, VCF, BED, bigWig tracks |
| **Transcriptomic** (mRNA) | Always 5'&rarr;3' in the **direction of transcription** | cDNA, the model's internal view, HGVS c. notation |

For **plus-strand genes** the two systems are numerically identical &mdash;
genomic position 100 comes before 101 in both the chromosome and the
transcript.

For **minus-strand genes** they run in opposite directions:

```
Genomic (plus-strand reference):

     gene_start (low coord)              gene_end (high coord)
     |                                   |
     5'===========================================3'  ← plus strand
     3'===========================================5'  ← minus strand
                                         |
                                         Transcription starts here (TSS)
                                         ↓
                                         5' ──────────────── 3'  mRNA

Position mapping (minus-strand gene):
  Transcriptomic pos 1  →  Genomic: gene_end      (highest coordinate)
  Transcriptomic pos 2  →  Genomic: gene_end - 1
  Transcriptomic pos N  →  Genomic: gene_start     (lowest coordinate)
```

**Consequence:** incrementing the transcript position *decrements* the
genomic coordinate.  Every tool in the pipeline must agree on which
coordinate system it operates in, or positions will be silently wrong.


### 1.1. Why This Matters for Splice Sites

A **donor site** is defined as the 5' end of an intron (the GT
dinucleotide in U2-type introns).  An **acceptor site** is the 3' end
(the AG dinucleotide).

For a **plus-strand gene**, the GT dinucleotide appears at the *lower*
genomic coordinate of the intron.  For a **minus-strand gene**, the same
GT dinucleotide (in transcript space) appears as its reverse complement
**AC** at the *higher* genomic coordinate when read from the plus-strand
FASTA:

```
Plus-strand intron (genomic view):
  ...exon]GT...intronic...AG[exon...
          ^^                ^^
          donor (5')        acceptor (3')

Minus-strand intron (genomic view, reading left to right on plus strand):
  ...exon]CT...intronic...AC[exon...
          ^^                ^^
          acceptor (3')     donor (5')

  Same intron in transcript (minus-strand, reading 5'→3'):
  ...exon]GT...intronic...AG[exon...
          ^^                ^^
          donor (5')        acceptor (3')
```

If you read the plus-strand FASTA and see `AC`, that is a **donor** on
the minus strand.  If you see `CT`, that is an **acceptor** on the minus
strand.  This is a common source of confusion.

---

## 2. Splice Prediction on Negative-Strand Genes

### 2.1. The Model Expects Transcript-Order Input

Deep learning splice models (SpliceAI, OpenSpliceAI, etc.) are trained
on transcript-order sequences &mdash; always 5'&rarr;3' in the direction
of transcription.  The model learns that a GT dinucleotide *in the input*
signals a donor and AG signals an acceptor.

For minus-strand genes, you must **reverse-complement** the genomic
sequence before feeding it to the model:

```python
# Genomic FASTA (always plus-strand)
genomic_seq = fasta[chrom][start:end]  # e.g., "...ACGTCTAC..."

# For minus-strand genes, convert to transcript order
if strand == "-":
    model_input = reverse_complement(genomic_seq)
else:
    model_input = genomic_seq
```

### 2.2. Mapping Predictions Back to Genomic Coordinates

The model outputs per-position scores in the same order as its input.
For minus-strand genes the input was reversed, so the output must be
reversed to recover genomic coordinate order:

```python
# Model output: [L, 3] with channels [neither, acceptor, donor]
predictions = model(model_input)

if strand == "-":
    # Reverse positions back to genomic order
    predictions = predictions[::-1]
```

### 2.3. Do NOT Swap Donor/Acceptor Channels

This is the most common mistake.  It is tempting to reason: "the model
detected donors and acceptors in transcript space; on the minus strand
donors become acceptors in genomic space."  **This reasoning is wrong.**

The model's output channels represent biological function:
- Channel "donor" = 5' splice site (beginning of intron)
- Channel "acceptor" = 3' splice site (end of intron)

These labels are **invariant to strand** because the input was already
reverse-complemented into transcript space.  A donor in the model's
output is a donor in the gene's biology, regardless of which genomic
strand the gene sits on.

After reversing the position axis, the donor and acceptor channels
are already correct.  This is confirmed by OpenSpliceAI's own variant
annotation pipeline (`openspliceai/variant/utils.py`), which reverses
positions but never swaps channels:

```python
# OpenSpliceAI convention (Keras)
if strand == "-":
    y_ref = y_ref[:, ::-1]   # reverse positions ONLY
    y_alt = y_alt[:, ::-1]
    # No channel swap — indices 1 (acceptor) and 2 (donor) unchanged

# Delta extraction uses the same channel indices for both strands:
idx_pa = (y_alt[:, 1] - y_ref[:, 1]).argmax()  # acceptor gain
idx_pd = (y_alt[:, 2] - y_ref[:, 2]).argmax()  # donor gain
```

### 2.4. Verification Technique

To verify your minus-strand handling is correct, check the dinucleotides
at high-scoring positions against the plus-strand FASTA:

| Model output | Strand | Expected dinucleotide on plus-strand FASTA |
|-------------|--------|---------------------------------------------|
| Donor > 0.9 | + | **GT** at that position |
| Donor > 0.9 | - | **AC** at that position (RC of GT) |
| Acceptor > 0.9 | + | **AG** at that position |
| Acceptor > 0.9 | - | **CT** at that position (RC of AG) |

Example validation for MYBPC3 (minus-strand, chr11):

```python
# All 33 donor-labeled positions showed AC dinucleotide → 100% correct
# All 33 acceptor-labeled positions showed CT dinucleotide → 100% correct
```

---

## 3. Multimodal Features and Coordinate Alignment

### 3.1. Genomic-Coordinate Features

Many features used in multimodal splice prediction are stored as
**genomic-coordinate signals** &mdash; bigWig files for conservation
(phastCons, phyloP), chromatin accessibility (ATAC-seq), histone
modifications (H3K36me3, H3K4me3), and others.

These signals are indexed by genomic position and are strand-agnostic in
storage (a conservation score at chr11:47332565 is the same regardless of
which strand you are analyzing).

### 3.2. The Alignment Problem

When the base model runs on a minus-strand gene, it operates in
transcript space.  But multimodal features live in genomic space.  The
features must be aligned to the same positions as the model's other
inputs.

There are two valid strategies:

**Strategy A: Everything in Genomic Order (used in training)**

During meta-layer training, all inputs are kept in genomic coordinate
order:
- DNA sequence: read directly from FASTA (no RC)
- Base scores: from precomputed parquets indexed by genomic position
- Multimodal features: extracted by genomic window

The meta-layer's sequence CNN learns the relevant patterns directly from
the genomic-order input.  It never sees reverse-complemented sequence.

**Strategy B: RC Sequence, Genomic Features (used in variant prediction)**

During variant effect prediction, the base model must run on
transcript-order sequence to produce fresh ref/alt scores.  The
multimodal features remain in genomic order.  The meta-layer receives:
- Sequence: genomic order (one-hot encoded from FASTA, no RC)
- Base scores: from the base model (RC'd internally, output reversed
  back to genomic order)
- Multimodal features: genomic order

The key is that **the meta-layer always sees everything in genomic
order**.  Only the base model's internal computation uses
transcript-order sequence.

### 3.3. BigWig Feature Caveats

Conservation and epigenomic bigWig files have specific behaviors on the
minus strand:

1. **Values are strand-agnostic**: `phastCons100way` at position X
   returns the same value regardless of query strand.  Conservation is a
   property of the position, not the strand.

2. **Feature windows must use genomic coordinates**: When extracting a
   window for a minus-strand gene, use `(chrom, genomic_start,
   genomic_end)` with start < end.  Do not reverse the query range.

3. **No reversal needed**: Because features are already in genomic order
   and the meta-layer expects genomic order, bigWig features require no
   special strand handling.

4. **Strand-specific signals**: Some tracks (e.g., RNA-seq coverage)
   may be strand-specific.  These require selecting the correct strand
   file but still do not need coordinate reversal.

---

## 4. Variant-Induced Splicing Patterns

### 4.1. Delta Score Framework

Variant effects on splicing are quantified by **delta scores**: the
difference between alternate-allele and reference-allele splice
probabilities at each position:

```
delta[i, c] = P_alt[i, c] - P_ref[i, c]
```

where `i` is the position and `c` is the channel (donor or acceptor).

Four summary statistics capture the worst-case effect (following
SpliceAI convention):

| Score | Definition | Interpretation |
|-------|-----------|----------------|
| **DS_DG** (Donor Gain) | max(delta[:, donor]) | New donor site created |
| **DS_DL** (Donor Loss) | max(-delta[:, donor]) | Existing donor site destroyed |
| **DS_AG** (Acceptor Gain) | max(delta[:, acceptor]) | New acceptor site created |
| **DS_AL** (Acceptor Loss) | max(-delta[:, acceptor]) | Existing acceptor site destroyed |

The overall pathogenicity score is often taken as the maximum of these
four values.  Variants with max |delta| > 0.5 are considered likely
splice-altering; those > 0.8 are high-confidence pathogenic.

### 4.2. Canonical Splice Site Disruption

The most common pathogenic splice variants directly disrupt the invariant
GT (donor) or AG (acceptor) dinucleotides at intron boundaries.  These
produce a characteristic **loss** pattern:

```
Donor GT disruption (e.g., G→A at the first position of GT):

  Position:   ... exon |G T ... intron ... A G| exon ...
  Ref score:        0.01 0.99                0.98 0.01
  Alt score:        0.01 0.05                0.98 0.01
  Delta:            0.00 -0.94               0.00 0.00
                         ^^^^^^
                         Strong donor loss (DS_DL ≈ 0.94)
```

**Real example** from our pipeline (MYBPC3, minus-strand):

```
chr11:47332565 C>A (G-of-GT in transcript)
  DS_DL = -0.68  (donor loss at the mutated site)
  DS_DG = +0.30  (cryptic donor gain 1bp away)
```

### 4.3. Compensatory Signals

A hallmark of splice-disrupting variants is the emergence of
**compensatory signals** &mdash; nearby cryptic splice sites that gain
strength when the canonical site is destroyed.  The splicing machinery
must choose *some* splice site, so when the primary site is weakened,
alternative sites become more competitive:

```
Before variant (reference):
  pos 100: donor = 0.99  ← canonical
  pos 110: donor = 0.05  ← cryptic (dormant)
  pos 135: donor = 0.02  ← cryptic (dormant)

After variant (alternate):
  pos 100: donor = 0.30  ← weakened canonical
  pos 110: donor = 0.48  ← cryptic activated   (Δ = +0.43)
  pos 135: donor = 0.37  ← cryptic activated   (Δ = +0.35)
```

This produces the characteristic "loss + gain" delta pattern:

| Delta type | Meaning | Biological consequence |
|-----------|---------|------------------------|
| Donor loss at canonical site | Primary splice site weakened | Exon skipping or intron retention |
| Donor gain at nearby position | Cryptic splice site activated | Altered exon boundary, possible frameshift |
| Multiple gains | Several cryptic sites compete | Complex mis-splicing, multiple aberrant isoforms |

**Real example** (MYBPC3 chr11:47333193 C>A):

```
DS_DL = -0.67  at position 47333193  (canonical donor destroyed)
DS_DG = +0.48  at position 47333182  (cryptic gain 10bp upstream)
DS_DG = +0.43  at position 47333194  (cryptic gain 2bp downstream)
DS_DG = +0.37  at position 47333227  (cryptic gain 35bp downstream)
```

Three cryptic donors activate simultaneously, suggesting this variant
could produce multiple aberrant transcript isoforms.

### 4.4. Common Pathogenic Patterns

| Pattern | Delta signature | Clinical examples |
|---------|----------------|-------------------|
| **Donor destruction** | DS_DL > 0.5, often with DS_DG > 0.2 | IVS+1 G>A/T mutations |
| **Acceptor destruction** | DS_AL > 0.5, often with DS_AG > 0.2 | IVS-1 G>A/C mutations |
| **Cryptic donor activation** | DS_DG > 0.5, minimal DS_DL | Deep intronic variants creating new GT |
| **Cryptic acceptor activation** | DS_AG > 0.5, minimal DS_AL | Deep intronic variants creating new AG |
| **Exon skipping** | DS_DL + DS_AL both > 0.3 | Loss of both splice sites flanking an exon |
| **Balanced compensation** | DS_DL &asymp; DS_DG (or DS_AL &asymp; DS_AG) | Strong cryptic site replaces canonical |

### 4.5. Distance from Variant

The **distance** between the variant and the affected splice site is
informative:

- **0-2 bp**: Direct disruption of the splice dinucleotide (GT/AG).
  Highest confidence pathogenic.
- **3-10 bp**: Disruption of the extended splice consensus (positions
  -3 to +6 for donors, -20 to +3 for acceptors).
- **10-50 bp**: May affect exonic/intronic splice enhancers or silencers
  (ESE/ESS/ISE/ISS).
- **50-500 bp**: Can create or destroy cryptic splice sites.
- **>500 bp**: Rare but documented; deep intronic variants can activate
  pseudoexons.

### 4.6. Interpreting Small vs. Large Deltas

| Max |delta| | Interpretation | Action |
|-------------|---------------|--------|
| < 0.1 | No significant splice effect predicted | Likely benign (for splicing) |
| 0.1 - 0.2 | Mild effect; possible subtle mis-splicing | Requires RNA evidence |
| 0.2 - 0.5 | Moderate effect; likely affects splicing | Investigate with RNA-seq |
| 0.5 - 0.8 | Strong effect; high confidence splice-altering | Clinically significant |
| > 0.8 | Near-complete disruption | Almost certainly pathogenic |

Note: these thresholds apply to the base model's delta scores.  The
meta-layer refines these predictions by incorporating multimodal context
(conservation, chromatin, RNA-seq junction support), which can
up- or down-weight the base model's signal.

---

## 5. Practical Checklist

When implementing a splice prediction pipeline that handles both strands:

- [ ] **Input to base model**: RC the sequence for minus-strand genes
- [ ] **Output positions**: Reverse the position axis after prediction
- [ ] **Channel semantics**: Do NOT swap donor/acceptor channels
- [ ] **Variant alleles**: Always specify ref/alt on the plus strand
      (matching the FASTA), regardless of gene strand
- [ ] **Multimodal features**: Extract using genomic coordinates
      (no reversal needed)
- [ ] **Validation**: Check dinucleotides at high-scoring positions
      against the FASTA (GT/AG for plus-strand; AC/CT for minus-strand)
- [ ] **Delta interpretation**: Look for loss + compensatory gain
      patterns; don't rely on a single channel

---

## 6. References

- Jaganathan et al. (2019). "Predicting Splicing from Primary Sequence
  with Deep Learning." *Cell*, 176(3), 535-548. The original SpliceAI
  paper establishing the delta score framework.
- Chao et al. (2025). "OpenSpliceAI improves the prediction of variant
  effects on mRNA splicing." *Genome Biology*. Updated architecture
  with class-wise temperature calibration.
- Mount (1982). "A catalogue of splice junction sequences." *Nucleic
  Acids Research*. Defines the consensus GT-AG rule and extended splice
  site motifs.
