# Virtual Transcripts — From Delta Scores to Junction-Level Predictions

**Created**: March 2026
**Prerequisite**: [01_alternative_splice_prediction_analysis.md](01_alternative_splice_prediction_analysis.md) (Formulations A–D, label hierarchy)

---

## Motivation

Formulations A–D in the prior note operate at the **per-position** level: they predict
how splice site probabilities change at individual nucleotides. But splicing is a
**junction-level** phenomenon — the spliceosome excises an intron defined by a specific
donor-acceptor *pair*. A high donor probability at position 150 and a high acceptor
probability at position 3200 are necessary but not sufficient to define a junction;
the two sites must actually be *paired* by the splicing machinery.

This note asks: what would it take to predict variant effects at the junction level,
and what exactly are the labels?

---

## The representation gap

### What SpliceAI actually computes

The model outputs a tensor `y[position, channel]` with three channels:

| Channel | Meaning |
|---------|---------|
| 0 | Background (non-splice) probability |
| 1 | Acceptor probability |
| 2 | Donor probability |

These are **marginal** per-position estimates. The donor score at position i is
computed independently of the acceptor score at position j. The model has no explicit
notion of "junction (i, j)".

### What the delta scoring pipeline extracts

`get_delta_scores()` in `openspliceai/variant/utils.py` (lines 508–539) reduces the
full prediction landscape to **four scalar summaries**:

```
idx_pa = (y_alt[:, 1] - y_ref[:, 1]).argmax()   # single best acceptor gain position
idx_na = (y_ref[:, 1] - y_alt[:, 1]).argmax()   # single best acceptor loss position
idx_pd = (y_alt[:, 2] - y_ref[:, 2]).argmax()   # single best donor gain position
idx_nd = (y_ref[:, 2] - y_alt[:, 2]).argmax()   # single best donor loss position
```

This discards two things:

1. **All non-maximum positions** — there may be multiple significant changes; only the
   argmax survives.
2. **Any notion of pairing** — the four positions are selected independently. The
   reported donor gain and acceptor gain have no guaranteed relationship to each other.

### What a junction requires

A junction is a **joint event**: donor at position D and acceptor at position A are
used *together* by the spliceosome, excising the intron [D+1, A-1]. A transcript is
an ordered set of non-overlapping junctions:

```
T = { (d_1, a_1), (d_2, a_2), ..., (d_n, a_n) }
    where a_k < d_{k+1} for all k   (exons don't overlap)
```

The reference transcript defines these pairs explicitly (from GTF via
`create_datafile.py:85–105`):

```python
for i in range(len(exons) - 1):
    donor    = exons[i].end          # last position of current exon
    acceptor = exons[i+1].start      # first position of next exon
```

Going from per-position probabilities to junction pairs requires solving an
**assignment problem** that the current pipeline does not address.

---

## The ambiguity problem: which pairs form junctions?

Consider a variant that produces these delta signals across a gene region:

```
Position    Δ(donor)    Δ(acceptor)    Reference role
────────    ────────    ───────────    ──────────────
  150        +0.82          —          (none)
  200        -0.91          —          known donor
  3100         —          +0.75        (none)
  3200         —          -0.88        known acceptor
```

The reference junction is **(200, 3200)**. But after the variant, what junctions exist?

| Candidate junction | Interpretation |
|-------------------|----------------|
| (200, 3200) removed | Reference junction lost — both donor and acceptor weakened |
| (150, 3200) | Donor shifts upstream by 50 bp; acceptor retained despite score drop |
| (200, 3100) | Donor retained; acceptor shifts upstream by 100 bp (shorter intron) |
| (150, 3100) | Both sites shift — entirely new junction |
| (150, 3200) + (200, 3100) | Two new junctions? Only if they define non-overlapping exons |

**The delta scores alone cannot distinguish these cases.** Each scenario produces a
different virtual transcript with different biological consequences (different protein
product, different mRNA stability, etc.).

### Why this is not just an implementation detail

The pairing ambiguity isn't a minor bookkeeping issue. It determines:

- **The exon that gets skipped or included** — and therefore the protein product
- **The reading frame** — a junction shift of non-multiple-of-3 bases causes a
  frameshift
- **The intron length** — which affects splicing efficiency and may determine whether
  splicing occurs at all
- **Whether the event is exon skipping, cryptic exon inclusion, intron retention, or
  alternative 5'/3' site usage** — these are biologically distinct mechanisms with
  different regulatory logic

---

## Three label strategies

The fundamental question for any supervised approach: what are the labels?

### Strategy 1: Per-junction modification labels (anchored to reference)

Start from the known reference transcript. For each reference junction (d_i, a_i),
define a categorical label:

```
RETAINED         — junction unchanged (both scores stable)
LOST             — junction abolished (donor or acceptor score drops below threshold)
SHIFTED_DONOR    — same acceptor, donor moves to position d' (with d' specified)
SHIFTED_ACCEPTOR — same donor, acceptor moves to position a' (with a' specified)
```

Plus a separate detection problem for novel junctions not near any reference junction.

**How to derive these labels**:

For a given variant, run SpliceAI on ref and alt sequences to get the full prediction
landscapes y_ref and y_alt. Then:

1. For each reference junction (d_i, a_i):
   - Compute donor delta: Δ_d = y_alt[d_i, 2] - y_ref[d_i, 2]
   - Compute acceptor delta: Δ_a = y_alt[a_i, 1] - y_ref[a_i, 1]
   - If both are near zero → RETAINED
   - If donor drops significantly → check nearby positions for donor gain → SHIFTED_DONOR or LOST
   - If acceptor drops significantly → check nearby positions for acceptor gain → SHIFTED_ACCEPTOR or LOST

2. For novel junctions: scan for position pairs where both donor and acceptor gain
   exceed a threshold and the pair is not near any reference junction.

**Strengths**:
- Well-defined for the common cases (most pathogenic variants affect 1–2 junctions)
- Labels can be constructed from existing SpliceAI predictions + GTF
- No external data beyond what we already have

**Weaknesses**:
- The labels are derived from SpliceAI predictions, which are themselves imperfect —
  we'd be training on the base model's own outputs (circular if not careful)
- The "nearby" search for shifted sites requires a distance threshold (how far to look?)
- Cannot discover truly novel exons far from reference structure
- **Pairing for novel junctions remains heuristic** — this is the core unsolved problem

### Strategy 2: Junction-level labels from RNA-seq

Use actual splice junction read counts from samples carrying the variant of interest.

**Data sources**:
- **GTEx**: ~17K samples with matched WGS + RNA-seq across 54 tissues. Variants in
  WGS can be associated with differential junction usage in RNA-seq from the same
  individuals.
- **TCGA**: Somatic variants with matched tumor RNA-seq.
- **SRA/recount3**: Large-scale reprocessed RNA-seq with junction quantification.

**Label definition**: For a variant V in gene G:

1. Identify individuals who carry V and individuals who don't (from WGS)
2. Quantify junction usage in both groups (from RNA-seq junction reads)
3. A junction (d, a) is **gained** if it appears significantly more often in carriers
4. A junction (d, a) is **lost** if it appears significantly less often in carriers
5. The **virtual transcript** is the reference transcript with gains added and losses
   removed

**What junction reads provide that delta scores don't**: A split read physically spans
the junction — it aligns partly to the upstream exon and partly to the downstream
exon. This is direct evidence that positions d and a were paired by the spliceosome.
No pairing heuristic needed.

**Strengths**:
- Ground truth for junction pairing — reads physically demonstrate the donor-acceptor pair
- Quantitative (read counts give PSI-like magnitude)
- Can discover entirely novel junctions not in any annotation

**Weaknesses**:
- Requires matching variant calls to RNA-seq in the same individuals
- Limited to variants present in available cohorts (common variants overrepresented)
- Tissue-specific: a junction may be used in brain but not liver
- Significant data engineering effort (variant calling + junction quantification + association)

### Strategy 3: Per-position prediction + learned junction assembly

Keep the per-position prediction framework but add an explicit **junction assembly**
stage that learns to pair donors with acceptors.

**Architecture**:

```
Stage 1 (per-position):
  sequence + variant → P(donor_i), P(acceptor_i)  for all i
  (This is what SpliceAI already does)

Stage 2 (junction assembly):
  Given all candidate donors {d : P(donor_d) > τ_d}
  and all candidate acceptors {a : P(acceptor_a) > τ_a},
  predict which (d, a) pairs form actual junctions.

  Inputs per candidate pair:
    - P(donor_d), P(acceptor_a)
    - genomic distance |a - d|
    - sequence features of the intron (branch point motifs, polypyrimidine tract)
    - reference transcript context (is this near a known junction?)

  Output: P(junction | d, a) — probability this pair is used
```

**Training signal for Stage 2**: Junction reads from RNA-seq (same data as Strategy 2,
but used to train a pairing model rather than as direct labels).

**Strengths**:
- Cleanly separates "where are splice sites?" (Stage 1, abundant labels) from "which
  sites pair?" (Stage 2, requires junction data)
- The pairing model can learn biological constraints (intron length distributions,
  U2 vs U12 intron types, branch point positioning)
- Generalizes: once trained, applies to any variant without needing per-variant RNA-seq

**Weaknesses**:
- Two-stage training complexity
- Stage 2 requires junction-level training data (back to RNA-seq)
- Candidate pair enumeration can be combinatorially large for genes with many
  candidate splice sites

---

## What "virtual transcript" means under each strategy

The virtual transcript T_virt(V) for a variant V, given reference transcript T_ref:

```
Strategy 1 (reference-anchored):
  T_virt = T_ref
           - {lost junctions}
           + {shifted junctions (donor or acceptor moved)}
           + {novel junctions found by heuristic scan}

  Label source: SpliceAI predictions (derived, potentially circular)

Strategy 2 (RNA-seq-grounded):
  T_virt = T_ref
           - {junctions with significantly fewer reads in variant carriers}
           + {junctions with significantly more reads in variant carriers}

  Label source: Junction read counts (ground truth)

Strategy 3 (learned assembly):
  T_virt = argmax_T  Π_{(d,a) ∈ T}  P(junction | d, a)
           subject to non-overlapping exon constraints

  Label source: Junction reads for training the pairing model
```

---

## Comparison

| Aspect | Strategy 1: Reference-anchored | Strategy 2: RNA-seq junctions | Strategy 3: Learned assembly |
|--------|-------------------------------|-------------------------------|------------------------------|
| Label source | SpliceAI predictions + GTF | Junction reads (GTEx/TCGA) | Junction reads (for training) |
| Pairing solved? | Heuristic (proximity) | Yes (reads are direct evidence) | Yes (learned) |
| Novel junctions | Limited (heuristic scan) | Yes (reads discover them) | Yes (if Stage 1 detects the sites) |
| Data requirement | Just SpliceAI + GTF | Matched WGS + RNA-seq cohort | Same as 2, but amortized |
| Circularity risk | High (training on own outputs) | None | None (Stage 2 trained on reads) |
| Per-variant RNA-seq needed at inference? | No | Yes (or a trained model) | No |
| Handles tissue specificity? | No | Yes (per-tissue junction counts) | Possible (tissue as input feature) |

---

## Key open questions

1. **Is Strategy 1 ever sufficient?** For the common case of a variant disrupting a
   canonical splice site (directly hitting GT or AG), the pairing is trivial: the
   affected junction is the one containing that site. The ambiguity only arises for
   variants that create cryptic sites or have distal effects. How large is this
   "trivial" fraction in practice?

2. **What fraction of SpliceVarDB variants have matched RNA-seq?** If most SpliceVarDB
   variants come from individuals not in GTEx/TCGA, Strategy 2 may have limited
   overlap. Alternatively, could we use population-level junction QTL data (sQTLs from
   GTEx) as a proxy?

3. **Is the junction assembly problem (Strategy 3) actually learnable?** The
   spliceosome's pairing rules are influenced by intron length, branch point location,
   exon definition vs. intron definition, and SR/hnRNP protein binding — can a model
   learn these from junction counts alone, or does it need additional features?

4. **Multi-isoform genes**: A gene may have multiple active transcripts simultaneously.
   The "virtual transcript" is really a set of virtual transcripts with usage
   proportions. Should the prediction target be a single most-likely transcript, or a
   distribution over transcript structures?

---

## Relationship to formulations A–D

This note extends the label hierarchy from `01_alternative_splice_prediction_analysis.md`
by adding a level between positions and magnitudes:

```
Level 0: Genome-wide       "Which positions are splice sites?"           (GTF)
Level 1: Variant-level     "Does this variant affect splicing?"          (SpliceVarDB)
Level 2: Effect type       "Gain or loss? Donor or acceptor?"            (inferred)
Level 3: Position-level    "At which position does the site occur?"      (absent)
Level 3.5: Junction-level  "Which donor pairs with which acceptor?"      ← NEW
Level 4: Magnitude         "How strong is the effect (PSI change)?"      (RNA-seq)
```

Formulations A–D all produce Level 3 outputs (per-position predictions). The virtual
transcript concept lives at Level 3.5: it requires not just knowing *where* splice
sites are, but *which ones pair*. This is a strictly harder problem.

The practical implication: any Formulation E approach that claims to predict virtual
transcripts must either (a) include an explicit junction assembly step, or (b) use
junction-level supervision to force the per-position predictions to be consistent
with actual pairing.

---

## Suggested next steps

1. **Quantify the trivial fraction**: For SpliceVarDB variants that directly hit GT/AG
   dinucleotides, the affected junction is unambiguous. Measure what percentage of
   splice-altering variants fall into this category — this sets the baseline for how
   far Strategy 1 alone can take us.

2. **Survey available junction data**: Check overlap between SpliceVarDB variants and
   GTEx sQTL data. If substantial, Strategy 2 becomes immediately viable for a subset.

3. **Prototype Strategy 1**: Implement the reference-anchored virtual transcript
   construction as a post-processing step on existing SpliceAI predictions. This
   doesn't require new training — just a pairing algorithm applied to the full
   prediction landscape (not just argmax). Evaluate manually on well-characterized
   variants (e.g., SMN2 exon 7 skipping, CFTR exon 9 skipping).

4. **Design Strategy 3 architecture**: If junction assembly is needed, sketch the
   Stage 2 model: what are the input features per candidate pair, what training data
   is available, what's the loss function?

---

## References within codebase

| Component | File |
|-----------|------|
| Delta score computation | `openspliceai/variant/utils.py:352–541` |
| Reference junction extraction from GTF | `openspliceai/create_data/create_datafile.py:85–105` |
| Per-position label encoding | `openspliceai/create_data/create_datafile.py:101–105` |
| Junction support labels (RNA-seq weighted) | `foundation_models/docs/evo2/junction_support_labels.md` |
| Splice site extraction pipeline | `splice_engine/base_layer/data/preparation.py` |
| SpliceAI model architecture (3-channel output) | `openspliceai/train_base/openspliceai.py:47–72` |
