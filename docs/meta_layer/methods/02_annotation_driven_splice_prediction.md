# Annotation-Driven Splice Prediction: From Label Sets to Latent Variables

## The Problem We're Solving

Canonical splice site prediction is solved. SpliceAI and its descendants achieve >99.5% accuracy on the canonical sites: GT-AG dinucleotides at exon boundaries, with clear sequence context rules. If the problem were just "is this position a splice donor or acceptor," we'd be done.

The real frontier is context-dependent splicing. A site is active in liver but silent in brain. It's used only under oxidative stress. It depends on the presence of an RNA-binding protein, or the chromatin state, or the developmental stage. These are the sites the base model struggles with because they violate the "sequence determines structure determines function" assumption.

This is where M2 lives. The canonical base model (M1) operates in a single sequence-to-structure namespace. M2's job is to ask: given that the base model sees this site as marginal (intermediate score, not high confidence), what multimodal context rescues it? Or dismisses it as noise?

To build M2, we need a training set of sites the base model has never reliably seen. That's not in MANE.

## The Annotation Landscape

MANE (Matched Annotation from NCBI and EBI) is a curated, minimal set: roughly one transcript per protein-coding gene, hand-validated, high confidence. As of GENCODE v47 (January 2025), that's 19,433 protein-coding genes with a single "best" isoform each. Perfect for a baseline. Useless for exploration.

Ensembl maintains a more comprehensive set, typically 5-10 transcripts per gene for protein-coding loci. It includes well-supported alternative isoforms, but still skews toward the common variants. For a gene like TTN (titin), you might see 10-15 isoforms in Ensembl.

GENCODE comprehensive is different in scope and ambition. It includes all Ensembl transcripts, plus manually curated alternatives, plus computational predictions, plus long non-coding RNAs. As of v47, GENCODE added 140,000+ new lncRNA transcripts and maintains those 19,433 protein-coding genes with 10-20+ transcripts each—sometimes 50+ for complex loci like TTN or DMD. GENCODE and Ensembl are joint projects managed by the same consortium, with slight timing differences in release cycles.

The key insight: more transcripts per gene means more annotated splice sites. But not all those sites are equal.

## The Set-Difference Trick

Define:
- A = MANE (canonical set, ~19K genes, ~1 isoform each)
- B = GENCODE comprehensive (full locus, ~19K genes, ~10-20+ isoforms each)

The candidates for M2 training are B \ A: all splice sites in the GENCODE comprehensive set that do not appear in any MANE isoform.

Why does this matter? Because the base model (call it the SpliceAI or OpenSpliceAI weights we're using as M1) was effectively trained on MANE-like data. It saw one isoform per gene. Its scores on B \ A will be systematically weak—not because the sequence is incomprehensible, but because the context is out-of-distribution.

The meta-layer's job: can we use tissue-specific junction reads, RBP binding predictions, conservation scores, or other modalities to say "this site is real" even though the base model is uncertain?

## Why GENCODE comprehensive \ MANE Beats Ensembl \ MANE

Both are valid training sets. GENCODE comprehensive \ MANE is better for M2 because it contains more of what we need: low-expression isoforms, retained introns, edge-case splice variants, and unannotated exons.

Ensembl is more conservative—it filters for evidence and has higher curation overhead. That's excellent for a baseline annotation. It's mediocre for building a meta-layer because the sites in Ensembl \ MANE are already somewhat "easy." They have multiple evidence traces, reasonable expression, and likely match existing splicing patterns.

GENCODE-only sites include singletons: transcripts found once or twice in a tissue, rare isoforms, computational predictions that haven't been validated, and cryptic splicing variants. These are the sites the base model genuinely fails on, and where multimodal context (or lack thereof) actually matters.

Put another way: Ensembl \ MANE gives you hard-negatives and modest-positives. GENCODE \ MANE gives you a spectrum from hard-positives (multiply evidenced) to noisy signals (single-experiment occurrences).

## The Label Quality Problem

Not all GENCODE sites are equally real. This is the uncomfortable truth that prevents naive binary labeling.

GENCODE includes:
- Manually curated human transcripts with RNA-seq and proteomics support (high confidence).
- Ensembl consensus transcripts, overlaid with manual review (high-moderate confidence).
- Computational predictions from long-read RNA-seq (SQANTI, IsoQuant) that haven't been curated (moderate-low confidence).
- Rare isoforms found in a single tissue or experimental condition (low confidence).
- Non-coding transcripts with less stringent validation than protein-coding (variable confidence).

The consequence: a site in GENCODE but not MANE could mean "this is a real alternative splicing pattern used in specific contexts," or it could mean "our pipeline found something that might exist." No way to know from the label alone.

This creates a labeling problem. If we treat all GENCODE-only sites as positive examples, we're training on noise. The recall will be inflated. If we treat them as negative examples, we're throwing away true positive evidence. The precision will be artificially high.

## Beyond Binary Labels: A Spectrum

The solution is to move beyond y ∈ {0, 1} and embrace a confidence-aware labeling scheme.

### Tier-Based Confidence

Assign each site to a tier based on annotation source:

```
Tier 0: In MANE
    -> High confidence (hand-curated, multi-evidence, >1% expression)

Tier 1: In GENCODE AND Ensembl, but not MANE
    -> Moderate-high confidence (multiple annotation projects agree)

Tier 2: In GENCODE-only
    -> Moderate-low confidence (computational or rare, not independently verified)
```

Within Tier 2, you could further stratify by annotation type (manual vs. computational) or by the number of supporting RNA-seq reads in the originating dataset.

### Junction-Weighted Labels

Even better: condition the label on multimodal support. GENCODE is a candidate generator. Junction support is a validator.

We already have the data: GTEx provides 353K tissue-specific junctions across 54 tissues, with PSI (percent spliced in) values and breadth metrics. A GENCODE-only site with strong junction support (PSI > 5%, breadth > 3 tissues) is almost certainly real. A GENCODE-only site with zero GTEx support is noise.

Define a splice_usage_score that combines:
- Annotation tier (0, 1, or 2)
- Junction support (PSI-weighted and breadth-weighted across tissues)
- Optional: proteomics evidence if available

```
splice_usage_score = w0 * tier_prior + w1 * psi_normalized + w2 * breadth_normalized
```

Tier 0 sites get label_strength = 1.0 automatically (they're canonical).

Tier 1 sites get label_strength based on junction overlap: if the site has supporting junctions in GTEx, bump it to 0.9. If not, 0.6.

Tier 2 sites get label_strength = junction support. If PSI > 10% in ≥2 tissues, 0.8. If PSI > 5% in ≥1 tissue, 0.5. If no GTEx support, 0.1 (weak but possible; maybe tissue-specific expression).

This doesn't require new data. It operationalizes what we already know: annotations are proxies for function; junction reads are stronger evidence.

## The Deeper Biological Reality

Annotations are tools. The real definition of "alternative splice site" is not "exists in a transcript database." It's "a site used in some contexts but not others."

That context is tissue, developmental stage, disease state, RBP expression, chromatin state, and upstream cis-elements. A site might be active in 10% of liver cells and silent in neurons. It might only fire under hypoxia. The presence of a single transcript in a database doesn't tell you any of that.

GENCODE is a candidate generator. It says "we found evidence that this exon boundary exists somewhere." The real splicing model's job is to learn which candidates activate under which conditions.

This is why tissue-specific context matters. It's why GTEx junction support matters. It's why M2 isn't just "base model plus annotation features"—it's "base model plus tissue context plus evidence of actual usage."

## State of the Art: Where M2 Fits

Recent models point in the same direction as AgenticSpliceAI—toward context-conditioned prediction—but through different architectures:

**AlphaGenome (DeepMind, Nature January 2026)** was the first to simultaneously predict splice sites, splice usage, and junctions at 1bp resolution from 1Mb input. It outperforms SpliceAI and Pangolin on 6 of 7 benchmarks, including an overall improvement in recall on rare variants.

**Splice Ninja (bioRxiv January 2026)** conditions predictions on 301 splicing factor expression levels from expression atlases. It predicts tissue-specific PSI and generalizes to unseen tissues by learning the relationship between RBP expression and splicing patterns.

**SpliceTransformer** and **TrASPr+BOS** take different angles: SpliceTransformer links splice site alterations to disease; TrASPr uses generative modeling and Bayesian optimization to design RNA sequences with desired tissue-specific splicing outcomes.

All of these point toward the same insight: sequence alone doesn't determine splicing. Context does. AlphaGenome and Splice Ninja condition on genomic or cellular context; TrASPr conditions on design objectives; SpliceTransformer surfaces the link between change and phenotype.

AgenticSpliceAI differs in one critical way: it separates the base model (M1: high-accuracy canonical prediction) from the context layer (M2-M4: increasingly sophisticated reasoning about why sites are used). This is a meta-learning architecture, not an end-to-end model. That separation lets us debug, extend, and reason about each layer independently.

## Building M2: The Experimental Setup

An important architectural constraint: every model in the pipeline — SpliceAI, OpenSpliceAI, M1-S, M2-S, M3-S — follows the same input-output protocol. Given an L-nucleotide sequence, the model outputs `[L, 3]` per-nucleotide scores (donor, acceptor, neither). M2 is not a different architecture. It's the same MetaSpliceModel trained and evaluated under different protocols.

M2a, for example, is an evaluation setting: take a trained model, run inference on Ensembl gene sequences, produce `[L, 3]` predictions, then evaluate specifically at positions in (Ensembl \ MANE). The set-difference filtering belongs in the evaluation script, not in the model.

With that grounding, the experimental setup:

**Reference genome:** GRCh38 (fixed).

**Annotations:**
- A = MANE (training annotation, canonical baseline)
- B = GENCODE comprehensive (evaluation annotation, full isoform space)

**Evaluation candidates:** B \ A — splice sites annotated in GENCODE but absent from MANE.

**Labels:** Generated by `build_splice_labels(gene_id, start, gene_len, splice_sites_df)` from the chosen annotation's `splice_sites_enhanced.tsv`. Each position gets one of: donor, acceptor, or neither. The label is the union across all transcripts — if a position is donor in any transcript, it's labeled donor.

**Training variants** (covered in the companion M2 formulations analysis):
- Train on MANE labels, evaluate on GENCODE \ MANE (pure evaluation protocol — M2a/b)
- Train on GENCODE labels with annotation-tier weighting (M2c)
- Train on GENCODE labels with junction-informed weighting (M2d)

**Features (all variants):** The MetaSpliceModel's three streams consume sequence (one-hot, dilated CNN), base model scores (OpenSpliceAI `[L, 3]`, MLP), and 9 dense multimodal channels (PhyloP, PhastCons, H3K36me3, H3K4me3, ATAC-seq, DNase-seq, junction log1p, junction support, RBP binding count).

**Baseline:** OpenSpliceAI base scores at B \ A sites. Expect low recall — the base model was trained on MANE-like data and has never seen these alternative sites as positive examples.

**Metrics:**
- Recall at alternative sites (B \ A) at different score thresholds
- Improvement over base model: Δ recall = recall(M2) - recall(base)
- Ablation by modality: disable individual multimodal channels to measure contribution
- Tiered evaluation: separate recall for Tier 1 (GENCODE ∩ Ensembl \ MANE) vs. Tier 2 (GENCODE-only)
- Blend alpha tracking: the learnable residual parameter indicates how much the meta-layer deviates from base scores at alternative sites

## Annotation as a Latent Variable

There's a theoretical frame underlying this entire approach:

P(site is real) = f(sequence, multimodal_features, annotation_confidence)

The annotation isn't a fixed label. It's a latent variable—something we observe with noise, something that correlates with the true underlying reality (whether the site is functionally used), but doesn't perfectly capture it.

This leads naturally to several extensions:

**Positive-Unlabeled (PU) Learning:** MANE sites are reliable positives. GENCODE\MANE sites are unlabeled—they might be positive or negative. Standard PU learning frameworks can handle this: learn a classifier that's robust to label noise, or learn both the classifier and the noise distribution jointly.

**Weak Supervision:** Multiple annotation sources (GENCODE, Ensembl, RefSeq, sample-specific transcriptomics) are weak signals. A probabilistic model can combine them—sites that appear in multiple sources are more likely to be real than sites that appear in one.

**Probabilistic Labeling:** Instead of uniform per-position loss weight, assign confidence weights based on annotation tier and junction evidence. A MANE site gets full weight. A GENCODE-only site with strong junction support gets weight 0.85. A GENCODE-only site with no junction evidence gets weight 0.3. The label remains discrete (donor/acceptor/neither — preserving the `[L, 3]` protocol), but the gradient contribution reflects confidence.

All of these assume that annotation is informative but imperfect. That's actually the right assumption. It pushes M2 away from "fit the annotation" and toward "use annotation as evidence to learn the underlying pattern."

## Next Steps

This analysis establishes why annotation choice is a modeling decision, not just a data collection detail. The next step—covered separately—is the variant analysis: given M2 trained on GENCODE \ MANE, how does it perform on real disease-associated variants? Do sites created or destroyed by a variant have higher M2 scores if they're predicted to be pathogenic? Can M2 contribute to variant effect prediction (M4)?

The key takeaway for now: the meta-layer works because the base model is incomplete. By choosing a label set that exposes the base model's blindspots, and by grounding labels in functional evidence (junction support, tissue breadth), we give M2 a meaningful task. Not "memorize which sites are in GENCODE"—that's trivial and useless. But "learn which sites are functionally used despite the base model's uncertainty." That's a real learning problem.
