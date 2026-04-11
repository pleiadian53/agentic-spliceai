# M2 Variant Formulations: Training and Evaluation Protocols

> **Naming convention update (April 2026)**: This document uses legacy
> names (M2a-f) from the original design. The updated convention is:
> - **M2a** → **Eval-Ensembl-Alt** (evaluation protocol)
> - **M2b** → **Eval-GENCODE-Alt** (evaluation protocol)
> - **M2c** → **M2-S** (model: Ensembl-trained meta-layer)
> - M2d-f remain as future training variants
>
> See [naming_convention.md](naming_convention.md) for full details.

## The Universal Protocol

Every model in the AgenticSpliceAI pipeline — base models (SpliceAI, OpenSpliceAI), meta models (M1-S, M2-S, M3-S), and any future extension — follows the same input-output contract:

```
DNA sequence (L nucleotides) → [L, 3] scores (donor, acceptor, neither)
```

M2 is not a different architecture. It is the same MetaSpliceModel (three-stream CNN, ~370K params, residual blending with base scores) trained and evaluated under different protocols. The three-stream architecture processes:

```
Stream A: DNA sequence [B, 4, L_seq] → dilated CNN → [B, H, L]
Stream B: base_scores + mm_features [B, 3+C, L] → CNN → [B, H, L]
Fusion → output head → logits [B, L, 3] → softmax → α×refined + (1-α)×base → [B, L, 3]
```

What changes across M2 variants is not the model shape. It's three things: what labels the model trains on, how loss is weighted, and where evaluation focuses.

## What M2a Actually Is

M2a is an evaluation setting. Take a trained model (M1-S or M2-S), run inference on Ensembl gene sequences, get `[L, 3]` per-nucleotide predictions, then evaluate specifically at the sites in Ensembl but not in MANE.

The flow:

1. Load `data/ensembl/GRCh38/splice_sites_enhanced.tsv` → full Ensembl labels
2. Load `data/mane/GRCh38/splice_sites_enhanced.tsv` → MANE labels
3. For each gene, `build_splice_labels(gene_id, start, gene_len, splice_sites_df)` constructs a `[L]` array (0=donor, 1=acceptor, 2=neither, remapped from the meta-layer convention)
4. Run model inference → `[L, 3]` predictions
5. Evaluate on all Ensembl sites (overall performance)
6. Evaluate on (Ensembl \ MANE) sites only — the M2a-specific metric

Step 6 tells us whether the meta-layer generalizes beyond the annotation source the base model was trained on. The filtering logic belongs in the evaluation script (`08_evaluate_sequence_model.py`, planned), not in training. The model's weights don't know about "Ensembl \ MANE" — they just produce `[L, 3]`. The evaluation carves out the interesting subset.

This matters because every M2 "variant" proposed below is either a training protocol change (different labels, different loss weighting) or an evaluation protocol change (different site filtering, different metrics). The architecture is constant.

---

## The Two Axes of Variation

**Training axis:** What labels does the model see, and how does it weight them?

The label source is `splice_sites_enhanced.tsv` from some annotation. The standard pipeline generates labels via `build_splice_labels()`, which takes the union across all transcripts in the annotation — if a position is donor in ANY transcript, it's labeled donor. The training protocol variations are:

- Which annotation generates the training labels (MANE alone, Ensembl, GENCODE comprehensive, or a combination)
- Whether all labeled positions are weighted equally, or some are upweighted/downweighted
- Whether the cross-entropy loss is standard or modified (ordinal, focal, sample-weighted)

**Evaluation axis:** Where do we measure performance?

- Overall recall/precision on the training annotation (sanity check)
- Recall at (Annotation_B \ Annotation_A) sites — the "alternative site" metric
- Per-tissue recall (if tissue-specific junction evidence is available for validation)
- Modality ablation — which input channels contribute most at alternative sites

These two axes are orthogonal. You can combine any training variant with any evaluation variant.

---

## M2a: Baseline Evaluation Protocol

**Training:** M1-S model trained on MANE-derived labels. All 9 multimodal channels enabled (including junction as feature). Standard cross-entropy on `[L, 3]`.

**Evaluation:** Filter predictions at positions in (Ensembl \ MANE). Measure recall and precision at these alternative sites. Compare to base model (OpenSpliceAI) scores at the same positions.

**What it tests:** Can a model trained on MANE labels, augmented with multimodal features (junction support, RBP binding, conservation, epigenetics), correctly predict splice sites that exist in Ensembl but weren't in MANE training data?

**Strengths:** Clean, already scoped. Ensembl is well-curated, so the evaluation sites are high-quality positives. Easy to implement — just add site-filtering to the existing evaluation pipeline.

**Limitations:** Ensembl is conservative (~5-10 isoforms/gene). The alternative sites in (Ensembl \ MANE) are the "easy" alternatives — well-supported, moderate expression, likely detectable from sequence context alone. The more interesting alternatives (rare isoforms, tissue-restricted, low expression) aren't in Ensembl.

---

## M2b: Expanded Evaluation on GENCODE Comprehensive

**Training:** Same as M2a — M1-S model trained on MANE labels, full multimodal stack.

**Evaluation:** Replace Ensembl with GENCODE comprehensive v47 as the evaluation annotation. Filter at (GENCODE \ MANE) sites.

**What changes:** The evaluation candidate space grows substantially. GENCODE v47 has 10-20+ transcripts per protein-coding gene (vs. Ensembl's 5-10), including low-expression isoforms, retained introns, computationally predicted transcripts, and the 140K+ lncRNA transcripts added in v47. The alternative sites in (GENCODE \ MANE) are harder and more diverse than (Ensembl \ MANE).

**Why it matters:** This is the evaluation that actually stress-tests the meta-layer. If multimodal features (junction support, conservation, RBP binding) help at Ensembl-only sites, that's expected — those sites are well-supported. If multimodal features also help at GENCODE-only sites, where the annotation is noisier and the base model signal is weaker, that's the stronger claim.

**Tiered evaluation within GENCODE \ MANE:** Not all GENCODE-only sites are equal. A natural stratification:

```
Tier 1: GENCODE ∩ Ensembl, but not MANE
    → Well-supported alternatives (both projects agree)
    → Expect moderate base model scores, strong multimodal rescue

Tier 2: GENCODE-only (not in Ensembl or MANE)
    → Rare isoforms, computational predictions, lncRNA splice sites
    → Expect low base model scores, variable multimodal signal
    → The real test of whether the meta-layer adds value
```

Report recall separately per tier. The gap between Tier 1 recall and Tier 2 recall tells you how much of the meta-layer's performance depends on annotation confidence vs. genuine pattern recognition.

**Implementation:** Generate `splice_sites_enhanced.tsv` from GENCODE v47 comprehensive annotation (if not already available). Load alongside MANE in the evaluation script. Site-filtering logic is identical to M2a — just swap the annotation source.

**Engineering effort:** Minimal. If the GENCODE v47 GTF is downloaded and the `splice_sites_enhanced.tsv` generation pipeline works for Ensembl, it works for GENCODE. The evaluation script changes are a few lines of filtering. Estimate 1-2 weeks, dominated by data prep.

---

## M2c: Training on GENCODE Labels with Confidence Weighting

**Training:** Train on GENCODE comprehensive labels (not MANE). Use `build_splice_labels()` with GENCODE's `splice_sites_enhanced.tsv`, so the model sees all annotated splice sites including rare alternatives. Apply sample weighting: sites in MANE get weight 1.0, sites in (GENCODE ∩ Ensembl \ MANE) get weight 0.8, sites in (GENCODE-only) get weight 0.5.

**Evaluation:** Same as M2b — evaluate at (GENCODE \ MANE) and stratify by tier.

**What changes relative to M2a/b:** The model now trains on alternative sites, not just evaluates on them. The MANE-trained M1-S has never seen GENCODE-only splice sites as positive labels. M2c exposes the model to these sites during training, weighted by annotation confidence.

**The key question:** Does training on a richer label set improve recall at alternative sites, or does the label noise from GENCODE-only sites degrade precision?

**Architecture impact:** None. Labels are still `[L]` arrays (0=donor, 1=acceptor, 2=neither). The only change is that more positions get labeled as donor/acceptor (because GENCODE has more annotated splice sites). Sample weighting is applied in the loss function via `torch.nn.CrossEntropyLoss(weight=...)` at the sample level, or by passing per-position weights.

**Practical weighting scheme:**

```python
# For each position in the label array:
if position in MANE_sites:
    weight = 1.0          # canonical, high confidence
elif position in ENSEMBL_sites:
    weight = 0.8          # well-supported alternative
elif position in GENCODE_sites:
    weight = 0.5          # GENCODE-only, uncertain
else:
    weight = 1.0          # negative (neither), full weight
```

This is implementable as a per-position weight tensor `[L]` passed to the loss function. No change to `MetaSpliceModel` or `build_splice_labels()` — just generate weights from the annotation overlap.

**Risks:** GENCODE-only sites include noise. Some fraction are computational artifacts or very low-confidence predictions. Training on them (even downweighted) might teach the model to predict splice sites where none exist, hurting precision on canonical sites. Ablation against M2a/b is essential: does M2c maintain M1-level precision on MANE while improving recall on alternatives?

**Engineering effort:** Low. 2-3 weeks. Requires generating annotation overlap maps (which sites are in MANE, which in Ensembl, which GENCODE-only) and adding a weight tensor to the data loader. The training loop already supports weighted loss.

---

## M2d: Junction-Informed Soft Labels

**Training:** Instead of binary labels from annotation, construct continuous label targets that integrate annotation presence with empirical junction evidence.

The label for each position becomes:

```python
# Still produces [L] array, but values are float in [0, 1] instead of discrete {0, 1, 2}
# Wait — this breaks the [L, 3] classification target.
```

Actually, this needs careful framing. The model outputs `[L, 3]` and trains with cross-entropy against discrete labels `[L]` (class indices). Continuous labels would require either:

**Option A: Sample-weighted cross-entropy.** Keep discrete labels (donor/acceptor/neither). Assign a confidence weight per position based on junction evidence. A GENCODE-only site with strong GTEx junction support (PSI > 5% in ≥3 tissues) gets weight 0.9. A GENCODE-only site with no junction support gets weight 0.2. The model still trains on classification, but attends more to high-evidence sites.

This is a refinement of M2c's weighting — instead of weighting by annotation tier alone, weight by annotation tier × junction evidence.

**Option B: Label smoothing with evidence-based targets.** For positions labeled as donor, instead of a hard target [1, 0, 0], use a soft target [0.9, 0.05, 0.05] for high-evidence sites and [0.6, 0.2, 0.2] for low-evidence sites. This is compatible with `[L, 3]` cross-entropy (use KL-divergence loss against soft distributions).

**The weighting function:**

```python
def compute_position_weight(
    annotation_tier: int,       # 0=MANE, 1=GENCODE∩Ensembl, 2=GENCODE-only
    junction_psi: float,        # max PSI across GTEx tissues, 0 if no support
    junction_breadth: int,      # number of tissues with PSI > 1%
) -> float:
    tier_prior = {0: 1.0, 1: 0.85, 2: 0.4}[annotation_tier]
    junction_signal = min(junction_psi / 50.0, 1.0)  # saturate at PSI=50%
    breadth_signal = min(junction_breadth / 10.0, 1.0)  # saturate at 10 tissues
    return tier_prior * 0.4 + junction_signal * 0.35 + breadth_signal * 0.25
```

**What this captures:** A GENCODE-only site with zero junction support gets weight ~0.16 (barely contributes to gradient). The same site with PSI=25% in 5 tissues gets weight ~0.58 (real signal, moderate confidence). A MANE site always gets weight ~0.80+ regardless of junction status (it's canonical).

**Advantages over M2c:** Junction evidence is empirical, not annotation-derived. It directly measures whether the spliceosome acts at this position. Weighting by junction support grounds the label confidence in biology rather than in which annotation pipeline flagged the site.

**Risks:** GTEx coverage is uneven. A site active in pancreatic islets or spleen may have no GTEx junction support — not because it's false, but because GTEx undersampled that tissue. The weighting function penalizes these sites unfairly. Mitigation: include a floor weight for all annotated sites (no position gets weight < 0.15), and document the GTEx tissue coverage bias.

**Relationship to junction-as-feature:** In M2, junction reads are input features (channels 6-7 of the multimodal stack). In M2d training, junction evidence also informs the label weights. These are different uses — the feature says "there's junction support here" at inference time, while the weight says "how much should the model trust this label" at training time. They don't create a circularity because the weight affects gradient magnitude, not feature values.

**Engineering effort:** Low-medium. 2-4 weeks. Requires mapping junction data to annotation sites (BED intersection), computing PSI and breadth per site, and modifying the data loader to pass per-position weights. The junction data is already available (GTEx v8, 353K junctions, 54 tissues).

---

## M2e: Tissue-Conditioned Input

**Training:** Add tissue context as an input conditioning signal. The model still outputs `[L, 3]`, but its predictions are influenced by which tissue context is provided.

Two implementation approaches, both preserving `[L, 3]` output:

**Approach 1: Tissue embedding as additional input channel.** Add a tissue identity as a dense channel (one-hot over 54 tissues, projected to a learned embedding of dimension d=8 or 16, then broadcast to [B, d, L]). Concatenate with the existing multimodal features in Stream B. The signal encoder input grows from `[B, 3+C, L]` to `[B, 3+C+d, L]`. Output remains `[L, 3]`.

**Approach 2: Splicing factor expression conditioning (Splice Ninja-style).** Instead of a discrete tissue ID, condition on a continuous vector of 301 splicing factor expression levels for the target tissue. This generalizes to unseen tissues — if you know the RBP expression profile, you can predict splicing without having trained on that tissue.

**Labels:** Per-tissue labels from GTEx. For each (gene, tissue) pair, `build_splice_labels()` uses only the splice sites with junction support in that tissue. A site active in liver gets labeled as donor in the liver-conditioned training example, but as "neither" in the brain-conditioned example for the same gene.

**What this does to M2 evaluation:** M2e enables tissue-specific recall. Instead of asking "can the model detect Ensembl-only sites?", ask "can the model detect sites active in liver but not brain?" Evaluation filters by (tissue-specific junction evidence) rather than (annotation set difference).

**Why this is the long-term differentiator:** AlphaGenome (Nature Jan 2026) predicts splice site usage from 1Mb sequence but has no tissue conditioning — it predicts a single usage level per site, not per-tissue. Splice Ninja (bioRxiv Jan 2026) conditions on splicing factor expression but doesn't incorporate multimodal regulatory features (conservation, RBP binding, chromatin accessibility) beyond the factors themselves. M2e would combine:

- Sequence context (Stream A, dilated CNN)
- Base model scores (from OpenSpliceAI, already captures sequence-level splicing rules)
- Multimodal regulatory features (conservation, epigenetic marks, RBP binding, chromatin)
- Tissue conditioning (either discrete or factor-expression-based)

This combination is novel. It separates what the sequence says (base model), what the regulatory landscape says (multimodal features), and what the cellular context says (tissue conditioning). The meta-learning framing — refine base predictions using context — is architecturally distinct from end-to-end models like AlphaGenome.

**Architecture impact:** Moderate. Adding an input channel doesn't change the output protocol. If tissue conditioning is implemented as an additional channel in Stream B, the only change is `mm_channels` in `MetaSpliceConfig` (from 9 to 9+d). If implemented as Splice Ninja-style factor expression conditioning, it requires a small embedding network to project 301 factors → d dimensions, then broadcast.

**Data pipeline impact:** Significant. Currently, `build_splice_labels()` takes a single `splice_sites_df` and unions across all transcripts. For tissue-conditioned training, you need per-tissue label arrays: only include splice sites with junction support in tissue t. This requires:

1. Mapping GTEx junctions to splice sites (BED intersection by tissue)
2. Generating per-tissue `splice_sites_enhanced.tsv` (or equivalent)
3. Modifying `SequenceLevelDataset` to sample (gene, tissue) pairs instead of just genes
4. Handling tissue imbalance (brain regions dominate GTEx; rare tissues have sparse data)

**Risks:**
- Data sparsity for rare tissues. GTEx has 54 tissues but uneven depth. If a tissue has <100 supported junctions across the genome, predictions for that tissue are unreliable.
- Training set explosion. With 54 tissues × ~20K genes, the effective training set is ~1M examples. Each requires generating tissue-specific labels. Caching becomes critical.
- Generalization to unseen tissues. The discrete tissue embedding approach can't predict for a tissue not in training. The factor expression approach can, but requires splicing factor expression data for the target tissue.

**Engineering effort:** High. 2-3 months. Dominated by data pipeline work (tissue-specific labels, caching strategy, factor expression integration) rather than model changes.

**Publication angle:** "Multimodal Meta-Learning for Tissue-Specific Splice Site Prediction." The pitch: we show that augmenting a canonical splice site predictor with a tissue-conditioned meta-layer, informed by conservation, RBP binding, and chromatin state, improves tissue-specific recall at alternative splice sites by X% over the unconditioned model and Y% over AlphaGenome's tissue-agnostic predictions. Validate on GTEx held-out tissues and clinically relevant tissue-specific variants.

---

## M2f: PU Learning on Annotation as Incomplete Positive Set

**Training:** Treat annotated splice sites (from any source) as an incomplete positive set. Unannotated positions are unlabeled, not confirmed negatives.

Standard PU learning frameworks (Kiryo et al. 2017, unbiased PU risk estimator) modify the loss function to account for the fact that negatives in the training set include hidden positives. The model output is still `[L, 3]` — the PU framework changes how loss is computed, not what the model predicts.

**What it assumes:** GENCODE (or any annotation) is a biased sample of the true splice site landscape. Highly expressed genes are over-annotated. Rare tissue-specific sites are under-annotated. The "neither" class in the training labels contains real splice sites that haven't been cataloged.

**Implementation:** Replace `CrossEntropyLoss` with a PU-aware loss. For the donor and acceptor classes, apply the unbiased PU risk:

```python
# Simplified PU risk for class k (donor or acceptor):
# R_pu(f) = π_k * R+(f) + max(0, R_u(f) - π_k * R+(f))
# where:
#   π_k = class prior (fraction of true positives among all positions)
#   R+(f) = risk on labeled positives
#   R_u(f) = risk on unlabeled data
```

The class prior π_k is the main hyperparameter. For donor sites, π is approximately (number of annotated donors) / (total positions). But this is a lower bound — the true π is higher because of unannotated sites.

**Advantages:** Principled. Doesn't require choosing annotation tiers or weighting schemes. The framework handles label incompleteness mathematically rather than heuristically.

**Risks:** Class prior estimation is notoriously difficult in genomics. The observation mechanism (which sites get annotated) is not random — it's biased toward highly expressed, well-studied genes. Standard PU prior estimators (Elkan & Noto 2008) assume random sampling, which doesn't hold. Mis-estimated priors cause systematic over- or under-prediction.

Additionally, the `[L, 3]` multiclass setting complicates PU learning. Most PU frameworks are designed for binary classification. With three classes (donor, acceptor, neither), you need either a one-vs-rest PU formulation (treat each splice type independently) or a multiclass PU extension, which is less mature.

**Where this belongs:** M2f has intellectual appeal but fits better at M3 (novel splice site prediction), where the model explicitly discovers unannotated sites. M3's framing — junction as target, predict whether a position has RNA-seq support without seeing junction data — is already a form of positive-unlabeled reasoning. Adding a formal PU loss to M3 is natural. Grafting it onto M2 (where junction data is available as input) is less well-motivated: if you have junction evidence, use it directly (M2d weighting) rather than treating the problem as PU.

**Engineering effort:** Medium. 2-3 weeks for the PU loss implementation, plus hyperparameter tuning for class priors. But debugging PU learning failures is time-consuming — the class prior sensitivity means small errors in estimation cascade into large performance gaps.

---

## Comparative Analysis

| Dimension | M2a | M2b | M2c | M2d | M2e | M2f |
|---|---|---|---|---|---|---|
| **What changes** | Eval set only | Eval set (bigger) | Training labels + weights | Training weights (junction) | Input conditioning | Loss function |
| **Annotation source** | Ensembl eval | GENCODE eval | GENCODE train | GENCODE train | GTEx per-tissue | Any |
| **Architecture change** | None | None | None | None | +tissue channels | None |
| **Output format** | [L, 3] | [L, 3] | [L, 3] | [L, 3] | [L, 3] | [L, 3] |
| **Label type** | Binary (class index) | Binary (class index) | Weighted binary | Weighted binary | Per-tissue binary | PU binary |
| **Engineering effort** | 1 week | 1-2 weeks | 2-3 weeks | 2-4 weeks | 2-3 months | 2-3 weeks |
| **Data pipeline change** | Eval script only | Eval script + GENCODE TSV | + weight tensor | + junction weights | Major (per-tissue labels) | Loss function only |
| **Risk** | Low | Low | Low-medium | Medium | High | High |
| **SoA differentiation** | Incremental | Incremental | Methodological | Methodological | Novel combination | Known technique |
| **Publication standalone** | No | No | Weak | Moderate | Strong | Weak |

---

## Strategic Verdict

### Ordering

**1. M2a → M2b (immediate, 1-2 weeks).** These are pure evaluation protocol changes. No training modification. Run the existing M1-S model, evaluate on Ensembl \ MANE (M2a), then on GENCODE \ MANE (M2b). This gives you the baseline: how well does the current model already generalize to alternative sites? If M1-S recall at Ensembl-only sites is already high (>80%), the meta-layer's multimodal features are working. If recall at GENCODE-only sites drops sharply, there's room for training protocol improvements.

**2. M2c (next, 2-3 weeks).** If M2b reveals weak recall at GENCODE-only sites, the next question is whether training on GENCODE labels helps. M2c trains on GENCODE-derived labels with confidence weighting by annotation tier. This is the smallest training change that expands the model's exposure to alternative sites. Go/no-go: if M2c recall on Tier 2 sites improves >5% over M2a/b (evaluated on the same GENCODE sites), the richer labels are helping.

**3. M2d (refinement, 2-4 weeks if M2c works).** If M2c shows that GENCODE training helps but Tier 2 sites remain noisy, junction weighting is the natural fix. M2d uses empirical junction evidence to modulate label confidence, grounding weights in biology rather than annotation provenance. This is a refinement of M2c, not an independent formulation — deploy it only if M2c's annotation-only weighting proves insufficient.

**4. M2e (long-term differentiator, 2-3 months).** Tissue conditioning is the formulation with the highest publication impact and the clearest differentiation from AlphaGenome and Splice Ninja. But it requires significant data pipeline engineering. Start prototyping in parallel with M2c/M2d, but don't let it block the M2a→M2b→M2c progression. Target a 5-10 tissue subset first (liver, brain cortex, heart, skeletal muscle, whole blood — well-sampled in GTEx) before scaling to 54.

**5. M2f (defer to M3).** PU learning is intellectually elegant but belongs at M3 where the model explicitly reasons about novel sites without junction input. For M2, where junction evidence is available as both feature and label validation, direct junction weighting (M2d) is more practical.

### The Most Publishable Contribution

M2e (tissue-conditioned meta-layer) is the publication target. The story: a meta-learning framework that combines canonical splice site predictions with tissue-specific regulatory context to predict context-dependent alternative splicing. Validated against AlphaGenome (no tissue input) and Splice Ninja (no multimodal regulatory features). The M2a→M2c progression provides ablation evidence, and M2e provides the punchline.

### The Highest Risk / Highest Reward

Also M2e. The data engineering is non-trivial and the tissue imbalance problem is real. If it works, it's novel. If it doesn't, you still have M2c/M2d as publishable increments over the baseline.

### The Most Immediately Useful

M2b. It costs almost nothing (evaluation filtering only), reveals how well the current model generalizes, and informs every subsequent decision. Run M2b first.

---

## Implementation Notes

**For M2a/M2b (evaluation only):**
- Add to the planned `08_evaluate_sequence_model.py`
- Load two annotation sources: training annotation (MANE) and evaluation annotation (Ensembl or GENCODE)
- Site-difference filtering: `eval_sites = ensembl_sites[~ensembl_sites.index.isin(mane_sites.index)]`
- Report: overall recall/precision, recall at alternative sites, stratified by tier if using GENCODE

**For M2c (confidence-weighted training):**
- Generate annotation overlap: for each position in GENCODE, flag whether it's also in MANE, also in Ensembl
- Store as an additional column in `splice_sites_enhanced.tsv` or as a separate weight file
- Modify `SequenceLevelDataset.__getitem__()` to return per-position weights alongside labels
- Pass weights to `CrossEntropyLoss(reduction='none')`, multiply by weights, then mean

**For M2d (junction-weighted):**
- Extend M2c weights with GTEx junction lookup: for each annotated site, query junction PSI and tissue breadth
- Compute composite weight = f(tier, psi, breadth) as described above
- Same data loader modification as M2c, just different weight values

**For M2e (tissue conditioning):**
- Start with discrete tissue embedding: add `tissue_id` field to dataset, project to d=16 via `nn.Embedding(54, 16)`
- Broadcast embedding to `[B, 16, L]`, concatenate with mm_features in Stream B
- Update `MetaSpliceConfig(mm_channels=9+16)` → no other architectural change needed
- Per-tissue labels: filter `splice_sites_enhanced.tsv` by junction support in tissue t
- Prototype on 5 tissues first; validate that tissue-specific recall exceeds the unconditioned model

**For all variants:**
- Freeze base model (OpenSpliceAI) weights. M2 only trains the meta-layer.
- Maintain separate test sets: canonical (MANE) for regression testing, alternative (GENCODE \ MANE) for M2-specific metrics.
- Log blend_alpha (the learnable residual mixing parameter) during training — it indicates how much the meta-layer deviates from the base model. If α→0 (sigmoid→0.5), the meta-layer contributes nothing. If α→1, it's fully overriding base scores. For alternative sites, expect α to increase during training as the meta-layer learns to correct base model weaknesses.

---

## Conclusion

M2 is not a model. It's a family of training and evaluation protocols applied to the same `[L, 3]` architecture. The progression from M2a (evaluation only) through M2e (tissue conditioning) represents increasing investment in label quality and context richness, with each step building on the previous one's empirical results.

The critical insight from Barnett's original formulation stands: the set difference (richer annotation \ MANE) defines the evaluation frontier. What changes across variants is how aggressively the training protocol exploits that richer annotation — from not at all (M2a/b, pure eval) to fully (M2e, tissue-conditioned labels).

Build from M2a outward. Let the data tell you when to stop.
