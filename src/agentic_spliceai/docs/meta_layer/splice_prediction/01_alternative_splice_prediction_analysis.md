# Alternative Splice Site Prediction — Deep Analysis

**Created**: March 2026
**Context**: Review of meta-layer approaches for predicting alternative splice sites induced by genetic variants and other perturbations

---

## Table of Contents

- [[#Prerequisite — SpliceVarDB and Trainable Signals]]
- [[#Approach 1 — Canonical Classification (Experiment 001)]]
- [[#Approach 2 — Paired Delta Prediction (Experiment 002)]]
- [[#Approach 3 — Binary Classification (Experiment 003)]]
- [[#Approach 4 — Validated Delta Prediction (Experiment 004)]]
- [[#What SpliceVarDB Actually Tells Us (and What It Doesn't)]]
- [[#Why This Is Not a Standard Supervised Learning Problem]]
- [[#Systematic ML Formulations]]
- [[#Recommendations and Roadmap]]

---

## Prerequisite — SpliceVarDB and Trainable Signals

Before dissecting the models, it's worth framing what "alternative splice site" means as a learning problem and what SpliceVarDB gives us.

### What SpliceVarDB provides

SpliceVarDB is a curated database of genetic variants with experimentally validated effects on splicing. Each record carries:

- **Genomic coordinates** (hg19 and hg38), parsed from strings like `"1-100107682-T-C"`
- **Classification** — one of three labels:
  - `"Splice-altering"` — experimentally confirmed to change splicing
  - `"Non-splice-altering"` — confirmed NOT to change splicing
  - `"Low-frequency"` — uncertain / insufficient evidence
- **Location** — exonic or intronic
- **Method** — the experimental validation technique used

> [!info] Implementation Reference
> See `meta_layer/data/splicevardb_loader.py` — `VariantRecord` dataclass (lines 27–58) and `SpliceVarDBLoader` class (lines 61–308)

### The labeling gap

The fundamental challenge is that base models (SpliceAI, OpenSpliceAI) are trained on **canonical** splice sites from GTF annotations — the known GT-AG sites at exon-intron boundaries. They have no training signal for:

1. Whether a **single nucleotide variant** activates or destroys a splice site
2. Whether a cryptic splice site emerges from a mutation
3. The **magnitude** of the effect (how much does donor/acceptor probability shift?)

SpliceVarDB fills this gap, but it provides a **categorical** label (altering vs. not), while what we actually want to predict is a **continuous delta** (how much do the scores change?). This mismatch between available labels and desired output shapes the entire experimental trajectory.

---

## Approach 1 — Canonical Classification (Experiment 001)

> [!summary] Outcome
> 99.11% classification accuracy — but only 17% variant detection rate. **Failed for the actual goal.**

### Label Preparation

Labels come directly from **GTF annotations** loaded via `ArtifactLoader`:

```python
# dataset.py:333-337
def _encode_label(self, splice_type: str) -> int:
    return LABEL_ENCODING.get(splice_type.lower(), LABEL_ENCODING['neither'])
```

where `LABEL_ENCODING = {'donor': 0, 'acceptor': 1, 'neither': 2, '': 2}` (defined in `feature_schema.py:196-201`).

Each sample is a **501nt window** centered on a genomic position. If that position is an annotated donor site, label = 0; if annotated acceptor, label = 1; otherwise label = 2. Classes are balanced via undersampling (`dataset.py:533-567`).

The 50+ numeric features are z-score normalized (`dataset.py:230-242`), and known leakage columns (`splice_type`, `pred_type`, `is_correct`, `error_type`, etc.) are explicitly excluded via `FeatureSchema.LEAKAGE_COLS`.

### Optimization Objective

**Cross-entropy with label smoothing (0.1) and inverse-frequency class weights**:

```python
# trainer.py:403-428
loss = F.cross_entropy(logits, labels, weight=self.class_weights, label_smoothing=0.1)
```

Class weights are computed as `total / (3 * count_per_class)`, standard inverse-frequency balancing.

### Architecture

`MetaSpliceModel` (`meta_splice_model.py:21-157`) is a two-stream multimodal network:

```
Sequence [B, 4, 501] ──→ CNN (kernels 3/5/7/11/15) ──→ GlobalAvgPool ──→ [B, 256]
                                                                                    ├── Fusion ──→ Classifier ──→ [B, 3]
Scores [B, ~50]       ──→ MLP (ScoreEncoder)         ──→                   [B, 256]
```

**Components:**
1. **Sequence stream**: CNN encoder processes `[B, 4, 501]` one-hot DNA → `[B, 256]` embedding via multi-scale convolutions with global average pooling
2. **Score stream**: MLP (`ScoreEncoder`) processes `[B, ~50]` numeric features → `[B, 256]`
3. **Fusion**: Concatenation (or optionally `CrossAttentionFusion` which does bidirectional attention between the two streams with residual connections)
4. **Head**: `Linear(512→256) → LayerNorm → GELU → Dropout → Linear(256→3)` producing class logits

**Training**: AdamW (lr=1e-4, weight_decay=0.01), cosine annealing LR, gradient clipping at 1.0, early stopping on `val_pr_auc` with patience=10.

### Assessment

> [!warning] Logically flawed for the stated goal
> This approach is logically flawed for variant detection, and the results confirm it.

The reason is a **distribution mismatch**: the model learns `P(donor | sequence, scores)` for canonical positions — a near-trivial task since the base model scores already encode this information with high accuracy. When asked to detect *changes* caused by variants, it has never seen variant-perturbed inputs during training. Worse, the label-smoothed cross-entropy objective encourages confident canonical predictions, actively suppressing sensitivity to subtle score perturbations that would signal variant effects.

**However**, this experiment is not useless. It established that:
- The multimodal fusion architecture works mechanically
- Leakage detection infrastructure is sound
- The feature schema is validated

It serves as an upper bound on "how well can we classify known sites" and a proof that **classification of known sites is the wrong proxy task** for variant detection.

---

## Approach 2 — Paired Delta Prediction (Experiment 002)

> [!summary] Outcome
> Best correlation: **r=0.38** (Gated CNN + Quantile loss). Moderate but insufficient.

### Label Preparation

Labels are now **continuous deltas** derived by running the base model on both reference and alternate sequences:

```python
# variant_dataset.py:227-364
delta = alt_scores - ref_scores  # [L, 3]
# Scan ±50bp window around variant for max absolute effect
max_donor_delta = donor_deltas[np.abs(donor_deltas).argmax()]
max_acceptor_delta = acceptor_deltas[np.abs(acceptor_deltas).argmax()]
```

Each sample is a `VariantSample` containing:
- `ref_sequence` and `alt_sequence` (501nt each, centered on variant)
- `delta_donor` and `delta_acceptor` (scalar: max absolute delta in ±50bp)
- `weight`: 2.0 for splice-altering, 1.0 for normal

> [!important] Critical limitation
> The target is the **base model's own prediction difference**, not ground truth.

### Optimization Objective

Multiple losses were tested (`delta_predictor_calibrated.py`):

| Loss | Mechanism | Result |
|------|-----------|--------|
| **Weighted MSE** | `per_sample_MSE * class_weight`, averaged | Baseline |
| **Quantile/Pinball (τ=0.9)** | Underpredictions penalized 9x more than overpredictions | **Best (r=0.38)** |
| **Output scaling** | Learnable multiplicative constant `delta * scale` | r=0.22 |
| **Temperature scaling** | `delta / exp(log_temperature)` | r=-0.03 |
| **Hybrid classification + regression** | Multi-task BCE + MSE | r=-0.07 |

The quantile loss at τ=0.9 is noteworthy:

```python
# delta_predictor_calibrated.py:226-233
loss = torch.where(
    error >= 0,
    tau * error,        # underprediction: penalized by τ=0.9
    (tau - 1) * error   # overprediction: penalized by 0.1
)
```

This forces the model to focus on capturing large deltas rather than regressing to the mean — important because most variants have near-zero delta.

### Architecture

**Siamese network** (`delta_predictor.py:35-169`):

```
ref_seq [B,4,501] ──→ SharedEncoder ──→ ref_emb [B,256]
                                                          diff = alt - ref ──→ DeltaHead ──→ [B,2]
alt_seq [B,4,501] ──→ SharedEncoder ──→ alt_emb [B,256]
```

- **SharedEncoder** (`delta_predictor.py:172-248`): 4 conv layers with exponentially dilated convolutions (dilation 1, 2, 4, 8), BatchNorm + ReLU, global average pooling, linear projection. Kaiming initialization on conv weights, Xavier on linear.
- **DeltaHead**: `Linear(256→256) → ReLU → Dropout → Linear(256→128) → ReLU → Dropout → Linear(128→2)`

The best variant uses `SimpleCNNDeltaPredictor` (a GatedCNN) wrapped in `QuantileDeltaPredictor`.

### Assessment

> [!note] Right intuition, poisoned target
> The Siamese architecture is the right geometric intuition — but the target is poisoned.

By learning `base_model(alt) - base_model(ref)`, you can never exceed the base model's accuracy. For non-splice-altering variants where the base model falsely predicts a delta, you're training on noise. This ceiling is visible in the r=0.38 plateau.

**What did help**:
- **Gated CNN with dilated convolutions** (r=0.36 vs. r=-0.04 for simple CNN)
- **Quantile loss at τ=0.9** (r=0.38 vs r=0.36) — biasing toward the upper quantile forces the model to preserve signal for rare large-effect variants rather than predicting zero for everything

**What failed**:
- **Multi-task** (classification + regression hybrid, r=-0.07) — the two tasks compete for representation capacity, and the classification head's gradient overwhelms the regression signal
- **Temperature scaling** (r=-0.03) — adjusting a global scalar can't fix position-dependent errors

> [!tip] Key lesson
> **Target quality is the bottleneck, not architecture or loss function.**

---

## Approach 3 — Binary Classification (Experiment 003)

> [!summary] Outcome
> AUC=0.61, F1=0.53. Better than random but not practically useful. Needs F1 > 0.7.

### Label Preparation

Labels are SpliceVarDB's **categorical classifications** used directly:

- Input: `alt_seq [B, 4, 501]` + `ref_base [B, 4]` + `alt_base [B, 4]` (one-hot encoded)
- Label: binary 0/1 from `variant.classification == "Splice-altering"`
- Balanced: 50/50 splice-altering vs. normal

This is the first approach to use **ground truth labels from SpliceVarDB** for training, not just evaluation.

### Optimization Objective

**Binary cross-entropy** with sigmoid output:

```python
# splice_classifier.py:253-255
return torch.sigmoid(logits)  # P(splice-altering) ∈ [0,1]
```

For the `UnifiedSpliceClassifier` multi-task variant:

```python
total_loss = binary_weight * BCE(p_splice_altering, label)
           + effect_weight * CE(effect_logits, effect_type)
```

with `binary_weight=1.0` and `effect_weight=0.5`.

### Architecture

**`SpliceInducingClassifier`** (`splice_classifier.py:165-275`):

```
alt_seq [B, 4, 501]
  ↓
GatedCNNEncoder (6 layers, 128 hidden)
  ├── Conv1d(4 → 128, k=1) initial projection
  ├── 6x GatedResidualBlock:
  │     Conv1d(128 → 256, k=15, dilation=2^(i%4))
  │     Split → [content, gate]
  │     out = content * sigmoid(gate)
  │     LayerNorm → Dropout → residual add
  ├── AdaptiveAvgPool1d(1) → [B, 128]
  ↓
seq_features [B, 128]

ref_base [B, 4] ┐
                 cat → [B, 8] → Linear(8→128) → ReLU → Dropout → Linear(128→128)
alt_base [B, 4] ┘
  ↓
var_features [B, 128]

[seq_features, var_features] → cat → [B, 256]
  ↓
Linear(256→128) → ReLU → Dropout → Linear(128→1) → sigmoid
  ↓
P(splice-altering) [B, 1]
```

The `GatedResidualBlock` is the workhorse — dilations cycle through 1, 2, 4, 8, 1, 2 giving a receptive field of ~465bp over 6 layers with kernel size 15. The gating mechanism (`out * sigmoid(gate)`) allows the network to selectively pass or suppress information at each position, similar to LSTM gating but applied spatially.

### Assessment

> [!note] Right question, insufficient input
> This is the right question to ask, but the sequence-only input may be insufficient to answer it.

AUC=0.61 (vs. 0.5 random) confirms there **is** a learnable signal — the 501nt context around a variant carries some information about whether it will affect splicing. But F1=0.53 means the model is barely better than a coin flip in practice.

**Why is the signal so weak?**

1. **501nt is too narrow.** Many splice-altering variants work by disrupting branch point sequences, exonic splicing enhancers/silencers, or regulatory elements that can be hundreds of nucleotides away. A 501nt window may simply not contain the relevant context.
2. **No base model scores as input.** Unlike Approach 1 which had 50+ precomputed features, this model only sees raw sequence + a 2-base variant description. It must independently learn the splice code from scratch.
3. **The variant embedding is minimal.** Concatenating two 4-dimensional one-hot vectors (ref + alt base) and projecting to 128 dimensions throws away positional context. The model knows *what* the mutation is but not *where* it sits relative to splice site motifs.

The **multi-step framework (Steps 2–4)** was never tested because the prerequisite F1 > 0.7 for Step 1 was not met. This was a sound gating decision.

---

## Approach 4 — Validated Delta Prediction (Experiment 004)

> [!summary] Outcome
> **r=0.41 (p=1.4e-07)** — best so far. +8% improvement over paired prediction.

### Label Preparation

This is where the experimental design gets most interesting. The **hybrid labeling strategy**:

```
If SpliceVarDB says "Splice-altering":
    target = base_model(alt) - base_model(ref)   # Trust the delta direction

If SpliceVarDB says "Normal":
    target = [0, 0, 0]                           # Override: no effect, period

If SpliceVarDB says "Low-frequency" or "Conflicting":
    SKIP                                          # Don't train on uncertainty
```

> [!tip] Key insight
> This is a **hybrid labeling strategy**: it uses the base model's continuous delta for confirmed splice-altering variants (where the base model is likely directionally correct, even if not calibrated), but overrides the base model with ground truth zeros for confirmed non-altering variants (where the base model may produce spurious deltas).

**Data**: ~2000 samples (1000 SA, 1000 Normal), balanced, train/test split by chromosome (1-20 train, 21-22 test).

### Optimization Objective

**MSE** against the validated delta targets, trained with AdamW (lr=5e-5, weight_decay=0.02) and OneCycleLR scheduler (max_lr=5e-4). 40 epochs, batch_size=32.

The delta head outputs raw values (no activation function on the final layer):

```python
# validated_delta_predictor.py:206-213
self.delta_head = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(hidden_dim // 2, 3)  # [Δ_donor, Δ_acceptor, Δ_neither]
)
```

### Architecture

**`ValidatedDeltaPredictor`** (`validated_delta_predictor.py:144-256`, ~3M parameters):

```
alt_seq [B, 4, 501]
  ↓
GatedCNNEncoder (6 layers, 128 hidden)
  ├── Conv1d(4 → 128, k=1)
  ├── 6x GatedResidualBlock(k=15, dilation=1,2,4,8,1,2)
  ├── AdaptiveAvgPool1d(1)
  ↓
seq_features [B, 128]

ref_base [B, 4] ┐
                 cat → [B, 8] → Linear(8→128) → ReLU → Dropout → Linear(128→128)
alt_base [B, 4] ┘
  ↓
var_features [B, 128]

[seq_features, var_features] → cat → [B, 256]
  ↓
Linear(256→128) → ReLU → Dropout → Linear(128→64) → ReLU → Linear(64→3)
  ↓
Δ = [Δ_donor, Δ_acceptor, Δ_neither]
```

Optionally includes base model scores as additional input (`include_base_scores=True` adds 3 dims to the variant input, enabling residual learning on top of the base model).

An attention variant (`ValidatedDeltaPredictorWithAttention`) replaces global average pooling with learned position attention:

```python
attn_logits = self.position_attention(per_pos_features).squeeze(-1)  # [B, L]
attention = F.softmax(attn_logits, dim=-1)
seq_features = torch.einsum('bl,blh->bh', attention, per_pos_features)  # weighted pool
```

### Assessment

> [!note] Advantage is in label engineering, not architecture
> The model architecture is nearly identical to Approach 3's `SpliceInducingClassifier`. The improvement comes almost entirely from cleaning up the training targets.

**Why validated targets work — asymmetric trust:**

For splice-altering variants, the base model's delta is a reasonable (if noisy) target — the direction is probably right even if the magnitude is off. For non-splice-altering variants, the base model's delta is unreliable noise — any predicted delta is a false positive by definition. By forcing zeros for the non-SA class, you remove approximately half the noise from the training signal.

> [!warning] Concerns
> 1. **r=0.41 is measured on SA samples only.** The model's ability to discriminate SA from non-SA (ROC-AUC=0.58) is barely better than Approach 3 (AUC=0.61).
> 2. **False negative blindness.** For SA variants where the base model predicts near-zero delta (false negatives), the validated target still uses `base_model(alt) - base_model(ref) ≈ 0`. The meta-layer has no mechanism to "discover" effects the base model completely misses.
> 3. **2000 samples is very small.** The full SpliceVarDB has ~50K variants. The r=0.41 should be taken with appropriate uncertainty bounds.
> 4. **At inference**, `final = base_scores + Δ`. This means the meta-layer is a residual correction — it can sharpen or attenuate base model predictions but fundamentally cannot detect novel splice sites the base model scores as zero.

---

## What SpliceVarDB Actually Tells Us (and What It Doesn't)

### What it provides per record

| Field | Example | Available? |
|-------|---------|------------|
| Genomic locus + variant | `chr1:100107682 T→C` | Yes |
| "Does it affect splicing?" | `Splice-altering` | Yes |
| Location (exonic/intronic) | `Intronic` | Yes |
| Experimental method | varies | Yes |
| **Where** the affected splice site is | *unknown* | **No** |
| **Gain or loss?** | *unknown* | **No** |
| **Donor or acceptor?** | *unknown* | **No** |
| **Magnitude** of the effect | *unknown* | **No** |
| **Resulting transcript** | *unknown* | **No** |

### What the variant might actually do

Knowing that `chr1:100107682 T→C` is "splice-altering" tells us *something happens to splicing* but not *what, where, or how much*. The variant might:

1. **Directly destroy** a canonical GT/AG dinucleotide (trivial case)
2. **Weaken** a splice site by disrupting the surrounding consensus motif
3. **Create a new cryptic splice site** nearby
4. **Disrupt an exonic splicing enhancer (ESE)**, causing exon skipping at a site hundreds of bp away
5. **Alter a branch point sequence**, changing acceptor site selection
6. **Modify RNA secondary structure**, exposing or hiding a splice site

Cases 1-2 are local and detectable within 501nt. Cases 3-6 may require much longer range context and are the truly "alternative" splice events that current approaches struggle with.

---

## Why This Is Not a Standard Supervised Learning Problem

### The label hierarchy

```
Level 0: Genome-wide       "Which positions are splice sites?"
                            → Labels: GTF annotations (complete for canonical sites)
                            → Supervision: Strong, but STATIC (reference genome only)

Level 1: Variant-level      "Does this variant affect splicing?"
                            → Labels: SpliceVarDB (binary, ~50K variants)
                            → Supervision: Weak — tells us IF, not WHERE/HOW

Level 2: Effect type        "Gain or loss? Donor or acceptor?"
                            → Labels: NOT in SpliceVarDB directly
                            → Can be partially INFERRED from base model deltas
                            → Supervision: Inferred, noisy

Level 3: Position-level     "At which position does the new/lost splice site occur?"
                            → Labels: Almost completely ABSENT
                            → Supervision: None (this is what we want to predict)

Level 4: Magnitude          "How strong is the effect (PSI change)?"
                            → Labels: Available in RNA-seq (not currently used)
                            → Supervision: Available but requires different data source
```

> [!important] The fundamental gap
> **We want to predict at Level 3 (positional) but our best labels exist at Level 1 (binary variant-level).** The current approaches try to bridge this gap through the base model's delta scores, which provide a noisy Level 3 signal — but only for effects the base model already partially detects.

---

## Systematic ML Formulations

### Formulation A: Conditional Splice Landscape Prediction

The most natural formulation. Instead of predicting "is this variant splice-altering?", predict the **full per-position splice site probability map conditioned on a perturbation**:

```
f(sequence, perturbation) → P(donor_i), P(acceptor_i)  for all positions i
```

where `perturbation` can be:
- A variant (SNV, indel)
- An epigenetic mark at position j
- A splicing factor binding event
- A condition label (stress, disease state)

**The delta** then falls out naturally:

```
Δ_i = f(sequence, perturbation)_i - f(sequence, null)_i
```

**Why this is better than the current approach**: The current system predicts deltas directly, which means it can only learn *corrections* to the base model. Formulation A predicts the full landscape, meaning it can discover splice sites the base model never scored. The delta is derived, not directly predicted.

**Training signal**:
- For `perturbation = null` (reference genome): Strong supervision from GTF annotations at Level 0
- For `perturbation = variant`: Weak supervision from SpliceVarDB at Level 1 — we know the landscape *should change* but not *where*

**Bridging the label gap (Levels 1→3)**:
1. **Consistency constraints**: For non-SA variants, the predicted landscape should be identical to reference: `f(seq, variant) ≈ f(seq, null)`. This is a self-supervised signal.
2. **Sparsity priors**: For SA variants, the difference `Δ` should be sparse (only a few positions change). Enforce with L1 regularization on the delta map.
3. **Motif constraints**: New splice sites should conform to known motifs (GT for donor, AG for acceptor, polypyrimidine tract for branch points). Physics-informed prior.

### Formulation B: Latent Splice State Model

Think of each position as having a latent "splice competence" that is modulated by context:

```
z_i = encoder(sequence_context_around_i)           # latent splice competence
s_i = modulator(z_i, perturbation)                  # perturbed state
P(donor_i), P(acceptor_i) = decoder(s_i)           # observed probabilities
```

> [!tip] Key insight
> The encoder learns position-level representations from canonical sites (Level 0, abundant labels), and the modulator learns how perturbations shift these representations (Level 1, scarce labels). This **separates "understanding splice sites" from "understanding perturbation effects"** — two very different learning problems with very different amounts of supervision.

**Training**:
- **Phase 1**: Train encoder + decoder on canonical splice sites (GTF labels, millions of examples)
- **Phase 2**: Freeze encoder, train modulator on SpliceVarDB variants

The modulator can be small because it only needs to learn a low-dimensional perturbation, while the heavy lifting of sequence understanding is done by the encoder with abundant labels.

### Formulation C: Contrastive Multi-Resolution Learning

Address the label gap directly by using **contrastive objectives** at multiple resolutions:

```
Resolution 1 (position-level):
  Known donor sites should have similar representations
  Known acceptor sites should have similar representations
  Donor ≠ Acceptor ≠ Neither
  → Supervision: GTF annotations (abundant)

Resolution 2 (variant-level):
  SA variants should produce representation SHIFTS
  Non-SA variants should produce NO representation shift
  → Supervision: SpliceVarDB (moderate)

Resolution 3 (effect-level):
  Similar effect types should cluster
  → Supervision: Inferred from base model deltas (noisy)
```

This formulation doesn't require Level 3 positional labels because it learns **representations** rather than **predictions** directly. The position-level prediction emerges from the learned representation space.

### Formulation D: Generalized Perturbation Framework

The observation that "external factors can be anything including variants, diseases, stress, epigenetic marks" points toward a more general formulation:

```
SpliceLandscape(position) = BaseModel(sequence) + Σ_k  δ_k(position, perturbation_k)
```

where each perturbation type `k` has its own delta function, but they share the same underlying sequence representation. This is a **multi-task perturbation model**:

| Task | Train On | Perturbation Type |
|------|----------|-------------------|
| Variant effects | SpliceVarDB | Genetic variants |
| Epigenetic effects | ENCODE methylation + splice junction data | Methylation, histone mods |
| Splicing factor effects | eCLIP + splice junction data | Protein binding |
| Disease/tissue state effects | GTEx tissue-specific splicing | Condition labels |

Each task has different supervision, but they all answer the same question: **how does this perturbation change the splice landscape?** The shared sequence encoder benefits from all tasks simultaneously (multi-task transfer).

### Comparison of Formulations

| Aspect | A: Conditional Landscape | B: Latent State | C: Contrastive | D: Perturbation Framework |
|--------|--------------------------|-----------------|----------------|--------------------------|
| Can discover novel sites | **Yes** | **Yes** | Indirectly | **Yes** |
| Handles label gap | Consistency + sparsity | Two-phase training | Multi-resolution | Multi-task |
| Non-variant perturbations | Natural | Natural | Requires redesign | **Primary design goal** |
| Data efficiency | Moderate | **High** (phase separation) | Moderate | Requires multiple data sources |
| Implementation complexity | Moderate | Moderate | High | High |

---

## Recommendations and Roadmap

### Where the current system stands

The current meta-layer is closest to **a simplified version of Formulation A**, but with critical limitations:

| Aspect | Current System | Ideal (Formulation A) |
|--------|---------------|----------------------|
| Predicts | Δ directly (residual) | Full landscape, Δ derived |
| Can discover novel sites | No (constrained by base model) | Yes |
| Context | 501nt | 5000nt+ |
| Variant supervision | SpliceVarDB binary → cleaned deltas | SpliceVarDB + consistency + sparsity constraints |
| Non-variant perturbations | Not supported | Naturally supported |

### Short-term (with current infrastructure)

- The Validated Delta Predictor (Approach 4) is the right starting point
- But reframe it: instead of predicting the scalar max-delta, predict a **per-position delta profile** `[L, 3]` — this naturally provides Level 3 information
- Add the **consistency constraint**: for non-SA variants, the loss penalizes any non-zero delta at any position. This is free supervision
- Add the **sparsity prior**: for SA variants, L1-regularize the delta profile. Real splice effects are sparse (1-3 positions change significantly)

### Medium-term (with GPU access)

- Move toward **Formulation B**: pre-train a position-level encoder on canonical sites genome-wide, then fine-tune a lightweight perturbation modulator on SpliceVarDB
- Extend context to **2000-5000nt** to capture branch points and regulatory elements
- Use **HyenaDNA** or equivalent as the pre-trained encoder backbone

### Long-term (full Formulation D)

- Integrate **RNA-seq junction data** (e.g., GTEx) as continuous Level 4 supervision
- Multi-task across perturbation types
- This is where "predicting alternative splicing from any external factor" becomes tractable

---

## Cross-Cutting Insights

### Experimental progression validates the design thinking

| Transition | What was learned |
|------------|-----------------|
| 001 → 002 | Classification of known sites is the wrong proxy task for variant detection |
| 002 → 004 | Target quality (not model capacity) is the bottleneck |
| 003 (parallel) | There IS a learnable signal from sequence alone (AUC > 0.5), but 501nt is insufficient |
| All → Future | The meta-layer is a **calibration layer**, not a **discovery engine** |

### The core limitation shared by all approaches

All four approaches can only refine what the base model already partially detects. True alternative splice site prediction — identifying cryptic splice sites, deep intronic activations, or complex multi-exon effects — requires:

1. **Longer context** (thousands of bp) to capture branch points, enhancers/silencers, and RNA secondary structure
2. **Direct sequence-to-function learning** rather than refining an existing model's outputs
3. **More diverse training signal** — RNA-seq data showing actual isoform usage, not just binary SA/non-SA labels
4. **Addressing the false negative problem** — variants that create entirely new splice sites the base model has never scored

---

> [!abstract] Bottom line
> The recognition that this is fundamentally a **weak-supervision problem with a label hierarchy** (not a standard input→label mapping) is the key insight that should drive the system design. The current experiments have established what *doesn't* work (canonical classification, raw base model targets) and what the signal looks like (validated targets, r=0.41). The next step is formalizing the problem structure itself and designing training objectives that bridge the gap between Level 1 labels (SpliceVarDB) and Level 3 predictions (positional splice landscapes).

---

## Related Documents

- [[experiments/001_canonical_classification/README|Experiment 001 — Canonical Classification]]
- [[experiments/002_delta_prediction/README|Experiment 002 — Paired Delta Prediction]]
- [[experiments/003_binary_classification/README|Experiment 003 — Binary Classification]]
- [[experiments/004_validated_delta/README|Experiment 004 — Validated Delta Prediction]]
- [[methods/VALIDATED_DELTA_PREDICTION|Validated Delta Prediction Method]]
- [[methods/PAIRED_DELTA_PREDICTION|Paired Delta Prediction Method]]
- [[methods/ROADMAP|Methodology Roadmap]]
