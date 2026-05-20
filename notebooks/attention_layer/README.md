# Multimodal Fusion via Attention — An Empirical Study

A series of notebooks studying attention mechanisms, their inductive biases,
and their fitness for multimodal fusion. Motivated by lessons from the
agentic-spliceai meta-layer (M1-S/M2-S), but conducted on a non-splice
substrate (ESM-2 + protein per-residue tasks) so architectural lessons
generalize and run quickly on M1 Mac.

---

## Motivation

The recent M1-S/M2-S experiments surfaced a question worth studying in
isolation: **when does attention help for multimodal fusion, and when does
it hurt?**

The case study is sharp:

| Model | Task | Fusion | Val PR-AUC |
|---|---|---|---|
| M1-S (v3) | Canonical splice sites | cat → 1×1 conv | **0.9954** |
| M2-S v2 (v3) | Alternative splice sites | cat → 1×1 conv | **0.833** |
| M2-S v4_xattn (H=64) | Alt splice sites + cross-attention fusion | global cross-attention | **0.6649** ← worse than cat-fusion, worse than base model alone |

The v4_xattn post-mortem (`dev/sessions/2026-05-18_*`) taught three things
attention *isn't*:

1. **Not free capacity.** v4_xattn at H=64 had 1.47M params; the model
   overfit on training donors/acceptors faster than it learned
   generalizable splicing grammar. Train_loss decoupled from val_pr_auc by
   epoch 5.
2. **Not always the right inductive bias.** Global attention gives every
   position equal access to every other position. For splice sites — where
   the biology is local (~100 bp consensus + branchpoint + polypyrimidine
   tract) and the positive class is rare (~0.1% of positions) — global
   attention dilutes informative signals with the majority class's noise.
3. **Not a drop-in upgrade.** The temperature/blend interaction (T flattens
   *both* meta and base contributions when applied post-blend), α-collapse,
   and gene-cache effects compounded silently. Each architectural lever
   changes how the model trains, not just what it computes.

But the experiments opened questions worth understanding rather than
guessing at:

- Why does **cross-attention** fail when **cat-fusion** succeeds on the
  same data?
- Would **local windowed attention** (Longformer-style) restore the right
  locality prior?
- When is **sparse / gated attention** the right mechanism, and when is it
  just more knobs to tune?
- **FiLM conditioning** — a simpler alternative to attention. When does it
  suffice?
- How do **multi-stream fusion strategies** compare on the *same task*,
  holding capacity constant?

This study takes a controlled approach: build the same per-position
prediction task under different attention/fusion mechanisms; measure,
visualize, and explain. The goal isn't a publishable architecture — it's
that the next M2-S architectural decision is informed rather than
speculative.

---

## Substrate choice — ESM-2 + per-residue protein prediction

The daily learning substrate moves away from splicing, for three reasons:

- M1 Mac compute constraint: full-genome retraining isn't feasible per
  notebook.
- M2-S's small positive class makes fusion-mechanism differences hard to
  disentangle from class-imbalance effects.
- Splice-specific biology can distract from the architectural question.

**Substrate: ESM-2 (8M or 35M) + protein secondary structure prediction.**

| Property | Why it fits |
|---|---|
| Per-residue prediction | Matches the per-position protocol of M2-S |
| Local signal | α-helices/β-sheets form over ~10–30 residues; same locality character |
| Pre-trained embeddings | ESM-2 8M is ~35 MB; runs on M1 in seconds |
| Auxiliary modalities for fusion | MSA conservation, biophysical features (hydrophobicity, charge), predicted SS from a second tool, AlphaFold-derived pLDDT/RSA |
| Established benchmarks | CB513, CASP12/13/14 — easy to compare |
| Mature data tooling | HuggingFace `facebook/esm2_t6_8M_UR50D`, `Rostlab/prot_bert`, etc. |
| Tractable | 8M ESM-2 + a small classifier head trains a notebook task in minutes |

**M2-S returns as a recurring case study** referenced throughout the
series, and as the final "graduation" notebook where the lessons are
applied back to splice prediction (without training — design only).

---

## Notebook series

Eight notebooks, organized as **foundations → bias analysis → application**.
Each produces a small, comparable artifact (a model, a plot, a results row)
so the series accumulates evidence rather than just text.

| # | Topic | Goal | Concrete output |
|---|---|---|---|
| **00** | Attention from scratch | Implement scaled dot-product attention, multi-head attention. Visualize what attention computes on a toy sequence. | Custom `attention.py`; attention-weight heatmaps; sanity-check that softmax(QKᵀ/√d) V matches `nn.MultiheadAttention`. |
| **01** | Self vs cross-attention | Same downstream task, two variants: protein self-attention over ESM-2 embeddings vs. cross-attention between ESM-2 and an auxiliary feature stream (e.g., per-residue biophysical descriptors). | Side-by-side training curves; per-residue accuracy table; discussion of when Q=K=V is appropriate vs. when streams should differ. |
| **02** | Positional encodings | Sinusoidal / learned / RoPE / ALiBi. Why attention is permutation-invariant without them. | Same model, four PE variants. Compare length-extrapolation behavior (train on L=256, eval on L=512). |
| **03** | The locality problem (M2-S post-mortem, in microcosm) | Build a controlled toy task with varying positive-class density (0.1% → 50%). Show when global attention hurts via the dilution mechanism we identified in v4_xattn. | Plot: PR-AUC vs. positive density for global cross-attention vs. local windowed (W=32) vs. a CNN baseline. Predict the crossover point. |
| **04** | Local windowed attention | Longformer-style sliding window. Two implementations: (a) masked SDPA, (b) explicit unfold-and-attend for tighter compute. Ablation over window sizes. | Local cross-attention module reusable in the meta-layer; results table at W ∈ {32, 64, 128, 256, 512}. |
| **05** | Sparse and gated attention | BigBird's random+window+global mixture; gated attention units (GAU); brief look at mixture-of-experts as gated attention generalization. | Two implementations on the same task; identify which sparsity pattern lifts performance, which is just more knobs. |
| **06** | FiLM conditioning | Feature-wise linear modulation — a cheaper alternative to attention for conditioning streams on each other (`y = γ(c) ⊙ x + β(c)`). | Replace cross-attention in notebook 01 with FiLM; compare param count, training time, and quality. Map out the regime where FiLM is preferable. |
| **07** | Multimodal fusion: head-to-head | Same task, same data, same total capacity — six fusion strategies side-by-side: (i) cat-fusion, (ii) late fusion, (iii) cross-attention, (iv) local cross-attention, (v) FiLM, (vi) gated fusion. | One results table; one decision tree for "when to use what" given task properties (signal density, modality count, capacity budget). |
| **08** | Graduation — M2-S redesign principles | Apply the lessons back to the splice task. Write the M2-S v3 architectural spec from a position of understanding rather than experimentation. | A design doc with: chosen fusion mechanism, predicted ablation results, a pre-mortem on the next M2-S architectural risks. **No training**, just informed prediction. |

---

## Layout

```
notebooks/attention_layer/
├── README.md                    # this file
├── lib/                         # shared utilities
│   ├── attention.py             # built up across notebooks (start in #00)
│   ├── esm2.py                  # ESM-2 loaders + per-residue feature extraction
│   ├── data.py                  # CB513 / CASP loader, splits
│   ├── fusion.py                # fusion modules from #07
│   └── eval.py                  # per-residue PR-AUC, F1, plot helpers
├── 00_attention_from_scratch.ipynb
├── 01_self_vs_cross_attention.ipynb
├── 02_positional_encodings.ipynb
├── 03_locality_problem.ipynb
├── 04_local_windowed_attention.ipynb
├── 05_sparse_gated_attention.ipynb
├── 06_film_conditioning.ipynb
├── 07_fusion_head_to_head.ipynb
└── 08_m2s_redesign_principles.md  # design doc, not a notebook
```

---

## Conventions

- **Tiny by default.** Each notebook uses ESM-2 8M and a ~1k-sample subset
  of the target dataset unless explicitly comparing to a "full" baseline.
  Aim for end-to-end runs in <2 min on M1 so the iteration loop stays fast.
- **Self-contained but cross-referencing.** Each notebook re-imports
  shared utilities from `lib/`; "see notebook XX" pointers replace
  re-derivation.
- **Each notebook ends with a "What I learned" cell** — plain text, no
  code. Captures the takeaway in 3–5 sentences so a skim is informative.
- **Reproducibility.** Pin random seeds at the top. Save final
  metrics/checkpoints into `notebooks/attention_layer/output/<nb_id>/`.
- **No production weights.** Anything that gets reused production-side
  graduates to `src/agentic_spliceai/splice_engine/meta_layer/models/` as
  its own module with tests.

---

## Sequencing + pacing

Roughly one notebook per session, ~2–3 hours each. Suggested pace:

- Notebooks **00–02** (foundations): one week. Builds the vocabulary +
  custom modules everything else relies on.
- Notebooks **03–04** (locality): one week. The heart of the bias
  analysis; directly connected to the M2-S failure.
- Notebooks **05–06** (sparse, gated, FiLM): one week. Comparative
  studies of alternatives.
- Notebooks **07–08** (synthesis): one week. Brings it all together.

Total: ~4 weeks at a relaxed pace; ~2 weeks at a focused pace.

---

## Optional explorations (after the core series)

These are tangential but related — fair game if the core series surfaces
specific questions:

- **Rotary position embeddings + ALiBi** for longer context (relevant when
  M2-S eventually needs >2 kb context).
- **Flash Attention 2** — a torch 2.5+ optional kernel; relevant only if
  the M2-S architecture goes deep into attention. Mostly an engineering
  topic, not a learning topic.
- **Cross-modal pretraining** (CLIP-style contrastive objectives between
  sequence and structure features) — a different way of getting modalities
  to talk that bypasses fusion entirely.
- **Perceiver / Perceiver IO** — fixed-size latent bottleneck; relevant if
  modality counts grow.
