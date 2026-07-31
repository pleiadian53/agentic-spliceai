# Meta-Layer Naming Convention

Definitive guide to model, artifact and evaluation-protocol naming in the
Agentic-SpliceAI meta-layer. All documents and code should follow these
conventions.

---

## The core rule: name the axis, never the ordinal

A meta-layer model is the product of **four independent choices**. Each one has
its own history, and they advance at different rates — a new corpus does not
imply a new architecture, and vice versa.

| Axis | Answers | Vocabulary | Example |
|------|---------|-----------|---------|
| **Variant** | which prediction problem | `M1-S` … `M4-S`, `M*-P` | `M2-S` |
| **Arch** | which neural network | *named for its mechanism* | `concat_fusion` |
| **Corpus** | which feature/label build | *named for what changed* | `cleanannot` |
| **Run** | which fit of the above three | slug or date | `baseline_repro` |

> **A bare `vN` is forbidden on every axis.** It carries no statement of which
> axis it belongs to, so the reader has to guess — and in this project the
> guess was wrong often enough to corrupt the artifact registry. See
> [What went wrong](#what-went-wrong-and-why-the-rule-exists).

**Canonical model ID:** `<variant>.<arch>.<corpus>[.<run>]`

```
m1s.concat_fusion.cleanannot     ← the promoted canonical-site refiner
m2s.concat_fusion.cleanannot     ← the promoted alternative-site model
m3s.concat_fusion.cleanannot     ← the novel-site ranker (status: research)
```

Read aloud: *"M2-S, concat-fusion architecture, clean-annotation corpus."*
Every token states its own axis, so there is nothing left to infer.

---

## Axis 1 — Variant (the prediction problem)

Pattern **M{task}-{level}**. Task number 1–4, ordered easiest to hardest;
level `S` (sequence-level) or `P` (position-level).

| Model | Training labels | Purpose |
|-------|----------------|---------|
| **M1-S** | MANE (~370K sites) | Canonical splice classification |
| **M1-P** | MANE | Position-level XGBoost baseline |
| **M2-S** | Ensembl (~2.8M sites) | Alternative splice site detection |
| **M3-S** | Ensembl, junction = target | Novel site discovery |
| **M4-S** | Perturbation pairs | Perturbation-induced splice changes |

- **M1-S vs M2-S**: same architecture, different training labels. M1-S sees
  only MANE canonical transcripts; M2-S sees the full Ensembl annotation.
- **M2-S is NOT "M1-S retrained on Ensembl"** — it is a distinct model for a
  different task.
- **M3-S** differs from M2-S in that junction features become the **target**
  (held out) rather than an input, forcing prediction without RNA-seq evidence.
- **M3-R** is a different model *family* — an XGBoost candidate refiner, not a
  meta-splice network. It has no position on the arch axis.

See [Model variants M1–M4](00_model_variants_m1_m4.md) for the full treatment.

---

## Axis 2 — Arch (the network)

Architectures are named for **the mechanism that distinguishes them**, never
numbered.

| `arch` | Distinguishing mechanism | Status |
|--------|-------------------------|--------|
| `concat_fusion` | Three streams (sequence, base scores, multimodal) concatenated into a 1×1 conv fusion stage; logit-space blend with base | **In use by every promoted model** |
| `xattn_fusion` | Same encoders, but sequence features (query) cross-attend to the merged base+multimodal signal, giving position-specific channel weighting | Implemented, **not promoted** |

Canonical list: `ARCH_REGISTRY` in
[`meta_layer/models/factory.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/meta_layer/models/factory.py).
Add an architecture by extending `build_model()` and that tuple.

!!! warning "The Python module names are frozen history — do not rename them"

    `config.pt` pickles the **fully-qualified class path**, so every trained
    checkpoint on disk resolves
    `...models.meta_splice_model_v3.MetaSpliceConfig` by literal string.
    Renaming the module or the dataclass silently breaks loading of every
    existing checkpoint.

    | Module (frozen) | Config class (frozen) | `arch` (the identifier) |
    |---|---|---|
    | `meta_splice_model_v3.py` | `MetaSpliceConfig` | `concat_fusion` |
    | `meta_splice_v4_xattn.py` | `MetaSpliceXAttnConfig` | `xattn_fusion` |

    The ordinals in those filenames are the *only* surviving `vN` in the
    meta-layer, and they are load-bearing artifacts of pickle, not names you
    should quote to anyone. Use the `arch` column.

The checkpoint's config class is the **authoritative** record of its
architecture — more reliable than a directory name or a registry line, both of
which are hand-maintained. `loader.arch_of_config(cfg)` performs that lookup.

Retired ordinal spellings (`v3`, `v4_xattn`) still resolve through
`factory.ARCH_ALIASES` with a deprecation warning, so old commands and pod job
files keep working.

---

## Axis 3 — Corpus (the feature/label build)

Corpus generations are named for **what changed**, in lineage order:

| `corpus` | What changed | Era |
|----------|-------------|-----|
| `encode_rbp` | First full multimodal build. ENCODE-only RBP channel; minus-strand annotation bug present | April 2026 |
| `neuronal_rbp` | RBP channel widened to the neuronal union (K562/HepG2 ∪ SH-SY5Y/H9 TARDBP) | 2026-05-22 |
| `cleanannot` | Minus-strand annotation fix — the current build | 2026-05-26 → |

A corpus change is a **data** event. It does not imply an architecture change,
and this is exactly where the old scheme misled: the promoted models jumped
from corpus `neuronal_rbp` to `cleanannot` while the architecture never moved.

---

## Axis 4 — Run (the individual fit)

A short descriptive slug or a date, for when variant + arch + corpus are all
identical and only the fit differs: `baseline_repro`, `confirmed`,
`20260530`. Never an ordinal alone.

**Hyperparameters are not a version axis.** Blend mode, hidden dim, dilation
schedule and channel selection all live in `config.pt` and are recoverable
from the checkpoint. Do not promote one into the name — see the retired
"v1 probability blend / v2 logit blend" scheme below.

---

## Where names live

| Surface | Uses | Example |
|---------|------|---------|
| `settings.yaml` `meta_models:` key | canonical ID | `m1s.concat_fusion.cleanannot` |
| `settings.yaml` `arch:` / `corpus:` | the axes, explicitly | `concat_fusion` / `cleanannot` |
| Checkpoint directory | *historical name, decoupled on purpose* | `output/meta_layer/m1s_v4_cleanannot` |
| MANIFEST `tags:` | `arch:` + `corpus:` namespaces | `arch:concat_fusion` `corpus:cleanannot` |
| `--arch` CLI flag | canonical arch name | `--arch concat_fusion` |

**The settings key and the directory name are deliberately decoupled.** Renaming
checkpoint directories would churn ~100 references across docs, example scripts
and published result files, and would have to be repeated on the GPU volume.
The key is the name you *say*; `dir:` is where the bytes *are*. Retired keys
resolve through `META_MODEL_ALIASES` in
[`resources/model_resources.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/resources/model_resources.py),
so every existing reference keeps working.

### Checking it

```bash
python scripts/check_meta_model_registry.py
```

Verifies that each key parses into its three axes, that the declared `arch` and
`variant` match what is actually pickled in the checkpoint's `config.pt`, that
`dir` exists, and that every retired alias still resolves. Exit code 0 = clean.

---

## Decoder: every artifact currently on disk

Use this table when reading older documents or `output/REGISTRY.md`.
The `arch` column was verified against each checkpoint's `config.pt`, not
inferred from the directory name.

| Directory | Canonical ID | Variant | Arch | Corpus | Role |
|-----------|-------------|---------|------|--------|------|
| `m1s_v4_cleanannot/` | `m1s.concat_fusion.cleanannot` | M1-S | `concat_fusion` | `cleanannot` | **promoted** |
| `m2s_v4_cleanannot/` | `m2s.concat_fusion.cleanannot` | M2-S | `concat_fusion` | `cleanannot` | **promoted** |
| `m3_v1/` | `m3s.concat_fusion.cleanannot` | M3-S | `concat_fusion` | `cleanannot` | research (novel-site ranker) |
| `m1s_v2_logit_blend/` | — | M1-S | `concat_fusion` | `encode_rbp` | baseline |
| `m2s_v2/` | — | M2-S | `concat_fusion` | `encode_rbp` | baseline |
| `m2s_v3_baseline_repro/` | — | M2-S | `concat_fusion` | `neuronal_rbp` | baseline; load-bearing for the cryptic-donor demo |
| `m3s_v1_1_confirmed/` | — | M3-S | `concat_fusion` | `cleanannot` | negative result (Tier 1) |
| `m3s/` | — | *(smoke)* | `concat_fusion` | — | **not a model** — a `--smoke` plumbing run |
| `m3r_candidate_refiner/` | — | M3-R | *(n/a — XGBoost)* | — | negative result (Tier 2) |

Three traps this table defuses:

- **`m3s/` is not the M3-S model.** It is a `--smoke` run with six modality
  channels excluded (`mm_channels=1`) whose `config.pt` nonetheless reads
  `variant=M3-S`. The M3-S deliverable is `m3_v1/`.
- **`m2s_v2/`'s `config.pt` reads `variant=M1-S`.** April checkpoints predate
  `_variant_for_mode()` (added 2026-05-16), so the field holds the dataclass
  default. It is genuinely an M2-S model; the field is not a claim.
- **`m1s_v2_logit_blend/`'s "v2" is doubly ambiguous** — it reads as both the
  retired blend-mode version and the corpus generation. Both happen to be true
  of that artifact. Every checkpoint on disk now uses `blend_mode=logit`.

---

## Evaluation protocols

**Models** are what you train — defined by the four axes above.
**Evaluation protocols** are how you test — defined by the test set and the
question asked. Any model can be evaluated with any protocol, so a model's
name must never encode the evaluation setting.

Pattern: **Eval-{test_set}**

| Protocol | Test set | Question answered |
|----------|----------|-------------------|
| **Eval-MANE** | MANE splice sites on test chroms | How well does the model classify canonical sites? |
| **Eval-Ensembl-Alt** | Ensembl \ MANE (set difference) | Can it detect alternative sites beyond MANE? |
| **Eval-GENCODE-Alt** | GENCODE \ MANE (set difference) | Broader alternative-site evaluation (curated) |
| **Eval-ClinVar** | ClinVar splice variants | Can delta scores separate pathogenic from benign? |
| **Eval-SpliceVarDB** | SpliceVarDB validated variants | Cross-validation against experimental evidence |

Results are described as **{Model} on {Protocol}**:

- "M1-S on Eval-MANE" → canonical classification
- "M2-S on Eval-Ensembl-Alt" → M2-S on its target task
- "M1-S on Eval-Ensembl-Alt" → testing M1-S out-of-distribution

---

## What went wrong, and why the rule exists

Before this convention, `v4` meant two unrelated things:

- `m1s_v4_cleanannot` — a **corpus** generation (the promoted model, running on
  the architecture then called "v3")
- `meta_splice_v4_xattn.py` — an **architecture** generation that was never
  promoted

Explaining "the promoted v4 model" therefore required naming which v4 you meant
and which v-numbered thing it was *not*. Worse, the ambiguity had already
produced concrete errors in the repository:

- `m3_v1/` was tagged `meta:v1` in `output/REGISTRY.md`, although its
  `train.log` reads `Arch: v3, variant: M3-S` on the cleanannot corpus. Under
  the documented meaning of the tag ("meta-layer model generation") it was
  simply wrong.
- `m1s_v2_logit_blend/`'s `produced_by` line read
  *"07_train_sequence_model.py (v2 architecture, April 2026)"* while its own
  `notes` field read *"v3 arch + ENCODE-only RBP"* — the manifest contradicted
  itself in adjacent fields.
- `m2s_v3_baseline_repro/` is the trap that hid all of this: its corpus
  generation "v3" *coincides* with its architecture "v3", so the wrong reading
  looks confirmed until it is applied to a different artifact.

Named axes make each of these unrepresentable: there is no way to write
`corpus:concat_fusion` and have it look plausible.

---

## Legacy names (deprecated)

| Old name | Current name | Notes |
|----------|-------------|-------|
| `meta:v2` / `meta:v3` / `meta:v4` (tags) | `corpus:encode_rbp` / `corpus:neuronal_rbp` / `corpus:cleanannot` | Ordinal tracked the corpus; the namespace implied the model as a whole |
| `--arch v3` | `--arch concat_fusion` | Alias retained with a warning |
| `--arch v4_xattn` | `--arch xattn_fusion` | Alias retained with a warning |
| `m1s_v4_cleanannot` (settings key) | `m1s.concat_fusion.cleanannot` | Directory name unchanged; key aliased |
| `m2s_v4_cleanannot` (settings key) | `m2s.concat_fusion.cleanannot` | Directory name unchanged; key aliased |
| `m3_v1` (settings key) | `m3s.concat_fusion.cleanannot` | Directory name unchanged; key aliased |
| `M1-S v1` / `M1-S v2` | *(retired)* | Was blend mode — now the `blend_mode` config field; all checkpoints use `logit` |
| M2a | **Eval-Ensembl-Alt** (protocol) | Was ambiguously both eval and model |
| M2b | **Eval-GENCODE-Alt** (protocol) | Same ambiguity |
| M2c | **M2-S** (model) | The Ensembl-trained model, not an eval variant |
| M2d | M2-S with junction weighting | Training variant, not a separate model code |
| M2e | Tissue-conditioned M2-S | Future extension |
| M1-S/MANE | **M1-S** | Redundant — M1-S is always MANE-trained |
| M1-S/Ensembl | **M2-S** | This IS the M2 model |

---

## Summary

```
Variant:    M1-S, M2-S, M3-S, M4-S      (which problem)
Arch:       concat_fusion, xattn_fusion (which network)
Corpus:     encode_rbp -> neuronal_rbp -> cleanannot  (which data)
Run:        descriptive slug or date    (which fit)

Model ID:   <variant>.<arch>.<corpus>[.<run>]
Protocols:  Eval-MANE, Eval-Ensembl-Alt, Eval-GENCODE-Alt, Eval-ClinVar
Results:    "{Model} on {Protocol}"
```

This separation ensures that:

1. Every version token declares which axis it belongs to.
2. A corpus rebuild never looks like an architecture change.
3. Evaluation protocols stay reusable across models.
4. New models or protocols can be added without renaming existing ones.
5. Results are always attributable to a specific model + protocol pair.
