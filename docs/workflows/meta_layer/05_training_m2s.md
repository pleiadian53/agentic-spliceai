# Stage 5 — Training M2-S (Alternative-Site)

**Pipeline position:** `features → ` **this stage** ` → evaluation → reporting`

M2-S is the *alternative-site* refiner. Where [M1-S](04_training_m1s.md) learns the curated canonical
(MANE) sites, M2-S is trained on the broader **Ensembl** annotation so it can recognize the
alternative sites MANE leaves out — competing donors/acceptors, minor isoforms, tissue-specific sites.

This doc is deliberately short: **M2-S uses the same trainer as M1-S.** Read [Stage 4](04_training_m1s.md)
first; below is only what changes.

!!! abstract "What differs from M1-S"
    **Labels:** Ensembl instead of MANE (auto-selected by `--mode m2`).
    **Base scores:** the Ensembl-windowed precomputed dir (`--base-scores-dir`).
    **Everything else** — script, architecture, channels, hyperparameters — is identical.

---

## The command

```bash
python examples/meta_layer/07_train_sequence_model.py \
    --mode m2 \
    --use-shards \
    --bigwig-cache <bigwig-cache-dir> \
    --base-scores-dir data/ensembl/GRCh38/openspliceai_eval/precomputed \
    --output-dir output/meta_layer/m2s_v4_cleanannot
```

`--mode m2` does two things automatically: it stamps the variant as `M2-S`, and it switches the label
source to **Ensembl** (`annotation_source = ensembl`). You still supply the Ensembl base-scores
directory explicitly with `--base-scores-dir`, because the base model scored the Ensembl gene windows
in [Stage 2](02_base_scoring.md#the-one-m2-s-wrinkle-same-model-different-windows).

!!! note "A discrepancy to be aware of"
    The pod runner `ops_train_m2s_pod.sh` invokes the trainer as `--mode m1 --annotation-source ensembl`.
    That trains on equivalent Ensembl data but stamps the config `variant = M1-S`. The promoted
    `m2s_v4_cleanannot` was produced with `--mode m2` (stamped `M2-S`). **Prefer `--mode m2`** — it
    sets the Ensembl label source for you and records the correct variant.

---

## Why M2-S is the harder model

M1-S is trained and tested on the same curated distribution, so it scores very high. M2-S targets the
**out-of-distribution** sites — the ones a MANE-only model systematically misses — which is a genuinely
harder problem. That shows up in the numbers: the promoted M2-S reaches a validation macro PR-AUC of
**0.953** (epoch 6), lower than M1-S's 0.998 but on a much broader, noisier label set. The right way to
judge M2-S is not the aggregate but its lift **on the alternative sites specifically** — which is what
the dedicated alt-site evaluation in [Stage 6](06_evaluation.md) measures.

The motivation and the architecture-search lessons behind M2-S are worth reading:

- [`examples/meta_layer/docs/ood_generalization.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/ood_generalization.md)
  — why the MANE-trained M1-S degrades on Ensembl/GENCODE sites.
- [`examples/meta_layer/docs/M2/architecture_and_inductive_bias_lessons.md`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/docs/M2/architecture_and_inductive_bias_lessons.md)
  — the key finding that leverage is upstream in features/labels, not in fancier fusion.

---

## Artifacts

Identical layout to [M1-S](04_training_m1s.md#artifacts-produced), under
`output/meta_layer/m2s_v4_cleanannot/`. The companion alternative-site evaluation writes to a sibling
`m2s_v4_cleanannot_alt_eval/` directory (produced by `09` in the next stage).

---

→ **Next: [Stage 6 — Evaluation](06_evaluation.md)**
