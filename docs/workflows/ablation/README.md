# Ablation Studies — measuring what each modality actually contributes

**Pipeline position:** `evaluation → ` **this stage** ` → reporting`

The meta layer fuses about ten heterogeneous data sources on top of a base model's scores. Ablation
answers the question that justifies that complexity: **which of those sources is doing work, and how
much?** Without it, a headline like "the meta layer recovers 5.3x more alternative sites" cannot be
attributed. The lift might come from the multimodal evidence, or simply from training on a richer
annotation.

!!! abstract "Inputs → Outputs"
    **Reads:** a trained checkpoint (`best.pt` + `config.pt`) and a `.npz` gene cache, or the
    per-position feature parquets for the tabular route.
    **Writes:** `eval_ablation_<channels>.json` per variant plus `eval_results.json` for the
    unablated baseline; or `ablation_comparison.json` for the tabular route.

---

## Two methods, two different questions

This is the distinction to get right before running anything, because the two are easy to conflate and
they do not measure the same thing.

| | Retrain without it | Zero it at inference |
|---|---|---|
| **Script** | [`03_modality_ablation.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/03_modality_ablation.py) | [`08_evaluate_sequence_model.py --zero-channels`](https://github.com/pleiadian53/agentic-spliceai/blob/main/examples/meta_layer/08_evaluate_sequence_model.py) |
| **Model** | XGBoost tabular baseline | the neural M\*-S checkpoints |
| **Question** | *Would a model built without this signal be worse?* | *Does the trained model lean on this signal?* |
| **Cost** | retrains per variant | one inference pass per variant |
| **Answers** | the design question | the attribution question |

They can disagree, and the disagreement is informative rather than a contradiction. A model trained
without junction support may learn to route around its absence and lose little, while the same model
trained *with* it may lean on it heavily. Zeroing measures dependence of a fixed model; retraining
measures necessity of the signal. Quote whichever matches the claim being made, and say which one it is.

---

## The channels

Nine dense multimodal channels, defined by `CHANNEL_NAMES` in
[`dense_feature_extractor.py`](https://github.com/pleiadian53/agentic-spliceai/blob/main/src/agentic_spliceai/splice_engine/features/dense_feature_extractor.py),
conventionally ablated as five biological groups plus an all-channel control:

| Group | Channels |
|-------|----------|
| conservation | `phylop_score`, `phastcons_score` |
| epigenetic | `h3k36me3_max`, `h3k4me3_max` |
| chromatin accessibility | `atac_max`, `dnase_max` |
| junction support | `junction_log1p`, `junction_has_support` |
| RBP binding | `rbp_n_bound` |
| *all* (control) | every channel, leaving sequence + base scores |

`--zero-channels` validates names against `CHANNEL_NAMES` and exits with an error on an unknown one, so
a typo fails immediately instead of silently zeroing nothing.

---

## Running it

### Locally, one variant

```bash
python examples/meta_layer/08_evaluate_sequence_model.py \
    --checkpoint output/meta_layer/<model-dir>/best.pt \
    --cache-dir <test-cache>/test \
    --output-dir output/<model>_ablation \
    --zero-channels phylop_score phastcons_score
# → output/<model>_ablation/eval_ablation_phylop_score_phastcons_score.json
```

Omit `--zero-channels` for the unablated baseline. Note the output filename is derived from the zeroed
channels, so the baseline lands in `eval_results.json`, not `eval_ablation_*.json`.

### The full sweep, on a GPU pod

Genome-scale caches make the full sweep a pod job. See
[GPU Pods](../meta_layer/08_gpu_pods.md) for provisioning, then:

```bash
# on the cluster
cd ~/sky_workdir
nohup bash examples/meta_layer/ops_ablation_m2s_pod.sh > <output-dir>/ablation.log 2>&1 &
```

`ops_ablation_m1s_pod.sh` does the same for M1-S against the MANE test cache. Both accept `CHECKPOINT`,
`CACHE_DIR` and `OUTPUT_DIR` as environment overrides, and both verify the checkpoint and cache exist
before the first evaluation rather than failing partway through a long run.

!!! tip "The sweep is CPU-bound, not GPU-bound"
    Each variant spends most of its wall-clock on gene-sequence extraction and paralog detection, not
    on inference. A single process was measured at 5.8 GB RSS, 131 threads and under 400 MiB of GPU
    memory. Running the variants **concurrently** rather than sequentially is therefore a large
    wall-clock win, provided per-process thread counts are capped so the box is not oversubscribed:

    ```bash
    export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12
    ```

    Stagger the launches. The setup phase reads the genome FASTA, and starting every worker at once
    puts all of them on that read simultaneously.

---

## Reading the results

Each JSON carries the same structure as a normal evaluation: `meta_model` and `base_model` blocks with
per-class and macro PR-AUC, precision/recall/F1, and `fn_reduction_pct`. Compare each variant against
the unablated baseline from the same run.

Report the **drop relative to the full model**, not the absolute number, and state the metric. A useful
table looks like:

| Zeroed | macro PR-AUC | Δ vs full |
|---|---|---|
| — (full model) | *baseline* | — |
| all multimodal | | the ceiling on what multimodal contributes |
| conservation | | |
| … | | |

Published findings belong in the [results series](../../meta_layer/results/README.md), not here. This
page is the method.

---

## Pitfalls

**Macro PR-AUC is inflated by the trivial class.** The macro average includes `neither`, which is
~99.5% of positions and near-perfectly predicted. A modality can look unimportant on macro PR-AUC while
mattering on the metric that reflects the task. Prefer the alternative-site or held-out numbers.

**A channel that is already zero in the cache gives a bit-identical result.** If an ablation reproduces
the baseline to every decimal place, that is not evidence the channel is uninformative. Check whether
the channel carries data in that cache before concluding anything. Identical-to-16-decimals means the
same computation ran, not that the signal was useless.

**Results are specific to one checkpoint and one corpus.** An ablation is a property of the model that
produced it. Re-run after any retrain, and record the checkpoint path and gene count alongside the
numbers so a stale table cannot be mistaken for a current one.

**Do not generalise across variants.** Modality importance shifts with the task. A signal that is
load-bearing for canonical sites may contribute nothing for novel-site ranking, and the reverse also
happens. Ablate the variant you intend to make claims about.

---

## Related

- [Stage 6 — Evaluation](../meta_layer/06_evaluation.md) — the unablated evaluation this builds on
- [Stage 8 — GPU Pods](../meta_layer/08_gpu_pods.md) — provisioning for the full sweep
- [Results & Findings](../../meta_layer/results/README.md) — where measured outcomes are published
