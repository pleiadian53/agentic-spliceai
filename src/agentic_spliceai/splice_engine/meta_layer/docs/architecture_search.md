# Architecture Search for the Sequence-Level Meta-Splice Model

This document tracks configurable architecture choices in `MetaSpliceModel`
(`meta_splice_model_v3.py`) and the experimental methodology for
comparing them.

---

## Configurable Axes

The `MetaSpliceConfig` dataclass exposes the following architecture choices.
Each axis can be evaluated independently or in combination.

### 1. Activation Function

```python
MetaSpliceConfig(activation="gelu")   # default
MetaSpliceConfig(activation="relu")
MetaSpliceConfig(activation="selu")
```

| Activation | BatchNorm | Dropout | Params | Init | Notes |
|------------|-----------|---------|--------|------|-------|
| **GELU** | BatchNorm1d | Standard | 367K | Default (Kaiming) | **Default**. Smoother gradient; avoids dead neurons; modern standard |
| **ReLU** | BatchNorm1d | Standard | 367K | Default (Kaiming) | SpliceAI baseline; risk of dead neurons with small H and extreme class imbalance |
| **SELU** | Identity (none) | AlphaDropout | 365K | LeCun normal | Self-normalizing; no BN needed; fixes MPS autograd bug |

**GELU** is the default — it avoids the dead neuron problem that ReLU
suffers from (particularly risky with small hidden dim H=32 and extreme
class imbalance where rare-class neurons can die early).

**GELU** is the standard activation in transformer architectures (BERT, GPT)
and has shown modest improvements over ReLU in many settings. It provides a
smooth approximation to ReLU that doesn't zero-out negative inputs entirely,
which can help gradient flow in deep networks. It retains BatchNorm.

**SELU** (Scaled Exponential Linear Unit) has a unique *self-normalizing*
property: given LeCun normal weight initialization and AlphaDropout, the
activations converge toward zero mean and unit variance without BatchNorm.
This is theoretically elegant but comes with caveats:

- The self-normalizing guarantee assumes no skip connections that break the
  variance flow. Our residual `out + residual` addition doubles variance at
  each block, which can destabilize SELU over 8+ blocks.
- SELU was originally proven for feedforward networks, not deeply stacked
  dilated convolutions.
- Empirical: SELU eliminates BatchNorm, which (a) reduces parameters by ~2K,
  and (b) sidesteps the PyTorch MPS autograd bug with BatchNorm1d backward
  through permuted tensors.

### 2. Stream Architecture

```python
MetaSpliceConfig(merge_base_scores=True)   # default: 2-stream
MetaSpliceConfig(merge_base_scores=False)  # legacy: 3-stream
```

| Mode | Streams | Base score handling | Params |
|------|---------|---------------------|--------|
| **Merged (2-stream)** | Sequence CNN + Signal CNN | 3 extra channels in signal stream | 367K |
| **Legacy (3-stream)** | Sequence CNN + Score MLP + Multimodal CNN | Dedicated MLP encoder (3 → H) | 369K |

The merged design treats base scores as just another modality — consistent
with the feature engineering framework where `base_scores` is modality #1.

### 3. Other Axes (future)

| Axis | Current | Candidates | Rationale |
|------|---------|------------|-----------|
| Hidden dim | 32 | 16, 64, 128 | Capacity vs memory tradeoff |
| Kernel size | 11 | 7, 15, 21 | Receptive field per block |
| Dilation rates | [1,1,1,1,4,4,4,4] | [1,2,4,8,...] | Exponential vs plateau dilation |
| Residual blend | Learned alpha | Fixed alpha, no blend | Value of base model as prior |
| Normalization | BN (relu/gelu), none (selu) | LayerNorm, GroupNorm | Alternatives to BatchNorm |

---

## Experimental Methodology

### Terminology

- **Ablation study**: Systematically removing or replacing components to
  measure their individual contribution. Answers: "How much does component X
  contribute to the overall performance?"
  Example: removing junction features from M1-P showed -48% FN reduction.

- **A/B test**: Comparing two specific configurations head-to-head on the
  same data split. Answers: "Is configuration A better than configuration B?"
  Example: ReLU+BN vs GELU+BN on M1-S.

- **Architecture search**: Exploring a space of configurations to find the
  best combination. Encompasses multiple A/B tests, potentially with
  automated search (grid, random, Bayesian).

An A/B test is the *evaluation method*; ablation is the *experimental design*;
architecture search is the *overall process*. Statistical significance
testing tells you whether the observed difference is real or noise.

### Protocol for Activation Function Comparison

**Controlled variables** (held constant across all runs):
- Same gene split (SpliceAI chromosome holdout, seed=42)
- Same training data (all MANE genes on training chromosomes)
- Same hyperparameters (lr=1e-3, batch_size=4, accum=4, epochs=50)
- Same loss (focal, gamma=2.0, class weights [166, 166, 1])
- Same random seed for weight initialization and data sampling

**Independent variable**: `activation` in {relu, gelu, selu}

**Metrics** (evaluated on held-out test chromosomes):
- Per-nucleotide accuracy
- Per-class PR-AUC (donor, acceptor)
- FN count (missed splice sites)
- FP count (false splice calls)
- Training loss convergence curve

**Statistical significance**: Since splice site prediction has millions of
test positions, even tiny metric differences are statistically significant.
The more meaningful question is *effect size* — does the improvement matter
in practice? Report both p-value (McNemar test on per-position predictions)
and absolute FN/FP reduction.

### Recommended Experiment Order

1. **Baseline**: GELU+BN (current default)
2. **ReLU+BN**: SpliceAI reference; compare to quantify dead neuron impact
3. **SELU (no BN)**: Higher risk, potentially bigger gain or regression
4. **Best activation + stream ablation**: merged vs legacy with best activation

Run 1-3 first. If SELU shows promise, investigate whether the residual
connection needs dampening (`out + 0.1 * residual`) to preserve
self-normalization.

---

## Training Script Integration

The training script `07_train_sequence_model.py` can be extended with an
`--activation` flag:

```bash
# Compare activations
python 07_train_sequence_model.py --mode m1 --activation relu  --output-dir output/meta_layer/m1s_relu
python 07_train_sequence_model.py --mode m1 --activation gelu  --output-dir output/meta_layer/m1s_gelu
python 07_train_sequence_model.py --mode m1 --activation selu  --output-dir output/meta_layer/m1s_selu
```

Results are saved as `best_metrics.json` in each output directory for
automated comparison.
