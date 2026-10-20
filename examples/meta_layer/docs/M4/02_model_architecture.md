# M4 Tutorial 2: Model Architecture

*Part 2 of 4. Previous: [1. Training data](01_training_data.md). Next: [3. Training and evaluation](03_training_and_evaluation.md).*

Part 1 defined one example $(e, R)$, its label $y = \Delta\mathrm{PSI}$, and a 229-column feature vector $X$. This part describes the models that consume it today, compares them with the conditional architecture M4 is meant to become, and works through a key design decision: should M4 predict the change directly, or predict the perturbed state and derive the change?

---

## 1. What a conditional M4 has to represent

Write the ideal model as

$$
\widehat{\Delta\mathrm{PSI}}_{e,R} = M4\big(h_e,\ z_R,\ c,\ p_0\big),
$$

| Symbol | Meaning |
|---|---|
| $h_e$ | the state of event $e$: its three sites, their sequence and multimodal context |
| $z_R$ | a representation (embedding) of the perturbed regulator $R$ |
| $c$ | the cellular context (cell line, tissue) |
| $p_0$ | the control-state PSI of the event |

The heart of it is an **interaction** between event and regulator,

$$
\text{effect}_{e,R} = g(h_e, z_R),
$$

because biology is specific: TDP-43 loss matters where UG-rich sequence sits near a site, while an SF3B1 perturbation acts through branch-point and 3′ splice-site geometry. A model with no interaction term can only learn "how volatile is this event" and "how strong is this regulator", never "this regulator acts on this kind of event".

## 2. What is implemented today

The current benchmark (`examples/meta_layer/17_m4_fewshot_benchmark.py`) compares a ladder of methods. None is the neural architecture above; they are deliberately simple, to establish what the data supports before building it.

| Method | What it learns | How $R$ enters |
|---|---|---|
| `zero` | nothing; predicts 0 | not at all |
| `volatility` | the event's mean $\Delta$PSI under other regulators | not at all (an event prior) |
| `pooled` | one gradient-boosted tree model (XGBoost, 400 trees, depth 6) over all training regulators | only through $R$'s eCLIP binding features |
| `xgb` | a small XGBoost (100 trees, depth 3) on $k$ examples of $R$ alone | implicitly, by training only on $R$'s data |
| `tabpfn` | TabPFN in context on $k$ examples of $R$ | implicitly, through the context set (part 4) |
| `pooled_tabpfn` | `pooled` plus a TabPFN correction fitted to its errors on the $k$ examples | binding features + context set |

Trees do model interactions between features, so `pooled` *can* learn "binding near the intron side of an acceptor matters", but only for the 104 regulators with eCLIP, and only through binding. It has no notion of regulator identity.

## 3. How the implementation maps onto the target

| Component | Target design | Current implementation | Gap |
|---|---|---|---|
| $h_e$, event state | learned encoder over sequence, base, and multimodal signals | 229 hand-built tabular features of the three sites (part 1) | no learned sequence encoder; no regulator-motif features, so "UG-rich context" is invisible |
| $z_R$, regulator embedding | a vector per regulator that can interact with $h_e$ | **none**; $R$ appears only as binding features or as the identity of a few-shot training set | the main gap |
| $c$, cell context | an embedding of cell type or tissue | **none, and not learnable here**: all 185 knockdowns are HepG2 | needs data from other cell types |
| $p_0$, control PSI | input to the model | an out-of-batch baseline PSI as a feature | consistent, and for good reason (section 5) |
| $g(h_e, z_R)$, interaction | explicit (bilinear, FiLM, attention) | implicit in trees, via binding features only | needs $z_R$ first |

So the implementation is consistent with the target in its *inputs* ($h_e$, $p_0$) and in its evaluation (held-out regulators), but it does not yet represent the regulator, which is the part that makes M4 conditional.

### 3.1 Where z_R could come from

A learned free embedding per regulator works for regulators seen in training but cannot be computed for a new one. For held-out regulators, $z_R$ has to be **computed from something observable**:

| Source | Available for | Gives the model |
|---|---|---|
| eCLIP binding profile (where $R$ binds genome-wide) | 104 of 185 | positional binding preferences |
| sequence motif (e.g. RNAcompete, eCLIP-derived) | most RBPs | what sequence $R$ recognises; pairs with motif features in $h_e$ |
| protein language-model embedding (e.g. ESM) of $R$ | all | domain architecture, family similarity |
| the $k$ labelled examples of $R$ | any regulator with data | the response itself (few-shot) |

The last row is what in-context learning does: TabPFN reads the $k$ examples and, in effect, infers a task representation for $R$ on the fly (part 4).

### 3.2 A first interaction model: factorisation

The simplest explicit interaction is bilinear,

$$
\Delta\mathrm{PSI}_{e,R} \approx \underbrace{a_e}_{\text{event volatility}} + \underbrace{b_R}_{\text{regulator strength}} + u_e^{\top} v_R ,
$$

which is low-rank matrix factorisation of the event × regulator matrix. The `volatility` baseline is the $a_e$ term alone. A new regulator's $v_R$ can be fitted from its $k$ examples (cold-start collaborative filtering), or predicted from the sources in section 3.1, giving a zero-shot $z_R = v_R$. Replacing $u_e$ with a function of the event features, $u_e = W h_e$, turns this into the general $g(h_e, z_R) = h_e^{\top} W^{\top} z_R$.

## 4. Should M4 predict ΔPSI directly?

Two formulations are natural.

**A. Direct regression of the change.**

$$
\widehat{\Delta\mathrm{PSI}} = \tanh\big(f(h_e, z_R, c)\big), \qquad \mathcal{L} = \mathrm{Huber}\big(\widehat{\Delta\mathrm{PSI}}, \Delta\mathrm{PSI}\big).
$$

It is simple and targets the quantity of interest. Its weakness is that $\tanh$ only bounds the change to $[-1, 1]$; the real bound depends on the starting point, $-p_0 \le \Delta\mathrm{PSI} \le 1 - p_0$. A model can predict $+0.4$ for an event already used 90% of the time.

**B. Predict the perturbed state, derive the change.**

$$
\widehat{\mathrm{PSI}}^{\text{KD}} = \sigma\big(f(h_e, z_R, c, p_0)\big), \qquad \widehat{\Delta\mathrm{PSI}} = \widehat{\mathrm{PSI}}^{\text{KD}} - p_0 .
$$

This guarantees $0 \le \widehat{\mathrm{PSI}}^{\text{KD}} \le 1$, so every predicted change is feasible.

The bound matters in this data: 45% of events have control PSI below 0.05 or above 0.95, and 16.6% of significant changes use more than 80% of the room available to them. Formulation B is the better foundation, with three refinements.

### 4.1 Refinement 1: put p₀ in as an offset, in log-odds

Rather than letting the network use $p_0$ freely, write

$$
\widehat{\mathrm{PSI}}^{\text{KD}} = \sigma\big(\operatorname{logit}(p_0) + \delta_{e,R}\big), \qquad \delta_{e,R} = g(h_e, z_R, c),
$$

where $\operatorname{logit}(p) = \log\frac{p}{1-p}$ and $\sigma$ is its inverse. Now $\delta_{e,R}$ is the regulator's effect on the **odds** of choosing the long form. One value of $\delta$ produces a large change for an event near 0.5 and a small one for an event near 0 or 1, which is the ceiling effect seen in the data, built in rather than learned. It also reads naturally: a regulator shifts the relative strength of two competing sites.

### 4.2 Refinement 2: p₀ must be an independent measurement

Formulation B invites using the experiment's own control PSI as $p_0$. That is the trap measured in part 1, section 5: a rule with no learning, $\widehat{\Delta\mathrm{PSI}} = -(\mathrm{PSI}^{\text{ctrl}} - \text{mean})$, reaches direction accuracy 0.691 and Spearman 0.297 on held-out data, ahead of every trained model, because the same control noise sits in the prediction and in the label. In training, $p_0$ must come from controls that share no noise with the label:

- the out-of-batch baseline already in $X$ (other control batches), or
- a **replicate split**: estimate $p_0$ from some control replicates and the label from the others.

With an independent $p_0$, the bound structure still carries honest signal: the same one-line rule using the out-of-batch baseline reaches direction accuracy 0.624. In the v1 benchmark that is higher than every learned zero-shot or few-shot method on new genes (pooled 0.520, TabPFN with 256 examples 0.565), which is the strongest argument for building the bound into the model.

### 4.3 Refinement 3: train on read counts, not on ΔPSI

PSI is a ratio of read counts, and its noise depends on how many reads there are. Instead of a Huber loss on $\Delta$PSI, fit the knockdown read counts directly. With $\psi = \widehat{\mathrm{PSI}}^{\text{KD}}$ and effective form lengths $L_I$, $L_S$, the probability that a read supports the long form is

$$
q = \frac{\psi L_I}{\psi L_I + (1 - \psi) L_S},
$$

and the inclusion reads $I$ out of $n = I + S$ follow a binomial (or, to absorb biological over-dispersion, a beta-binomial) distribution. The loss is the negative log-likelihood,

$$
\mathcal{L} = -\log \operatorname{BetaBinomial}(I \mid n, q, \phi).
$$

An event with 5 reads then contributes little, and an event with 500 reads a lot, automatically and without a hand-set read threshold. rMATS's own statistics likewise treat junction reads as binomial draws given PSI, with PSI varying across replicates. The Huber formulation remains a useful simple baseline.

### 4.4 Recommendation

| Choice | Recommendation |
|---|---|
| Output | perturbed PSI via a log-odds offset from $p_0$; report $\Delta$PSI as a derived quantity |
| $p_0$ | an independent baseline (out-of-batch now; replicate split when feasible) |
| Loss | beta-binomial on knockdown read counts; Huber on $\Delta$PSI as the baseline to beat |
| Interaction | start with factorisation (section 3.2), then $g(h_e, z_R)$ with $z_R$ from computable sources |
| Context $c$ | keep the slot; it is constant in this corpus |
| Evaluation | always on $\Delta$PSI against held-out regulators and genes, with control batches kept whole |

---

## Summary

- Today's M4 models consume a strong event representation but have **no regulator embedding**; the regulator enters only through its eCLIP binding or through few-shot examples.
- The target design needs $z_R$ computable for unseen regulators, and an explicit interaction $g(h_e, z_R)$; factorisation is the natural first step.
- Predicting the perturbed PSI through a log-odds offset from an **independent** $p_0$, trained on read counts, respects the bounds of PSI, matches the ceiling effect, and avoids the control-noise trap.

**Next:** [Part 3, training and evaluation](03_training_and_evaluation.md).
