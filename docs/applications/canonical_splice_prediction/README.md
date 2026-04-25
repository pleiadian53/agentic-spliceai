# Canonical Splice Prediction

**Goals served**: adaptive splice prediction (foundation for all other applications)

**Tier**: Mature

**Last updated**: 2026-04

---

## Problem

Establish a **pluggable baseline splice-site prediction layer**. Given
a gene symbol, chromosome, or the whole genome, produce per-nucleotide
acceptor and donor probabilities — **not tied to any one model**. The
architectural novelty is the
[`BasePredictor` protocol](../../../src/agentic_spliceai/applications/base_layer/protocol.py):
any model — classical (SpliceAI, OpenSpliceAI), foundation-model-derived
(e.g., the SpliceBERT-based classifier trained via
[`examples/foundation_models/07a_direct_shard_splice_predictor.py`](../../../examples/foundation_models/07a_direct_shard_splice_predictor.py)),
or any future models — can be registered and served through the same CLI
and API surface as long as its output satisfies the **per-nucleotide
3-class scoring contract** (neither / acceptor / donor probabilities
aligned to genomic positions).

Without this abstraction, the base layer would be a wrapper over two
existing tools and nothing more. With it, the base layer is the
extension point that lets new prediction models (foundation-model
fine-tunes, architectural variants, ensemble strategies) plug into all
downstream applications — meta layer, variant analysis, isoform
discovery — with no downstream code changes.

**The foundation that all downstream layers build on:** every consumer
(meta layer, variant analysis, isoform discovery) reads the
same typed `PredictionResult`, so swapping or adding a predictor is a
registration concern, not an integration one.

## User-facing functionality

- Register any splice predictor satisfying the 3-class per-nt protocol
  (decorator-based for built-ins, YAML manifest for foundation-model
  checkpoints and external models)
- Predict splice sites for one or more genes by symbol, across a
  chromosome, or at whole-genome scale
- Currently registered predictors include SpliceAI (TF, GRCh37/Ensembl),
  OpenSpliceAI (PyTorch, GRCh38/MANE), and the SpliceBERT + dilated-CNN
  classifier (PyTorch, GRCh38/MANE); the set is open-ended
- Chunked, resumable genome-scale workflows with per-chromosome parquet outputs
- Evaluate predictions against MANE annotations (TP/FP/FN per splice type)
- Cross-predictor benchmarking on the same input through a uniform CLI

## Driving examples

- [`examples/base_layer/01_phase1_prediction.py`](../../../examples/base_layer/01_phase1_prediction.py) — single-gene prediction, smoke test
- [`examples/base_layer/02_chromosome_prediction.py`](../../../examples/base_layer/02_chromosome_prediction.py) — chromosome-wide prediction
- [`examples/base_layer/03_prediction_with_evaluation.py`](../../../examples/base_layer/03_prediction_with_evaluation.py) — prediction + evaluation against MANE
- [`examples/base_layer/04_chunked_prediction.py`](../../../examples/base_layer/04_chunked_prediction.py) — chunked workflow with checkpointing
- [`examples/base_layer/05_genome_precomputation.py`](../../../examples/base_layer/05_genome_precomputation.py) — whole-genome precomputation
- [`examples/base_layer/EVALUATION_GUIDE.md`](../../../examples/base_layer/EVALUATION_GUIDE.md) — evaluation methodology

Supporting notebooks:

- [`notebooks/base_layer/01_phase1_basics/`](../../../notebooks/base_layer/01_phase1_basics/) — phase 1 fundamentals walkthrough

## `src/` surface

**Library (stable):**

- `agentic_spliceai.splice_engine.base_layer.BaseModelRunner` — unified predictor interface
- `agentic_spliceai.splice_engine.base_layer.workflows.PredictionWorkflow` — chunked/checkpointed orchestration
- `agentic_spliceai.splice_engine.base_layer.io.ArtifactManager` — atomic per-chromosome parquet writes
- `agentic_spliceai.splice_engine.resources.registry` — model/genome resource resolution
- `agentic_spliceai.splice_engine.cli.predict` — infrastructure CLI

**Application package** (`src/agentic_spliceai/applications/base_layer/`):

- `BasePredictor` protocol — per-nucleotide 3-class scores (neither / acceptor / donor)
- Plugin `registry` — built-in (`@register_predictor`) + YAML manifest for foundation-model-derived predictors
- Built-in adapters: `predictors/spliceai.py`, `predictors/openspliceai.py` (wrap `BaseModelRunner`)
- `predictors/from_checkpoint.py` — foundation-model-derived predictors (e.g., SpliceBERT + SpliceClassifier head trained via `examples/foundation_models/07a`)
- `runner.py`, `evaluator.py` — packaged versions of the `examples/base_layer/` orchestration
- `tracking.py` — optional wandb integration for evaluation runs
- `cli.py` — unified subcommand CLI

**CLI entry points:**

- `agentic-spliceai-predict` — infrastructure CLI (pre-existing, single predictor)
- `agentic-spliceai-base` — **application CLI** (pluggable predictors):
  ```bash
  agentic-spliceai-base list-predictors
  agentic-spliceai-base predict --predictor openspliceai --genes BRCA1 TP53
  agentic-spliceai-base predict --predictor splicebert_classifier --genes BRCA1
  agentic-spliceai-base evaluate --predictor openspliceai --chromosomes 21 22 \
      --track --tracking-project agentic-spliceai
  ```

**Data-preparation pre-flight** (built-in on `predict` and `evaluate`):

Before running, the CLI calls
[`applications.data_preparation.get_status()`](../../../src/agentic_spliceai/applications/data_preparation/status.py)
on the predictor's canonical build directory (e.g., `data/mane/GRCh38/`
for OpenSpliceAI). Missing artifacts produce a warning with the exact
gap-fill command; `--strict-preflight` turns the warning into a hard
failure; `--skip-preflight` disables the check.

**Registered predictors (as of 2026-04-19):**

| Name | Build | Annotation | Backend |
|---|---|---|---|
| `spliceai` | GRCh37 | Ensembl | TensorFlow ensemble (5 models) |
| `openspliceai` | GRCh38 | MANE | PyTorch ensemble (5 models), MPS-capable |
| `splicebert_classifier` | GRCh38 | MANE | Foundation-model-derived (SpliceBERT → dilated-CNN head) |

Adding a new predictor: drop a module under `predictors/` with a `@register_predictor` decorator for built-ins, or add a `factory:` entry to `configs/predictors.yaml` for externally-trained classifiers. The protocol contract is the only integration point.

## Evaluation

- **Dataset**: MANE annotations (GRCh38) for OpenSpliceAI; Ensembl for SpliceAI
- **Scale verified**: chr22 (423 genes, 17.6M positions, 5 chunks, 12.4 min on MPS)
- **Key metrics**: TP/FP/FN per splice type, acceptor/donor delta scores
- **Cross-model consistency**: 17,571 shared protein-coding genes between SpliceAI and OpenSpliceAI

## Maturity tier and signals

**Current tier**: Mature

**Signals supporting the tier**:

- 5 example scripts covering progressive scale (gene → chromosome → genome)
- Stable CLI surface (`agentic-spliceai-predict`) since Phase 1 completion
- Verified at genome scale on both MacBook M1 (MPS) and A40 GPUs
- Consumed by every downstream application (meta layer, variant analysis, features)
- Full-genome feature pipeline (24/24 chromosomes) built on top
- **Packaged application** at `src/agentic_spliceai/applications/base_layer/` exposing a `BasePredictor` protocol + plugin registry + dedicated CLI (`agentic-spliceai-base`). Three predictors registered (SpliceAI, OpenSpliceAI, SpliceBERT-classifier).
- Smoke-tested locally on M1 16GB: BRCA1 via SpliceBERT (81K positions, 20.6s); TP53 via OpenSpliceAI (19K positions, 2.7s)

## Graduation signals

**To advance to Product, the application needs**:

- Versioned artifact packaging for base model weights
- Inference-path test suite in `tests/`
- Stable deployment guide (e.g., container image, service manifest)
- Decision to take on external-user maintenance commitment

Currently closest candidate for product promotion if that decision is
taken — but there is no active push to promote.

## Known limitations

- Conservation bigWig streaming is the bottleneck (~2 hrs for 562K positions)
- pyBigWig has no timeout — laptop standby kills remote connections
- Each registered predictor declares its own genomic build and annotation
  source (e.g., SpliceAI → GRCh37/Ensembl, OpenSpliceAI → GRCh38/MANE,
  SpliceBERT-classifier → GRCh38/MANE). Cross-predictor comparisons must
  respect these constraints; the registry surfaces them explicitly but
  does not translate coordinates between builds.
- Base layer alone captures only ~10% of biologically active splice sites (motivation for meta layer)

## Related

- [Adaptive Splice Prediction](../adaptive_splice_prediction/README.md) — meta layer built on these predictions
- [Variant Effect Analysis](../variant_analysis/README.md) — uses base layer for ref/alt delta computation
- [Splice Prediction Guide](../../tutorials/SPLICE_PREDICTION_GUIDE.md)
- [Base Layer Architecture](../../base_layer/)
- [Roadmap: Phase 1 + Phase 3](../../ROADMAP.md)
