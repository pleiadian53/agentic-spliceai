# Novel Isoform Discovery (M3)

**Goals served**: isoform discovery (primary project goal)

**Tier**: Incubating

**Last updated**: 2026-04

---

## Problem

Canonical annotations (MANE, RefSeq) capture only ~10% of biologically
active splice sites. The remaining ~90% include disease-specific,
tissue-specific, and variant-induced isoforms — the targets of greatest
scientific and therapeutic interest. The M3 task is to identify these
**novel splice sites** that are absent from canonical annotations but
supported by orthogonal evidence (cross-species conservation, RNA-seq
junction reads, chromatin features).

Critically, the M3 model must be trained with junction reads held out as
a **label**, not a feature — otherwise it leaks information about the
very sites it should discover.

## User-facing functionality (planned)

- Predict high-confidence novel splice sites genome-wide, ranked by
  multimodal evidence
- Filter by tissue context (brain, immune, cancer) via GTEx junction
  overlap
- Produce per-site evidence cards (conservation, RBP binding, chromatin,
  foundation-model embeddings)
- Feed ranked candidates into downstream isoform assembly and agentic
  validation

## Driving examples (planned)

- (none yet — M3 training not started)

The existing `meta_m3_novel.yaml` feature config (junction excluded)
provides the feature-pipeline side; the training and inference scripts
are pending.

## `src/` surface (existing, ready for M3 training)

- `agentic_spliceai.splice_engine.features.*` — already supports
  junction-as-label via the `meta_m3_novel` config profile
- `agentic_spliceai.splice_engine.meta_layer.core.feature_schema` — M3
  variant already scoped
- Training / inference modules — to be added, likely extending the
  existing `sequence_model` and `training/*` infrastructure

## Evaluation (planned)

- **Dataset**: MANE-negative, annotation-rich positions with junction
  support held out as label
- **Baselines**: base model (canonical only), M1-S v2, random forest on
  conservation + chromatin
- **Key metrics**: recall of GTEx-supported non-canonical junctions,
  conservation enrichment of top-K candidates, literature-confirmed rate
- **Validation**: cross-tissue RNA-seq junction verification (GTEx v8
  already available at `data/mane/GRCh38/junction_data/`)

## Maturity tier and signals

**Current tier**: Incubating

**Signals supporting the tier**:

- Feature config (`meta_m3_novel.yaml`) exists with junction excluded
- GTEx v8 junction data aggregated (353K junctions, 54 tissues)
- M3-S training architecture scoped in meta-layer design docs
- No scripts yet, no results yet — pre-implementation

## Graduation signals

**To advance to Active, the application needs**:

- First training script working end-to-end (even at small scale)
- Baseline evaluation against random forest or M1-S-trained-on-negatives
- Reproducible benchmark on held-out chromosome(s)
- Initial results committed to `examples/meta_layer/results/`

## Known limitations (anticipated)

- Label definition is non-trivial: "novel" is a negative space (MANE-negative
  but evidence-positive). Label noise will dominate.
- GTEx junction support varies by tissue and depth — low-expression
  novel sites will be missed
- Adjacent application (M4 induced sites) overlaps conceptually but has
  a different training signal

## Related

- [Adaptive Splice Prediction](../adaptive_splice_prediction/README.md) — prerequisite for M3 training (shares infrastructure)
- [Multimodal Feature Engineering](../multimodal_features/README.md) — `meta_m3_novel.yaml` config
- [Agentic Validation](../agentic_validation/README.md) — downstream consumer of M3 candidates
- [Meta Layer Methods](../../meta_layer/methods/) — M1-M4 framework including M3 design
- [Use cases: Neurology](../use_cases.md#3-neurology-brain-specific-isoforms), [Oncology](../use_cases.md#1-oncology-cancer-specific-isoform-targets) — intended clinical applications
- [Roadmap: Phase 9](../../ROADMAP.md)
