# Genomic Data Preparation

**Goals served**: cross-cutting (foundation for all other applications)

**Tier**: Active

**Last updated**: 2026-04

---

## Problem

Every downstream application depends on consistent, validated genomic
data: reference FASTA, gene annotations (GTF), splice-site ground truth,
chromosome splits, and resource resolution. Inconsistent coordinate
systems (0-based vs 1-based, chr prefix vs bare), mismatched genome
builds (GRCh37 vs GRCh38), or silent annotation-source drift are common
failure modes that cascade into every model.

This application provides a stable data-preparation layer with explicit
resource manifests, coordinate-system guardrails, and validated MANE /
Ensembl annotation sources.

## User-facing functionality

- Extract per-gene sequence and annotations from a reference genome
- Produce splice-site ground truth TSVs (donor/acceptor/neither) aligned
  to MANE or Ensembl
- Validate MANE vs Ensembl metadata consistency for a chosen gene set
- Generate balanced chromosome splits matching the SpliceAI convention
- Resolve model/genome/annotation resources via a central registry

## Driving examples

- [`examples/data_preparation/01_prepare_gene_data.py`](../../../examples/data_preparation/01_prepare_gene_data.py) — gene data extraction
- [`examples/data_preparation/02_prepare_splice_sites.py`](../../../examples/data_preparation/02_prepare_splice_sites.py) — splice-site annotation generation
- [`examples/data_preparation/03_full_data_pipeline.py`](../../../examples/data_preparation/03_full_data_pipeline.py) — end-to-end preparation pipeline
- [`examples/data_preparation/04_generate_ground_truth.py`](../../../examples/data_preparation/04_generate_ground_truth.py) — ground truth TSV generation
- [`examples/data_preparation/validate_mane_metadata.py`](../../../examples/data_preparation/validate_mane_metadata.py) — MANE validation utility

## `src/` surface

**Library (stable):**

- `agentic_spliceai.splice_engine.base_layer.data.preparation` — gene and splice-site extraction
- `agentic_spliceai.splice_engine.resources.schema.ensure_chrom_column` — boundary guardrail (canonical `chrom` column)
- `agentic_spliceai.splice_engine.resources.registry` — model/genome/annotation resource resolver
- `agentic_spliceai.splice_engine.resources.model_resources.get_model_resources` — per-model config lookup
- `agentic_spliceai.splice_engine.eval.splitting` — balanced chromosome split helpers

**Application package** (`src/agentic_spliceai/applications/data_preparation/`):

- `manifest.py` — versioned `IngestManifest` (inputs + artifacts + SHA-256 hashes + timestamps)
- `pipeline.py` — `prepare_build()` orchestrator + `resolve_canonical_output_dir()` helper
- `status.py` — `DataPrepStatus` readiness query (what's done / missing / stale)
- `steps.py` — thin wrappers over library calls (gene_features, splice_sites, chromosome_split, validate)
- `cli.py` — unified subcommand CLI

**CLI entry points:**

- `agentic-spliceai-prepare` — infrastructure CLI (pre-existing)
- `agentic-spliceai-ingest` — **application CLI** (ingestion layer):
  ```bash
  agentic-spliceai-ingest list-builds
  agentic-spliceai-ingest status --canonical --build GRCh38 --annotation-source mane
  agentic-spliceai-ingest prepare --inplace --build GRCh38 --annotation-source mane --dry-run
  agentic-spliceai-ingest prepare --build GRCh38 --annotation-source mane \
      --output-dir output/ingest/my_run
  agentic-spliceai-ingest validate --build GRCh38 --annotation-source mane \
      --output-dir data/mane/GRCh38
  ```

**Production-safety guards:**

- `--output-dir` is **required** for throwaway runs
- `--inplace` is the opt-in flag for writing to the resource-manager-resolved canonical dir (`data/<source>/<build>/`)
- Existing artifacts are **preserved** by default; regeneration requires `--force`
- `status --canonical` and `validate` are fully read-only

**Artifacts produced** (per output dir):

| Name | File | Step | Purpose |
|---|---|---|---|
| `gene_features` | `gene_features.parquet` or `.tsv` | gene_features | Gene metadata table |
| `splice_sites` | `splice_sites_enhanced.tsv` | splice_sites | Donor/acceptor ground truth |
| `chromosome_split` | `chromosome_split.json` | chromosome_split | SpliceAI-convention train/test split |
| `ingest_manifest` | `ingest_manifest.json` | (automatic) | Versioned record: inputs + artifacts + hashes + timestamps |

**Optional experiment tracking** (silent fallback when wandb is not installed
or no API key is set):

```bash
agentic-spliceai-ingest prepare --inplace \
    --build GRCh38 --annotation-source mane \
    --track --tracking-project agentic-spliceai-data-preparation
```

Logs per-step durations, rows, success flags, and the resulting
`ingest_manifest.json` as a W&B artifact. Shared with base_layer and
multimodal_features via
[`applications._common.tracking`](../../../src/agentic_spliceai/applications/_common/tracking.py).

## Evaluation

- **Genome builds**: GRCh37 (no chr prefix, MT only), GRCh38 (chr prefix, M+MT)
- **Annotation sources**: MANE (~19K genes, OpenSpliceAI-compatible), Ensembl (~57K genes including pseudogenes)
- **Cross-model genes**: 17,571 shared protein-coding genes across SpliceAI + OpenSpliceAI
- **Aliases**: `hg38 → GRCh38`, `hg19 → GRCh37` resolved by registry

## Maturity tier and signals

**Current tier**: Active (moving toward Mature)

**Signals supporting the tier**:

- 4 numbered scripts with stable args
- Shared by every downstream application
- Canonical schema columns (`chrom`, `splice_type`) enforced at I/O boundaries
- Genome-wide annotation extraction (no `force_extract` required anymore)
- MANE metadata validation utility exists
- **Packaged application** at `src/agentic_spliceai/applications/data_preparation/` exposing a versioned manifest, readiness API, and dedicated CLI (`agentic-spliceai-ingest`)
- **Production-path completeness check**: `status --canonical` detects which artifacts are present in `data/<source>/<build>/` without modifying them. Verified on `data/mane/GRCh38/` (gene_features + splice_sites present, chromosome_split gap-filled non-destructively)

## Graduation signals

**To advance to Mature, the application needs**:

- Per-chromosome sequence caching (`gene_sequence_{chrom}.parquet`) surfaced as a `step_gene_sequences` step (currently lazy-loaded by base predictors)
- Inference-path tests for the pipeline + status API
- Modality-side equivalent for the meta-layer (planned: `src/agentic_spliceai/applications/multimodal_features/`)
- Declared coordinate-system invariants in the API
- Integration hook from base-layer CLI (call `get_status()` before running, surface warnings when incomplete)

## Known limitations

- `DYLD_LIBRARY_PATH` in `.zshrc` can cause torch import failures during prep (documented in `dev/errors/`)
- GTF parsing is memory-bound for very large annotations; use streaming mode
- MANE and Ensembl use different transcript IDs — cross-reference requires explicit mapping
- Legacy `seqname` alias preserved for backward compatibility — do not remove without a deprecation cycle

## Related

- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — primary consumer
- [Multimodal Feature Engineering](../multimodal_features/README.md) — consumes annotations and resources
- [Variant Effect Analysis](../variant_analysis/README.md) — consumes reference FASTA and annotations
- [Splice Prediction Guide](../../tutorials/SPLICE_PREDICTION_GUIDE.md)
- [Roadmap: Phase 2](../../ROADMAP.md)
