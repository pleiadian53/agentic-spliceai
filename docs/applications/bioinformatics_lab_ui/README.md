# Bioinformatics Lab UI

**Goals served**: all goals (interactive exploration interface)

**Tier**: Mature

**Last updated**: 2026-04

---

## Problem

Much of splice-prediction research is exploratory: comparing base-model
predictions for a gene, inspecting a chromosome-wide distribution,
cross-referencing metrics across training runs. A CLI or notebook is
inefficient for these tasks. Domain scientists benefit from an interactive
web interface with gene browsing, on-demand prediction, and model-comparison
dashboards — all backed by the same `src/` library as the rest of the
system.

## User-facing functionality

- Browse and search ~19K genes with filters (chromosome, biotype, length)
- Request on-demand splice prediction for any gene with per-model
  switching (SpliceAI vs OpenSpliceAI)
- Visualize 3-track Plotly genome view (annotations, predictions, delta)
- Compare evaluation runs across models and thresholds in a metrics
  dashboard
- Peak-preserving downsampling for genes > 10K positions
- LRU-cached predictions keyed by `(gene, model)`

## Driving examples

- [`server/bio/`](../../../server/bio/) — FastAPI + Jinja2 + Plotly.js web service (port 8005)
- [`notebooks/bioinfo_ui/01_serpina1_copd_use_case/`](../../../notebooks/bioinfo_ui/01_serpina1_copd_use_case/) — SERPINA1 (COPD), UNC13A/STMN2 (ALS) demo notebook

Start the server:

```bash
mamba run -n agentic-spliceai python -m server.bio.app
# Browse: http://localhost:8005/
```

Pages:

- `/` — Gene Browser
- `/genome/{gene}` — Genome View with on-demand prediction
- `/metrics` — Metrics Dashboard

API endpoints:

**Gene browser + genome view:**

- `/api/genes`, `/api/genes/stats`, `/api/genes/chromosomes`
- `/api/genome/{gene}/predict?model=X&threshold=T`
- `/api/metrics/runs`, `/api/metrics/{run_id}`, `/api/metrics/compare`

**Ingestion-layer readiness** (Phase D3 — read-only wrappers over the
[data_preparation](../genomic_data_preparation/README.md) and
[multimodal_features](../multimodal_features/README.md) applications'
`get_status()` APIs; see [`server/bio/ingest_api.py`](../../../server/bio/ingest_api.py)):

- `/api/ingest/health` — liveness check
- `/api/ingest/data-prep/builds` — list configured base-model / build entries
- `/api/ingest/data-prep/status?build=GRCh38&annotation_source=mane` — canonical-path check
- `/api/ingest/data-prep/status?output_dir=/path` — explicit output-dir check
- `/api/ingest/features/profiles` — feature-profile catalog (default, full_stack, …)
- `/api/ingest/features/tracks?build=GRCh38&modality=conservation` — external-track catalog (UCSC URLs, ENCODE accessions)
- `/api/ingest/features/status?build=GRCh38&chromosomes=1,2,21,22` — per-chromosome feature-parquet readiness

All ingestion endpoints are **read-only**. Writes (running `prepare`)
remain CLI-only; a future async-job endpoint would be a separate router.

## `src/` surface

- `server.bio.app` — FastAPI app entry
- `server.bio.bio_service` — prediction cache, service logic
- `server.bio.config` — `MAX_CACHED_PREDICTIONS`, color palette, etc.
- `agentic_spliceai.splice_engine.base_layer.*` — prediction backend
- `agentic_spliceai.splice_engine.data.preparation` — annotation extraction
- `agentic_spliceai.splice_engine.eval.*` — metrics aggregation

Design patterns (see [`CLAUDE.md`](../../../CLAUDE.md) for details):

- LRU prediction cache (OrderedDict in `bio_service.py`), keyed by
  `(gene, model)`, threshold only affects classification (raw predictions
  cached)
- Peak-preserving downsampling for sparse peaky data (never naive
  `[::factor]`)
- `reload_dirs=["server/bio"]` in `app.py` prevents data writes from
  triggering uvicorn restart

## Evaluation

- **Genes browsed**: ~19K (filtered to canonical chromosomes)
- **Demo notebook**: SERPINA1 (COPD), UNC13A/STMN2 (ALS) clinical use cases
- **Throughput**: prediction cache hit rate dominates interactive workflows
- No formal benchmark — quality measured by interactive usability

## Maturity tier and signals

**Current tier**: Mature

**Signals supporting the tier**:

- Running FastAPI service with stable endpoint contracts
- Demo notebook demonstrating end-to-end clinical use cases
- LRU cache + downsampling patterns documented and tested in practice
- Used as the primary exploratory interface across sessions
- Phase 2.5 marked complete in [ROADMAP.md](../../ROADMAP.md)

## Graduation signals

**To advance to Product, the application needs**:

- Deployment guide (containerization, reverse proxy, auth)
- External hosting decision (intranet only today)
- Versioned API contract with deprecation policy
- Inference-path tests covering all endpoints
- Rate limiting / resource quotas for on-demand prediction

## Known limitations

- Single-process service; on-demand prediction blocks the event loop
  during model inference
- No authentication or access control — intended for local/dev use
- Prediction cache is in-memory only — cold start loses cache on restart
- No multi-user session support; threshold changes affect all users
- Plotly rendering for very large genes relies on downsampling; exact
  per-position view requires dedicated endpoint

## Related

- [Canonical Splice Prediction](../canonical_splice_prediction/README.md) — prediction backend
- [Use cases](../use_cases.md) — clinical scenarios the Lab UI is designed to support
- [Roadmap: Phase 2.5](../../ROADMAP.md)
- [Notebooks](../../../notebooks/bioinfo_ui/) — demonstration notebooks
