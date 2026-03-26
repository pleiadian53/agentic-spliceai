# API Endpoints Reference

This document covers the current HTTP API endpoints for Agentic-SpliceAI services. As the project matures, APIs will grow to cover data preparation, multimodal feature engineering, foundation model inference, and architecture-specific layers.

---

## Splice Analysis Service (port 8004)

The splice analysis service provides AI-powered analysis of splice site data.

### GET /analyses

List available analysis templates.

**Response**: Array of template objects with `name`, `description`, and `parameters` fields.

### POST /analyze/template

Generate analysis using a predefined template.

```bash
curl -X POST http://localhost:8004/analyze/template \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/splice_sites_enhanced.tsv",
    "analysis_type": "high_alternative_splicing",
    "model": "gpt-4o-mini"
  }'
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_path` | string | yes | Path to splice site data file (TSV, CSV, Parquet) |
| `analysis_type` | string | yes | Template name (e.g., `high_alternative_splicing`, `splice_site_genomic_view`, `exon_complexity`, `strand_bias`, `gene_transcript_diversity`) |
| `model` | string | no | LLM model to use (default: `gpt-4o-mini`) |

### POST /analyze/exploratory

Generate a custom exploratory analysis from a free-text research question.

```bash
curl -X POST http://localhost:8004/analyze/exploratory \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/splice_sites_enhanced.tsv",
    "research_question": "What is the relationship between gene length and splice site density?",
    "model": "gpt-4o"
  }'
```

**Parameters**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dataset_path` | string | yes | Path to splice site data file |
| `research_question` | string | yes | Free-text research question |
| `model` | string | no | LLM model to use (default: `gpt-4o-mini`) |

---

## Bioinformatics Lab (port 8005)

Interactive web UI for gene browsing, splice site visualization, and evaluation metrics.

### Gene Browser

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Gene Browser page (HTML) |
| GET | `/api/genes` | List genes with filtering and pagination |
| GET | `/api/genes/stats` | Summary statistics (gene counts, chromosome distribution) |
| GET | `/api/genes/chromosomes` | List available chromosomes |

**`/api/genes` query parameters**: `search`, `chromosome`, `strand`, `page`, `per_page`, `sort_by`, `sort_order`

### Genome View

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/genome/{gene}` | Genome View page for a specific gene (HTML) |
| GET | `/api/genome/{gene}/predict` | Run splice site prediction for a gene |

**`/api/genome/{gene}/predict` query parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | `openspliceai` | Base model (`spliceai` or `openspliceai`) |
| `threshold` | float | `0.5` | Classification threshold for splice site calls |

### Metrics Dashboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/metrics` | Metrics Dashboard page (HTML) |
| GET | `/api/metrics/runs` | List available evaluation runs |
| GET | `/api/metrics/{run_id}` | Get detailed metrics for a specific run |
| GET | `/api/metrics/compare` | Compare metrics across multiple runs |

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes (for splice service) | OpenAI API key for LLM-powered analysis |
| `SPLICE_AGENT_PORT` | No | Splice service port (default: 8004) |
| `SPLICE_AGENT_HOST` | No | Splice service host (default: 0.0.0.0) |

### Supported LLM Models

The splice analysis service supports any OpenAI-compatible model, including:
- `gpt-4o-mini` (default, fast and cost-effective)
- `gpt-4o` (higher quality analysis)
- `gpt-5-mini` (latest generation)

---

## Data Format

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `chrom` | string | Chromosome (e.g., `chr1`, `chrX`) |
| `position` | int | Genomic position (1-based) |
| `splice_type` | string | Splice site type (`donor` or `acceptor`) |
| `strand` | string | Strand (`+` or `-`) |

### Optional Columns

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | string | Gene symbol (e.g., `BRCA1`) |
| `gene_id` | string | Ensembl gene ID (e.g., `ENSG00000012048`) |
| `transcript_id` | string | Ensembl transcript ID (e.g., `ENST00000357654`) |
| `exon_rank` | int | Exon number within the transcript |

### Example TSV Data

```tsv
chrom	position	splice_type	strand	gene_name
chr17	43044295	donor	-	BRCA1
chr17	43044294	acceptor	-	BRCA1
chr17	43045802	donor	-	BRCA1
```

---

See also: `docs/system_design/configuration_system.md` for Pydantic-based configuration patterns
