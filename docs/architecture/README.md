# Architecture — Multi-Layer Pipeline to Novel Isoforms

Three-layer architecture enabling progression from canonical prediction to novel isoform discovery:

```mermaid
%%{init: {'theme':'base', 'themeVariables': { 'fontSize':'15px','fontFamily':'ui-sans-serif, system-ui, sans-serif'}}}%%

graph TB
    subgraph AGENTIC["<b>🤖 AGENTIC LAYER</b> - Clinical Translation & Validation"]
        direction TB
        LIT["<b>📚 Literature Mining</b><br/>PubMed • arXiv<br/>Splice Databases"]:::agent
        EXP["<b>🧬 Expression Evidence</b><br/>GTEx • TCGA<br/>RNA-seq Junctions"]:::agent
        CLIN["<b>🏥 Clinical Integration</b><br/>ClinVar • COSMIC<br/>Disease Associations"]:::agent

        NEXUS["<b>🎯 Nexus Research Agent</b><br/>(Orchestrator)<br/>━━━━━━━━━━━━━━━<br/>• Evidence Aggregation<br/>• Validation Workflows<br/>• Drug Target Assessment<br/>• Report Generation"]:::orchestrator

        LIT --> NEXUS
        EXP --> NEXUS
        CLIN --> NEXUS

        OUTPUT1["<b>✅ OUTPUT</b><br/>Validated Novel Isoforms<br/>Drug Target Reports"]:::output
        NEXUS --> OUTPUT1
    end

    subgraph META["<b>🧠 META LAYER</b> - Adaptive Context-Aware Prediction"]
        direction TB

        MULTIMODAL["<b>🎨 Multimodal Evidence Fusion</b><br/>9 Modalities • 100 Features"]:::metalayer

        BASE["<b>📊 Base Scores</b><br/>Foundation Model<br/>Predictions (43)"]:::input
        SEQ["<b>🧬 Sequence + Genomic</b><br/>DNA Context • GC<br/>Conservation (19)"]:::input
        EPI["<b>🧪 Epigenetic + Chromatin</b><br/>H3K36me3 • H3K4me3<br/>ATAC-seq (18)"]:::input
        RNA["<b>🔬 RNA Evidence</b><br/>Junction Reads • RBP<br/>eCLIP Binding (20)"]:::input

        BASE --> MULTIMODAL
        SEQ --> MULTIMODAL
        EPI --> MULTIMODAL
        RNA --> MULTIMODAL

        FUSION["<b>⚡ Fusion Predictor</b><br/>+ Delta Scorer<br/>━━━━━━━━━━━━━━━<br/>Δ = Meta - Base<br/>High Δ → Novel Site!"]:::fusion

        MULTIMODAL --> FUSION

        DETECTOR["<b>🔍 Novel Site Detector</b><br/>━━━━━━━━━━━━━━━<br/>• High-confidence Filtering<br/>• Context Clustering<br/>• Multi-factor Scoring"]:::discovery

        RECON["<b>🧩 Isoform Reconstruction</b><br/>━━━━━━━━━━━━━━━<br/>• Transcript Assembly<br/>• ORF Validation<br/>• Functional Annotation"]:::discovery

        FUSION --> DETECTOR
        DETECTOR --> RECON

        OUTPUT2["<b>✅ OUTPUT</b><br/>Novel Splice Sites<br/>Reconstructed Isoforms"]:::output
        RECON --> OUTPUT2
    end

    subgraph BASE_LAYER["<b>🔬 BASE LAYER</b> - Foundation Models (Extensible)"]
        direction TB

        RUNNER["<b>⚙️ Base Model Runner</b><br/>Standardized I/O Protocol"]:::baselayer

        SA["<b>SpliceAI</b><br/>GRCh37<br/>Pre-trained"]:::foundation
        OSA["<b>OpenSpliceAI</b><br/>GRCh38/MANE<br/>Pre-trained"]:::foundation
        EXT["<b>Extensible</b><br/>Evo • GPT-based<br/>Any New Model"]:::foundation

        RUNNER --> SA
        RUNNER --> OSA
        RUNNER --> EXT

        RESOURCES["<b>📂 Genomic Resources</b><br/>━━━━━━━━━━━━━━━<br/>• GTF/FASTA Loading<br/>• Sequence Extraction<br/>• Splice Annotation<br/>• Resource Registry"]:::resources

        RESOURCES --> RUNNER

        OUTPUT3["<b>✅ OUTPUT</b><br/>Per-Nucleotide Scores<br/>Canonical Baseline (~10%)"]:::output
        SA --> OUTPUT3
        OSA --> OUTPUT3
        EXT --> OUTPUT3
    end

    FINAL["<b>🎉 NOVEL ISOFORM CATALOG</b><br/>━━━━━━━━━━━━━━━━━━━━━<br/>✓ Disease-Specific Isoforms<br/>✓ Variant-Induced Splicing<br/>✓ Tissue-Specific Transcripts<br/>✓ Druggable Targets + Evidence<br/>✓ Biomarker Candidates"]:::final

    OUTPUT3 --> META
    OUTPUT2 --> AGENTIC
    OUTPUT1 --> FINAL

    classDef agent fill:#0891b2,stroke:#0e7490,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef orchestrator fill:#7c3aed,stroke:#6d28d9,stroke-width:4px,color:#ffffff,font-weight:bold
    classDef metalayer fill:#8b5cf6,stroke:#7c3aed,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef input fill:#06b6d4,stroke:#0891b2,stroke-width:2px,color:#ffffff
    classDef fusion fill:#d946ef,stroke:#c026d3,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef discovery fill:#059669,stroke:#047857,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef baselayer fill:#1e40af,stroke:#1e3a8a,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef foundation fill:#3b82f6,stroke:#2563eb,stroke-width:2px,color:#ffffff
    classDef resources fill:#475569,stroke:#334155,stroke-width:2px,color:#ffffff
    classDef output fill:#ea580c,stroke:#c2410c,stroke-width:3px,color:#ffffff,font-weight:bold
    classDef final fill:#d97706,stroke:#b45309,stroke-width:4px,color:#ffffff,font-weight:bold,font-size:16px
```

---

## Layer Responsibilities

| Layer | Purpose | Output | Status |
|-------|---------|--------|--------|
| **Base Layer** | Canonical splice prediction (MANE) | Baseline scores for ~10% of sites | Done |
| **Feature Engineering** | Multimodal evidence fusion | 9-modality, 100-column enriched features | Done |
| **Foundation Models** | Evo2/SpliceBERT splice classification | Per-nucleotide embeddings + classifiers | Experimental |
| **Meta Layer** | Context-aware adaptive prediction (M1-M4) | Novel sites (90% beyond MANE) | Active |
| **Agentic Layer** | Multi-source validation + reports | Validated isoforms + drug targets | Planned |

---

## Feature Engineering

The multimodal pipeline fuses 9 data modalities into 100 feature columns per genomic position via a YAML-driven workflow:

| Modality | Columns | Source |
|----------|---------|--------|
| base_scores | 43 | Foundation model predictions (SpliceAI/OpenSpliceAI) |
| annotation | 3 | Ground truth splice labels |
| sequence | 3 | DNA context via pyfaidx |
| genomic | 4 | GC content, CpG density, dinucleotides |
| conservation | 9 | PhyloP/PhastCons (UCSC bigWig) |
| epigenetic | 12 | H3K36me3/H3K4me3 ChIP-seq (ENCODE) |
| junction | 12 | GTEx RNA-seq junction evidence |
| rbp_eclip | 8 | ENCODE RBP eCLIP binding peaks |
| chrom_access | 6 | ENCODE ATAC-seq chromatin accessibility |

See [`docs/multimodal_feature_engineering/feature_catalog.md`](../multimodal_feature_engineering/feature_catalog.md) for the complete feature reference and [`examples/features/docs/`](../../examples/features/docs/) for per-modality tutorials.

---

## Delta Score Analysis

The key innovation for novel isoform discovery is the delta score -- the difference between meta layer and base layer predictions:

```python
delta_score = meta_prediction - base_prediction

if delta_score > 0.3:  # High confidence
    # This splice site is context-dependent!
    # -> Novel isoform candidate
    # -> Not in MANE canonical set
    # -> Validate with RNA-seq, literature, conservation
```

- **Base layer** (SpliceAI/OpenSpliceAI): Trained on canonical annotations, detects ~10% of sites
- **Meta layer** (Context-aware): Learns from variants, disease, tissue context, detects the other 90%
- **Delta score** = Confidence that this is a real novel isoform, not noise

---

## Project Structure

```text
agentic-spliceai/
├── src/
│   ├── agentic_spliceai/
│   │   │
│   │   ├── splice_engine/           # Core splice prediction engine
│   │   │   │
│   │   │   ├── config/              # Configuration management
│   │   │   │   ├── genomic_config.py    # Config dataclass & loader
│   │   │   │   └── settings.yaml        # Default settings
│   │   │   │
│   │   │   ├── resources/           # Genomic resource management
│   │   │   │   ├── registry.py          # Path resolution for GTF/FASTA/models
│   │   │   │   └── schema.py            # Column standardization (splice_type, chrom)
│   │   │   │
│   │   │   ├── utils/               # Shared utilities
│   │   │   │   ├── dataframe.py         # DataFrame operations
│   │   │   │   ├── display.py           # Printing & formatting
│   │   │   │   └── filesystem.py        # File I/O helpers
│   │   │   │
│   │   │   ├── base_layer/          # Base model predictions
│   │   │   │   ├── models/              # Model configs + runner
│   │   │   │   │   ├── config.py            # BaseModelConfig, WorkflowConfig
│   │   │   │   │   └── runner.py            # BaseModelRunner
│   │   │   │   ├── prediction/          # Core prediction logic
│   │   │   │   ├── workflows/           # Chunked prediction pipeline
│   │   │   │   │   └── prediction.py        # PredictionWorkflow (checkpointing, resume)
│   │   │   │   ├── io/                  # Artifact management
│   │   │   │   │   └── artifacts.py         # ArtifactManager (atomic writes, mode-aware)
│   │   │   │   └── data/                # Data types & preparation
│   │   │   │
│   │   │   ├── features/            # Multimodal feature engineering
│   │   │   │   ├── pipeline.py          # FeaturePipeline (dependency resolution)
│   │   │   │   ├── workflow.py          # FeatureWorkflow (genome-scale)
│   │   │   │   ├── modality.py          # Modality protocol (ABC)
│   │   │   │   ├── verification.py      # Position alignment verification
│   │   │   │   └── modalities/          # 9 modalities:
│   │   │   │       ├── base_scores.py       # 43 engineered features
│   │   │   │       ├── annotation.py        # Ground truth labels (3)
│   │   │   │       ├── sequence.py          # DNA context via pyfaidx (3)
│   │   │   │       ├── genomic.py           # GC content, CpG, dinucs (4)
│   │   │   │       ├── conservation.py      # PhyloP/PhastCons bigWig (9)
│   │   │   │       ├── epigenetic.py        # H3K36me3/H3K4me3 ChIP-seq (12)
│   │   │   │       ├── junction.py          # GTEx RNA-seq junctions (12)
│   │   │   │       ├── rbp_eclip.py         # ENCODE RBP eCLIP binding (8)
│   │   │   │       └── chrom_access.py      # ENCODE ATAC-seq accessibility (6)
│   │   │   │
│   │   │   ├── eval/                # Cross-layer evaluation
│   │   │   │   ├── metrics.py           # TP/FP/FN, sensitivity, specificity
│   │   │   │   ├── output.py            # EvaluationOutputWriter
│   │   │   │   └── display.py           # Result visualization
│   │   │   │
│   │   │   ├── data/                # Cross-layer data utilities
│   │   │   │   └── sampling.py          # Balanced train/test sampling
│   │   │   │
│   │   │   ├── meta_layer/          # Meta-learning layer
│   │   │   │   ├── core/                # Configuration & schema
│   │   │   │   │   ├── config.py            # MetaLayerConfig
│   │   │   │   │   └── feature_schema.py    # Feature definitions (8 column groups)
│   │   │   │   ├── models/              # Neural network models
│   │   │   │   ├── training/            # Training pipeline
│   │   │   │   └── workflows/           # Meta-layer workflows
│   │   │   │
│   │   │   └── cli/                 # CLI entry points
│   │   │       ├── predict.py           # agentic-spliceai-predict
│   │   │       └── prepare.py           # agentic-spliceai-prepare
│   │   │
│   │   ├── agents/                  # Agentic workflows (WIP)
│   │   ├── server/                  # FastAPI splice service
│   │   └── analysis/                # Analysis tools & templates
│   │
│   └── nexus/                       # Research agent package
│       ├── agents/                      # Multi-agent pipeline
│       │   ├── research/                    # Research orchestrator
│       │   ├── planner/                     # Research planning
│       │   ├── researcher/                  # Information gathering
│       │   ├── writer/                      # Report writing
│       │   └── editor/                      # Report refinement
│       ├── core/                        # Core utilities
│       ├── cli/                         # CLI interface
│       └── templates/                   # Report templates
│
├── foundation_models/               # Experimental sub-project (own pyproject.toml)
│   ├── foundation_models/
│   │   ├── evo2/                        # Evo2-based exon classifier
│   │   │   ├── config.py                    # Evo2Config (device auto-detect)
│   │   │   ├── model.py                     # HuggingFace wrapper
│   │   │   ├── embedder.py                  # Chunked extraction + HDF5 cache
│   │   │   └── classifier.py               # ExonClassifier (linear/MLP/CNN/LSTM)
│   │   └── utils/                       # Quantization, chunking
│   ├── configs/skypilot/               # SkyPilot cloud deployment (RunPod)
│   ├── examples/                        # Learning path (01-05)
│   └── docs/                            # Sub-project documentation
│
├── server/                          # Standalone FastAPI services
│   ├── bio/                             # Bioinformatics Lab UI (port 8005)
│   │   ├── app.py                           # FastAPI + Jinja2 entry point
│   │   ├── bio_service.py                   # Core service (LRU cache, predictions)
│   │   └── templates/                       # HTML templates (Gene Browser, etc.)
│   ├── splice_service/                  # Splice prediction API (port 8004)
│   └── chart_service/                   # Chart/viz API (port 8003)
│
├── examples/                        # Learning path examples
│   ├── base_layer/                      # 5 scripts: prediction -> precomputation
│   ├── features/                        # 4 scripts: base scores -> genome-scale
│   ├── foundation_models/               # 5 scripts: resource check -> orchestrate
│   └── data_preparation/               # Data prep & ground truth generation
│
├── data/                            # Data directory (symlinked)
│   ├── ensembl/GRCh37/                  # Ensembl annotations
│   ├── mane/GRCh38/                     # MANE annotations
│   └── models/                          # Pre-trained model weights
│
├── notebooks/                       # Jupyter analysis & demos
├── docs/                            # Public documentation (MkDocs)
├── scripts/                         # Utility scripts
├── tests/                           # Unit tests
└── pyproject.toml                   # Package configuration
```

---

## Related Documentation

- [Package Organization](PACKAGE_ORGANIZATION.md) -- How the codebase is structured
- [Structure Guide](STRUCTURE.md) -- Directory structure overview
- [Processing Architecture](../base_layer/PROCESSING_ARCHITECTURE.md) -- Base layer architecture
- [Configuration System](../system_design/configuration_system.md) -- Pydantic-based configuration patterns

---

Last Updated: March 2026
