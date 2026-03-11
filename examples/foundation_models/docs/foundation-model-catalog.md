# Foundation Model Catalog for Splice Site Prediction

Reference guide for genomic foundation models supported by the GPU runner infrastructure.
Each model is declared as a dependency profile in `foundation_models/configs/gpu_config.yaml`
and can be selected via `--model <name>` on any ops script.

Only the active model's dependencies are installed — the rest are inert declarations.

```bash
# Use default (evo2)
python examples/foundation_models/ops_provision_cluster.py

# Use a different model
python examples/foundation_models/ops_provision_cluster.py --model splicebert

# No model deps (just infrastructure)
python examples/foundation_models/ops_provision_cluster.py --model none
```

---

## Evo2 (Arc Institute) — Primary

| Field | Value |
|-------|-------|
| **Profile key** | `evo2` |
| **pip** | `evo2` |
| **Sizes** | 7B (14 GB VRAM), 40B (40 GB VRAM) |
| **Architecture** | StripedHyena (Mamba SSM + Attention hybrid) |
| **Context** | Up to 1M bp at single-nucleotide resolution |
| **HuggingFace** | `arcinstitute/evo2-7b`, `arcinstitute/evo2-40b` |
| **Paper** | Nguyen et al. 2025, "Sequence modeling and design from molecular to genome scale with Evo 2" |
| **License** | Apache 2.0 |

### Why Evo2

- Causal DNA language model trained on 9.3 trillion tokens across all domains of life
- Learns genomic structure (coding regions, splice sites, regulatory elements) from sequence alone
- Embeddings from intermediate layers (e.g., `blocks.26`) are highly informative for
  downstream classification — no fine-tuning needed for many tasks
- Our current primary model for the sparse exon classifier (Brixi et al. 2026, Section 4.3.9)

### Embedding Extraction

```python
from evo2 import Evo2
import torch

model = Evo2("evo2_7b")
input_ids = torch.tensor(
    model.tokenizer.tokenize("ACGTACGT..."), dtype=torch.int
).unsqueeze(0).to("cuda:0")

# Extract embeddings from an intermediate layer
outputs, embeddings = model(
    input_ids, return_embeddings=True, layer_names=["blocks.26.mlp.l3"]
)
# embeddings["blocks.26.mlp.l3"] has shape [1, seq_len, 4096] for 7b
```

### Key Notes

- **BFloat16 required** for training/fine-tuning (Mamba layers overflow in FP16, causing NaN loss)
- **FP8 patch needed** on GPUs < compute 8.9 (A40, A100) — monkey-patch `te.fp8_autocast`
- Intermediate embeddings outperform final-layer embeddings for downstream tasks
- Hidden dim: 4096 (7b), 8192 (40b) — probed at runtime via `_probe_hidden_dim()`

### Use Cases

- Sparse exon classifier: point-query embeddings at 1,500 random positions per gene
- Dense per-nucleotide predictor: sliding-window embeddings for full chromosomes
- Zero-shot variant effect prediction (see `notebooks/brca1/` in Evo2 repo)
- Sequence generation and design

---

## SpliceBERT (Sun Yat-sen University)

| Field | Value |
|-------|-------|
| **Profile key** | `splicebert` |
| **pip** | `multimolecule` |
| **Size** | 19.4M parameters |
| **Architecture** | BERT encoder (masked language model) |
| **Context** | 64–1024 nt (pre-mRNA sequences) |
| **HuggingFace** | `multimolecule/splicebert`, `multimolecule/splicebert.510nt` |
| **Paper** | Chen & Zheng 2024, Briefings in Bioinformatics 25(3) |
| **License** | AGPL-3.0 (via MultiMolecule) |

### Why SpliceBERT

- Pre-trained on 2 million pre-mRNA sequences from 72 vertebrate species (65 billion nucleotides)
- Purpose-built for RNA splicing — unlike general DNA models, it was trained exclusively on
  pre-mRNA sequences which include splice site context
- Outperforms DNABERT on human and non-human splice site prediction tasks
- Lightweight (19.4M params) — runs on consumer GPUs or even CPU
- Cross-species generalization: trained on 72 vertebrates, transfers to unseen species

### Embedding Extraction

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("multimolecule/splicebert", trust_remote_code=True)
model = AutoModel.from_pretrained("multimolecule/splicebert", trust_remote_code=True)

# SpliceBERT expects RNA-style input (T is fine, internally handled)
sequence = "ACGTACGTNNACGT"
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Per-nucleotide embeddings: [1, seq_len, 512]
embeddings = outputs.last_hidden_state

# Pooled embedding (CLS token): [1, 512]
cls_embedding = embeddings[:, 0, :]
```

### Key Notes

- Minimum sequence length: 64 nt (trained on 64–1024 nt windows)
- Tokenization: single-nucleotide (N=5, A=6, C=7, G=8, T/U=9)
- The `multimolecule` package is a general framework that also hosts other RNA models
- Hidden dim: 512

### Use Cases

- Splice site classification (donor/acceptor/neither) — directly comparable to our pipeline
- Zero-shot variant effect prediction on splicing
- Branchpoint prediction
- Cross-species splice site transfer learning (72 vertebrates in training set)
- Lightweight alternative to Evo2 when VRAM is limited or task is RNA-specific

---

## AlphaGenome (Google DeepMind)

| Field | Value |
|-------|-------|
| **Profile key** | `alphagenome` |
| **pip** | `alphagenome` |
| **Architecture** | Multi-head transformer with specialized output heads |
| **Context** | 1 Mb (1,048,576 bp) |
| **HuggingFace** | `google/alphagenome-all-folds` |
| **Paper** | Cheng et al. 2025, Nature 2026; Nature Genetics 57, 949-961 (2025) |
| **License** | Non-commercial (research only) |

### Why AlphaGenome

- State-of-the-art on 25 of 26 variant effect prediction evaluations
- Predicts thousands of functional genomic tracks from a single 1 Mb input including:
  - Splice site positions (donor/acceptor)
  - Splice site usage (tissue-specific quantitative)
  - Splice junction coordinates and strength (which donor connects to which acceptor)
- Multi-modal output: chromatin accessibility, histone marks, gene expression, and splicing
  from a single model
- API-based — no local GPU needed for inference (but local weights also available)

### Usage (API Mode)

```python
from alphagenome.data import genome
from alphagenome.models import dna_client

# API key from https://deepmind.google.com/science/alphagenome
model = dna_client.create("YOUR_API_KEY")

interval = genome.Interval(chromosome="chr22", start=35677410, end=36725986)
variant = genome.Variant(
    chromosome="chr22",
    position=36201698,
    reference_bases="A",
    alternate_bases="C",
)

# Request splice-related outputs
outputs = model.predict_variant(
    interval=interval,
    variant=variant,
    ontology_terms=["UBERON:0001157"],  # tissue ontology
    requested_outputs=[
        dna_client.OutputType.SPLICE_SITES,
        dna_client.OutputType.SPLICE_SITE_USAGE,
        dna_client.OutputType.SPLICE_JUNCTIONS,
    ],
)

# Access splice junction predictions (sashimi-style arcs)
junctions = outputs.splice_junctions
```

### Usage (Local Weights)

```python
# Source code + weights available for non-commercial use
# pip install git+https://github.com/google-deepmind/alphagenome_research.git
# Weights: google/alphagenome-all-folds on HuggingFace
```

### Key Notes

- **Non-commercial license** — cannot be used in production/clinical settings
- API mode requires internet access and an API key
- Local inference requires significant compute (weights are large)
- Outputs are pre-computed tracks, not raw embeddings — different usage pattern than Evo2/SpliceBERT
- Particularly strong for variant effect prediction (VEP) on splicing

### Use Cases

- Variant effect prediction: "does this SNP disrupt a splice site?"
- Splice junction discovery: identify which donors connect to which acceptors
- Tissue-specific splice site usage quantification
- Benchmarking: compare our Evo2-based predictions against AlphaGenome as an oracle
- Multi-modal analysis: correlate splice predictions with chromatin/expression tracks

---

## HyenaDNA (Hazy Research / Stanford)

| Field | Value |
|-------|-------|
| **Profile key** | `hyenadna` |
| **pip** | `hyena-dna` |
| **Sizes** | Tiny (1.6M), Small (7M), Medium (70M), Large (300M) |
| **Architecture** | Hyena operator (sub-quadratic long-range convolution) |
| **Context** | Up to 1M bp at single-nucleotide resolution |
| **HuggingFace** | `LongSafari/hyenadna-small-32k-seqlen`, etc. |
| **Paper** | Nguyen et al. 2023, NeurIPS 2023 |
| **License** | Apache 2.0 |

### Why HyenaDNA

- Pioneered single-nucleotide resolution at long context (up to 1M bp)
- Sub-quadratic scaling via Hyena operator — much cheaper than full attention at long contexts
- Multiple model sizes — from tiny (1.6M, runs on CPU) to large (300M)
- Well-supported HuggingFace integration with character-level DNA tokenizer

### Embedding Extraction

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "LongSafari/hyenadna-medium-450k-seqlen", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "LongSafari/hyenadna-medium-450k-seqlen", trust_remote_code=True
)

sequence = "ACGTACGT" * 1000
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Per-nucleotide embeddings
embeddings = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
```

### Key Notes

- Character-level tokenizer: A, C, G, T, N as individual tokens
- Pre-trained on human reference genome only (not multi-species like SpliceBERT)
- Good baseline for long-range dependency tasks where context length matters most
- Hidden dim varies by model size (e.g., 256 for medium)

### Use Cases

- Long-range splice site prediction (where 100kb+ context matters)
- Embedding extraction for downstream classifiers (similar pattern to Evo2)
- Lightweight alternative for tasks where smaller models suffice
- Benchmarking: context length ablation studies

---

## Pangolin (Broad Institute / Columbia)

| Field | Value |
|-------|-------|
| **Profile key** | `pangolin` |
| **pip** | `pangolin-splice` |
| **Architecture** | CNN (similar to SpliceAI) with tissue-specific heads |
| **Context** | 5,000 bp |
| **GitHub** | `tkzeng/Pangolin` |
| **Paper** | Zeng & Li 2022, Genome Biology 23:263 |
| **License** | MIT |

### Why Pangolin

- Directly predicts splice site **strength** (not just presence/absence)
- Tissue-specific predictions: heart, liver, brain, testis
- Outperforms SpliceAI and MMSplice on splice site usage prediction
- VCF-based variant analysis built in — takes VCF + FASTA + gene annotation as input

### Usage

```bash
# Command-line variant analysis
pangolin variants.vcf hg38.fa gencode.db output_prefix

# Requires: reference FASTA, gffutils database (created via create_db.py)
```

### Key Notes

- CNN architecture — no embeddings to extract (outputs are scores, not representations)
- 5,000 bp context — much shorter than Evo2/HyenaDNA
- Tissue-specific output heads are the key differentiator from SpliceAI
- Best suited for variant effect prediction rather than embedding-based classification

### Use Cases

- Tissue-specific splice variant effect prediction
- Clinical variant interpretation (VCF analysis)
- Comparison baseline against our tissue-agnostic Evo2 approach
- Complement our predictions with tissue-specific context

---

## DNABERT-2 (MAGICS Lab)

| Field | Value |
|-------|-------|
| **Profile key** | `dnabert` |
| **pip** | `dnabert2` |
| **Size** | 117M parameters |
| **Architecture** | BERT with BPE tokenization + ALiBi positional encoding |
| **Context** | ~4,000 bp (BPE-dependent) |
| **HuggingFace** | `zhihan1996/DNABERT-2-117M` |
| **Paper** | Zhou et al. 2024, ICLR 2024 |
| **License** | BSD-3-Clause |

### Why DNABERT-2

- Multi-species pre-training (not human-only)
- BPE tokenization replaces k-mer tokenization — more efficient and flexible
- ALiBi positional encoding allows some length generalization
- 21x fewer parameters and 56x less GPU time than DNABERT-1
- Strong baseline on the GUE (Genome Understanding Evaluation) benchmark

### Embedding Extraction

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M", trust_remote_code=True
)

sequence = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
inputs = tokenizer(sequence, return_tensors="pt")["input_ids"]
hidden_states = model(inputs)[0]  # [1, seq_len_tokens, 768]

# Mean-pooled embedding
embedding = torch.mean(hidden_states[0], dim=0)  # [768]
```

### Key Notes

- BPE tokenization means token count != nucleotide count — embeddings are per-token, not per-nucleotide
- Hidden dim: 768
- Moderate context length (~4K bp) — shorter than Evo2 but sufficient for many splice tasks
- Well-suited as a general-purpose DNA encoder baseline

### Use Cases

- Multi-species splice site prediction
- General DNA sequence classification baseline
- Transfer learning to non-human genomes
- Benchmarking: compare BPE-based encoding vs single-nucleotide models

---

## Comparison Matrix

| Model | Params | Context | Per-nt Embeddings | Splice-Specific | Tissue-Specific | License | Min GPU |
|-------|--------|---------|-------------------|-----------------|-----------------|---------|---------|
| **Evo2 7b** | 7B | 1M bp | Yes | No (general) | No | Apache 2.0 | A40 (48 GB) |
| **SpliceBERT** | 19.4M | 1024 nt | Yes | **Yes** | No | AGPL-3.0 | Any (CPU OK) |
| **AlphaGenome** | Large | 1M bp | No (tracks) | **Yes** | **Yes** | Non-commercial | API or large |
| **HyenaDNA** | 1.6M–300M | 1M bp | Yes | No (general) | No | Apache 2.0 | Tiny: CPU |
| **Pangolin** | ~10M | 5K bp | No (scores) | **Yes** | **Yes** | MIT | Any (CPU OK) |
| **DNABERT-2** | 117M | ~4K bp | Per-token | No (general) | No | BSD-3 | Any (CPU OK) |

### Recommended Combinations

**For our sparse exon classifier** (current focus):
- Primary: Evo2 7b (frozen embeddings → MLP classifier)
- Validation: SpliceBERT (lightweight, splice-specific — good sanity check)

**For variant effect prediction**:
- Primary: AlphaGenome (SOTA VEP, if non-commercial is acceptable)
- Alternative: Pangolin (tissue-specific, open license)

**For multi-species generalization**:
- SpliceBERT (72 vertebrates) or DNABERT-2 (multi-species BPE)

**For ultra-long-range context**:
- Evo2 (1M bp, richest representations) or HyenaDNA (1M bp, lighter)

---

## Adding a New Model

1. Add a profile to `foundation_models/configs/gpu_config.yaml`:

```yaml
models:
  my_model:
    pip: "my-package extra-dep"
    description: "My model — short description"
```

2. Select it on any ops script:

```bash
python examples/foundation_models/ops_provision_cluster.py --model my_model
python examples/foundation_models/ops_run_pipeline.py --execute --model my_model \
    -- python your_training_script.py
```

3. For models without pip packages (GitHub-only), use `extra_setup`:

```bash
python examples/foundation_models/ops_run_pipeline.py --execute --model none \
    --extra-setup "pip install git+https://github.com/org/model.git" \
    -- python your_script.py
```

---

## References

- Evo2: [GitHub](https://github.com/ArcInstitute/evo2) | [PyPI](https://pypi.org/project/evo2/)
- SpliceBERT: [GitHub](https://github.com/chenkenbio/SpliceBERT) | [Paper](https://academic.oup.com/bib/article/25/3/bbae163/7644137) | [MultiMolecule](https://multimolecule.danling.org/models/splicebert/)
- AlphaGenome: [GitHub](https://github.com/google-deepmind/alphagenome) | [Paper](https://www.nature.com/articles/s41586-025-10014-0) | [API Docs](https://www.alphagenomedocs.com/colabs/quick_start.html)
- HyenaDNA: [GitHub](https://github.com/HazyResearch/hyena-dna) | [HuggingFace](https://huggingface.co/LongSafari/hyenadna-small-32k-seqlen)
- Pangolin: [GitHub](https://github.com/tkzeng/Pangolin) | [Paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02664-4)
- DNABERT-2: [GitHub](https://github.com/MAGICS-LAB/DNABERT_2) | [HuggingFace](https://huggingface.co/zhihan1996/DNABERT-2-117M) | [Paper (ICLR 2024)](https://openreview.net/forum?id=oMLQB4EZE1d)
- SpliceTransformer: [GitHub](https://github.com/ShenLab-Genomics/SpliceTransformer) | [Paper](https://www.nature.com/articles/s41467-024-53088-6) (not in gpu_config — no pip package)
- Orthrus: [GitHub](https://github.com/bowang-lab/Orthrus) | [Paper](https://www.biorxiv.org/content/10.1101/2024.10.10.617658v1) (not in gpu_config — no pip package)
- Benchmarking study: [bioRxiv 2026](https://www.biorxiv.org/content/10.64898/2026.02.22.707219v1)
