# Agentic Memory for Scientific Discovery

**A Tutorial on Using Persistent Memory in AI-Powered Research Workflows**

---

## 🎯 What is Agentic Memory?

**Agentic memory** enables AI agents to remember, learn, and build knowledge across sessions—transforming them from stateless tools into cumulative research collaborators.

### The Problem

Traditional AI agents are like researchers with amnesia:
- ❌ Forget everything between sessions
- ❌ Can't learn from past experiments
- ❌ Can't connect findings across time
- ❌ Can't provide historical context

**Result**: You spend time explaining context every session.

---

### The Solution

With agentic memory, agents become knowledge-building partners:
- ✅ Remember experimental results
- ✅ Learn what works (and what doesn't)
- ✅ Connect patterns across studies
- ✅ Provide full provenance for every claim

**Result**: Agents that get smarter over time, just like human researchers.

---

## 🧬 Why This Matters for Genomics

### Use Case 1: Novel Isoform Discovery

**Without Memory**:
```
Day 1: "I found a novel splice site in BRCA1 exon 11"
Day 2: Agent forgets → starts validation from scratch
Day 3: Repeat validation → waste time
```

**With Memory**:
```
Day 1: Agent stores discovery + evidence
Day 2: Agent remembers → "I validated this last week, confidence 0.92"
Day 3: Agent builds on it → "Let's check nearby exons for similar patterns"
```

---

### Use Case 2: Experiment Tracking

**Without Memory**:
```
Researcher: "Did adding chromatin modality help?"
Agent: "I don't have that information"
Researcher: *Searches through 50 log files manually*
```

**With Memory**:
```
Researcher: "Did adding chromatin modality help?"
Agent: "Yes. In 3 experiments (Feb 2026), F1 improved by 8% (0.87→0.91).
        Especially effective for tissue-specific splicing.
        Evidence: experiments/chromatin_ablation.json"
```

---

### Use Case 3: Literature Knowledge

**Without Memory**:
```
Researcher: "Has anyone found cryptic sites in BRCA1 exon 11?"
Agent: *Searches PubMed again (5 minutes)*
```

**With Memory**:
```
Researcher: "Has anyone found cryptic sites in BRCA1 exon 11?"
Agent: "Yes. Wang et al. (2023, Cell) reported 127 novel junctions 
        including BRCA1 exon 11. I stored this in 
        memory/literature/Wang2023Cell.md last month.
        Would you like the full citation?"
```

---

## 🏗️ How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│         Your Research Workflow              │
│  (Run experiments, analyze data, write)     │
└──────────────────┬──────────────────────────┘
                   │
                   │ stores results
                   ▼
┌─────────────────────────────────────────────┐
│           Agentic Memory Layer              │
│                                             │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  Experiments │  │  Discoveries │       │
│  │     (.md)    │  │     (.md)    │       │
│  └──────────────┘  └──────────────┘       │
│                                             │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  Literature  │  │  Hypotheses  │       │
│  │     (.md)    │  │     (.md)    │       │
│  └──────────────┘  └──────────────┘       │
└──────────────────┬──────────────────────────┘
                   │
                   │ traces to
                   ▼
┌─────────────────────────────────────────────┐
│          Raw Data & Artifacts               │
│  (logs, configs, papers, plots, tables)     │
└─────────────────────────────────────────────┘
```

**Key Principle**: Memory preserves the full chain from **claim → evidence → raw data**.

---

### Memory as Markdown Files

Memory is stored as **human-readable Markdown files** (not opaque databases):

```
memory/
├── experiments/
│   ├── chromatin_modality_2026Q1.md
│   └── histone_marks_ablation.md
├── discoveries/
│   ├── brca1_novel_isoforms.md
│   └── tp53_cryptic_sites.md
├── literature/
│   ├── wang2023_cell_splicing.md
│   └── key_papers_alternative_splicing.md
└── hypotheses/
    └── high_delta_predicts_novel.md
```

**Benefits**:
- ✅ **Human-readable**: You can open and read any memory file
- ✅ **Git-versionable**: Track how knowledge evolves over time
- ✅ **Editable**: Correct mistakes or add notes
- ✅ **Transparent**: See exactly what the agent remembers

---

## 📝 Example: Storing an Experiment

### Step 1: Run Your Experiment
```python
from agentic_spliceai import MetaModel

# Run experiment
model = MetaModel(modalities=["base", "sequence", "chromatin"])
results = model.train(data="chr21", epochs=50)

print(f"F1 Score: {results.f1}")  # 0.91
```

---

### Step 2: Store in Memory
```python
from agentic_spliceai.memory import AgenticMemory

memory = AgenticMemory()

# Store experiment
memory.store_experiment({
    "date": "2026-02-15",
    "hypothesis": "Chromatin modality improves tissue-specific predictions",
    "config": {
        "model": "meta-v2",
        "modalities": ["base", "sequence", "chromatin"],
        "test_set": "chr21"
    },
    "results": {
        "f1": 0.91,
        "precision": 0.89,
        "recall": 0.93
    },
    "conclusion": "Chromatin effective (+8% F1 improvement)",
    "artifacts": [
        "logs/chromatin_exp.log",
        "plots/chromatin_roc.png"
    ]
})
```

---

### Step 3: Memory File Created

**File**: `memory/experiments/chromatin_modality_2026Q1.md`

```markdown
# Chromatin Modality Experiment (Q1 2026)

## Experiment: chromatin-chr21-validation
**Date**: 2026-02-15  
**Hypothesis**: Chromatin modality improves tissue-specific predictions

### Setup
- Model: meta-v2
- Modalities: base + sequence + chromatin (ATAC-seq)
- Test set: chr21 validation

### Results
- **F1**: 0.91 (baseline: 0.83, improvement: +8%)
- **Precision**: 0.89
- **Recall**: 0.93

### Conclusion
Chromatin accessibility significantly improves predictions for 
tissue-specific splice sites. Most effective in genes with 
dynamic chromatin state (e.g., immune response genes).

### Evidence
- [logs/chromatin_exp.log]
- [plots/chromatin_roc.png]
- [configs/chromatin_exp.yaml]

### Related
- See: memory/modality_effectiveness/chromatin.md
```

---

## 🔍 Example: Querying Memory

### Natural Language Queries

```python
# Ask a question
results = memory.query(
    "What have we learned about chromatin modality effectiveness?"
)

# Agent finds relevant memories
for result in results:
    print(f"Category: {result.category}")
    print(f"Finding: {result.content}")
    print(f"Evidence: {result.links}")
    print("---")
```

**Output**:
```
Category: experiments
Finding: Chromatin modality improved F1 by 8% (0.83 → 0.91) in chr21 validation
Evidence: [logs/chromatin_exp.log, plots/chromatin_roc.png]
---

Category: modality_effectiveness
Finding: Chromatin most effective for tissue-specific splicing (F1=0.89)
         Least effective for housekeeping genes (F1=0.71)
Evidence: [47 experiments across 12 tissues]
---

Category: literature
Finding: Wang et al. (2023) showed chromatin remodeling activates cryptic sites
Evidence: [papers/Wang_2023_Cell.pdf]
---
```

---

### Complex Logical Queries

MemU uses **LLM-based search** (not just embeddings), enabling complex logic:

```python
# Find experiments where:
# - Chromatin was used
# - AND F1 > 0.9
# - AND tested on cancer samples
results = memory.query(
    "experiments with chromatin modality, F1 above 0.9, on cancer samples"
)
```

**Why this matters**: Semantic search would miss the logical constraints (F1 > 0.9, cancer-only).

---

## 🔬 Example: Novel Discovery Workflow

### Full Workflow with Memory

```python
from agentic_spliceai import MetaModel, NexusAgent, AgenticMemory

# Initialize
model = MetaModel()
agent = NexusAgent(memory=AgenticMemory())

# 1. Make predictions
predictions = model.predict(gene="BRCA1")

# 2. Find high-delta sites (potential novel isoforms)
novel_sites = [s for s in predictions if s.delta_score > 0.5]

# 3. Validate with memory context
for site in novel_sites:
    # Agent searches memory for similar cases
    similar = agent.memory.query(
        f"novel splice sites in {site.gene} with high chromatin signal"
    )
    
    # Agent validates with historical context
    validation = agent.validate(
        site,
        context=similar,  # ← Uses past discoveries!
        sources=["literature", "rnaseq", "databases"]
    )
    
    # 4. Store discovery if high confidence
    if validation.confidence > 0.8:
        agent.memory.store_discovery(
            site=site,
            evidence=validation.evidence,
            confidence=validation.confidence
        )
        
        print(f"✅ Novel site discovered: {site.location}")
        print(f"   Confidence: {validation.confidence}")
        print(f"   Evidence: {len(validation.evidence)} sources")
        print(f"   Similar cases: {len(similar)} in memory")
```

**Output**:
```
✅ Novel site discovered: chr17:43,094,582
   Confidence: 0.92
   Evidence: 4 sources (literature + RNA-seq + databases + chromatin)
   Similar cases: 3 in memory (BRCA1 exon 10, exon 12)
```

---

### What Got Stored

**File**: `memory/discoveries/brca1_novel_isoforms.md`

```markdown
# BRCA1 Novel Isoforms

## Discovery: BRCA1_exon11_cryptic_donor_20260220

**Location**: chr17:43,094,582 (GRCh38)  
**Type**: Cryptic donor (GT motif)  
**Discovery Date**: 2026-02-20

### Prediction Scores
- Base model: 0.15 (LOW - not in training data)
- Meta model: 0.88 (HIGH - strong multimodal evidence)
- **Delta: 0.73** (large gap → novel signal)

### Evidence
**Sequence**: AGGTAAGT (canonical GT donor, 0.90 strength)  
**Chromatin**: 0.82 (open in breast tissue)  
**Histone**: H3K36me3 = 0.76 (active transcription)  
**RNA-seq**: 24 junction reads in TCGA-BRCA  
**Literature**: Confirmed by Venkitaraman (2014, Nature)

### Validation
- **Confidence**: 0.92 (HIGH)
- **Agent**: nexus-v2
- **Status**: ✅ Validated
- **Clinical**: Likely pathogenic (familial breast cancer)

### Provenance
- [rnaseq/TCGA-BRCA/junction_counts.tsv]
- [papers/Venkitaraman_2014_Nature.pdf]
- [predictions/meta-v2/BRCA1_predictions.tsv]
- [igv_screenshots/BRCA1_exon11_junction.png]
```

---

## 🧪 Scientific Workflows

### Workflow 1: Hypothesis Testing

```python
# Propose hypothesis
agent.memory.store_hypothesis({
    "title": "High-delta sites predict novel isoforms",
    "definition": "Sites with delta > 0.5 are enriched for novel sites",
    "rationale": "Base model hasn't seen them, meta model has context",
    "proposed": "2026-01-30"
})

# Run experiments
for experiment in ["chr21", "brca1", "tp53"]:
    results = run_experiment(experiment)
    agent.memory.add_experiment_to_hypothesis(
        hypothesis_id="high_delta_predicts_novel",
        experiment=results
    )

# Query hypothesis status
status = agent.memory.query("status of high-delta hypothesis")
# → "Validated in 3/3 experiments. Precision = 0.85, Recall = 0.73"
```

---

### Workflow 2: Error Pattern Analysis

```python
# Analyze false positives
fps = model.get_false_positives(test_set="chr21")

# Store pattern
agent.memory.store_error_pattern({
    "type": "false_positive",
    "pattern": "High FP rate in repetitive regions",
    "frequency": 0.23,  # 23% of all FPs
    "cause": "Base model trained on canonical sites",
    "mitigation": "Add repeat masking + chromatin filter",
    "effectiveness": "78% FP reduction"
})

# Later experiments automatically benefit
agent.memory.query("how to reduce false positives?")
# → Returns stored mitigation strategies
```

---

### Workflow 3: Cross-Study Validation

```python
# Discovery in Study 1 (BRCA1)
agent.memory.store_discovery({
    "gene": "BRCA1",
    "site": "chr17:43,094,582",
    "evidence": ["TCGA-BRCA RNA-seq"],
    "confidence": 0.85
})

# Study 2 (TP53) - Agent checks for similar patterns
similar = agent.memory.query(
    "discoveries with cryptic donors and chromatin evidence"
)

# Agent: "Found similar pattern in BRCA1. Let me check TP53..."
# → Cross-validates across studies
# → Builds confidence through independent replication
```

---

## 📊 Benefits for Your Research

### 1. **Time Savings**
- No more searching through old log files
- No more re-reading papers you've already processed
- No more explaining context to tools each session

**Estimate**: Save 2-5 hours/week

---

### 2. **Better Science**
- **Reproducibility**: Full provenance for every claim
- **Cumulative learning**: Build on past experiments
- **Cross-validation**: Connect findings across studies
- **Error avoidance**: Remember what didn't work

---

### 3. **Publication Ready**
Memory files can directly become:
- **Methods sections** (full experimental details)
- **Supplementary materials** (all evidence chains)
- **Data availability statements** (all artifacts linked)

---

### 4. **Collaborative**
- Share memory with lab members
- Build institutional knowledge
- Onboard new researchers faster

---

## 🚀 Getting Started

### Installation

```bash
# Install Agentic-SpliceAI with memory support
pip install agentic-spliceai[memory]

# Initialize memory
agentic-spliceai memory init --dir ./memory
```

---

### Basic Usage

```python
from agentic_spliceai.memory import AgenticMemory

# Create memory
memory = AgenticMemory()

# Store something
memory.store(
    category="experiments",
    item={"name": "My first experiment", "result": "success"}
)

# Query it
results = memory.query("my first experiment")
print(results)  # Finds it!
```

---

### CLI Commands

```bash
# Search memory
agentic-spliceai memory search \
  --query "chromatin modality effectiveness"

# Store experiment
agentic-spliceai memory store \
  --category experiments \
  --file experiment_results.json

# Export for paper
agentic-spliceai memory export \
  --categories discoveries,literature \
  --gene BRCA1 \
  --output supplementary_materials/
```

---

## 🎓 Best Practices

### 1. **Store Early, Store Often**
Don't wait until "final results"—store intermediate findings too.

**Why**: Negative results and failed experiments are valuable knowledge!

---

### 2. **Always Link to Raw Data**
Every memory item should link back to source files.

```python
memory.store({
    "finding": "Chromatin improved F1 by 8%",
    "evidence": [
        "logs/experiment_123.log",  # ← Link to raw
        "results/metrics.json"
    ]
})
```

---

### 3. **Use Descriptive Categories**
Organize memory by scientific theme, not tool structure.

**Good**: `memory/tissue_specific_splicing/`  
**Bad**: `memory/model_outputs/`

---

### 4. **Review High-Stakes Memories**
Validate discoveries before storing as "confirmed."

```python
if validation.confidence > 0.9:
    memory.store(item, reviewed=True, reviewer="expert@uni.edu")
```

---

### 5. **Query Before You Start**
Check memory before running experiments.

```python
# Before experiment
prior_work = memory.query("chromatin modality on chr21")
if prior_work:
    print("We tried this before! Here's what we learned:")
    print(prior_work)
```

---

## 📚 Advanced Topics

### Memory Provenance Chains

Every claim traces back through full evidence chain:

```
Discovery: "BRCA1 exon 11 cryptic donor"
    ↓
Evidence: "24 RNA-seq junction reads"
    ↓
Experiment: "TCGA-BRCA analysis"
    ↓
Raw Data: "tcga_brca_junctions.bam"
```

Query any level:
```python
memory.trace_provenance(discovery_id="BRCA1_exon11_cryptic")
# Returns full chain with file paths
```

---

### Memory Versioning

Memory files are Markdown → version with Git:

```bash
git log memory/discoveries/brca1_novel_isoforms.md
# See how knowledge evolved over time
```

---

### Collaborative Memory

Share memory across team:

```bash
# Clone lab memory
git clone lab-server:/memory ./shared_memory

# Use it
memory = AgenticMemory(root_dir="./shared_memory")

# Contribute discoveries
memory.store_discovery(...)
git commit -m "Added TP53 cryptic site validation"
git push
```

---

## 🔗 Learn More

- **Framework**: MemU (https://github.com/NevaMind-AI/memU)
- **Integration**: See `dev/planning/agentic_workflows/MEMORY_LAYER.md`
- **Use Cases**: See `docs/agency/MEMORY_PATTERNS.md`
- **API Reference**: See `docs/api/memory.md`

---

## 🎉 Summary

**Agentic memory transforms AI agents from tools into research collaborators** that:
- ✅ Remember what you've learned
- ✅ Build cumulative knowledge
- ✅ Provide full provenance
- ✅ Get smarter over time

**The result**: Better science, faster discovery, stronger publications.

---

**Try it today**: `pip install agentic-spliceai[memory]`

**Questions?** Open an issue on GitHub or email research@agentic-spliceai.org

---

*"The best research assistant is one that remembers everything you've taught it."*
