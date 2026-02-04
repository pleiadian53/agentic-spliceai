# Agentic AI Documentation

**Autonomous agents for scientific discovery and validation**

---

## 📚 Documentation Index

| Document | Purpose | Audience |
|----------|---------|----------|
| **AGENTIC_MEMORY_TUTORIAL.md** | Tutorial on persistent memory for agents | All users |
| **MEMORY_PATTERNS.md** | Practical patterns and examples | Researchers |
| **AGENT_WORKFLOWS.md** | Complete workflow examples | Advanced users |
| **API_REFERENCE.md** | API documentation | Developers |

---

## 🤖 What Are Agentic Workflows?

**Agentic AI** = Autonomous agents that can:
- 🔍 **Research**: Search literature, databases, prior experiments
- ✅ **Validate**: Assess biological plausibility, gather evidence
- 🧠 **Reason**: Connect patterns, propose hypotheses, design experiments
- 📊 **Report**: Generate summaries with full provenance
- 🧬 **Remember**: Build cumulative knowledge over time

---

## 🎯 Key Features in Agentic-SpliceAI

### 1. **Nexus Research Agent**
Searches multiple knowledge sources:
- PubMed (biomedical literature)
- ClinVar (clinical variants)
- GTEx (tissue expression)
- SpliceVarDB (splice variants)
- Your own experiments (via memory!)

**Example**:
```python
agent = NexusAgent()
evidence = agent.research(
    query="BRCA1 cryptic splice sites in breast cancer",
    sources=["pubmed", "clinvar", "memory"]
)
```

---

### 2. **Validation Agent**
Validates predictions with multi-source evidence:
```python
validation = agent.validate(
    site=novel_splice_site,
    criteria=["sequence_motif", "conservation", "expression", "literature"],
    confidence_threshold=0.8
)
```

---

### 3. **Agentic Memory** 🔬 NEW!
Persistent knowledge across sessions:
```python
# Store discovery
agent.memory.store_discovery(site, evidence)

# Query later
prior_discoveries = agent.memory.query(
    "validated BRCA1 isoforms with RNA-seq support"
)
```

**Learn more**: See `AGENTIC_MEMORY_TUTORIAL.md`

---

## 🔬 Scientific Use Cases

### Use Case 1: Novel Isoform Discovery
**Challenge**: How to validate novel splice sites?

**Solution**:
1. Meta-model predicts high-delta sites (novelty candidates)
2. Agent searches literature for similar cases
3. Agent validates with RNA-seq, databases
4. Agent stores discovery with full provenance
5. Agent remembers for future cross-validation

**Example**: See `MEMORY_PATTERNS.md` → Pattern 2

---

### Use Case 2: Variant Interpretation
**Challenge**: Is this VUS pathogenic?

**Solution**:
1. Agent checks memory for similar variants
2. Agent searches ClinVar, literature
3. Agent assesses splice disruption
4. Agent provides interpretation with confidence score
5. Agent stores decision for future reference

**Example**: See `MEMORY_PATTERNS.md` → Template 3

---

### Use Case 3: Experiment Design
**Challenge**: What experiment to run next?

**Solution**:
1. Agent queries memory: What have we tried?
2. Agent searches literature: What's been published?
3. Agent identifies gaps
4. Agent proposes optimal next experiment
5. Agent tracks results for future iterations

**Example**: See `MEMORY_PATTERNS.md` → Pattern 4

---

## 🚀 Getting Started

### Quick Start

```python
from agentic_spliceai import NexusAgent, AgenticMemory

# Initialize agent with memory
agent = NexusAgent(memory=AgenticMemory())

# Research a gene
findings = agent.research(gene="BRCA1", focus="alternative_splicing")

# Validate a discovery
validation = agent.validate(site=novel_site)

# Query memory
prior_work = agent.memory.query("BRCA1 splice variants")
```

---

### CLI Commands

```bash
# Research
agentic-spliceai agentic research \
  --gene BRCA1 \
  --query "pathogenic splice variants"

# Validate
agentic-spliceai agentic validate \
  --predictions predictions.tsv \
  --mode comprehensive

# Memory
agentic-spliceai memory search \
  --query "chromatin modality effectiveness"
```

---

## 📖 Tutorials

### Beginner
1. **AGENTIC_MEMORY_TUTORIAL.md** - Start here!
   - What is agentic memory?
   - Why it matters for genomics
   - Basic usage examples

### Intermediate
2. **MEMORY_PATTERNS.md** - Practical patterns
   - 8 memory patterns with code
   - Scientific workflow templates
   - Best practices

### Advanced
3. **Agent Development Guide** (Coming soon)
   - Custom agent creation
   - Memory schema design
   - Multi-agent coordination

---

## 🔗 Architecture

### How Agents Use Memory

```
┌─────────────────────────────────────┐
│         Research Question           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│     Agent Queries Memory First      │
│  "Have we seen this before?"        │
└──────────────┬──────────────────────┘
               │
               ├─ YES → Use prior knowledge as context
               │         (faster, more accurate)
               │
               └─ NO  → Search external sources
                        (PubMed, databases)
                        Then STORE for future
```

**Key Benefit**: Agent gets faster and smarter over time.

---

## 🎓 Key Concepts

### Concept 1: Memory ≠ Vector Database
**Traditional RAG**: Embed documents → Semantic search → Top-K chunks  
**Agentic Memory**: Structured facts → LLM reasoning → Relevant knowledge

**Why different**: Scientific queries require **logical reasoning**, not just semantic similarity.

---

### Concept 2: Provenance is Critical
Every fact links back to source:
```
Claim: "Chromatin improves F1 by 8%"
  ↓
Evidence: Experiment chromatin_chr21 (2026-02-15)
  ↓
Artifacts: logs/chromatin_chr21.log, results/metrics.json
  ↓
Raw Data: config/chromatin_chr21.yaml, data/chr21_validation.tsv
```

**Why critical**: Scientific reproducibility requires full audit trails.

---

### Concept 3: Memory is Collaborative
Multiple agents and researchers contribute to shared memory:

```
Discovery Agent → Finds novel sites → Stores in memory
Validation Agent → Reads from memory → Validates sites → Updates memory
Clinical Agent → Reads validated sites → Assesses pathogenicity → Updates memory
Researcher → Reviews memory → Approves high-confidence discoveries
```

**Result**: Knowledge accumulates through collective intelligence.

---

## 📊 Success Stories (Coming Soon)

### Case Study 1: BRCA1 Isoform Discovery
- Discovered 7 novel cryptic sites using delta scores
- Validated with memory context in 50% less time
- Full provenance chain for publication

### Case Study 2: Chromatin Modality Optimization
- Tracked 47 experiments across 12 tissues
- Agent learned tissue-specific effectiveness patterns
- Optimized compute allocation (skip low-ROI modalities)

### Case Study 3: Collaborative Lab Memory
- 3 researchers contributed to shared memory
- Cross-validated discoveries independently
- Published with complete reproducibility

---

## 🔗 Related Documentation

**In Agentic-SpliceAI**:
- **Isoform Discovery**: `docs/isoform_discovery/README.md`
- **Meta-Layer**: `docs/meta_layer/README.md` (future)
- **CLI Reference**: `docs/cli/README.md`

**In Meta-SpliceAI** (Reference):
- **Agentic Workflows**: `meta-spliceai/docs/agentic_workflows/`
- **Nexus Agent**: `meta-spliceai/docs/nexus_agent/`

**External**:
- **MemU Framework**: https://github.com/NevaMind-AI/memU

---

## 🎯 Quick Reference

### Common Commands
```bash
# Initialize memory
agentic-spliceai memory init

# Store experiment
agentic-spliceai memory store --category experiments --file result.json

# Search memory
agentic-spliceai memory search --query "your question"

# Export for paper
agentic-spliceai memory export --gene BRCA1 --output supplementary/

# View statistics
agentic-spliceai memory stats
```

---

### Python API
```python
from agentic_spliceai.memory import AgenticMemory

memory = AgenticMemory()

# Store
memory.store(category="...", item={...})

# Query
results = memory.query("...")

# Trace
provenance = memory.trace(item_id="...")
```

---

## 🆘 Support

- **GitHub Issues**: https://github.com/pleiadian53/agentic-spliceai/issues
- **Documentation**: https://agentic-spliceai.readthedocs.io (future)
- **Discussions**: https://github.com/pleiadian53/agentic-spliceai/discussions

---

**Ready to get started?** Read `AGENTIC_MEMORY_TUTORIAL.md` next! 🚀

---

*"Transform your AI agents from tools into research collaborators."*
