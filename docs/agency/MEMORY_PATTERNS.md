# Memory Patterns for Scientific Workflows

**A Guide to Using Agentic Memory in Genomics Research**

---

## 🎯 Overview

This document provides **concrete patterns** for using agentic memory in splice site prediction, isoform discovery, and variant interpretation workflows.

**Target Audience**: Researchers, bioinformaticians, computational biologists

---

## 📚 Pattern Catalog

### Pattern 1: Experiment Memory Chain

**Use Case**: Track ablation studies, remember what works

**Example**: Testing multimodal fusion

```python
from agentic_spliceai.memory import AgenticMemory

memory = AgenticMemory()

# Baseline experiment
baseline = train_model(modalities=["base", "sequence"])
memory.store_experiment({
    "name": "baseline_chr21",
    "modalities": ["base", "sequence"],
    "f1": 0.83,
    "conclusion": "Baseline performance established"
})

# Add chromatin
chromatin = train_model(modalities=["base", "sequence", "chromatin"])
memory.store_experiment({
    "name": "chromatin_chr21",
    "modalities": ["base", "sequence", "chromatin"],
    "f1": 0.91,
    "delta_from_baseline": +0.08,
    "conclusion": "Chromatin improves tissue-specific prediction"
})

# Add histone marks
histone = train_model(modalities=["base", "sequence", "chromatin", "histone"])
memory.store_experiment({
    "name": "histone_chr21",
    "modalities": ["base", "sequence", "chromatin", "histone"],
    "f1": 0.93,
    "delta_from_chromatin": +0.02,
    "conclusion": "Histone marks provide marginal improvement"
})

# Query accumulated knowledge
insights = memory.query("which modalities improved performance most?")
# → "Chromatin: +8%, Histone: +2%. Chromatin has best ROI."
```

**Memory Benefit**: Agent learns which modalities to prioritize for future experiments.

---

### Pattern 2: Discovery Validation Loop

**Use Case**: Build confidence through cross-validation

**Example**: Novel splice site in multiple contexts

```python
# Discovery in Study 1 (BRCA1 tumor samples)
site = {
    "gene": "BRCA1",
    "location": "chr17:43,094,582",
    "type": "cryptic_donor"
}

# Initial validation
validation1 = agent.validate(site, context="TCGA-BRCA")
memory.store_discovery(
    site=site,
    evidence=validation1,
    confidence=0.75,  # Medium confidence
    note="Found in tumor RNA-seq, needs more validation"
)

# Study 2 (GTEx normal breast)
# Agent checks memory first
prior = memory.query("BRCA1 chr17:43,094,582")
# → Finds Study 1 discovery

validation2 = agent.validate(site, context="GTEx-Breast", prior=prior)
memory.update_discovery(
    site=site,
    new_evidence=validation2,
    confidence=0.88,  # Higher confidence (two sources)
    note="Confirmed in normal tissue too"
)

# Study 3 (Literature search)
literature = agent.search_literature(site, prior=memory.get_all(site))
memory.update_discovery(
    site=site,
    new_evidence=literature,
    confidence=0.94,  # Very high confidence (three sources)
    status="validated",
    note="Validated by Wang et al. 2023"
)
```

**Memory Benefit**: Confidence builds over time through independent validation. Agent tracks the full evidence chain.

---

### Pattern 3: Error Pattern Learning

**Use Case**: Learn from mistakes, avoid repeated errors

**Example**: False positives in repetitive regions

```python
# Week 1: Notice error pattern
fps = analyze_false_positives(results)
memory.store_error_pattern({
    "pattern": "FPs clustered in Alu repeats",
    "frequency": 0.23,
    "cause": "Sequence similarity to canonical sites",
    "first_observed": "2026-02-01"
})

# Week 2: Test mitigation
result = apply_repeat_filter(model)
memory.update_error_pattern({
    "pattern": "FPs in Alu repeats",
    "mitigation": "RepeatMasker filter",
    "effectiveness": "Reduced FPs by 12%",
    "side_effects": "May lose rare cryptic sites in repeats"
})

# Week 3: Improve mitigation
result = apply_chromatin_filter(model)
memory.update_error_pattern({
    "pattern": "FPs in Alu repeats",
    "mitigation": "Chromatin accessibility filter",
    "effectiveness": "Reduced FPs by 18% (combined with repeat mask)",
    "recommended": True
})

# Future experiments automatically apply learned mitigation
future_predictions = model.predict(
    gene="TP53",
    filters=memory.get_recommended_filters()  # ← Uses learned knowledge!
)
```

**Memory Benefit**: Agent learns best practices from experience, applies them automatically.

---

### Pattern 4: Modality Effectiveness Tracking

**Use Case**: Learn when to trust which evidence sources

**Example**: Building modality profiles

```python
# Experiment across tissues
for tissue in ["brain", "liver", "breast", "heart"]:
    result = train_tissue_specific_model(
        tissue=tissue,
        modalities=["base", "sequence", "chromatin"]
    )
    
    memory.store_modality_effectiveness({
        "modality": "chromatin",
        "context": f"tissue_specific_{tissue}",
        "f1": result.f1,
        "analysis": result.feature_importance["chromatin"]
    })

# Later: Agent learns patterns
effectiveness = memory.query(
    "chromatin modality effectiveness across tissues"
)

# Agent discovers: "Chromatin most effective in brain (F1=0.92) 
#                   and immune (F1=0.89), less in liver (F1=0.74)"

# Future experiments use this knowledge
if tissue == "brain":
    modalities = ["base", "sequence", "chromatin"]  # High priority!
elif tissue == "liver":
    modalities = ["base", "sequence"]  # Skip chromatin (low ROI)
```

**Memory Benefit**: Agent learns resource allocation strategy (which modalities worth the compute cost).

---

### Pattern 5: Literature Knowledge Base

**Use Case**: Build cumulative literature knowledge

**Example**: Mining alternative splicing papers

```python
# Week 1: Read and store key paper
paper = extract_from_pdf("Wang_2023_Cell.pdf")
memory.store_literature({
    "citation": "Wang et al., Cell 2023",
    "key_finding": "Chromatin remodeling activates cryptic sites in cancer",
    "datasets": ["TCGA pan-cancer"],
    "methods": ["RNA-seq", "ChIP-seq"],
    "relevance": "Supports chromatin modality for cancer-specific isoforms"
})

# Week 5: New experiment validates paper's finding
result = test_chromatin_in_cancer()
memory.link_experiment_to_literature(
    experiment_id="chromatin_cancer_2026",
    paper_id="Wang2023Cell",
    relationship="validates"
)

# Week 10: Writing paper
evidence = memory.query("evidence for chromatin modality in cancer")
# → Returns: Your experiment + Wang 2023 + full provenance
# → Can cite both in paper with confidence
```

**Memory Benefit**: Agent builds reusable literature knowledge base, connects your findings to prior art.

---

## 🔬 Advanced Patterns

### Pattern 6: Hypothesis Evolution

**Use Case**: Track how hypotheses evolve with new evidence

```python
# Initial hypothesis (weak)
memory.store_hypothesis({
    "id": "delta_predicts_novel",
    "version": 1,
    "statement": "Delta > 0.5 predicts novel sites",
    "confidence": 0.5,
    "based_on": "Intuition + small pilot (n=10)"
})

# Experiment 1 (supports)
result1 = validate_on_chr21()
memory.evolve_hypothesis({
    "id": "delta_predicts_novel",
    "version": 2,
    "confidence": 0.7,
    "evidence": [result1],
    "note": "Validated on chr21 (AUC=0.84)"
})

# Experiment 2 (strongly supports)
result2 = validate_on_brca1_tp53()
memory.evolve_hypothesis({
    "id": "delta_predicts_novel",
    "version": 3,
    "confidence": 0.9,
    "evidence": [result1, result2],
    "note": "Strong validation across multiple genes"
})

# Publication: Show full evolution
hypothesis_history = memory.get_hypothesis_evolution("delta_predicts_novel")
# → Version 1 (intuition) → Version 2 (chr21) → Version 3 (validated)
# → Full story for Methods section
```

---

### Pattern 7: Multi-Agent Knowledge Sharing

**Use Case**: Coordinate multiple specialized agents

**Example**: Discovery → Validation → Clinical pipeline

```python
# Agent 1: Discovery Agent
discovery_agent = DiscoveryAgent(memory=shared_memory)
novel_sites = discovery_agent.find_novel_sites(gene="BRCA1")

for site in novel_sites:
    shared_memory.store({
        "category": "pending_validation",
        "site": site,
        "discovered_by": "discovery_agent"
    })

# Agent 2: Validation Agent (reads from shared memory)
validation_agent = ValidationAgent(memory=shared_memory)
pending = shared_memory.query("sites pending validation")

for site in pending:
    validation = validation_agent.validate(site)
    shared_memory.move(
        from_category="pending_validation",
        to_category="validated_discoveries" if validation.confidence > 0.8 else "low_confidence",
        update={"validation": validation}
    )

# Agent 3: Clinical Agent (reads validated discoveries)
clinical_agent = ClinicalAgent(memory=shared_memory)
validated = shared_memory.query("validated discoveries with high delta")

for discovery in validated:
    clinical = clinical_agent.assess_pathogenicity(discovery)
    shared_memory.update({
        "discovery": discovery,
        "clinical_assessment": clinical,
        "assessed_by": "clinical_agent"
    })
```

**Memory Benefit**: Agents coordinate through shared memory, each adds their expertise to the same knowledge base.

---

### Pattern 8: Time-Series Analysis

**Use Case**: Track metrics over time, detect trends

**Example**: Model performance evolution

```python
# Store monthly performance
for month in ["2026-01", "2026-02", "2026-03"]:
    result = evaluate_model(month=month)
    memory.store({
        "category": "model_performance",
        "date": month,
        "f1": result.f1,
        "novel_sites_detected": result.novel_count,
        "validation_rate": result.validation_rate
    })

# Query trend
trend = memory.query("model performance over time")
# → Agent detects: "F1 improving (+5% per month), 
#                   validation rate stable (0.85)"

# Visualize
memory.plot_metric_over_time(
    metric="f1",
    category="model_performance",
    output="reports/model_improvement.png"
)
```

---

## 🎯 Scientific Workflow Templates

### Template 1: Modality Ablation Study

```python
def run_ablation_study(gene: str, modalities: list):
    """Systematic modality ablation with memory tracking."""
    
    memory = AgenticMemory()
    results = {}
    
    # Baseline
    baseline = train_model(modalities=["base", "sequence"])
    memory.store_experiment({
        "study": f"ablation_{gene}",
        "variant": "baseline",
        "modalities": ["base", "sequence"],
        "f1": baseline.f1
    })
    results["baseline"] = baseline.f1
    
    # Add each modality
    for modality in modalities:
        current_modalities = ["base", "sequence", modality]
        model = train_model(modalities=current_modalities)
        
        delta = model.f1 - baseline.f1
        
        memory.store_experiment({
            "study": f"ablation_{gene}",
            "variant": f"+{modality}",
            "modalities": current_modalities,
            "f1": model.f1,
            "delta": delta,
            "interpretation": f"{modality} contributes {delta:+.2%}"
        })
        
        results[modality] = model.f1
    
    # Generate report from memory
    report = memory.generate_ablation_report(study=f"ablation_{gene}")
    return report
```

---

### Template 2: Discovery Pipeline with Memory

```python
def discover_novel_isoforms(gene: str, context: dict):
    """Full discovery pipeline with memory integration."""
    
    agent = NexusAgent(memory=AgenticMemory())
    
    # Step 1: Check if we've analyzed this gene before
    prior = agent.memory.query(f"discoveries in {gene}")
    if prior:
        print(f"Found {len(prior)} prior discoveries in {gene}")
    
    # Step 2: Run meta-model
    predictions = meta_model.predict(gene=gene, context=context)
    
    # Step 3: Flag high-delta sites
    candidates = [s for s in predictions if s.delta_score > 0.5]
    
    # Step 4: Validate each candidate
    discoveries = []
    for site in candidates:
        # Check memory for similar patterns
        similar = agent.memory.query(
            f"sites with delta > 0.5 in {context['tissue']} tissue"
        )
        
        # Validate with context
        validation = agent.validate(
            site,
            similar_cases=similar,
            sources=["literature", "rnaseq", "databases"]
        )
        
        # Store if high confidence
        if validation.confidence > 0.8:
            agent.memory.store_discovery(site, validation)
            discoveries.append(site)
    
    # Step 5: Generate report
    report = agent.memory.generate_discovery_report(
        gene=gene,
        discoveries=discoveries,
        include_provenance=True
    )
    
    return report
```

---

### Template 3: Variant Interpretation with Historical Context

```python
def interpret_variant(variant: dict):
    """Interpret variant with memory of similar cases."""
    
    agent = NexusAgent(memory=AgenticMemory())
    
    # Check memory for similar variants
    similar = agent.memory.query(
        f"pathogenic variants affecting {variant['gene']} splice sites"
    )
    
    # Check if we've seen this exact variant
    exact = agent.memory.query(
        f"variant {variant['hgvs']} in {variant['gene']}"
    )
    
    if exact:
        print(f"We've analyzed this variant before!")
        print(f"Prior assessment: {exact[0]['pathogenicity']}")
        print(f"Evidence: {exact[0]['evidence']}")
        return exact[0]
    
    # New variant - validate with context
    interpretation = agent.validate_variant(
        variant,
        similar_cases=similar  # Historical context helps!
    )
    
    # Store for future
    agent.memory.store({
        "category": "variant_interpretations",
        "variant": variant,
        "interpretation": interpretation,
        "assessed_date": datetime.now()
    })
    
    return interpretation
```

---

### Template 4: Literature-Guided Experiments

**Use Case**: Design experiments based on literature memory

```python
def design_next_experiment(research_question: str):
    """Agent designs experiment using literature memory."""
    
    agent = NexusAgent(memory=AgenticMemory())
    
    # Query literature memory
    relevant_papers = agent.memory.query(
        f"literature about {research_question}"
    )
    
    # Query past experiments
    past_experiments = agent.memory.query(
        f"experiments related to {research_question}"
    )
    
    # Agent reasons: What's been tried? What worked? What's missing?
    design = agent.llm.propose_experiment(
        research_question=research_question,
        literature=relevant_papers,
        past_experiments=past_experiments,
        prompt="""
        Based on:
        1. What the literature has shown
        2. What we've already tried
        3. What gaps remain
        
        Propose the next experiment that will maximize learning.
        """
    )
    
    return design
```

**Example Output**:
```
Research Question: "Can epigenetic marks predict cryptic site activation?"

Literature Memory:
- Wang 2023: H3K27ac marks active enhancers
- Smith 2022: H3K36me3 marks exons
- Jones 2024: Cancer cells show altered H3K27ac

Past Experiments:
- Tried chromatin (F1=0.91) ✓
- Tried conservation (F1=0.85) ✓
- Haven't tried histone marks yet ✗

Proposed Experiment:
"Test H3K27ac + H3K36me3 as joint modality for cryptic site prediction 
 in cancer samples. Hypothesis: H3K27ac will distinguish activated 
 cryptic sites from dormant ones. Expected improvement: +5-10% F1."

Rationale: Literature strongly supports, we have the data (ENCODE), 
           and it fills a gap in our current experiments.
```

---

## 🧪 Practical Examples

### Example 1: Daily Research Workflow

```python
# Morning: Check what happened yesterday
yesterday = memory.query("experiments completed yesterday")
agent.summarize(yesterday)

# Run today's experiment
result = train_new_model_variant()
memory.store_experiment(result)

# Query for insights
insights = memory.query("what patterns emerged this week?")
agent.analyze(insights)

# Evening: Generate weekly summary
summary = memory.generate_summary(timeframe="this_week")
```

---

### Example 2: Pre-Experiment Check

```python
def before_experiment(config: dict):
    """Check memory before spending compute."""
    
    # Have we tried this before?
    similar = memory.query(
        f"experiments with modalities {config['modalities']} on {config['test_set']}"
    )
    
    if similar:
        print("⚠️  Similar experiment found!")
        print(f"   {similar[0]['name']}: F1 = {similar[0]['f1']}")
        print("   Consider adjusting hyperparameters or skipping.")
        
        user_input = input("Continue anyway? (y/n): ")
        if user_input.lower() != 'y':
            return "Experiment skipped (duplicate)"
    
    # Proceed
    return run_experiment(config)
```

**Benefit**: Avoid wasting compute on duplicate experiments.

---

### Example 3: Evidence Aggregation for Papers

```python
def prepare_paper_evidence(gene: str):
    """Aggregate all evidence for a gene into paper-ready format."""
    
    # Query all relevant memory
    discoveries = memory.query(f"validated discoveries in {gene}")
    experiments = memory.query(f"experiments validating {gene} sites")
    literature = memory.query(f"literature supporting {gene} findings")
    
    # Generate structured output
    evidence = {
        "discoveries": discoveries,
        "experiments": experiments,
        "literature": literature,
        "full_provenance": memory.trace_all(gene)
    }
    
    # Export to Markdown (ready for Supplementary Materials)
    memory.export(
        evidence,
        format="markdown",
        output=f"supplementary/{gene}_evidence.md"
    )
    
    return evidence
```

**Output**: `supplementary/BRCA1_evidence.md`
```markdown
# BRCA1: Evidence Summary

## Discoveries (3)
1. Cryptic donor at exon 11 (validated, confidence 0.94)
2. Novel acceptor at exon 18 (validated, confidence 0.87)
3. Exon 7 skipping isoform (pending validation, confidence 0.72)

## Experimental Evidence (5)
- chromatin_ablation_2026-02-15: F1 = 0.91 (+8%)
- histone_marks_2026-02-20: F1 = 0.93 (+2%)
- ...

## Literature Support (3)
- Wang et al. 2023, Cell: Chromatin remodeling in cancer
- Venkitaraman 2014, Nature: BRCA1 exon 11 cryptic donor
- ...

## Full Provenance
[Discovery 1] → [Experiment chromatin_2026] → [TCGA-BRCA RNA-seq] → [Raw BAM files]
[Discovery 1] → [Literature Wang2023] → [Paper PDF] → [Figure 3B]
```

**Benefit**: One command generates complete supplementary materials!

---

## 📊 Memory Query Patterns

### Query Type 1: Factual Lookup
```python
# Simple fact retrieval
memory.query("What was the F1 score in experiment chromatin_chr21?")
# → "0.91"
```

---

### Query Type 2: Logical Reasoning
```python
# Complex logic
memory.query(
    "experiments where chromatin improved F1 by more than 5% "
    "AND were tested on cancer samples"
)
# → LLM reasons over memory files, returns matching experiments
```

---

### Query Type 3: Temporal
```python
# Time-based
memory.query("what did we learn about histone marks last month?")
# → Returns experiments, papers, insights from February
```

---

### Query Type 4: Comparative
```python
# Comparison
memory.query("compare chromatin vs histone modality effectiveness")
# → Agent synthesizes from multiple memory files
```

---

### Query Type 5: Provenance Trace
```python
# Full evidence chain
memory.trace("BRCA1 exon 11 cryptic donor discovery")
# → Discovery → Validation → Experiments → Papers → Raw data (full chain)
```

---

## 🎯 Integration with Agentic-SpliceAI

### When to Use Memory

| Task | Use Memory? | Why |
|------|-------------|-----|
| **Running predictions** | Optional | Memory won't affect predictions |
| **Validating discoveries** | ✅ Yes | Use prior validations as context |
| **Designing experiments** | ✅ Yes | Learn from past experiments |
| **Writing papers** | ✅ Yes | Export provenance chains |
| **Training new models** | ✅ Yes | Apply learned best practices |
| **Debugging errors** | ✅ Yes | Check error pattern memory |

---

### Recommended Workflow

```python
# 1. Initialize agent with memory
agent = NexusAgent(memory=AgenticMemory())

# 2. Query memory before acting
context = agent.memory.query("relevant context for this task")

# 3. Perform task (prediction, validation, etc.)
result = agent.perform_task(context=context)

# 4. Store results in memory
agent.memory.store(result)

# 5. Query insights
insights = agent.memory.query("what did we learn?")
```

---

## 📚 Resources

### Documentation
- **Tutorial**: This document
- **Implementation**: `dev/planning/agentic_workflows/MEMORY_LAYER.md`
- **Research Notes**: `dev/research/agency/agentic_memory.md`
- **Integration Strategy**: `dev/research/agency/integration_plans/MEMU_INTEGRATION_STRATEGY.md`

### Framework
- **MemU**: https://github.com/NevaMind-AI/memU
- **Examples**: See MemU repo `examples/`
- **API Docs**: See MemU repo `docs/`

### Related Work
- **LangMem**: Memory for LangChain agents
- **GPT Memory**: OpenAI's memory feature
- **Mem0**: Another memory framework
- **MemU**: Best for scientific workflows (file-system, provenance)

---

## 🎉 Getting Started

### Quick Start (3 Steps)

**Step 1**: Install
```bash
pip install agentic-spliceai[memory]
```

**Step 2**: Initialize
```python
from agentic_spliceai.memory import AgenticMemory
memory = AgenticMemory()
```

**Step 3**: Use
```python
# Store
memory.store(category="experiments", item={...})

# Query
results = memory.query("your question")

# Enjoy cumulative intelligence! 🎉
```

---

**Questions?** Open an issue on GitHub or see the full documentation.

---

*"Make your AI agents remember. Make your science cumulative."*
