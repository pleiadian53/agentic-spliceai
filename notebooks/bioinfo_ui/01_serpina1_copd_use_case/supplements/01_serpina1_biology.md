# SERPINA1: Biology and Clinical Relevance

## Alpha-1 Antitrypsin (AAT)

**SERPINA1** (Serpin Family A Member 1) encodes **alpha-1 antitrypsin (AAT)**, the most abundant serine protease inhibitor in human plasma. AAT is primarily synthesized in the liver and secreted into the bloodstream, where it protects the lungs from neutrophil elastase — a protease released during the inflammatory response.

- **Protein size**: 394 amino acids, 52 kDa glycoprotein
- **Normal plasma concentration**: 1.0–2.0 g/L
- **Primary target**: Neutrophil elastase (NE) in the lower respiratory tract
- **Mechanism**: AAT binds and inactivates NE, preventing destruction of alveolar tissue

## Gene Structure

- **Location**: Chromosome 14q32.13
- **Span**: ~12.2 kb
- **Exons**: 5 exons (4 coding exons + 1 non-coding exon at the 5' end)
- **Canonical transcript**: ENST00000440909 / NM_000295

**Exon-intron organization**:
| Exon | Region | Function |
|------|--------|----------|
| 1 (Ia, Ib, Ic) | 5' UTR | Non-coding, contains multiple transcription start sites |
| 2 | Coding | Signal peptide + N-terminal region |
| 3 | Coding | Beta-sheet A (core structural region) |
| 4 | Coding | Beta-sheet B + hinge region |
| 5 | Coding | Reactive center loop (RCL) — the business end |

The **reactive center loop** (encoded by exon 5) is the functional domain that directly interacts with neutrophil elastase. Most common pathogenic missense variants occur in or near this region.

## Alpha-1 Antitrypsin Deficiency (AATD)

AATD is one of the most common genetic disorders in people of European descent:
- **Prevalence**: ~1 in 2,500 individuals of European ancestry
- **Inheritance**: Autosomal co-dominant
- **Underdiagnosed**: Estimated >90% of affected individuals are undiagnosed

### Common Pathogenic Variants

| Variant | Allele Name | Mechanism | Effect |
|---------|-------------|-----------|--------|
| E342K (Glu342Lys) | Z allele | Protein misfolding, polymerization in liver | ~85% reduction in plasma AAT |
| E264V (Glu264Val) | S allele | Reduced stability | ~40% reduction in plasma AAT |
| Null variants | Various | Splice site disruption, frameshifts, nonsense | Complete loss of AAT production |

### Clinical Consequences

**Lung disease (COPD/emphysema)**:
- Without sufficient AAT, neutrophil elastase destroys alveolar walls
- Leads to early-onset emphysema (typically 30s-40s, especially in smokers)
- AATD is the most common genetic cause of COPD in young adults

**Liver disease**:
- Z allele causes AAT to polymerize within hepatocytes
- Can lead to neonatal hepatitis, cirrhosis, hepatocellular carcinoma

## Why Splice Site Prediction Matters for SERPINA1

Splice site variants in SERPINA1 are particularly clinically significant:

1. **Null alleles**: Splice site mutations that disrupt exon-intron boundaries can cause complete loss of functional AAT protein. These are often more severe than missense variants (Z, S) because they produce no protein at all rather than reduced/misfolded protein.

2. **Diagnostic challenge**: Splice site variants may not be detected by standard protein-level testing (isoelectric focusing, nephelometry) because they present as null alleles — indistinguishable from other null mechanisms without genetic testing.

3. **Cryptic splice sites**: Some SERPINA1 variants create novel splice sites within exons or introns, leading to aberrant mRNA processing. These can be difficult to predict without computational tools.

4. **Therapeutic implications**: Patients with splice site variants may benefit from different therapeutic approaches (e.g., AAT augmentation therapy, RNA-targeted therapies) compared to those with polymerization variants (Z allele).

## ALS-Related Genes: UNC13A and STMN2

Two additional genes of interest for splice site analysis in the context of neurodegeneration:

### UNC13A
- **Function**: Synaptic vesicle priming protein, essential for neurotransmitter release
- **Location**: Chromosome 19p13.11
- **ALS relevance**: A common intronic variant (rs12608932) creates a cryptic splice site in intron 20-21. When TDP-43 is depleted (as happens in ALS/FTD), this cryptic exon is included, producing a non-functional truncated protein. This makes UNC13A one of the most important genetic modifiers of ALS/FTD.
- **Splice site significance**: The cryptic exon inclusion is a direct consequence of a splice site being created by a SNP, making accurate splice site prediction critical for understanding disease risk.

### STMN2
- **Function**: Stathmin-2 (also called SCG10), a microtubule-regulatory protein essential for axonal regeneration
- **Location**: Chromosome 8p21.3
- **ALS relevance**: Similar to UNC13A, TDP-43 loss leads to a cryptic exon inclusion in STMN2 (between exons 1 and 2), producing a truncated non-functional protein. Reduced STMN2 is a hallmark of ALS motor neurons and a potential therapeutic target.
- **Splice site significance**: The cryptic splice site in STMN2 is a key example of how aberrant splicing directly contributes to neurodegeneration. Detecting these cryptic sites computationally could aid early diagnosis and drug development.

## References

- Crystal RG. Alpha 1-antitrypsin deficiency, emphysema, and liver disease. J Clin Invest. 1990;85(5):1343-1352.
- Stoller JK, Aboussouan LS. A review of alpha1-antitrypsin deficiency. Am J Respir Crit Care Med. 2012;185(3):246-259.
- Brown et al. TDP-43 loss and ALS-risk SNPs drive mis-splicing and depletion of UNC13A. Nature. 2022;603:131-137.
- Klim et al. ALS-implicated protein TDP-43 sustains levels of STMN2, a mediator of motor neuron growth and repair. Nat Neurosci. 2019;22:167-179.
