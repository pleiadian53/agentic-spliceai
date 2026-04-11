# Variant Analysis Examples

**Status**: Phase 1A+1B complete, Phase 2-3 planned
**Model**: M1-S v2 (logit-space blend, PR-AUC 0.9954)

---

## Scripts

### Phase 1A: Single-Variant Delta Prediction

```bash
# Single variant via CLI
python 01_single_variant_delta.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --chrom chr11 --pos 47333193 --ref C --alt A \
    --gene MYBPC3 --strand - --plot output/delta_plot.png

# Batch from YAML config
python 01_single_variant_delta.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --config test_variants.yaml --no-multimodal
```

### Phase 1B: Splice Consequence Prediction

```bash
# With consequence classification + JSON output
python 01b_splice_consequences.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --config test_variants.yaml \
    --no-multimodal --device cpu \
    --json results/consequences.json
```

### Phase 2: ClinVar & MutSpliceDB Benchmarking

```bash
# Download and filter ClinVar VCF
python 02_clinvar_download.py --output-dir data/clinvar/

# Benchmark on ClinVar (pathogenic vs benign)
python 03_clinvar_benchmark.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
    --clinvar data/clinvar/clinvar_splice_snvs.parquet

# Benchmark on MutSpliceDB (RNA-seq validated variants)
python 04_mutsplicedb_benchmark.py \
    --checkpoint output/meta_layer/m1s_v2_logit_blend/best.pt \
    --fasta data/mane/GRCh38/Homo_sapiens.GRCh38.dna.primary_assembly.fa
```

### Data Preparation Utilities

```bash
# Parse raw MutSpliceDB CSV into splice_sites_induced.tsv
# (one-time setup — data already exists at data/mutsplicedb/)
python ../../scripts/data/parse_mutsplicedb.py
```

## Test Variants

`test_variants.yaml` contains 13 validated splice site mutations across
10 disease genes (PSMD2, MYBPC3, BRCA1, TP53, DMD, CFTR, NF1, ATM,
BRCA2, MLH1), covering both strands and multiple consequence types.

## Results

- [Variant effect validation](results/variant_effect_validation.md) —
  13 disease-gene variants + 4 SpliceAI paper RNA-seq validated cases

## Documentation

- [Development roadmap](docs/development_roadmap.md) — Phase 1-5 plan,
  architecture decisions, what's completed vs planned
- [Saturation mutagenesis application](../../docs/applications/variant_analysis/saturation_mutagenesis_and_validation.md) —
  Phase 3 plan for gene-wide vulnerability mapping

## Related

- [Negative strand tutorial](../../docs/variant_analysis/negative_strand_and_variant_effects.md) — coordinate systems and strand handling
- [OOD generalization](../meta_layer/docs/ood_generalization.md) — model limitations on unseen genes
- [M1-S v2 results](../meta_layer/results/m1s_v2_logit_blend_results.md) — logit blend training results

---

**Last Updated**: April 2026
