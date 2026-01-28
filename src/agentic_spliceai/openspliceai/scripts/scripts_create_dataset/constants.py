CL_max=10000
# Maximum nucleotide context length (CL_max/2 on either side of the 
# position of interest)
# CL_max should be an even number

SL=5000
# Sequence length of SpliceAIs (SL+CL will be the input length and
# SL will be the output length)

# step 1 input
# ref_genome='/genomes/Homo_sapiens/UCSC/hg19/Sequence/WholeGenomeFasta/genome.fa'
# ref_genome='/Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/hg19/hg19.fa'
ref_genome='/home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna'


# step 1 output dir
data_dir='/home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_SpliceAI27_test/'
# splice_table='../../results/train_test_dataset_spliceAI/canonical_dataset.txt'
splice_table='./canonical_dataset.txt'

# Input details
# sequence='../../results/train_test_dataset_spliceAI/canonical_sequence.txt'
sequence='./canonical_sequence.txt'
# Output details
