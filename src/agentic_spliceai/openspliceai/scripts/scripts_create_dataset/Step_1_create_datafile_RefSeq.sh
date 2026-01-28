python Step_1_create_datafile.py \
--genome-fasta  /Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/NCBI_Refseq_chr_fixed/GCF_000001405.40_GRCh38.p14_genomic.fna \
--annotation-gff /Users/chaokuan-hao/Documents/Projects/ref_genome/homo_sapiens/NCBI_Refseq_chr_fixed/GCF_000001405.40_GRCh38.p14_genomic.gff \
--output-dir /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/ \
--parse-type all_isoforms \
> Step_1_create_datafile_RefSeq.log 2> Step_1_create_datafile_RefSeq_error.log




python Step_1_create_datafile.py \
--genome-fasta  /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna \
--annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/ \
--parse-type maximum