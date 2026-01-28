python Step_2_create_dataset.py \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/datafile_train.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/datafile_test.h5 \
--output-dir /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/ \
> Step_2_create_dataset_RefSeq.log 2> Step_2_create_dataset_RefSeq_error.log









python Step_2_create_dataset.py \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/datafile_train.h5 \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/datafile_test.h5 \
--output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/


# spliceai-toolkit create-data \
# --genome-fasta  /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/GRCh38/GCF_000001405.40_GRCh38.p14_genomic.fna \
# --annotation-gff /home/kchao10/data_ssalzbe1/khchao/ref_genome/homo_sapiens/MANE/v1.3/MANE.GRCh38.v1.3.refseq_genomic.gff \
# --output-dir /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE_test/ \
# --parse-type maximum