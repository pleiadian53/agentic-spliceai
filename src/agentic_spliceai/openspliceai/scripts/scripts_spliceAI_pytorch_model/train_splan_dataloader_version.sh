python train_splan_dataloader_version.py \
--flanking-size 80 \
--exp-num small_dataset_noshuffle \
--training-target MANE \
--train-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_train_pytorch.pth \
--test-dataset /home/kchao10/data_ssalzbe1/khchao/data/train_test_dataset_MANE/dataset_test_pytorch.pth \
--project-root /home/kchao10/data_ssalzbe1/khchao/spliceAI-toolkit/ \
--project-name Merged_MANE_genes_to_pytorch_dataset \
> dataset_creation.log 2> dataset_creation_error.log
# --dataset-shuffle \
