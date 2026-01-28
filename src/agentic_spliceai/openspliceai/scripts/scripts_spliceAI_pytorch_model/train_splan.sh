##########################################
# Train RefSeq dataset h5py_version
##########################################
python train_splan.py --flanking-size 80 \
--exp-num small_dataset_h5py_version \
--training-target RefSeq \
--train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_train.h5 \
--test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_RefSeq/dataset_test.h5 \
--project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
--project-name RefSeq_h5py_dataset \
--model DNALocalTransformer \
> train_splan.log 2> train_splan_error.log

# --exp-num full_dataset_h5py_version \

# ##########################################
# # Train MANE dataset h5py_version
# ##########################################
# python train_splan.py --flanking-size 80 \
# --exp-num full_dataset_h5py_version \
# --training-target MANE \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_MANE/dataset_train_all.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_MANE/dataset_test_0.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name MANE_h5py_dataset \
# > dataset_creation.log 2> dataset_creation_error.log

# ##########################################
# # Train SpliceAI27 dataset h5py_version
# ##########################################
# python train_splan.py --flanking-size 80 \
# # --exp-num medium_dataset_h5py_version \
# --exp-num full_dataset_h5py_version \
# --training-target SpliceAI27 \
# --train-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_SpliceAI27/dataset_train_all.h5 \
# --test-dataset /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/results/train_test_dataset_SpliceAI27/dataset_test_0.h5 \
# --project-root /Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/ \
# --project-name SpliceAI27_h5py_dataset \
# > dataset_creation.log 2> dataset_creation_error.log