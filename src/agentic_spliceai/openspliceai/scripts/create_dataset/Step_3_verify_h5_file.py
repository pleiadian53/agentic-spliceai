import h5py
import os

project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/"
output_dir = f"{project_root}results/train_test_dataset_MANE/"
os.makedirs(output_dir, exist_ok=True)

# for type in ['test']:
for type in ['train', 'test']:
    input_file = output_dir + f'datafile_{type}.h5'
    output_file = output_dir + f'dataset_{type}_500.h5'

    # filename = "datafile_train_all.h5"
    # filename = "datafile_test_0.h5"

    with h5py.File(output_file, "r") as f:
        # List all groups
        print(("Keys: %s" % list(f.keys())))
        a_group_key = list(f.keys())[0]
        print(("a_group_key: ", a_group_key))
        for key in f.keys():
            # print(key)
            # Get the data
            print(f[key])
            # data = list(f[a_group_key])
            # # print(("data: ", data))
            # print(("data: ", len(data)))
            # print(("data: ", data[1].shape))
