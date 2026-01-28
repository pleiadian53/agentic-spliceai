# import h5py
# # filename = "datafile_train_all.h5"
# # filename = "dataset_train_all.h5"
# filename = "dataset_test_0.h5"
# # filename = "datafile_test_0.h5"

# with h5py.File(filename, "r") as f:
#     # List all groups
#     print(("Keys: %s" % list(f.keys())))
#     a_group_key = list(f.keys())[3]
#     print(("a_group_key: ", a_group_key))

#     # Get the data
#     data = list(f[a_group_key])
#     # print(("data: ", data))
#     print(("data: ", len(data)))


import h5py
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from constants import *
import time 

start_time = time.time()

assert sys.argv[1] in ['train', 'test', 'all']

# hf = h5py.File('./dataset_train_all.h5', 'r')
hf = h5py.File(data_dir + 'dataset'
                + '_' + sys.argv[1]
                + '.h5', 'r')

donor_motif_dict = {}
acceptor_motif_dict = {}

print(hf.keys())
for i in range(len(hf.keys())//2):
    x_idx = f'X{i}'
    y_idx = f'Y{i}'
    torch.from_numpy(hf[x_idx][:]).shape
    torch.from_numpy(hf[x_idx][:])[0].float()
    torch.from_numpy(hf[y_idx][:]).shape

    x = torch.from_numpy(hf[x_idx][:]).float()
    y = torch.from_numpy(hf[y_idx][:])

    print(f"x[0].shape: {x.shape}")
    # print(f"x[0].shape: {x[0,6231,:]}")
    # print(f"y[0].shape: {y[0].shape}")
    # print(f"y[0].shape: {y[0]}")

    # labels_ls = y[0].transpose(1, 2).reshape(-1, 3)
    labels = y[0].argmax(dim=2)  # If labels are one-hot encoded across the 3 classes
    labels = labels.view(-1)  # Flattening labels from [size, 3, 5000] to [size*5000]

    # print(f"logits before: {x[:, 5000:10000, :]}")

    logits = x[:, 5000:10000, :].reshape(-1, 4)  # Flattening logits from [size, 3, 5000] to [size*5000, 3]

    # print(f"logits.shape: {logits.shape}")
    # print(f"logits: {logits}")
    # print(f"logits: {logits[6231,:]}")

    # print(f"labels.shape: {labels.shape}")
    # print(f"labels: {labels}")

    # Donor site
    d_index_positions = [index for index, value in enumerate(labels.tolist()) if value == 2]
    # print("d_index_positions: ", d_index_positions)
    # Acceptor site
    a_index_positions = [index for index, value in enumerate(labels.tolist()) if value == 1]
    # print("a_index_positions: ", a_index_positions)

    # Convert one-hot encoding to numerical labels
    # Here, the mapping is: N -> 0, A -> 1, C -> 2, G -> 3, T -> 4
    numerical_labels = np.argmax(logits.tolist(), axis=1)
    # print("numerical_labels: ", numerical_labels)
    # print("numerical_labels.shape: ", numerical_labels.shape)

    nucleotides = np.array(['A', 'C', 'G', 'T'])
    # Convert the numerical labels to nucleotide bases using the simplified example
    nucleotide_labels = nucleotides[numerical_labels]

    # print("nucleotide_labels: ", nucleotide_labels)
    # print("nucleotide_labels.shape: ", nucleotide_labels.shape)

    # Get Donor motif:
    for d in d_index_positions:
        # print("d: ", d)
        d_dinuc = "" 
        d_dinuc = d_dinuc.join(nucleotide_labels[d+1:d+3])
        if d_dinuc in donor_motif_dict:
            donor_motif_dict[d_dinuc] += 1
        else:
            donor_motif_dict[d_dinuc] = 1
        # print("d_dinuc: ", d_dinuc)
        # print("nucleotide_labels[d+1:d+3]: ", nucleotide_labels[d+1:d+3])

    # Get Acceptor motif:
    for a in a_index_positions:
        # print("a: ", a)
        a_dinuc = "" 
        a_dinuc = a_dinuc.join(nucleotide_labels[a-2:a])
        if a_dinuc in acceptor_motif_dict:
            acceptor_motif_dict[a_dinuc] += 1
        else:
            acceptor_motif_dict[a_dinuc] = 1
        # print("a_dinuc: ", a_dinuc)
        # print("nucleotide_labels[a-2:a]: ", nucleotide_labels[a-2:a])


print("Donor motifs: ")
for key, value in donor_motif_dict.items():
    print(f"{key}: {value}")

print("\n\n")
print("Acceptor motifs: ")
for key, value in acceptor_motif_dict.items():
    print(f"{key}: {value}")

    # from collections import Counter

    # # Count each distinct element in the list
    # element_count = Counter(labels.tolist())
    # print(element_count)
