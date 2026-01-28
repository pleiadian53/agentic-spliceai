"""
Filename: verify_h5_file.py
Author: Kuan-Hao Chao
Date: 2025-03-20
Description: Functions to process sequences to/from .h5 datasets.
"""

import h5py
import torch
import matplotlib.pyplot as plt
import time 
import h5py

def verify_h5(args):
    """
    Verifies the integrity of the created testing and/or training h5 datasets.
    """
    # record start time for benchmark
    print("--- Step 3: Verifying integrity of h5 file ... ---")
    start_time = time.time()

    # construct the filename and open h5 file
    dataset_ls = [] 
    if args.chr_split == 'test':
        dataset_ls.append('test')
    elif args.chr_split == 'train-test':
        dataset_ls.append('test')
        dataset_ls.append('train')
    for dataset_type in dataset_ls:
        if args.biotype =="non-coding":
            filename = f"{args.output_dir}/dataset_{dataset_type}_ncRNA.h5"
            figname = f"{args.output_dir}/verify_{dataset_type}_ncRNA.png"
        elif args.biotype =="protein-coding":
            filename = f"{args.output_dir}/dataset_{dataset_type}.h5"
            figname = f"{args.output_dir}/verify_{dataset_type}.png"

        print(f'Verifying {filename}...')
        with h5py.File(filename, 'r') as hf:
            # print the available dataset keys in the file
            print(f"Dataset keys: {list(hf.keys())}\n\n")

            # convert datasets to PyTorch tensors and display their shapes
            X0_tensor = torch.from_numpy(hf['X0'][:]).float()  
            Y0_tensor = torch.from_numpy(hf['Y0'][:]) 
            print(f"X0 shape: {X0_tensor.shape}, Y0 shape: {Y0_tensor.shape}")

            # process a specific dataset ('X3') for visualization
            x = torch.from_numpy(hf['X3'][:]).float()  
            y = torch.from_numpy(hf['Y3'][:])  
            print(f"x[0].shape: {x[0].shape}, y[0].shape: {y[0].shape}")

            # plot the sum of the last entry in the 'X3' dataset along its rows
            fig = plt.figure(figsize=(7, 3))
            ax = fig.add_subplot(111)
            ax.set_xlim([0, 15000])
            ax.plot(x[-1].sum(axis=1))  # Sum over rows and plot
            fig.savefig(figname, dpi=300, bbox_inches='tight')
            plt.close(fig)

            # additional info
            print("x[0].sum(axis=1): ", len(x[len(x)-1].sum(axis=1))) # Length of row sum in the last 'X3' entry
            print("x[0].sum(axis=0): ", len(x[len(x)-1].sum(axis=0))) # Length of column sum in the last 'X3' entry
            print(f"(x[0].sum(axis=1) == 0).sum(): {(x[0].sum(axis=1) == 0).sum()}") # Number of zero-sum rows in the first 'X3' entry

    # END
    print(("--- %s seconds ---" % (time.time() - start_time)))