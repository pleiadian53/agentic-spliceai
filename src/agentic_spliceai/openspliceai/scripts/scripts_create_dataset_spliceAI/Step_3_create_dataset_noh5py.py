import h5py
import numpy as np
import sys, os
import time
import argparse
# from utils import *
from math import ceil
from constants import *
import torch
from torch.utils.data import Dataset, DataLoader


# One-hot encoding of the inputs: 
# 0 is for padding, 
# 1: A;  2: C;  3: G;  4: T
IN_MAP = np.asarray([[0, 0, 0, 0],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

# One-hot encoding of the outputs: 
# 0: no splice;  1: acceptor;  2: donor;  -1: padding
OUT_MAP = np.asarray([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])


def replace_non_acgt_to_n(input_string):
    """
    Use a generator expression to go through each character in the input string.
    If the character is in the set of allowed characters, keep it as is.
    Otherwise, replace it with 'N'.
    """
    # Define the set of allowed characters
    allowed_chars = {'A', 'C', 'G', 'T'}    
    return ''.join(char if char in allowed_chars else 'N' for char in input_string)


def create_datapoints(seq, strand, label):
    """
    This function first converts the sequence into an integer array, where
    A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
    negative, then reverse complementing is done. The labels 
    are directly used as they are, converted into an array of integers,
    where 0, 1, 2 correspond to no splicing, acceptor, donor 
    respectively. It then calls reformat_data and one_hot_encode
    and returns X, Y which can be used by Pytorch Model.
    """
    # I do not need to reverse complement the sequence, as the sequence is already reverse complemented in the previous step
    seq = 'N' * (CL_max // 2) + seq + 'N' * (CL_max // 2)
    seq = seq.upper().replace('A', '1').replace('C', '2')
    seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')
    # Convert label string to array of integers
    label_array = np.array(list(map(int, list(label))))
    X0 = np.asarray(list(map(int, list(seq))))
    Y0 = label_array
    Y0 = [Y0]
    Xd, Yd = reformat_data(X0, Y0)
    X, Y = one_hot_encode(Xd, Yd)
    return X, Y


def ceil_div(x, y):
    return int(ceil(float(x)/y))


def reformat_data(X0, Y0):
    """
    This function converts X0, Y0 of the create_datapoints function into
    blocks such that the data is broken down into data points where the
    input is a sequence of length SL+CL_max corresponding to SL nucleotides
    of interest and CL_max context nucleotides, the output is a sequence of
    length SL corresponding to the splicing information of the nucleotides
    of interest. The CL_max context nucleotides are such that they are
    CL_max/2 on either side of the SL nucleotides of interest.
    """
    num_points = ceil_div(len(Y0[0]), SL)
    Xd = np.zeros((num_points, SL+CL_max))
    Yd = [-np.ones((num_points, SL)) for t in range(1)]
    X0 = np.pad(X0, [0, SL], 'constant', constant_values=0)
    Y0 = [np.pad(Y0[t], [0, SL], 'constant', constant_values=-1)
         for t in range(1)]
    for i in range(num_points):
        Xd[i] = X0[SL*i:CL_max+SL*(i+1)]
    for t in range(1):
        for i in range(num_points):
            Yd[t][i] = Y0[t][SL*i:SL*(i+1)]
    return Xd, Yd


def one_hot_encode(Xd, Yd):
    return IN_MAP[Xd.astype('int8')], \
           [OUT_MAP[Yd[t].astype('int8')] for t in range(1)]

    # return IN_MAP[Xd.asdata_type('int8')], \
    #        [OUT_MAP[Yd[t].asdata_type('int8')] for t in range(1)]

donor_motif_counts = {}  # Initialize counts
acceptor_motif_counts = {}  # Initialize counts

def check_and_count_motifs(seq, labels, strand):
    """
    Check sequences for donor and acceptor motifs based on labels and strand,
    and return their counts in a dictionary.
    """    
    global donor_motif_counts, acceptor_motif_counts
    for i, label in enumerate(labels):
        if label in [1, 2]:  # Check only labeled positions
            if strand == '+':  # For forward strand
                motif = str(seq[i:i+2]) if i > 0 else ""  # Extract preceding 2-base motif
                if label == 2:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1
                elif label == 1:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1
            elif strand == '-':  # For reverse strand, after adjustment
                motif = str(seq[i:i+2]) if i < len(seq) - 1 else ""  # Extract following 2-base motif
                if label == 2:
                    if motif not in donor_motif_counts:
                        donor_motif_counts[motif] = 0
                    donor_motif_counts[motif] += 1    
                elif label == 1:
                    if motif not in acceptor_motif_counts:
                        acceptor_motif_counts[motif] = 0
                    acceptor_motif_counts[motif] += 1
                    

def print_motif_counts():
    global donor_motif_counts, acceptor_motif_counts
    print("Donor motifs:")
    for motif, count in donor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nAcceptor motifs:")
    for motif, count in acceptor_motif_counts.items():
        print(f"{motif}: {count}")
    print("\nTotal donor motifs: ", sum(donor_motif_counts.values()))
    print("Total acceptor motifs: ", sum(acceptor_motif_counts.values()))


class SpliceAIDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.STRAND = f['STRAND'][:]
            self.TX_START = f['TX_START'][:]
            self.TX_END = f['TX_END'][:]
            self.SEQ = f['SEQ'][:]
            self.LABEL = f['LABEL'][:]
        
        # Assuming create_datapoints and replace_non_acgt_to_n are defined elsewhere
        self.X_data = []
        self.Y_data = []
        counter = 0
        print("len(self.SEQ): ", len(self.SEQ))
        for idx in range(len(self.SEQ)):
            seq_decode = self.SEQ[idx].decode('ascii')
            strand_decode = self.STRAND[idx].decode('ascii')
            label_decode = self.LABEL[idx].decode('ascii')
            fixed_seq = replace_non_acgt_to_n(seq_decode)
            X, Y = create_datapoints(fixed_seq, strand_decode, label_decode)
            print("X_tensor.shape: ", X.shape)
            print("Y_tensor.shape: ", Y[0].shape)
            self.X_data.extend(X)
            self.Y_data.extend(Y[0])
            counter += 1
            if counter > 200:
                break
        # self.X_data = torch.tensor((np.asarray(self.X_data)))#.asdata_type('int8')
        # self.Y_data = torch.tensor((np.asarray(self.Y_data)))#.asdata_type('int8')
        print("self.X_data.shape: ", len(self.X_data))
        print("self.Y_data.shape: ", len(self.Y_data))
        # # print("X_data: ", self.X_data)
        # # print("Y_data: ", self.Y_data)
        
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        # Convert to tensor when accessing the item
        X_tensor = torch.tensor(self.X_data[idx], dtype=torch.int8)
        Y_tensor = torch.tensor(self.Y_data[idx], dtype=torch.int8)
        return X_tensor, Y_tensor
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', '-p', type=str)
    args = parser.parse_args()
    project_root = "/Users/chaokuan-hao/Documents/Projects/spliceAI-toolkit/"
    # project_root = args.project_root
    output_dir = f"{project_root}train_test_dataset_/"
    os.makedirs(output_dir, exist_ok=True)
    for data_type in ['train', 'test']:
        # data_type = 'test'
        print(("--- Processing %s ... ---" % data_type))
        start_time = time.time()
        input_file = output_dir + f'datafile_{data_type}.h5'
        # output_file = output_dir + f'dataset_{data_type}_500.h5'
        print("\tReading datafile.h5 ... ")

        dataset = SpliceAIDataset(input_file)
        # Save dataset
        torch.save(dataset, f'{output_dir}dataset_{data_type}_pytorch.pth')

        dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
        # print("dataloader: ", dataloader)
        # for X_batch, Y_batch in dataloader:
            # print("Batch X shape:", X_batch.shape)
            # print("Batch Y shape:", Y_batch.shape)
        
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()