import h5py
import numpy as np
import sys, os
import time
# from utils import *
from math import ceil
from constants import *
import argparse

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


# def create_datapoints(seq, strand, tx_start, tx_end, jn_start, jn_end):
#     # This function first converts the sequence into an integer array, where
#     # A, C, G, T, N are mapped to 1, 2, 3, 4, 0 respectively. If the strand is
#     # negative, then reverse complementing is done. The splice junctions 
#     # are also converted into an array of integers, where 0, 1, 2, -1 
#     # correspond to no splicing, acceptor, donor and missing information
#     # respectively. It then calls reformat_data and one_hot_encode
#     # and returns X, Y which can be used by Keras models.

#     seq = 'N'*(CL_max//2) + seq[CL_max//2:-CL_max//2] + 'N'*(CL_max//2)
#     # Context being provided on the RNA and not the DNA

#     seq = seq.upper().replace('A', '1').replace('C', '2')
#     seq = seq.replace('G', '3').replace('T', '4').replace('N', '0')

#     tx_start = int(tx_start)
#     tx_end = int(tx_end) 

#     # jn_start = map(lambda x: map(int, re.split(',', x)[:-1]), jn_start)
#     # jn_end = map(lambda x: map(int, re.split(',', x)[:-1]), jn_end)

#     # Convert jn_start and jn_end maps to lists of lists
#     jn_start = list(map(lambda x: list(map(int, re.split(',', x)[:-1])), jn_start))
#     jn_end = list(map(lambda x: list(map(int, re.split(',', x)[:-1])), jn_end))

#     if strand == '+':

#         # X0 = np.asarray(map(int, list(seq)))
#         X0 = np.asarray(list(map(int, list(seq))))
#         Y0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

#         for t in range(1):
            
#             if len(jn_start[t]) > 0:
#                 Y0[t] = np.zeros(tx_end-tx_start+1)
#                 for c in jn_start[t]:
#                     if tx_start <= c <= tx_end:
#                         Y0[t][c-tx_start] = 2
#                 for c in jn_end[t]:
#                     if tx_start <= c <= tx_end:
#                         Y0[t][c-tx_start] = 1
#                     # Ignoring junctions outside annotated tx start/end sites
                     
#     elif strand == '-':

#         # X0 = (5-np.asarray(map(int, list(seq[::-1])))) % 5  # Reverse complement
#         X0 = (5-np.asarray(list(map(int, list(seq[::-1]))))) % 5
#         Y0 = [-np.ones(tx_end-tx_start+1) for t in range(1)]

#         for t in range(1):

#             if len(jn_start[t]) > 0:
#                 Y0[t] = np.zeros(tx_end-tx_start+1)
#                 for c in jn_end[t]:
#                     if tx_start <= c <= tx_end:
#                         Y0[t][tx_end-c] = 2
#                 for c in jn_start[t]:
#                     if tx_start <= c <= tx_end:
#                         Y0[t][tx_end-c] = 1

#     Xd, Yd = reformat_data(X0, Y0)
#     X, Y = one_hot_encode(Xd, Yd)

#     return X, Y


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
            if label == 2:
                # Donor site
                d_motif = str(seq[i+1:i+3])
                if d_motif not in donor_motif_counts:
                    donor_motif_counts[d_motif] = 0
                donor_motif_counts[d_motif] += 1
            elif label == 1:
                # Acceptor site
                a_motif = str(seq[i-2:i])
                if a_motif not in acceptor_motif_counts:
                    acceptor_motif_counts[a_motif] = 0
                acceptor_motif_counts[a_motif] += 1
                    

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


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--training-target', '-t', type=str, default="MANE")
    parser.add_argument('--train-dataset', '-train', type=str)
    parser.add_argument('--test-dataset', '-test', type=str)
    parser.add_argument('--output-dir', '-o', type=str)
    args = parser.parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # for type in ['test']:
    for type in ['train', 'test']:
        print(("--- Processing %s ... ---" % type))
        start_time = time.time()
        if type == 'train':
            input_file = args.train_dataset
        elif type == 'test':
            input_file = args.test_dataset
        output_file = f"{os.path.dirname(input_file)}/dataset_{type}.h5"
        print("\tReading datafile.h5 ... ")
        h5f = h5py.File(input_file, 'r')
        STRAND = h5f['STRAND'][:]
        TX_START = h5f['TX_START'][:]
        TX_END = h5f['TX_END'][:]
        SEQ = h5f['SEQ'][:]
        LABEL = h5f['LABEL'][:]
        h5f.close()

        seq_num = SEQ.shape[0]
        print("seq_num: ", seq_num)
        # print("STRAND.shape[0]: ", STRAND.shape[0])
        print("STRAND.shape[0]: ", STRAND)
        print("TX_START.shape[0]: ", TX_START.shape[0])
        print("TX_END.shape[0]: ", TX_END.shape[0])
        print("LABEL.shape[0]: ", LABEL.shape[0])
        # Check motif
        for idx in range(seq_num):
            label_decode = LABEL[idx].decode('ascii')
            seq_decode = SEQ[idx].decode('ascii')
            strand_decode = STRAND[idx].decode('ascii')
            label_int = [int(char) for char in label_decode]
            check_and_count_motifs(seq_decode, label_int, strand_decode)
        print_motif_counts()
        
        # print("\tCreating dataset.h5 ... ")
        # h5f2 = h5py.File(output_file, 'w')
        # CHUNK_SIZE = 100
        # # Create dataset
        # for i in range(seq_num//CHUNK_SIZE):
        #     # Each dataset has CHUNK_SIZE genes
        #     if (i+1) == seq_num//CHUNK_SIZE:
        #         NEW_CHUNK_SIZE = CHUNK_SIZE + seq_num%CHUNK_SIZE
        #     else:
        #         NEW_CHUNK_SIZE = CHUNK_SIZE
        #     X_batch = []
        #     Y_batch = [[] for t in range(1)]
        #     for j in range(NEW_CHUNK_SIZE):
        #         idx = i*CHUNK_SIZE + j
        #         seq_decode = SEQ[idx].decode('ascii')
        #         strand_decode = STRAND[idx].decode('ascii')
        #         tx_start_decode = TX_START[idx].decode('ascii')
        #         tx_end_decode = TX_END[idx].decode('ascii')
        #         label_decode = LABEL[idx].decode('ascii')
        #         fixed_seq = replace_non_acgt_to_n(seq_decode)
        #         X, Y = create_datapoints(fixed_seq, strand_decode, label_decode)            
        #         X_batch.extend(X)
        #         for t in range(1):
        #             Y_batch[t].extend(Y[t])
        #     X_batch = np.asarray(X_batch).astype('int8')
        #     print("X_batch.shape: ", X_batch.shape)
            
        #     for t in range(1):
        #         Y_batch[t] = np.asarray(Y_batch[t]).astype('int8')
        #     print("len(Y_batch[0]): ", len(Y_batch[0]))
        #     h5f2.create_dataset('X' + str(i), data=X_batch)
        #     h5f2.create_dataset('Y' + str(i), data=Y_batch)
        # h5f2.close()
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()