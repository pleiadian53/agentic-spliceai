import torch
from torch.utils.data import Dataset

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
    