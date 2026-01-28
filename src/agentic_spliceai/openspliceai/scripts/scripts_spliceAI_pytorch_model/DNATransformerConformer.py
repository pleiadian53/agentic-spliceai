import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, kernel_size=33):
        super(ConformerBlock, self).__init__()
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Convolution for local context
        src2 = src.permute(1, 2, 0)  # Change to (batch, features, seq_len) for Conv1d
        src2 = F.gelu(self.conv1(src2))
        src2 = src2.permute(2, 0, 1)  # Back to (seq_len, batch, features)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Multi-head attention for global context
        src2, _ = self.attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm2(src)
        
        # Feedforward network
        src2 = self.ff(src)
        src = src + self.dropout2(src2)
        src = self.norm3(src)
        
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5080):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class DNATransformerConformer(nn.Module):
    def __init__(self, ninp, nhead, nhid, nconformers, dropout=0.5, kernel_size=33):
        super(DNATransformerConformer, self).__init__()
        self.model_type = 'Transformer+Conformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.encoder = nn.Linear(4, ninp)  # Project one-hot vectors to embedding dimension
        
        # Conformer blocks
        self.conformers = nn.ModuleList([ConformerBlock(ninp, nhead, nhid, dropout, kernel_size) for _ in range(nconformers)])
        
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, 3)  # Output layer

    def forward(self, src):
        src = src.permute(0, 2, 1)  # Adjust shape to (batch_size, sequence_length, nucleotide_types)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        for conformer in self.conformers:
            src = conformer(src)
        output = self.decoder(src)
        output = output.permute(0, 2, 1)  # Adjust shape to (batch_size, nucleotide_types, sequence_length)
        output = F.softmax(output, dim=1)
        return output



# # Model instantiation example
# ninp = 512 # embedding dimension
# nhead = 8 # number of attention heads
# nhid = 2048 # dimension of the feedforward network model in nn.TransformerEncoder
# nconformers = 6 # number of nn.TransformerEncoderLayer
# dropout = 0.1 # dropout rate
# kernel_size = 33  # Kernel size for the convolutional layers in the Conformer blocks

# model = DNATransformerWithConformer(ninp=ninp, nhead=nhead, nhid=nhid, nconformers=nconformers, dropout=dropout, kernel_size=kernel_size)

# print("model: ", model)

# batch_size = 10
# sequence_length = 5000
# nucleotide_types = 4
# # Generate a random input sequence
# input_sequence = torch.randint(high=nucleotide_types, size=(batch_size, sequence_length, nucleotide_types)).float()
# # input_sequence = F.one_hot(input_sequence.argmax(dim=-1), num_classes=nucleotide_types).float()
# input_sequence = input_sequence.permute(0, 2, 1)  # Adjust shape to (batch_size, nucleotide_types, sequence_length)

# print("input_sequence: ", input_sequence.shape)
# # Feed the input sequence into the model
# output = model(input_sequence)

# output.shape
# print("output: ", output.shape)
