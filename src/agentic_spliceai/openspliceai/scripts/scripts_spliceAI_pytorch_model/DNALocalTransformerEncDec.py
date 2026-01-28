import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class DNALocalTransformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, device, forward_expansion, dropout, window_size=80, sequence_length=5000):
        super(DNALocalTransformer, self).__init__()
        self.device = device
        self.embed_size = embed_size
        self.window_size = window_size
        self.sequence_length = sequence_length

        # Embedding layer to map 4 nucleotide channels into an embedding space
        self.embedding = nn.Linear(4, embed_size)
        # Positional embeddings to add information about the position in the sequence
        self.position_embedding = nn.Parameter(torch.rand(window_size, embed_size))

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=heads,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer to map from embed_size to 3 output channels (donor, acceptor, neither)
        self.output_layer = nn.Linear(embed_size, 3)

    def forward(self, x):
        batch_size, _, seq_len = x.shape

        # Initialize an output tensor for the entire sequence with zeros
        output_tensor = torch.zeros(batch_size, 3, seq_len).to(self.device)

        # Process each window
        step_size = self.window_size // 2  # 50% overlap
        for start in range(0, seq_len, step_size):
            end = start + self.window_size
            # print("start: ", start, "; end: ", end, "; seq_len: ", seq_len)
            if end > seq_len:
                break  # Skip the last window if it's not full; alternative: pad it

            window = x[:, :, start:end]  # Shape: [batch_size, 4, window_size]
            window = window.permute(0, 2, 1)  # Change to [batch_size, window_size, 4] for linear layer
            window = self.embedding(window)  # Apply embedding

            # Add positional embeddings
            pos_embeddings = self.position_embedding.unsqueeze(0).expand(batch_size, -1, -1)

            window += pos_embeddings
            # Process the window with the transformer encoder
            encoded_window = self.transformer_encoder(window.permute(1, 0, 2))  # Shape: [window_size, batch_size, embed_size]
            decoded_window = self.transformer_decoder(encoded_window, encoded_window)  # Self-attention
            decoded_window = decoded_window.permute(1, 0, 2)  # Back to [batch_size, window_size, embed_size]

            # Apply the output layer
            window_output = self.output_layer(encoded_window)  # Shape: [batch_size, window_size, 3]
            window_output = F.softmax(window_output, dim=-1)
            # Combine the output of this window with the overall output tensor
            output_tensor[:, :, start:end] = window_output.transpose(1, 2)

        return output_tensor  # Final shape: [batch_size, 3, sequence_length]





# # Model parameters setup
# embed_size = 512
# num_layers = 2
# heads = 8
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# forward_expansion = 4
# dropout = 0.1

# # Example model instantiation
# model = DNALocalTransformer(embed_size, num_layers, heads, device, forward_expansion, dropout)
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

# Note: The forward method currently concatenates the processed windows without handling overlaps correctly.
# You'll need to implement a method to properly aggregate these overlapping windows into a continuous sequence output.
