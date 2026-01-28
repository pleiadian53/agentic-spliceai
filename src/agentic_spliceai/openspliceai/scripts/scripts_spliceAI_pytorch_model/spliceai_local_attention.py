import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, channel_size, seq_length, window_size=80, feature_size=512):
        super(LocalAttention, self).__init__()
        self.channel_size = channel_size
        self.seq_length = seq_length
        self.window_size = window_size
        self.feature_size = feature_size
        
        # Define the query, key, value transformations
        self.query = nn.Linear(window_size, feature_size)
        self.key = nn.Linear(window_size, feature_size)
        self.value = nn.Linear(window_size, feature_size)
        
        # Assuming a fully connected layer for processing after attention
        self.fc = nn.Linear(feature_size, channel_size)  # Adjust output features as needed

    def forward(self, x):
        # x shape: [batch_size, channels, seq_length]
        batch_size, channels, seq_length = x.shape
        
        # Initialize output tensor
        output = torch.zeros_like(x)
        
        # Loop through each position in the sequence
        for i in range(seq_length):
            # Calculate start and end indices for the local window
            start = max(0, i - self.window_size // 2)
            end = min(seq_length, i + self.window_size // 2)

            print(start, " - ", end)
            
            # Extract the local window for query, key, value
            local_window = x[:, :, start:end]
            
            # Transform query, key, value
            q = self.query(local_window)
            k = self.key(local_window)
            v = self.value(local_window)
            
            # Compute attention scores
            attn_scores = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
            
            # Apply attention to values
            attn_output = torch.bmm(attn_scores, v)
            
            # Post-processing (optional, depending on your architecture)
            attn_output = attn_output.mean(dim=1)  # Example aggregation
            
            # Pass through a final fully connected layer (reshape as needed)
            attn_output = self.fc(attn_output.view(batch_size, -1))
            
            # Update output tensor
            output[:, :, i] = attn_output
        
        return output



# Model parameters setup
embed_size = 512
num_layers = 2
heads = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
forward_expansion = 4
dropout = 0.1



batch_size = 10
sequence_length = 5000
nucleotide_types = 4

# Example model instantiation
model = LocalAttention(embed_size, sequence_length)
print("model: ", model)
# Generate a random input sequence
input_sequence = torch.randint(high=nucleotide_types, size=(batch_size, sequence_length, nucleotide_types)).float()
# input_sequence = F.one_hot(input_sequence.argmax(dim=-1), num_classes=nucleotide_types).float()
input_sequence = input_sequence.permute(0, 2, 1)  # Adjust shape to (batch_size, nucleotide_types, sequence_length)

print("input_sequence: ", input_sequence.shape)
# Feed the input sequence into the model
output = model(input_sequence)

output.shape
print("output: ", output.shape)
