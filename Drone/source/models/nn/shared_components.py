import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out) if out.size(0) > 1 else out
        out = F.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out) if out.size(0) > 1 else out
        out += residual
        out = F.leaky_relu(out)
        return out

class AttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.context_vector = nn.Parameter(torch.rand(hidden_dim))

    def forward(self, x):
        attention_weights = torch.matmul(x, self.context_vector)
        attention_weights = F.softmax(attention_weights, dim=0)
        attended_state = x * attention_weights.unsqueeze(1)
        return attended_state

# Usage Example in ICM or Other Networks
if __name__ == "__main__":
    # Example input dimensions
    input_dim = 128
    hidden_dim = 128

    # Instantiate and test ResidualBlock
    residual_block = ResidualBlock(input_dim, hidden_dim)
    test_input = torch.rand(1, input_dim)
    print(f"Residual Block Output: {residual_block(test_input).shape}")

    # Instantiate and test AttentionLayer
    attention_layer = AttentionLayer(input_dim, hidden_dim)
    print(f"Attention Layer Output: {attention_layer(test_input).shape}")
