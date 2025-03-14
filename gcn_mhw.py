import torch
import torch.nn as nn

class GraphConv(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x, adj_matrix):
        return torch.relu(self.fc(torch.matmul(adj_matrix, x)))

class GCNMHW(nn.Module):
    """Graph Convolution Network with Mexican Hat Wavelet"""
    def __init__(self, in_features=3, hidden_dim=128):
        super(GCNMHW, self).__init__()
        self.conv1 = GraphConv(in_features, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

    def forward(self, x, adj_matrix):
        x = self.conv1(x, adj_matrix)
        return self.conv2(x, adj_matrix)

if __name__ == "__main__":
    model = GCNMHW()
    dummy_pose = torch.randn(1, 17, 3)
    adj_matrix = torch.eye(17)  # Identity matrix as placeholder
    output = model(dummy_pose, adj_matrix)
    print("GCN-MHW Output Shape:", output.shape)
