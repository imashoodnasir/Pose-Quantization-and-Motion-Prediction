import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze and Excitation Block"""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        se_weight = torch.sigmoid(self.fc2(torch.relu(self.fc1(x.mean(dim=-1)))))
        return x * se_weight.unsqueeze(-1)

class SEViT(nn.Module):
    """Vision Transformer with SE Block"""
    def __init__(self, embed_dim=256):
        super(SEViT, self).__init__()
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.se = SEBlock(embed_dim)
        self.transformer = nn.Transformer(embed_dim, nhead=8, num_encoder_layers=6)
        self.fc_out = nn.Linear(embed_dim, 512)  # Output feature size

    def forward(self, x):
        x = self.conv1(x).flatten(2).permute(2, 0, 1)  # Reshape for transformer
        x = self.se(x)
        x = self.transformer(x, x)
        return self.fc_out(x.mean(dim=0))

if __name__ == "__main__":
    model = SEViT()
    dummy_input = torch.randn(1, 3, 384, 384)
    output = model(dummy_input)
    print("SE-ViT Output Shape:", output.shape)
