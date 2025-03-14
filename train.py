import torch.optim as optim
from se_vit import SEViT
from gcn_mhw import GCNMHW
from music_synchronization import sync_loss
from data_preprocessing import AISTDataset

dataset = AISTDataset()
model_vit = SEViT()
optimizer = optim.Adam(model_vit.parameters(), lr=0.001)

for epoch in range(10):  # Placeholder
    for video, pose, music in dataset:
        optimizer.zero_grad()
        features = model_vit(torch.tensor(video).float())
        loss = sync_loss(features, torch.tensor(pose).float())
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
