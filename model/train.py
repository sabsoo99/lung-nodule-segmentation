from unet3d import UNet3D
import torch
from torch.utils.data import DataLoader, TensorDataset

def train(model, dataloader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        for x, y in dataloader:
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Example
if __name__ == '__main__':
    model = UNet3D()
    dummy_data = torch.rand(4, 1, 64, 64, 64)
    dummy_labels = torch.rand(4, 1, 64, 64, 64)
    dataset = TensorDataset(dummy_data, dummy_labels)
    loader = DataLoader(dataset, batch_size=2)
    train(model, loader)
