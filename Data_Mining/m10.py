import torch
from torch_geometric_temporal.dataset import METRLADatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.nn import STGCN
import matplotlib.pyplot as plt

# Load dataset
loader = METRLADatasetLoader()
dataset = loader.get_dataset()

# Split dataset into train and test
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Initialize model
# input_features = 1 (speed), output_features = 1 (predicted speed)
model = STGCN(node_features=1, hidden_channels=32, out_channels=1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    total_loss = 0
    for snapshot in train_dataset:
        optimizer.zero_grad()
        y_hat = model(snapshot.x, snapshot.edge_index)
        loss = criterion(y_hat, snapshot.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataset)

def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for snapshot in test_dataset:
            y_hat = model(snapshot.x, snapshot.edge_index)
            loss = criterion(y_hat, snapshot.y)
            total_loss += loss.item()
    return total_loss / len(test_dataset)

# Training loop
epochs = 20
for epoch in range(epochs):
    loss = train()
    test_loss = test()
    print(f"Epoch {epoch+1}, Train Loss: {loss:.4f}, Test Loss: {test_loss:.4f}")

# Visualize prediction vs ground truth for last snapshot in test
model.eval()
with torch.no_grad():
    snapshot = test_dataset[-1]
    y_pred = model(snapshot.x, snapshot.edge_index)

plt.figure(figsize=(10,5))
plt.plot(snapshot.y.cpu().numpy(), label='Ground Truth')
plt.plot(y_pred.cpu().numpy(), label='Prediction')
plt.legend()
plt.title('Traffic Speed Prediction on METR-LA')
plt.show()
