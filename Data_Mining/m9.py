import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class STGCN(torch.nn.Module):
    def __init__(self, in_channels, gcn_hidden_channels, lstm_hidden_channels, out_channels):
        super(STGCN, self).__init__()
        # Spatial component: GCN layers
        self.gcn1 = GCNConv(in_channels, gcn_hidden_channels)
        self.gcn2 = GCNConv(gcn_hidden_channels, gcn_hidden_channels)

        # Temporal component: LSTM
        self.lstm = torch.nn.LSTM(gcn_hidden_channels, lstm_hidden_channels, batch_first=True)

        # Output layer
        self.linear = torch.nn.Linear(lstm_hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # x shape: [batch_size, time_steps, num_nodes, features]
        batch_size, time_steps, num_nodes, in_channels = x.shape

        # Reshape for GCN: merge batch & time dims
        x = x.view(batch_size * time_steps, num_nodes, in_channels)

        # Apply GCN for each graph snapshot
        # PyG GCNConv expects input [num_nodes, features]
        # So we process each graph snapshot separately
        gcn_outputs = []
        for i in range(batch_size * time_steps):
            xi = x[i]
            xi = self.gcn1(xi, edge_index)
            xi = F.relu(xi)
            xi = self.gcn2(xi, edge_index)
            gcn_outputs.append(xi)

        gcn_outputs = torch.stack(gcn_outputs)  # shape: [batch_size*time_steps, num_nodes, gcn_hidden_channels]

        # Restore shape: [batch_size, time_steps, num_nodes, gcn_hidden_channels]
        gcn_outputs = gcn_outputs.view(batch_size, time_steps, num_nodes, -1)

        # Now process temporal info per node:
        # We want to feed each node's feature sequence to LSTM
        # So permute dims to [batch_size*num_nodes, time_steps, gcn_hidden_channels]
        gcn_outputs = gcn_outputs.permute(0, 2, 1, 3)  # [batch_size, num_nodes, time_steps, features]
        gcn_outputs = gcn_outputs.reshape(batch_size * num_nodes, time_steps, -1)

        # LSTM over temporal sequence
        lstm_out, (h_n, c_n) = self.lstm(gcn_outputs)  # lstm_out shape: [batch_size*num_nodes, time_steps, lstm_hidden_channels]

        # Take last time step output for prediction
        last_out = lstm_out[:, -1, :]  # [batch_size*num_nodes, lstm_hidden_channels]

        # Final linear layer
        out = self.linear(last_out)  # [batch_size*num_nodes, out_channels]

        # Reshape back to [batch_size, num_nodes, out_channels]
        out = out.view(batch_size, num_nodes, -1)
        return out


# Dummy example usage:

num_nodes = 5
in_features = 3
time_steps = 4
batch_size = 2
out_features = 1

# Sample edge_index for a small graph (COO format)
edge_index = torch.tensor([[0,1,2,3,4,1,2],[1,0,3,2,1,4,3]], dtype=torch.long)

# Random features: [batch_size, time_steps, num_nodes, in_features]
x = torch.randn(batch_size, time_steps, num_nodes, in_features)

model = STGCN(in_channels=in_features, gcn_hidden_channels=8, lstm_hidden_channels=16, out_channels=out_features)
out = model(x, edge_index)
print(out)
print("Output shape:", out.shape)  # Expect: [batch_size, num_nodes, out_features]
