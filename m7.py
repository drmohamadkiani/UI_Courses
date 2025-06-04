import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling

# Load Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Remove edge_index from data to prepare for link prediction
data = train_test_split_edges(data)

# Define GCN encoder to generate node embeddings
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Link prediction model
class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.encoder = GCNEncoder(in_channels, 64)

    def decode(self, z, edge_index):
        # Dot product between node embeddings of each edge
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def decode_all(self, z):
        # Predict scores for all node pairs (optional)
        prob_adj = torch.matmul(z, z.t())
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, pos_edge_index, neg_edge_index):
        z = self.encoder(x, edge_index)
        # Positive edge scores
        pos_score = self.decode(z, pos_edge_index)
        # Negative edge scores
        neg_score = self.decode(z, neg_edge_index)
        return pos_score, neg_score

# Instantiate model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LinkPredictor(dataset.num_node_features).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=pos_edge_index.size(1)
    )
    pos_score, neg_score = model(data.x, data.train_pos_edge_index, pos_edge_index, neg_edge_index)

    # Labels: 1 for positive edges, 0 for negative
    pos_label = torch.ones(pos_score.size(0), device=device)
    neg_label = torch.zeros(neg_score.size(0), device=device)
    loss = F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]),
                                              torch.cat([pos_label, neg_label]))
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test(pos_edge_index, neg_edge_index):
    model.eval()
    z = model.encoder(data.x, data.train_pos_edge_index)
    pos_score = model.decode(z, pos_edge_index).sigmoid()
    neg_score = model.decode(z, neg_edge_index).sigmoid()
    # Combine and evaluate accuracy/AUC
    from sklearn.metrics import roc_auc_score
    y_true = torch.cat([torch.ones(pos_score.size(0)), torch.zeros(neg_score.size(0))])
    y_score = torch.cat([pos_score, neg_score])
    return roc_auc_score(y_true.cpu(), y_score.cpu())

# Training loop
for epoch in range(1, 101):
    loss = train()
    val_auc = test(data.val_pos_edge_index, data.val_neg_edge_index)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

# Final test
test_auc = test(data.test_pos_edge_index, data.test_neg_edge_index)
print(f'Test AUC: {test_auc:.4f}')
