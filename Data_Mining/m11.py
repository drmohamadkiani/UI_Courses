from ogb.nodeproppred import NodePropPredDataset
import torch
from torch.serialization import add_safe_globals
from torch_geometric.data.data import Data

add_safe_globals({'torch_geometric.data.data.Data': Data})
from torch_geometric.data.data import DataEdgeAttr
add_safe_globals({'torch_geometric.data.data.DataEdgeAttr': DataEdgeAttr})

import torch
from torch_geometric.data import Data
from ogb.nodeproppred import NodePropPredDataset


dataset = NodePropPredDataset(name='ogbn-products')
split_idx = dataset.get_idx_split()

graph, labels = dataset[0]


edge_index = torch.tensor(graph['edge_index'], dtype=torch.long)
x = torch.tensor(graph['node_feat'], dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long).squeeze()

data = Data(x=x, edge_index=edge_index, y=y)

train_idx = split_idx["train"]
valid_idx = split_idx["valid"]
test_idx = split_idx["test"]

print(data)
print(f"# Train samples: {train_idx.shape[0]}")

