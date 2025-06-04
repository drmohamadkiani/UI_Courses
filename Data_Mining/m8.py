import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, k_hop_subgraph
import matplotlib.pyplot as plt
import networkx as nx

# Load dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Choose a center node randomly
center_node = 50  # you can also do: torch.randint(0, data.num_nodes, (1,)).item()

# Get subgraph around that node (3-hop neighborhood)
subset, edge_index, _, _ = k_hop_subgraph(center_node, num_hops=1, edge_index=data.edge_index)

# Create subgraph data
from torch_geometric.data import Data
sub_data = Data(x=data.x[subset], edge_index=edge_index)

# Convert to networkx graph
G = to_networkx(sub_data, to_undirected=True)

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=150, node_color='lightcoral', edge_color='gray')
plt.title(f"2-hop Subgraph Around Node {center_node}")
plt.show()
