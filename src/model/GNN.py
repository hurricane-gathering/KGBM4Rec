from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
import networkx as nx

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__(aggr='mean')  # 聚合方式为mean
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
  
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
   
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return self.lin(aggr_out)

gnn_model = GNN(in_channels=3, out_channels=2)

out = gnn_model(data.x, data.edge_index)
print(out)


G = nx.Graph()
edge_index_np = data.edge_index.numpy()
G.add_edges_from(list(zip(edge_index_np[0], edge_index_np[1])))

nx.draw(G, with_labels=True, node_color='lightblue', node_size=600)
plt.show()
