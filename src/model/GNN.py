from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
import networkx as nx

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__(aggr='mean')  # 聚合方式为mean
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 将自环加入边集中
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # 进行消息传递
        return self.propagate(edge_index, x=x)
    
    def message(self, x_j):
        # 消息函数，返回每个邻居的特征
        return x_j

    def update(self, aggr_out):
        # 更新函数，使用线性层处理聚合后的特征
        return self.lin(aggr_out)

# 创建 GNN 模型
gnn_model = GNN(in_channels=3, out_channels=2)

# 前向传播
out = gnn_model(data.x, data.edge_index)
print(out)


# 转换为networkx图
G = nx.Graph()
edge_index_np = data.edge_index.numpy()
G.add_edges_from(list(zip(edge_index_np[0], edge_index_np[1])))

# 绘制图
nx.draw(G, with_labels=True, node_color='lightblue', node_size=600)
plt.show()
