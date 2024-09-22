import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# 假设有 4 个节点和 5 条边
# 边表示为 (source, target) 格式
edge_index = torch.tensor([[0, 1, 1, 2, 3],
                           [1, 0, 2, 1, 1]], dtype=torch.long)

# 每个节点的特征向量 (4 个节点, 每个节点 3 维特征)
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=torch.float)

# 创建图数据对象
data = Data(x=x, edge_index=edge_index)




class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        # 定义两层GCN卷积层
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        # 从Data对象获取节点特征和边信息
        x, edge_index = data.x, data.edge_index

        # 第一层卷积 + 激活函数（ReLU）
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        
        return x

# 定义模型，输入维度是3，隐藏层维度是4，输出维度是2
model = GCN(in_channels=3, hidden_channels=4, out_channels=2)

# 使用 Adam 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 损失函数：Cross-Entropy Loss
loss_fn = torch.nn.CrossEntropyLoss()

# 假设我们有节点标签（例如用于分类任务），这是一个4节点的标签
labels = torch.tensor([0, 1, 0, 1])

# 训练模型
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    out = model(data)
    
    # 计算损失（假设所有节点都有标签）
    loss = loss_fn(out, labels)
    
    # 反向传播和优化
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
