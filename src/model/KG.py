import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 创建一个有向图
G = nx.DiGraph()

# 添加节点（实体）和边（关系）
G.add_node('Entity1')
G.add_node('Entity2')
G.add_edge('Entity1', 'Entity2', relation='related_to')

# 查看图中的所有节点和边
print(G.nodes)
print(G.edges(data=True))


# 使用NetworkX计算最短路径
shortest_path = nx.shortest_path(G, source='Entity1', target='Entity2')
print("Shortest path:", shortest_path)



class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head, relation, tail):
        # 获取头实体、关系、尾实体的嵌入向量
        head_embed = self.entity_embeddings(head)
        relation_embed = self.relation_embeddings(relation)
        tail_embed = self.entity_embeddings(tail)

        # TransE的基本公式：头 + 关系 ≈ 尾
        score = torch.norm(head_embed + relation_embed - tail_embed, p=1, dim=1)
        return score

# 创建模型实例
model = TransE(num_entities=1000, num_relations=50, embedding_dim=100)


# 预测关系：给定头实体和尾实体，预测关系
def predict_relation(head, tail, model):
    relations = torch.arange(num_relations)
    scores = model(head, relations, tail)
    predicted_relation = torch.argmin(scores).item()
    return predicted_relation

# 示例调用
head = torch.tensor([1])  # 假设实体ID为1
tail = torch.tensor([2])  # 假设实体ID为2
predicted_relation = predict_relation(head, tail, model)
print("Predicted relation:", predicted_relation)




# 画出知识图谱
nx.draw(G, with_labels=True)
plt.show()
