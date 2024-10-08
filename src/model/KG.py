import networkx as nx
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

G = nx.DiGraph()

G.add_node('Entity1')
G.add_node('Entity2')
G.add_edge('Entity1', 'Entity2', relation='related_to')

print(G.nodes)
print(G.edges(data=True))


shortest_path = nx.shortest_path(G, source='Entity1', target='Entity2')
print("Shortest path:", shortest_path)



class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(TransE, self).__init__()
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

    def forward(self, head, relation, tail):
        head_embed = self.entity_embeddings(head)
        relation_embed = self.relation_embeddings(relation)
        tail_embed = self.entity_embeddings(tail)

        score = torch.norm(head_embed + relation_embed - tail_embed, p=1, dim=1)
        return score

model = TransE(num_entities=1000, num_relations=50, embedding_dim=100)


def predict_relation(head, tail, model):
    relations = torch.arange(num_relations)
    scores = model(head, relations, tail)
    predicted_relation = torch.argmin(scores).item()
    return predicted_relation

head = torch.tensor([1])  
tail = torch.tensor([2])  
predicted_relation = predict_relation(head, tail, model)
print("Predicted relation:", predicted_relation)


nx.draw(G, with_labels=True)
plt.show()
