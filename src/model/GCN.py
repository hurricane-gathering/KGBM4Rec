import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


edge_index = torch.tensor([[0, 1, 1, 2, 3],
                           [1, 0, 2, 1, 1]], dtype=torch.long)

x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10, 11, 12]], dtype=torch.float)


data = Data(x=x, edge_index=edge_index)

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
       
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
       
        x, edge_index = data.x, data.edge_index

       
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        
        x = self.conv2(x, edge_index)
        
        return x

model = GCN(in_channels=3, hidden_channels=4, out_channels=2)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


loss_fn = torch.nn.CrossEntropyLoss()

labels = torch.tensor([0, 1, 0, 1])


for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    

    out = model(data)
    
    loss = loss_fn(out, labels)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
