import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn, optim
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from itertools import compress 

print("Downloading dataset...")
orig_dataset = Planetoid(root='/local/scratch/', name='Cora')
print("Downloading pickled clustering...")
G_list = nx.read_gpickle("g_list.gpickle")                                      # from hierarchical clustering
print("Done!")

num_steps = 200                                                                 # proxy for num_scales, e.g. 200 is 3 scales
G_indices = list(range(0, len(G_list)-1, num_steps))                            # -1 since last graph has no edges so GCN cant work properly on it                             
dataset = [None] * len(G_indices)


# networkx removed all the features and other dataset info, so have to reinsert them 
for i, j in zip(list(range(0, len(G_indices))), G_indices):                                                     
  dataset[i] = torch_geometric.utils.from_networkx(G_list[j])
  dataset[i].x = orig_dataset[0].x
  dataset[i].y = orig_dataset[0].y
  dataset[i].train_mask = orig_dataset[0].train_mask
  dataset[i].val_mask = orig_dataset[0].val_mask
  dataset[i].test_mask = orig_dataset[0].test_mask
  dataset[i].num_classes = orig_dataset.num_classes


class GCNN(torch.nn.Module):
    def __init__(self):
        super(GCNN, self).__init__()
        self.conv1 = GCNConv(dataset[0].num_node_features, dataset[0].num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        

        return F.log_softmax(x, dim=1)

out_list = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for i in list(range(0, len(G_indices))):
  model = GCNN().to(device)
  print("Training GCN scale "+ str(i))
  data = dataset[i].to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


  model.train()
  for epoch in range(300):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()                
      optimizer.step()

  
  out_list.append(out)


out_flat = torch.flatten(torch.cat(out_list, dim=0))
out_nodes = torch.split(out_flat, dataset[0].num_classes, dim=0)
labels = torch.cat([dataset[0].y]*(len(dataset)))
labels = labels.tolist()

train_data = []
for i in range(len(out_nodes)):
   train_data.append([out_nodes[i], labels[i]])


# filtering so only "train mask = true" nodes are put into training dataloader
train_data = list(compress(train_data, (torch.cat([dataset[0].train_mask]*(len(dataset)))).tolist()   ))

trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=100)
i1, l1 = next(iter(trainloader))

class FC(torch.nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(dataset[0].num_classes, 30)
        self.fc2 = nn.Linear(30, dataset[0].num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = x.detach()
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = FC().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.NLLLoss()

model.train()
for epoch in range(10):
  print("FC layer training epoch: " + str(epoch))
  for i, l in trainloader:
    i = i.to(device)
    l = l.to(device)
    optimizer.zero_grad()
    output = model(i)
    loss = criterion(output, l)
    loss.backward()
    optimizer.step()

test_data = []
for i in range(len(out_nodes)):
   test_data.append([out_nodes[i], labels[i]])


# filtering so only "test mask = true" nodes are put into testing dataloader
test_data = list(compress(test_data, (torch.cat([dataset[0].test_mask]*(len(dataset)))).tolist()   ))


testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=100)


model.eval()
correct = 0

for i, l in testloader:
    i = i.to(device)
    l = l.to(device)
    pred = model(i).max(1)[1]
    correct += pred.eq(l).sum().item()

print("Accuracy: " + str(correct / len(test_data)))
