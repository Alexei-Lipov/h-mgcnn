import community as cmt
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/local/scratch', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()
dendo = cmt.generate_dendrogram(G)
file = open("Louvain_Modularities.txt","w") 
for level in range(len(dendo) - 1):
  partition = cmt.partition_at_level(dendo, level)
  print("partition at level", level, "is", partition)
  file.write(str(level) + "," + str(cmt.modularity(partition, G))) 
  file.write('\n')

file.close()
