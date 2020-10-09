import networkx as nx
import torch_geometric
from torch_geometric.datasets import Planetoid
import markov_clustering as mc
import random
import matplotlib.pyplot as plt

dataset = Planetoid(root='/local/scratch', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()

matrix = nx.to_scipy_sparse_matrix(G)

result = mc.run_mcl(matrix, inflation = 1.3)          
clusters = mc.get_clusters(result)   
print(clusters)

file = open("Markov_Modularities.txt","w") 
# perform clustering using different inflation values from 1.1 and 2.6
# for each clustering run, calculate the modularity
for inflation in [i / 10 for i in range(11, 26)]:
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    Q = mc.modularity(matrix=result, clusters=clusters)
    print("inflation:", inflation, "modularity:", Q)
    file.write(str(inflation) + "," + str(Q)) 
    file.write('\n')
 


file.close()
