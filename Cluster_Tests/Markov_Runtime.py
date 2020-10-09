# importing the required module 
import timeit 

# code snippet to be executed only once 
mysetup = '''
import networkx as nx
import torch_geometric
from torch_geometric.datasets import Planetoid
import markov_clustering as mc

dataset = Planetoid(root='/local/scratch/', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()
'''

# code snippet whose execution time is to be measured 
mycode = ''' 
matrix = nx.to_scipy_sparse_matrix(G)

for inflation in [i / 10 for i in range(11, 26)]:
    result = mc.run_mcl(matrix, inflation=inflation)
    clusters = mc.get_clusters(result)
    Q = mc.modularity(matrix=result, clusters=clusters)
    print("inflation:", inflation, "modularity:", Q)
		
'''

# timeit statement 
print(timeit.timeit(setup = mysetup, 
					stmt = mycode, 
					number = 1)  / 1)
