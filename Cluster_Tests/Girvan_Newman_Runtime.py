# importing the required module 
import timeit 

# code snippet to be executed only once 
mysetup = '''
import networkx as nx
import torch_geometric
from networkx.algorithms import community
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/local/scratch', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()

'''

# code snippet whose execution time is to be measured 
mycode = ''' 
communities_generator = community.girvan_newman(G)

for i in communities_generator:
  print(community.modularity(G, i))

'''

# timeit statement 
print(timeit.timeit(setup = mysetup, 
					stmt = mycode, 
					number = 1)  / 1)
