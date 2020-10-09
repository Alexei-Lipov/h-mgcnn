# importing the required module 
import timeit 

# code snippet to be executed only once 
mysetup = '''
import networkx as nx
import torch_geometric
from torch_geometric.datasets import Planetoid
import community as cmt

dataset = Planetoid(root='/local/scratch/', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()

'''

# code snippet whose execution time is to be measured 
mycode = ''' 
partition = cmt.best_partition(G)
'''

# timeit statement 
print(timeit.timeit(setup = mysetup, 
					stmt = mycode, 
					number = 100)  / 100)
