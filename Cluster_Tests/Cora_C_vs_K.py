import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric
from torch_geometric.datasets import Planetoid
import community as cmt
import numpy as np

dataset = Planetoid(root='/local/scratch/', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()

font = {'size'   : 15}

plt.rc('font', **font)

c = nx.clustering(G)

k = nx.degree(G)

y = list(c.values())

x = [x[1] for x in k]

x1 = np.arange(1, 175, 0.05)

np.shape(x1)[0]

y1 = np.ones(np.shape(x1)[0],)*(4) / x1

plt.loglog(x,y,'kx', markersize = 4 )
plt.loglog(x1,y1, color = '#f76c6c')
plt.xlim(1, 100)
plt.ylim(0.01,1.2)
plt.xlabel("$k$")
plt.ylabel("$C(k)$")
plt.tight_layout()
plt.savefig("C_vs_K_Plot.png",dpi=1000)
