import networkx as nx
import math
import csv
import random as rand
import sys
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import torch_geometric
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/local/scratch', name='Cora')

G = torch_geometric.utils.to_networkx(dataset[0])

G = G.to_undirected()

fig = plt.figure(1,figsize=(20,20))
nx.draw(G,node_size=60,font_size=8, with_labels=False, edge_color="white")
fig.set_facecolor("#00000F")
plt.show()

# Girvan-Newman implementation from http://www.kazemjahanbakhsh.com/codes/cmty.html  - I have modified it 

_DEBUG_ = False

# This method reads the graph structure from the input file
def buildG(G, file_, delimiter_):
    #construct the weighted version of the contact graph from cgraph.dat file
    #reader = csv.reader(open("/home/kazem/Data/UCI/karate.txt"), delimiter=" ")
    reader = csv.reader(open(file_), delimiter=delimiter_)
    for line in reader:
        if len(line) > 2:
            if float(line[2]) != 0.0:
                #line format: u,v,w
                G.add_edge(int(line[0]),int(line[1]),weight=float(line[2]))
        else:
            #line format: u,v
            G.add_edge(int(line[0]),int(line[1]),weight=1.0)


# This method keeps removing edges from Graph until one of the connected components of Graph splits into two
# compute the edge betweenness
def CmtyGirvanNewmanStep(G):
    if _DEBUG_:
        print("Running CmtyGirvanNewmanStep method ...")
    init_ncomp = nx.number_connected_components(G)    #no of components
    ncomp = init_ncomp
    while ncomp <= init_ncomp:
        bw = nx.edge_betweenness_centrality(G, weight='weight')    #edge betweenness for G
        #find the edge with max centrality
        max_ = max(bw.values())
        #find the edge with the highest centrality and remove all of them if there is more than one!
        for k, v in bw.items():
            if float(v) == max_:
                G.remove_edge(k[0],k[1])    #remove the central edge
        ncomp = nx.number_connected_components(G)    #recalculate the no of components
            #added return G here (Alex)

# This method compute the modularity of current split
def _GirvanNewmanGetModularity(G, deg_, m_, Num_com_list):
    New_A = nx.adj_matrix(G)
    New_deg = {}
    New_deg = UpdateDeg(New_A, G.nodes())
    #Let's compute the Q
    comps = nx.connected_components(G)    #list of components    
    print('No of communities in decomposed G:', nx.number_connected_components(G))
    Num_com_list.append(nx.number_connected_components(G))
    Mod = 0    #Modularity of a given partitionning
    for c in comps:
        EWC = 0    #no of edges within a community
        RE = 0    #no of random edges
        for u in c:
            EWC += New_deg[u]
            RE += deg_[u]        #count the probability of a random edge
        Mod += ( float(EWC) - float(RE*RE)/float(2*m_) )
    Mod = Mod/float(2*m_)
    if _DEBUG_:
        print("Modularity:", Mod)
    return Mod, Num_com_list


def UpdateDeg(A, nodes):
    deg_dict = {}
    n = len(nodes)  #len(A) ---> some ppl get issues when trying len() on sparse matrixes!
    B = A.sum(axis = 1)
    i = 0
    for node_id in list(nodes):
        deg_dict[node_id] = B[i, 0]
        i += 1
    return deg_dict


# This method runs GirvanNewman algorithm and find the best community split by maximizing modularity measure
def runGirvanNewman(G, Orig_deg, m_):
    #let's find the best split of the graph
    BestQ = 0.0
    Q = 0.0
    G_list = []
    Q_list = []
    Num_com_list = []
    while True:    
        CmtyGirvanNewmanStep(G)         #added G= here (Alex)
        print(G.number_of_edges())
        G_list.append(G.copy())
        print(G_list[-1].number_of_edges())
        print(list(nx.connected_components(G)))
        Q, Num_com_list = _GirvanNewmanGetModularity(G, Orig_deg, m_, Num_com_list);
        print("Modularity of decomposed G:", Q)
        Q_list.append(Q)
        if Q > BestQ:
            BestQ = Q
            Bestcomps = list(nx.connected_components(G))    #Best Split
            print("Components:", Bestcomps)
        if G.number_of_edges() == 0:
            break
    if BestQ > 0.0:
        print("Max modularity (Q):", BestQ)
        print("Graph communities:", Bestcomps)
    else:
        print("Max modularity (Q):", BestQ)
    return G_list, Q_list, Num_com_list


#G = nx.Graph()  #let's create the graph first
#buildG(G, "/content/graph.txt", ',')

if _DEBUG_:
    print('G nodes:', G.nodes())
    print('G no of nodes:', G.number_of_nodes())

n = G.number_of_nodes()    #|V|
A = nx.adj_matrix(G)    #adjacency matrix

m_ = 0.0    #the weighted version for number of edges
for i in range(0,n):
    for j in range(0,n):
        m_ += A[i,j]
m_ = m_/2.0
if _DEBUG_:
    print("m:", m_)

#calculate the weighted degree for each node
Orig_deg = {}
Orig_deg = UpdateDeg(A, G.nodes())

#run Newman alg
G_list, Q_list, Num_com_list = runGirvanNewman(G, Orig_deg, m_)

print(G_list, Q_list, Num_com_list)

nx.write_gpickle(G_list, "g_list.gpickle")

import csv

with open('Girvan_Newman_Modularities.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(Num_com_list, Q_list))
