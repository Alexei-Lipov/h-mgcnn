# README

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)

## Dependencies

* Python 3.7
* PyTorch 1.5
* PyTorch Geometric 1.5.0
* NetworkX 2.4
* python-louvain
* markov_clustering
* Matplotlib 3.2.1
* NumPy 1.18.4

## Code provided:

### Cluster_Tests folder:

* "Girvan_Newman_Modularity.py" - Executes the Girvan-Newman algorithm on the truncated Cora network. Outputs a list of NetworkX graphs from the dendrogram into a "g_list.gpickle" file. Outputs the number of communities and modularity for each partition into a "Girvan_Newman_Modularities.csv" file. 

* "Girvan_Newman_Runtime.py" - Returns the execution time of the Girvan-Newman algorithm on the truncated Cora network.

* "Markov_Modularity.py" - Executes the Markov Clustering algorithm on the truncated Cora network. Outputs the inflation parameter and modularity for each partition into a "Markov_Modularities.txt" file. 

* "Markov_Runtime.py" - Returns the execution time of the Girvan-Newman algorithm on the truncated Cora network.

* "Louvain_Modularity.py" - Executes the Louvain algorithm on the truncated Cora network. Outputs the dendrogram level and modularity for each partition into a "Louvain_Modularities.txt" file. 

* "Louvain_Runtime.py" - Returns the execution time of the Louvain algorithm on the truncated Cora network.

* "Cora_C_vs_K.py" - Calculates the clustering coefficients and degrees of the nodes in the truncated Cora network. Outputs a plot of these into the "C_vs_K_Plot.png" file.

### Architecture_Tests folder:

* "MGCNN_Depth_1.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the one graph convolutional layer version. 

* "MGCNN_Depth_2.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the two graph convolutional layer version. 

* "MGCNN_Depth_2_N.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the two graph convolutional layer version with noise. Alter the variable noise to control the fraction of training set nodes which are to be corrupted with random binary valued vectors. 

* "MGCNN_Depth_3.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the three graph convolutional layer version. 

* "MGCNN_Depth_4.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the four graph convolutional layer version. 

* "MGCNN_Depth_5.py" - Loads "g_list.gpickle" generated from "Girvan_Newman_Modularity.py". Executes architecture as described in the report. Returns the accuracy. To change the number of scales, alter the num_steps variable on line 16. To change the number of hidden nodes, alter the second argument of nn.Linear on line 88 and the first argument of nn.Linear on line 89. This is the five graph convolutional layer version. 
