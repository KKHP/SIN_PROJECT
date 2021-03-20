from collections import defaultdict import random
import time
import networkx as nx
from CELF import readData




choice = int(input('1-graph from text file \n 2-random graph \n')) 
if choice==1:
    graph_file=input('enter the name/directory of thefile: ') novert, noedge, graph, outdeg =
    read_data(graph_file) G = nx.DiGraph()
    G.add_nodes_from(graph)
    G.add_edges_from(((u, v, data)for u, nbrs in graph.items()for v, data in nbrs.items()))
elif choice==2:
    novert=int(input('enter the number of vertices for Erdős-Rényi graph: '))
    prob=float(input('enter the probability for node establishment: '))
    G=nx.fast_gnp_random_graph(novert, prob, seed=None, directed=True)
    noedge=nx.number_of_edges(G)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] =
        random.random()
    outdeg={}
    for i in G.nodes():
        outdeg[i]=G.out_degre
        e(i)
    print(noedge)
seed_size=int(input('enter the no of seeds required: '))
model=input('Propagation models:\nIC - INDEPENDENT CASCADE\nLT - LINEAR THRESHOLD\nenter: ')
start_time = time.time()
getseeds(G, novert, seed_size, outdeg, model)
print("execution time = ",time.time() - start_time,"ms")    
