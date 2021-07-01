import time
import networkx as nx
import random
from assets import Getseeds as gs, read_data as rd

if __name__ == '__main__':
    choice = int(input('1-graph from text file \n 2-random graph \n'))
    if choice == 1:
        graph_file = input('enter the name/directory of thefile: ')
        novert, noedge, graph, outdeg = rd.read_data(graph_file)
        G = nx.DiGraph()
        G.add_nodes_from(graph)
        G.add_edges_from(((u, v, data) for u, nbrs in graph.items() for v, data in nbrs.items()))
    elif choice == 2:
        novert = int(input('enter the number of vertices for Erdos-Renghi graph: '))
        prob = float(input('enter the probability for node establishment: '))
        G = nx.fast_gnp_random_graph(novert, prob, seed=None, directed=True)
        noedge = nx.number_of_edges(G)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.random()
        outdeg = {}
        for i in G.nodes():
            outdeg[i] = G.out_degree(i)
        print(noedge)
    seed_size = int(input('enter the no of seeds required: '))
    model = input('Propagation models:\nIC - INDEPENDENT CASCADE\nLT - LINEAR THRESHOLD\nenter: ')
    start_time = time.time()
    gs.getseeds(G, novert, seed_size, outdeg, model)
    print("execution time = ", time.time() - start_time, "s")
    #nx.draw_circular(G, node_color='bisque', with_labels=True)
