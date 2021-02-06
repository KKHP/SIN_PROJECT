def getseeds(G, novert, seed_size, outdeg, model): 
    final_seeds = []
    total_influence = 0
    final_seeds = inffind(G, novert, seed_size, outdeg, model) 
    total_influence = calculate_average(G, final_seeds, model)
    print("Hence the seeds selected for highest influence are\n ", final_seeds)
    choice=int(input('1-graph from txt file \n2-random graph \n')) 
    if choice==1:
        graph_file=input('enter the name/directory of the file: ') 
        novert, noedge, graph, outdeg = read_data(graph_file) 
        G = nx.DiGraph() 
        G.add_nodes_from(graph)
    G.add_edges_from(((u, v, data)for u, nbrs in graph.items()for v, data in nbrs.items())) 
    elif choice==2:
        novert=int(input('enter the number of vertices for Erdős-Rényi graph: ')) 
        prob=float(input('enter the probability for node establishment: ')) 
        G=nx.fast_gnp_random_graph(novert, prob, seed=None, directed=True) 
        noedge=nx.number_of_edges(G)
        for (u, v) in G.edges():
            G.edges[u,v]['weight'] = random.random()
        outdeg={}
        for i in G.nodes(): 
            outdeg[i]=G.out_degre 
            e(i)
        print(noedge)

