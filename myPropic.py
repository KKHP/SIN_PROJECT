def propic(graph, novert, seed_size, outdeg): 
    test_count = 0
    seeds = []
    s_n_influnece = defaultdict(float)
    G=graph
    if G.is_directed():
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.out_degree()} G = G.reverse()
    else:
        s = 1.0 / (len(G) - 1.0)
    degreecentrality = {n: d * s for n, d in G.degree()} path_length = nx.single_source_shortest_path_length nodes = G.nodes
    closeness_centrality = {} 
    for n in nodes:
        sp = dict(path_length(G, n)) 
        totsp = sum(sp.values())
    if totsp > 0.0 and len(G) > 1: closeness_centrality[n] = (len(sp) - 1.0) / totsp
        s = (len(sp) - 1.0) / (len(G) - 1) closeness_centrality[n] *= s
    else:
        closeness_centrality[n] = 0.0 
    while len(seeds) < seed_size:
        if len(seeds) == 0:
            for node in range(1, novert + 1): 
                s_n_influnece[node] = 0
                if node in outdeg.keys():
                    s_n_influnece[node] = (s_n_influnece[node] + ICpropmodel(graph, seeds+[node]))/novert
                    if not closeness_centrality[node]==0:
                        s_n_influnece[node]=s_n_influnece[node]*0.4+(1/closeness_centrality[node])*0.4+degree centrality[node]*0.2
                    else:
                        s_n_influnece[node]=s_n_influnece[node]*0.4+(0)*0.2+degreecentrality[node]*0.4 max_seed = max(s_n_influnece, key=s_n_influnece.get) 
                        s_n_influnece.pop(max_seed)
                        seeds.append(max_seed) 
                        test_count+=1 
                elif len(seeds)!= 0:
                    prev_best = max(s_n_influnece, key=s_n_influnece.get) 
                    s_n_influnece[prev_best] = 0
                    marginal_profit = ICpropmodel(graph, seeds + [prev_best]) - ICpropmodel(graph,seeds)
    s_n_influnece[prev_best] += marginal_profit if not closeness_centrality[prev_best]==0:
s_n_influnece[prev_best]=s_n_influnece[prev_best]*0.4+(1/closeness_centrality[prev_best
])*0.4+degreecentrality[prev_best]*0.2 else:

s_n_influnece[prev_best]=s_n_influnece[prev_best]*0.4+(0)*0.2+degreecentrality[prev_be st]*0.4
current_seed = max(s_n_influnece, key=s_n_influnece.get) if current_seed == prev_best:
seeds.append(current_seed) s_n_influnece.pop(current_seed)
else:
continu          e return seeds


#intendation under process

