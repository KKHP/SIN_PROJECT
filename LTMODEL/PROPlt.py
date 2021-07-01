from collections import defaultdict
import networkx as nx
from LTMODEL import LTpropmodel as ltpm

SN_INFLUENCE_PER = 0.4
CENTRALITY_PER = 0.4


def proplt(graph, novert, seed_size, outdeg):
    seeds = []
    s_n_influnece = defaultdict(float)
    G = graph
    if G.is_directed():
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.out_degree()}
        G = G.reverse()
    else:
        s = 1.0 / (len(G) - 1.0)
        degreecentrality = {n: d * s for n, d in G.degree()}
    path_length = nx.single_source_shortest_path_length
    nodes = G.nodes
    closeness_centrality = {}
    for n in nodes:
        sp = dict(path_length(G, n))
        totsp = sum(sp.values())
        if totsp > 0.0 and len(G) > 1:
            closeness_centrality[n] = (len(sp) - 1.0) / totsp
            s = (len(sp) - 1.0) / (len(G) - 1)
            closeness_centrality[n] *= s
        else:
            closeness_centrality[n] = 0.0
    while len(seeds) < seed_size:
        if len(seeds) == 0:
            for node in range(1, novert + 1):
                s_n_influnece[node] = 0
                if node in outdeg:
                    single_node = []
                    single_node.append(node)
                    s_n_influnece[node] = (s_n_influnece[node] + ltpm.ltpropmodel(graph, single_node)) / novert
                    if not closeness_centrality[node] == 0:
                        s_n_influnece[node] = s_n_influnece[node] * SN_INFLUENCE_PER + (
                                    1 / closeness_centrality[node]) * CENTRALITY_PER + degreecentrality[node] * (
                                                          1 - (CENTRALITY_PER + SN_INFLUENCE_PER))
                    else:
                        s_n_influnece[node] = s_n_influnece[node] * SN_INFLUENCE_PER + (0) * (
                                    1 - (CENTRALITY_PER + SN_INFLUENCE_PER)) + degreecentrality[node] * CENTRALITY_PER
            max_seed = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece.pop(max_seed)
            seeds.append(max_seed)
        else:
            prev_best = max(s_n_influnece, key=s_n_influnece.get)
            s_n_influnece[prev_best] = 0
            new_seeds = seeds + [prev_best]
            marginal_profit = ltpm.ltpropmodel(graph, new_seeds) - ltpm.ltpropmodel(graph, seeds)
            s_n_influnece[prev_best] = s_n_influnece[prev_best] + marginal_profit
            if not closeness_centrality[prev_best] == 0:
                s_n_influnece[prev_best] = s_n_influnece[prev_best] * SN_INFLUENCE_PER + (
                            1 / closeness_centrality[prev_best]) * CENTRALITY_PER + degreecentrality[prev_best] * (
                                                       1 - (CENTRALITY_PER + SN_INFLUENCE_PER))
            else:
                s_n_influnece[prev_best] = s_n_influnece[prev_best] * SN_INFLUENCE_PER + (0) * (
                            1 - (CENTRALITY_PER + SN_INFLUENCE_PER)) + degreecentrality[prev_best] * CENTRALITY_PER
            current_seed = max(s_n_influnece, key=s_n_influnece.get)
            if current_seed == prev_best:
                seeds.append(current_seed)
                s_n_influnece.pop(current_seed)
            else:
                continue
    return seeds
