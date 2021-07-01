from collections import defaultdict
import random


def ltpropmodel(graph, seeds):
    inf = seeds[:]
    qu = inf[:]
    pre_node_record = defaultdict(float)
    threshold = defaultdict(float)
    while len(qu) != 0:
        node = qu.pop(0)
        for element in graph[node]:
            if element not in inf:
                if threshold[element] == 0:
                    threshold[element] = random.random()
                pre_node_record[element] = pre_node_record[element] + graph[node][element]['weight']
                if pre_node_record[element] >= threshold[element]:
                    inf.append(element)
                    qu.append(element)
    noofinfl = len(inf)
    return noofinfl
