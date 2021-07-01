import random


def icpropmodel(graph, seeds):
    inf = seeds[:]
    qu = inf[:]
    while len(qu) != 0:
        node = qu.pop(0)
        for element in graph[node]:
            if element not in inf:
                probility = random.random()
                if probility <= graph[node][element]['weight']:
                    inf.append(element)
                    qu.append(element)
    noofinfl = len(inf)
    return noofinfl
