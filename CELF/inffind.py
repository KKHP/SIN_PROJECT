def inffind(graph, novert, seed_size, outdeg, model): 
    if model == "IC":
        seeds = propic(graph, novert, seed_size, outdeg) 
    else:
        seeds = proplt(graph, novert, seed_size, outdeg) 
    return seeds
