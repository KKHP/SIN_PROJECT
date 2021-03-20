def read_data(graph_file): 
    f1 = open(graph_file, 'r')
    first_line = f1.readline().split() 
    novert = int(first_line[0]) 
    noedge = int(first_line[1]) 
    graph = defaultdict(dict) 
    outdeg = defaultdict(int)
    for line in f1.readlines(): 
        data = line.split() 
        outdeg[int(data[0])] += 1 
        if float(data[2])>0:
            graph[int(data[0])][int(data[1])] ={'weight': float(data[2])} 
        elif float(data[2])<0:
            graph[int(data[0])][int(data[1])] ={'weight': -1*float(data[2])} 
    return novert, noedge, graph, outdeg
