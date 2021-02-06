def calculate_average(graph, seeds, model): 
    if model == "IC":
        count = 0
        total_influence = 0 
        while count < 1000:
            total_influence += ICpropmodel(graph, seeds) 
            count += 1
        IC_average = total_influence/count 
        average_result = IC_average
    else:
        count = 0
        total_influence = 0 
        while count < 1000:
            total_influence += LTpropmodel(graph, seeds) 
            count += 1
        LT_average = total_influence / count 
        average_result = LT_average
    return average_result
