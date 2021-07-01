from ICMODEL import ICpropmodel as icp
from LTMODEL import LTpropmodel as ilt

def calculate_average(graph, seeds, model):
    if model == "IC":
        count = 0
        total_influence = 0
        while count < 1000:
            total_influence += icp.icpropmodel(graph, seeds)
            count += 1
        IC_average = total_influence/count
        average_result = IC_average
    else:
        count = 0
        total_influence = 0
        while count < 1000:
            total_influence += ilt.ltpropmodel(graph, seeds)
            count += 1
        LT_average = total_influence / count
        average_result = LT_average
    return average_result