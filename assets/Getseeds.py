from assets import choosemodel as cm
from assets import calcAvg as avg


def getseeds(G, novert, seed_size, outdeg, model):
    final_seeds = []
    total_influence = 0
    final_seeds = cm.inffind(G, novert, seed_size, outdeg, model)
    total_influence = avg.calculate_average(G, final_seeds, model)
    print("Hence the seeds selected for highest influence are\n ", final_seeds)
