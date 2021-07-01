from ICMODEL import PROPic as pic
from LTMODEL import PROPlt as plt


def inffind(graph, novert, seed_size, outdeg, model):
    if model == "IC":
        seeds = pic.propic(graph, novert, seed_size, outdeg)
    else:
        seeds = plt.proplt(graph, novert, seed_size, outdeg)
    return seeds
