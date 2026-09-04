# emopylab 2026
# Reference:
# K. Deb. An efficient constraint handling method for genetic algorithms.
# Computer Methods in Applied Mechanics and Engineering, 2000, 186(2-4): 311-338.
from ._common import fitness_single


def FitnessSingle(Population):
    return fitness_single(Population)
