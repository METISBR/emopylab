# emopylab 2026
from ._common import roulette_wheel_selection

def RouletteWheelSelection(N, Fitness, **kwargs):
    return roulette_wheel_selection(N, Fitness, **kwargs)
