# emopylab 2026
"""Self-organizing multiobjective evolutionary algorithm.\n\nReference:\nH. Zhang, A. Zhou, S. Song, Q. Zhang, X. Gao, and J. Zhang. A self- organizing multiobjective evolutionary algorithm. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 792-806.\n"""

from __future__ import annotations

from algorithms.gde3.gde3 import GDE3


ALGORITHM_FLAGS = {'SMEA': {'integer', 'multi', 'real'}}


class SMEA(GDE3):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
