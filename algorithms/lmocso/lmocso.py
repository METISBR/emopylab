# emopylab 2026
"""Large-scale multi-objective competitive swarm optimization algorithm.\n\nReference:\nY. Tian, X. Zheng, X. Zhang, and Y. Jin. Efficient large-scale multi- objective optimization based on a competitive swarm optimizer. IEEE Transactions on Cybernetics, 2020, 50(8): 3696-3708.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'LMOCSO': {'constrained', 'integer', 'large', 'many', 'multi', 'real'}}


class LMOCSO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
