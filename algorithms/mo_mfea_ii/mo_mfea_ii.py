# emopylab 2026
"""Multi-objective multifactorial evolutionary algorithm II.\n\nReference:\nK. K. Bali, A. Gupta, Y. Ong, and P. S. Tan. Cognizant multitasking in multiobjective multifactorial evolution: MO-MFEA-II. IEEE Transactions on Cybernetics, 2021, 51(4): 1784-1796.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOMFEAII': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MOMFEAII(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
