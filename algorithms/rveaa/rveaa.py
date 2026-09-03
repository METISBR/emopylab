# emopylab 2026
"""RVEA embedded with the reference vector regeneration strategy.\n\nReference:\nR. Cheng, Y. Jin, M. Olhofer, and B. Sendhoff. A reference vector guided evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 773-791.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'RVEAa': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class RVEAa(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
