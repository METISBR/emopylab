# emopylab 2026
"""Dynamic constrained NSGA-III.\n\nReference:\nR. Jiao, S. Zeng, C. Li, S. Yang, and Y. S. Ong. Handling constrained many-objective optimization problems via problem transformation. IEEE Transactions on Cybernetics, 2021, 51(10): 4834-4847.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DCNSGAIII': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class DCNSGAIII(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
