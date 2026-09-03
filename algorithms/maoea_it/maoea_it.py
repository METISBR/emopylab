# emopylab 2026
"""Many-objective evolutionary algorithms based on an independent two-stage.\n\nReference:\nY. Sun, B. Xue, M. Zhang, and G. G. Yen. A new two-stage evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2019, 23(5): 748-761.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MaOEAIT': {'constrained', 'integer', 'many', 'multi', 'real'}}


class MaOEAIT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
