# emopylab 2026
"""c-DPEA (Constrained dual-population evolutionary algorithm).

Reference:
M. Ming, A. Trivedi, R. Wang, and D. Srinivasan. A dual-population based evolutionary algorithm for constrained multi-objective optimization. IEEE TEC, 2021, 25(4): 739-753.
"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'cDPEA': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class cDPEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, dual_population=True, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.dual_population = dual_population
