# emopylab 2026
"""Three-population based constrained many-objective co-evolutionary algorithm.\n\nReference:\nY. Tian, Z. Shi, Y. Zhang, L. Zhang, H. Zhang, and X. Zhang. Solving optimal power flow problems via a constrained many-objective co-evolutionary algorithm. Frontiers in Energy Research, 2023, 11: 1293193.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'TPCMaO': {'binary', 'constrained', 'integer', 'label', 'many', 'permutation', 'real'}}


class TPCMaO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
