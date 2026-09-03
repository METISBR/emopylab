# emopylab 2026
"""Pattern mining based multi-objective evolutionary algorithm.\n\nReference:\nY. Tian, C. Lu, X. Zhang, F. Cheng, and Y. Jin. A pattern mining based evolutionary algorithm for large-scale sparse multi-objective optimization problems. IEEE Transactions on Cybernetics, 2022, 52(7): 6784-6797.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'PMMOEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class PMMOEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
