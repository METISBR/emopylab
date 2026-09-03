# emopylab 2026
"""Multi-objective evolutionary algorithm with nonzero detection.\n\nReference:\nX. Wang, R. Cheng, and Y. Jin. Sparse large-scale multiobjective optimization by identifying nonzero decision variables. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2024, 54(10): 6280-6292.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEANZD': {'constrained', 'large', 'many', 'multi', 'real', 'sparse'}}


class MOEANZD(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
