# emopylab 2026
"""Many-objective evolutionary algorithm based on dominance and.\n\nReference:\nK. Li, K. Deb, Q. Zhang, and S. Kwong. An evolutionary many-objective optimization algorithm based on dominance and decomposition. IEEE Transactions Evolutionary Computation, 2015, 19(5): 694-716.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEADD': {'binary', 'constrained', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class MOEADD(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
