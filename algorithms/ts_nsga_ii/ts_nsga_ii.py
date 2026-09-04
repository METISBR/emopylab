# emopylab 2026
"""Two stage NSGA-II.\n\nReference:\nF. Ming, W. Gong, and L. Wang. A two-stage evolutionary algorithm with balanced convergence and diversity for many-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(10): 6222-6234.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'TSNSGAII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class TSNSGAII(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
