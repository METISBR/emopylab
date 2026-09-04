# emopylab 2026
"""Reference point dominance-based NSGA-II.\n\nReference:\nM. Elarbi, S. Bechikh, A. Gupta, L. B. Said, and Y. S. Ong. A new decomposition-based NSGA-II for many-objective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2018, 48(7): 1191-1210.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'RPDNSGAII': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class RPDNSGAII(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
