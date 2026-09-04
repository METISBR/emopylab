# emopylab 2026
"""theta-dominance based evolutionary algorithm.\n\nReference:\nY. Yuan, H. Xu, B. Wang, and X. Yao. A new dominance relation-based evolutionary algorithm for many-objective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(1): 16-37.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'tDEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class tDEA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
