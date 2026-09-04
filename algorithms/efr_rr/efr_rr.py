# emopylab 2026
"""Ensemble fitness ranking with a ranking restriction scheme.\n\nReference:\nY. Yuan, H. Xu, B. Wang, B. Zhang, and X. Yao. Balancing convergence and diversity in decomposition-based many-objective optimizers. IEEE Transactions on Evolutionary Computation, 2016, 20(2): 180-198.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'EFRRR': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class EFRRR(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
