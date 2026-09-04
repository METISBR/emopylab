# emopylab 2026
"""Coevolutionary multimodal multi-objective evolutionary algorithm.\n\nReference:\nW. Li, X. Yao, K. Li, R. Wang, T. Zhang, and L. Wang. Coevolutionary framework for generalized multimodal multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2023, 10(7): 1544-1556.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'CoMMEA': {'binary', 'integer', 'label', 'multi', 'multimodal', 'permutation', 'real'}}


class CoMMEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
