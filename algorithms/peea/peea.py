# emopylab 2026
"""Pareto front shape estimation based evolutionary algorithm.\n\nReference:\nL. Li, G. G. Yen, A. Sahoo, L. Chang, and T. Gu. On the estimation of pareto front and dimensional similarity in many-objective evolutionary algorithm. Information Sciences, 2021, 563: 375-400.\n"""

from __future__ import annotations

from algorithms.moo.nsga2 import NSGA2


ALGORITHM_FLAGS = {'PeEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class PeEA(NSGA2):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
