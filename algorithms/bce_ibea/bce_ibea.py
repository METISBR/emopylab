# emopylab 2026
"""BCE-IBEA (Bi-criterion evolution based IBEA).

Reference:
M. Li, S. Yang, and X. Liu. Pareto or non-Pareto: Bi-criterion evolution in multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2016, 20(5): 645-665.
"""

from __future__ import annotations

from algorithms.e_moea.e_moea import eMOEA


ALGORITHM_FLAGS = {'BCEIBEA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class BCEIBEA(eMOEA):
    def __init__(self, pop_size: int = 100, sampling=None, kappa=0.05, **kwargs):
        if sampling is None:
            super().__init__(pop_size=pop_size, **kwargs)
        else:
            super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        self.kappa = kappa
