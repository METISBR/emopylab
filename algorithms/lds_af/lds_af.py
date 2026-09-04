# emopylab 2026
"""Low-dimensional surrogate aggregation function.\n\nReference:\nH. Gu, H. Wang, C. He, B. Yuan, and Y. Jin. Large-scale multiobjective evolutionary algorithm guided by low-dimensional surrogates of scalarization functions. Evolutionary Computation, 2025, 33(3): 309-334.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'LDSAF': {'expensive', 'integer', 'large', 'multi', 'real'}}


class LDSAF(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
