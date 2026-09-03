# emopylab 2026
"""Robust multi-objective evolutionary algorithm with decision variable assortment.\n\nReference:\nJ. Liu, Y. Liu, Y. Jin, and F. Li. A decision variable assortment-based evolutionary algorithm for dominance robust multiobjective optimization. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2022, 52(5): 3360-3375.\n"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'RMOEADVA': {'integer', 'multi', 'real', 'robust'}}


class RMOEADVA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
