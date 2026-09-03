# emopylab 2026
"""Constrained multiobjective evolutionary Bayesian optimization based on decomposition.

Reference:
Z. Zhang, Y. Wang, G.Sun, and T. Pang. A novel evolutionary Bayesian optimization algorithm based on decomposition for expensive constrained multiobjective optimization problems. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 2025.
"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'CMOEBOD': {'constrained', 'expensive', 'many', 'multi', 'real'}}


class CMOEBOD(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
