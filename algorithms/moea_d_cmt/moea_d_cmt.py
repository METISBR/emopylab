# emopylab 2026
"""MOEA/D with competitive multitasking.\n\nReference:\nX. Chu, F. Ming, and W. Gong. Competitive multitasking for computational resource allocation in evolutionary constrained multi-objective optimization. IEEE Transactions on Evolutionary Computation, 2025, 29(3): 809-821.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOEADCMT': {'constrained', 'multi', 'real'}}


class MOEADCMT(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
