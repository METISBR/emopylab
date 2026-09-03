# emopylab 2026
"""Evolutionary algorithm with cascade clustering and reference point incremental learning.

Reference:
H. Ge, M. Zhao, L. Sun, Z. Wang, G. Tan, Q. Zhang, and C. L. P. Chen. A many-objective evolutionary algorithm with two interacting processes: Cascade clustering and reference point incremental learning. IEEE Transactions on Evolutionary Computation, 2019, 23(4): 572-586.
"""

from __future__ import annotations

from algorithms.moead_de.moead_de import MOEADDE


ALGORITHM_FLAGS = {'CLIA': {'binary', 'integer', 'label', 'many', 'multi', 'permutation', 'real'}}


class CLIA(MOEADDE):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
