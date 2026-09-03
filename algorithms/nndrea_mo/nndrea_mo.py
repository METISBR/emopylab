# emopylab 2026
"""Evolutionary algorithm with neural network-based dimensionality reduction.\n\nReference:\nY. Tian, L. Wang, S. Yang, J. Ding, Y. Jin, and X. Zhang. Neural network-based dimensionality reduction for large-scale binary optimization with millions of variables. IEEE Transactions on Evolutionary Computation, 2025, 29(6): 2328-2342.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'NNDREAMO': {'binary', 'constrained', 'large', 'multi', 'sparse'}}


class NNDREAMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
