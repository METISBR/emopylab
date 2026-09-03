# emopylab 2026
"""Multi-objective conjugate gradient and differential evolution algorithm.\n\nReference:\nY. Tian, H. Chen, H. Ma, X. Zhang, K. C. Tan, and Y. Jin. Integrating conjugate gradients into evolutionary algorithms for large-scale continuous multi-objective optimization. IEEE/CAA Journal of Automatica Sinica, 2022, 9(10): 1801-1817.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MOCGDE': {'constrained', 'large', 'many', 'multi', 'real'}}


class MOCGDE(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
