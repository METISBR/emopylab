# emopylab 2026
"""Paired offspring generation based constrained evolutionary algorithm.\n\nReference:\nC. He, R. Cheng, Y. Tian, X. Zhang, K. C. Tan, and Y. Jin. Paired offspring generation for constrained large-scale multiobjective optimization. IEEE Transactions on Evolutionary Computation, 2021, 25(3): 448-462. --------------------------------------------------------------------------\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'POCEA': {'constrained', 'integer', 'large', 'multi', 'real'}}


class POCEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
