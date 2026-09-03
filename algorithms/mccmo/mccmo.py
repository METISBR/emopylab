# emopylab 2026
"""Multi-population coevolutionary constrained multi-objective optimization.\n\nReference:\nJ. Zou, R. Sun, Y. Liu, Y. Hu, S. Yang, J. Zheng, and K. Li. A multi-population evolutionary algorithm using new cooperative mechanism for solving multi-objective problems with multi-constraint. IEEE Transactions on Evolutionary Computation, 2024, 28(1): 267-280.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MCCMO': {'binary', 'constrained', 'integer', 'label', 'multi', 'permutation', 'real'}}


class MCCMO(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
