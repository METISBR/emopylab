# emopylab 2026
"""Dual-population with variable auxiliary population size.\n\nReference:\nJ. Liang, Z. Chen, Y. Wang, X. Ban, K. Qiao, and K. Yu. A dual-population constrained multi-objective evolutionary algorithm with variable auxiliary population size. Complex & Intelligent Systems, 2023, 9: 5907-5922.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'DPVAPS': {'constrained', 'integer', 'large', 'multi', 'real'}}


class DPVAPS(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
