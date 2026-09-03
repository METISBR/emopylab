# emopylab 2026
"""Sparsity-guided elitism co-evolutionary framework.\n\nReference:\nC. Wu, Y. Tian, Y. Zhang, H. Jiang, and X. Zhang. A sparsity-guided elitism co-evolutionary framework for sparse large-scale multi-objective optimization. Proceedings of the IEEE Congress on Evolutionary Computation, 2023.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'SGECF': {'binary', 'constrained', 'large', 'multi', 'real', 'sparse'}}


class SGECF(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
