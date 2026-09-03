# emopylab 2026
"""Multi-stage knowledge-guided evolutionary algorithm.\n\nReference:\nZ. Ding, L. Chen, D. Sun, and X. Zhang. A multi-stage knowledge-guided evolutionary algorithm for sparse multi-objective optimization problems. Swarm and Evolutionary Computation, 2022, 73: 101119.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'MSKEA': {'binary', 'constrained', 'integer', 'large', 'multi', 'real', 'sparse'}}


class MSKEA(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
