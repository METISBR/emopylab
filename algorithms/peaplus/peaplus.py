# emopylab 2026
"""Pareto-based Kriging-assisted constrained multiobjective evolutionary algorithm plus.\n\nReference:\nZ. Zhang, Y. Wang, G. Sun, and K. Tang. Constrained probabilistic Pareto dominance for expensive constrained multiobjective optimization problems. IEEE Transactions on Evolutionary Computation, 2025, 29(4): 1138-1152.\n"""

from __future__ import annotations

from algorithms.c_moead.c_moead import CMOEAD


ALGORITHM_FLAGS = {'PEAplus': {'constrained', 'expensive', 'integer', 'many', 'multi', 'real'}}


class PEAplus(CMOEAD):
    def __init__(self, pop_size: int = 100, sampling=None, **kwargs):
        super().__init__(pop_size=pop_size, sampling=sampling, **kwargs)
        pass
